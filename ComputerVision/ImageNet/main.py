# Copyright 2023 Cerebras Systems Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import datetime
import gc
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import math
import sys
import time
import warnings

import torch
import torch.utils.data
from torch import nn
from utils import (
    MetricLogger,
    SmoothedValue,
    accuracy,
    get_base_parser,
    get_optimizer_and_lr_scheduler,
    get_tensorboard,
    init_distributed_mode,
    init_tensorboard,
    is_main_process,
    load_data,
    mkdir,
    num_model_parameters,
    reduce_across_processes,
    save_on_master,
    set_seed,
)

from models import resnet_basic, resnet_bottleneck

sys.path.append("../")
sys.path.append("../../")

from cbsparse.sparse.utils import (
    TrainProfiler,
    add_mask_optimizer_specific_args_cv,
    create_mask_optimizer,
)


def log_tboard_dict(log_dict, itr, pre):
    writer = get_tensorboard()
    for k, v in log_dict.items():
        writer.add_scalar(f"{pre}/{k}", v, itr)


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    args,
    scaler=None,
    mask_optimizer=None,
    profiler=None,
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter(
        "img/s", SmoothedValue(window_size=10, fmt="{value}")
    )

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        samples = image.size(0) * args.world_size

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        if profiler is not None:
            profiler.update_profile(samples)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer"s assigned params
                # if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad_norm
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad_norm
                )
            optimizer.step()

        if mask_optimizer is not None:
            mask_optimizer.step()
            if mask_optimizer.sparsity_updated:
                profiler.update_profiler_dict_sparsity(mask_optimizer)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(
            loss=loss.item(), lr=optimizer.param_groups[0]["lr"]
        )
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(
            batch_size / (time.time() - start_time)
        )

    if is_main_process():
        # metrics for training
        log_dict = {
            "acc1": metric_logger.meters["acc1"].global_avg,
            "acc5": metric_logger.meters["acc5"].global_avg,
            "loss": metric_logger.meters["loss"].global_avg,
            "lr": metric_logger.meters["lr"].value,
            "img_per_sec": metric_logger.meters["img/s"].value,
        }
        if scaler is not None:
            log_dict["loss_scale"] = scaler.get_scale()
        log_tboard_dict(log_dict, itr=epoch, pre="train")

        # metrics for efficiency
        log_dict_flops = {}
        if profiler is not None:
            log_dict_flops["avg_dense_flops"] = profiler.avg_dense_flops_used()
            if mask_optimizer is not None:
                log_dict_flops[
                    "avg_sparse_flops"
                ] = profiler.avg_sparse_flops_used()

        log_tboard_dict(log_dict_flops, itr=epoch, pre="flops")


def evaluate(
    model, criterion, data_loader, device, epoch, print_freq=500, log_suffix=""
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(
            data_loader, print_freq, header
        ):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples,"
            f" but {num_processed_samples} samples were used for the validation"
            " , which might bias the results. Try adjusting the batch size"
            " and / or the world size. Setting the world size to 1"
            " is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    if is_main_process():
        log_dict = {
            "acc1": metric_logger.meters["acc1"].global_avg,
            "acc5": metric_logger.meters["acc5"].global_avg,
            "loss": metric_logger.meters["loss"].global_avg,
        }
        log_tboard_dict(log_dict, itr=epoch, pre="eval")

    return metric_logger.acc1.global_avg


def get_model(args):
    model = None
    if "_18" in args.model or "_34" in args.model:
        # flop budget for r34 :: 7355270501
        # flop budget for r18 :: 3647128359
        model = resnet_basic.Model.get_model_from_name(args)
    elif "_50" in args.model or "_101" in args.model or "_152" in args.model:
        model = resnet_bottleneck.Model.get_model_from_name(args)
    else:
        raise ValueError("Model not supported currently !!")

    if model:
        print("-" * 60)
        print(model)
        num_params = num_model_parameters(model)
        print(f"Create {args.model} with {num_params} parameters")

    return model


def get_mask_optimizer(
    args,
    mask_optimizer,
    train_loader,
    model,
    optimizer,
    adjust_flops_budget=False,
):
    print("-" * 60)
    # Get profile for the dense model, so that we can match these FLOPs
    # with the binary search in case of sparse model
    profiler = None
    x, y = next(iter(train_loader))
    profiler = TrainProfiler(
        model,
        tuple(list(x.shape[1:])),
        as_strings=False,
        mask_optimizer=mask_optimizer,
        print_per_layer_stat=False,
        verbose=False,
        global_verbose=False,
    )

    # num_updates are used to terminate mask-updates (eg. RiGL)
    num_updates = (
        len(train_loader) * args.epochs
    ) // args.gradient_accumulation_steps

    # Binary search to maintain FLOP budget
    # "uniform" is also there since we did not sparisfy first and last layer
    if adjust_flops_budget:
        low = 0.0
        high = 0.9999
        total_flops = args.flop_budget
        tolerance = 1e-3
        sparsity_increment = 1e-4

        upper_bound = total_flops + (total_flops * tolerance)
        lower_bound = total_flops - (total_flops * tolerance)

        log_str = f"Running binary search given fixed budget: {total_flops}"
        log_str += (
            f" given min budget {lower_bound} and max budget {upper_bound}"
        )
        print(log_str)

        while low <= high:
            # start with a copy of the model
            model_copy = copy.deepcopy(model)

            mid = low + (high - low) / 2
            args.maskopt_sparsity = float(mid)

            # profile model flops / params
            mask_optimizer_test = create_mask_optimizer(
                args, model_copy, optimizer, num_updates, profiler=profiler,
            )
            sparse_flops, profiler = _get_sparse_flops(
                train_loader,
                model_copy,
                mask_optimizer_test,
                global_verbose=False,
            )

            if lower_bound <= sparse_flops <= upper_bound:
                break
            elif sparse_flops < upper_bound:
                high = mid - sparsity_increment
            else:
                low = mid + sparsity_increment

            for group in mask_optimizer_test.param_groups:
                for idx, p in enumerate(group["params"]):
                    setattr(p, "_has_rigl_backward_hook", False)

            del model_copy
            gc.collect()
            torch.cuda.empty_cache()

    mask_optimizer = create_mask_optimizer(
        args, model, optimizer, num_updates, profiler=profiler,
    )

    # Check the Sparse FLOPs vs Dense FLOPs
    sparse_flops, profiler = _get_sparse_flops(
        train_loader, model, mask_optimizer, global_verbose=True
    )

    if args.maskopt:
        gc.collect()
        torch.cuda.empty_cache()

    return mask_optimizer, profiler


def _get_sparse_flops(
    _train_loader, _model, _mask_optimizer, global_verbose=False
):
    x, y = next(iter(_train_loader))
    _train_profiler = TrainProfiler(
        _model,
        tuple(list(x.shape[1:])),
        as_strings=False,
        mask_optimizer=_mask_optimizer,
        print_per_layer_stat=False,
        verbose=False,
        global_verbose=global_verbose,
    )
    sparse_flops = _train_profiler.complexity_profile["sparse"]["flops"]
    return sparse_flops, _train_profiler


def main(args):
    if args.output_dir:
        mkdir(args.output_dir)
    init_tensorboard(os.path.join(args.output_dir, "tboard_logs"))

    init_distributed_mode(args)
    print(args)
    print("-" * 60)

    if args.seed is not None:
        _rank = args.rank if args.distributed else 0
        set_seed(seed=args.seed, rank=_rank)

    device = torch.device(args.device)

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    data_loader, data_loader_test, train_sampler, test_sampler = load_data(
        train_dir, val_dir, args
    )

    model = get_model(args)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer, lr_scheduler = get_optimizer_and_lr_scheduler(args, model)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    mask_optimizer = None
    profiler = None
    if args.maskopt:
        adjust_flops = args.sift_scaling and args.sift_family == "sparse_wide"
        mask_optimizer, profiler = get_mask_optimizer(
            args, mask_optimizer, data_loader, model, optimizer, adjust_flops,
        )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )
        model_without_ddp = model.module

    if args.resume_ckpt:
        checkpoint = torch.load(args.resume_ckpt, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])
        if mask_optimizer is not None:
            mask_optimizer.load_state_dict(checkpoint["mask_optimizer"])
        if profiler is not None:
            profiler.load_state_dict(checkpoint["profiler"])

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # determinism for loader
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=data_loader,
            device=device,
            epoch=epoch,
            args=args,
            scaler=scaler,
            mask_optimizer=mask_optimizer,
            profiler=profiler,
        )
        lr_scheduler.step()
        evaluate(model, criterion, data_loader_test, device=device, epoch=epoch)

        # save artifacts
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            if mask_optimizer is not None:
                checkpoint["mask_optimizer"] = mask_optimizer.state_dict()
            if profiler is not None:
                checkpoint["profiler"] = profiler.state_dict()

            if (epoch + 1) % 10 == 0:
                save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, f"model_{epoch}.pth"),
                )
            save_on_master(
                checkpoint, os.path.join(args.output_dir, "checkpoint.pth")
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def prepare_args_for_sparsity(args):
    if args.maskopt:
        try:
            args.maskopt_sparsity = float(args.maskopt_sparsity)
        except ValueError:
            pass
    else:
        args.maskopt_sparsity = 0.0

    if args.sift_scaling:
        if not args.maskopt:
            raise ValueError("Cannot run SIFT scaling without using sparsity")

        # we need to use round because of precision issues in subtraction / division
        # using standard python. For eg, if use int instead of round, at 99% sparsity
        # we get base_scaling = 99, whereas we want it to be 100
        base_scaling = round(1 / (1.0 - args.maskopt_sparsity))
        args.base_scaling = base_scaling
        if args.sift_family == "sparse_wide":
            args.scaling_factor = math.sqrt(base_scaling)
            assert args.block_name in [
                "Block",
                "PreActBlock",
            ], f"expected one of Block or PreActBlock, got {args.block_name}"
        elif args.sift_family == "sparse_parallel":
            args.scaling_factor = base_scaling
            assert (
                args.block_name == "BlockISOFlopParallel"
            ), f"expected BlockISOFlopParallel, got {args.block_name}"


def get_args():
    parser = get_base_parser()
    temp_args, _ = parser.parse_known_args()
    if temp_args.maskopt:
        parser = add_mask_optimizer_specific_args_cv(parser)

    args = parser.parse_args()
    prepare_args_for_sparsity(args)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
