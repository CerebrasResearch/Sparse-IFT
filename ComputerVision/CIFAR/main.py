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
import argparse
import copy
import gc
import os
import sys
import time
import copy
import warnings

sys.path.append('../')
sys.path.append('../../')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import torch.nn as nn
from utils import (
    AverageMeter,
    accuracy,
    compute_explicit_weight_decay_low_rank,
    get_data_loaders_for_runtime,
    get_optimizer_and_lr_scheduler,
    load_checkpoint,
    num_model_parameters,
    save_checkpoint,
    set_seed,
)

from models import cifar_resnet18

from cbsparse.sparse.utils import (
    TrainProfiler,
    add_mask_optimizer_specific_args_cv,
    create_mask_optimizer,
)
from cbutils.tboard_utils import get_tensorboard, init_tensorboard


warnings.filterwarnings("ignore", category=UserWarning)


def log_tboard_dict(log_dict, itr, pre, post=''):
    writer = get_tensorboard()
    for k, v in log_dict.items():
        writer.add_scalar(f'{pre}/{k}{post}', v, itr)


def train(
    args,
    model,
    train_loader,
    optimizer,
    train_loss_fn,
    epoch,
    global_step=0,
    mask_optimizer=None,
    train_profiler=None,
):
    model.train()

    ## all stats / meters
    samples_per_itr = args.batch_size
    losses_m = AverageMeter()
    acc1_m = AverageMeter()
    acc5_m = AverageMeter()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = model(data)

        loss = train_loss_fn(output, target)
        if args.compute_joint_wd:
            assert (
                args.uv_layer == 'linear'
                or args.block_name == 'BlockISOFlopDoping'
            )
            loss_wd = compute_explicit_weight_decay_low_rank(
                model,
                args.weight_decay,
                no_wd_bn_bias=True,
                apply_on_folded=True,
            )
            loss = loss + loss_wd

        acc1, acc5 = accuracy(
            output,
            target.argmax(axis=1) if len(target.shape) > 1 else target,
            topk=(1, 5),
        )

        loss.backward()
        optimizer.step()

        if mask_optimizer is not None:
            mask_optimizer.step()

        if train_profiler is not None:
            train_profiler.update_profile(samples_per_itr)

            if mask_optimizer is not None and mask_optimizer.sparsity_updated:
                train_profiler.update_profiler_dict_sparsity(mask_optimizer)

        losses_m.update(loss.item(), data.size(0))
        acc1_m.update(acc1.item(), data.size(0))
        acc5_m.update(acc5.item(), data.size(0))

        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{:>5}/{} ({:2.0f}%)] '
                'Loss: {:.6f}, Accuracy: {:.3f}%'.format(
                    epoch,
                    batch_idx * len(data),
                    args.train_data_len,
                    100.0 * batch_idx / train_loader.__len__(),
                    loss.item(),
                    acc1.item(),
                )
            )

            if args.log_tboard:
                log_dict = {
                    'acc1': acc1_m.avg,
                    'acc5': acc5_m.avg,
                    'loss_avg': losses_m.avg,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                }
                if train_profiler is not None:
                    dflop = train_profiler.dense_flops_used()
                    log_tboard_dict(
                        log_dict, dflop, 'train_flop', post='_per_dflop'
                    )
                    if mask_optimizer is not None:
                        sflop = train_profiler.sparse_flops_used()
                        log_tboard_dict(
                            log_dict, sflop, 'train_flop', post='_per_sflop'
                        )

                    log_dict['dflop'] = dflop
                    log_dict[
                        'avg_dflop_persample'
                    ] = train_profiler.avg_dense_flops_used()
                    if mask_optimizer is not None:
                        log_dict[
                            'avg_sflop_persample'
                        ] = train_profiler.avg_sparse_flops_used()
                        log_dict['sflop'] = sflop

                for idx, param_group in enumerate(optimizer.param_groups):
                    log_dict[f'lr_g{idx}'] = param_group['lr']

                if mask_optimizer is not None:
                    mask_dict = mask_optimizer.get_logging_dict()
                    log_tboard_dict(mask_dict, global_step, 'maskopt_metrics')

                log_tboard_dict(log_dict, global_step, 'train')

        global_step += 1

    if args.log_tboard:
        log_epoch_dict = {
            'acc1': acc1_m.avg,
            'acc5': acc5_m.avg,
            'total_loss': losses_m.avg,
        }
        log_tboard_dict(log_epoch_dict, epoch, 'train', '_ep')

    # training summary
    print(
        '\nTrain summary for Epoch {} -- '
        'Total Loss: {loss.avg:>6.4f}, '
        'Acc@1: {acc1.avg:>7.4f}'.format(epoch, loss=losses_m, acc1=acc1_m,)
    )
    return global_step


def evaluate(
    args, model, test_loader, validate_loss_fn, epoch=0, is_test_set=False
):
    model.eval()

    losses_m = AverageMeter()
    acc1_m = AverageMeter()
    acc5_m = AverageMeter()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss = validate_loss_fn(output, target)
            acc1, acc5 = accuracy(
                output,
                target.argmax(axis=1) if len(target.shape) > 1 else target,
                topk=(1, 5),
            )

            losses_m.update(test_loss.item(), data.size(0))
            acc1_m.update(acc1.item(), output.size(0))
            acc5_m.update(acc5.item(), output.size(0))

    log_name = 'Test ' if is_test_set else 'Eval '
    print(
        '{} summary for Epoch {} -- '
        'Total Loss: {loss.avg:>6.4f}, '
        'Acc@1: {acc1.avg:>7.4f}'.format(
            log_name, epoch, loss=losses_m, acc1=acc1_m,
        )
    )

    if args.log_tboard:
        log_dict = {
            'acc1': acc1_m.avg,
            'acc5': acc5_m.avg,
            'loss': losses_m.avg,
        }
        log_tboard_dict(log_dict, epoch, 'val', '_ep')

    return acc1_m.avg


def test(args, model, test_loader, model_ckpt):
    """Example command to ensure test for model / fused model. The first
    command will save the checkpoints to a local folder called `save` which we
    can use to then run this test.

    $ python main.py --model cifar_resnet_18 --epochs 2

    ```
    Model test summary -- Acc@1: 26.1200  Acc@5: 56.8900
    ```

    Note that this will run the test and then exit -- which is desirable
    behavior for testing.
    """

    model.eval()
    load_checkpoint(model, model_ckpt)

    acc1_m = AverageMeter()
    acc5_m = AverageMeter()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)

            acc1, acc5 = accuracy(
                output,
                target.argmax(axis=1) if len(target.shape) > 1 else target,
                topk=(1, 5),
            )
            acc1_m.update(acc1.item(), output.size(0))
            acc5_m.update(acc5.item(), output.size(0))

    print('-' * 60)
    print(
        'Model test summary -- '
        'Acc@1: {acc1.avg:>7.4f}  Acc@5: {acc5.avg:>7.4f}'.format(
            acc1=acc1_m, acc5=acc5_m,
        )
    )

    print('-' * 60)


def get_mask_optimizer(
    args,
    mask_optimizer,
    train_loader,
    model,
    optimizer,
    match_target_flops=False,
):
    '''
    Args:
        match_target_flops (bool) : If True, we will perform binary search to match
            the target FLOPs. Note, doing so, discards the passed in sparsity.
    '''

    print('-' * 60)
    # Get profile for the dense model, so that we can match these FLOPs with the binary search in case of sparse model
    dense_train_profiler = None
    x, y = next(iter(train_loader))
    dense_train_profiler = TrainProfiler(
        model,
        tuple(list(x.shape[1:])),
        as_strings=False,
        mask_optimizer=mask_optimizer,
        print_per_layer_stat=False,
        verbose=False,
        global_verbose=False,
    )
    # dense_train_profiler is used to compute ERK FLOPs later in MaskOpt

    # num_updates are used to terminate mask-updates (eg. RiGL)
    num_updates = (
        len(train_loader) * args.epochs
    ) // args.gradient_accumulation_steps

    pai_masks = []  # pruning at init masks

    if args.maskopt_mask_distribution in ['snip', 'grasp', 'force']:
        args.maskopt_mask_init_method = args.maskopt_mask_distribution

    # Binary search to maintain FLOP budget
    # 'uniform' is also there since we did not sparisfy first and last layer
    if match_target_flops:
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
                args,
                model_copy,
                optimizer,
                num_updates,
                profiler=dense_train_profiler,
            )
            sparse_flops = _get_sparse_flops(
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
                for idx, p in enumerate(group['params']):
                    setattr(p, '_has_rigl_backward_hook', False)

            del model_copy
            gc.collect()
            torch.cuda.empty_cache()

    mask_optimizer = create_mask_optimizer(
        args, model, optimizer, num_updates, profiler=dense_train_profiler,
    )

    # Check the Sparse FLOPs vs Dense FLOPs
    sparse_flops = _get_sparse_flops(
        train_loader, model, mask_optimizer, global_verbose=True
    )

    if args.maskopt:
        del pai_masks
        gc.collect()
        torch.cuda.empty_cache()

    return mask_optimizer


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
    return sparse_flops


def load_model_opt_state_for_resume(
    args, model, optimizer, mask_optimizer, global_step
):
    if os.path.isfile(args.resume):
        print("loading checkpoint '{}'".format(args.resume))

        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["global_step"]

        if checkpoint["mask_optimizer"] and not args.dense_warmup:
            mask_optimizer.load_state_dict(checkpoint["mask_optimizer"])

        print(
            f"loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
        )
    else:
        print("no checkpoint found at '{}'".format(args.resume))
    return global_step


def get_model(args):
    model = None
    if "_18" in args.model or "_34" in args.model:
        model = cifar_resnet18.Model.get_model_from_name(args)
    else:
        raise ValueError("Model not supported currently !!")

    if model:
        print("-" * 60)
        model = model.to(args.device)
        num_params = num_model_parameters(model)
        print(model)
        print(f"Create {args.model} with {num_params} parameters")

    return model


def prepare_args_for_sparsity(args):
    if args.maskopt:
        try:
            args.maskopt_sparsity = float(args.maskopt_sparsity)
        except ValueError:
            pass
    else:
        args.maskopt_sparsity = 0.0

    if args.sift_scaling:
        if not args.maskopt and args.sift_family not in  ['sparse_factorized', 'sparse_wide_factorized']:
            raise ValueError("Cannot run SIFT scaling without using sparsity")

        # we need to use round because of precision issues in subtraction / division
        # using standard python. For eg, if use int instead of round, at 99% sparsity
        # we get base_scaling = 99, whereas we want it to be 100
        if args.maskopt_sparsity:
            args.base_scaling = round(1 / (1.0 - args.maskopt_sparsity))
        else:
            args.base_scaling = round(1/ (1.0 - args.sparse_width_scaling))

        if args.sift_family == "sparse_wide":
            assert args.block_name in [
                "Block",
                "PreActBlock",
            ], f"expected one of Block or PreActBlock, got {args.block_name}"


def main():
    # fmt: off
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Sparse Training Runs')

    # Runtime
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--output_dir', type=str, default="save/",
                        help='path to save the final model')
    parser.add_argument('--save_every_epoch', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log_tboard', action='store_false', default=True,
                        help='log training and validation metrics to tensorbaord')
    parser.add_argument('--resume', type=str, help='checkpoint to resume run from')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='epoch to start training runs from')
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help='gradient accumulation (default: 1')

    # Data
    parser.add_argument('--data', type=str, default='cifar100')
    parser.add_argument('--data_dir', type=str, default='/cb/datasets/cv/scratch/cifar/',
                        help='data dir for base dataset, defaults to CIFAR100')
    parser.add_argument('--input_size', default=[3, 32, 32], nargs=3, type=int, metavar='N N N',
                        help='Input all image dimensions (d h w, e.g. \
                        --input_size 3 224 224), uses model default if empty')
    parser.add_argument('--num_classes', type=int, default=100, metavar='N',
                        help='number of label classes (Model default if None)')
    parser.add_argument('--max_threads', type=int, default=4, help='How many threads to use for data loading.')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='label smoothing coefficient for training / validation')
    parser.add_argument('--validation_split', type=float, default=0.0,
                        help='Portion of train dataset to split as validation')

    # Model
    parser.add_argument('--model', type=str, default='cifar_resnet_18',
                        help='model to train')
    parser.add_argument('--block_name', type=str, default='Block',
                         choices=['Block', 'PreActBlock', 'BlockISOFlopParallel', 'BlockISOFlopFactorized', 'BlockISOFlopWideFactorized', 'BlockISOFlopDoping'],
                         help='Choose the block in ResNet architecture. Default is the original block.')
    parser.add_argument('--num_parallel_branch', type=int, default=1,
                        help='Num parallel branches in ISO-FLOP parallel block.')
    parser.add_argument('--zero_init_residual', action='store_false', default=True,
                        help='Initialize layer before residuals with zero weights')

    # Optimizer
    parser.add_argument("--optimizer", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--nesterov", action='store_false', default=True,
                        help='apply nesterov for sgd optimizer')
    parser.add_argument("--filter_norm_and_bias", action='store_true', default=False,
                        help='filter parameters based on weight decays')
    parser.add_argument('--weight_decay', type=float, default=5.0e-4,
                        help='set overall weight decay for training')
    parser.add_argument('--compute_joint_wd', action='store_true', default=False,
                        help='enables explicit wd computation')

    # Learning Rate
    parser.add_argument('--lr_scheduler', default='multistep', type=str,
                        help='the lr scheduler (default: cosineannealinglr)')
    parser.add_argument('--lr_step_size', default=30, type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr_min', default=0.0, type=float,
                        help='minimum lr of lr schedule (default: 0.0)')
    parser.add_argument('--lr_milestones', nargs='+', type=int,
                        help='milestones for multisteplr')
    parser.add_argument("--lr_warmup_epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr_warmup_method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr_warmup_decay", default=0.01, type=float, help="the decay for lr")

    # RigL args
    parser.add_argument('--maskopt', choices=['rigl', 'set', 'static'], default=None,
                        help='enable dynamic mask optimization (default: None)')
    parser.add_argument('--dense_warmup', action='store_true', default=False,
                        help='enables dense warmup')

    # Pruning at Init
    parser.add_argument('--num_steps', default=1, type=int,
                        help='number of steps for pruning at init (default: 1')
    parser.add_argument('--flop_budget', type=float, default=None)
    # 138538103 for MobileNetv2

    # Testing only
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='run test only and return')
    parser.add_argument('--test_checkpoint', type=str, default='',
                        help='checkpoint to run test on')

    # Sift args
    parser.add_argument('--sift_scaling', action='store_true', default=False,
            help='Run sift for imagenet models')
    parser.add_argument('--sift_family', type=str, default=None,
            help='Which sift family to run')
    parser.add_argument('--sparse_u_block', action='store_true', default=False,
            help='Make only the U block sparse for SF family')
    parser.add_argument('--uv_layer', type=str, default='linear',
            help='Type of intermediate layer between U & V in SF family')
    parser.add_argument('--sp_use_internal_relu6', action='store_true', default=False,
            help='Use ReLU6 internally in the network')
    parser.add_argument('--sp_use_layer_specialization', action='store_true', default=False,
            help='Use specialized version of Sparse Parallel, where there is no sparsity in depthwise convs')
    parser.add_argument('--sparse_width_scaling', default=0.0, type=float,
                        help='sparsity used to get width scaling factor. Note. same as maskopt_sparsity.')
    # fmt: on

    temp_args, _ = parser.parse_known_args()
    if temp_args.maskopt:
        parser = add_mask_optimizer_specific_args_cv(parser)

    args = parser.parse_args()
    prepare_args_for_sparsity(args)
    print(f"Training with args :: {args}")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.log_tboard:
        init_tensorboard(os.path.join(args.output_dir, "tboard_logs"))
    os.makedirs(os.path.join(args.output_dir, "checkpoints"))

    # fix random seed for reproducibility
    set_seed(args.seed)

    # data loaders
    train_loader, valid_loader, test_loader = get_data_loaders_for_runtime(args)
    loader_eval = valid_loader if valid_loader is not None else test_loader
    # models
    model = get_model(args)

    # Debugging purposes.
    if args.sift_family in ['sparse_factorized', 'sparse_wide_factorized'] and args.maskopt_sparsity >= 0.0:
        # Get profile for the dense model, so that we can match these FLOPs with the binary search in case of sparse model
        dense_train_profiler = None
        x, y = next(iter(train_loader))
        dense_train_profiler = TrainProfiler(
            model,
            tuple(list(x.shape[1:])),
            as_strings=False,
            mask_optimizer=None,
            print_per_layer_stat=True,
            verbose=True,
            global_verbose=True,
        )
        print(args.flop_budget)

    # run testing if needed with given `test_checkpoint`
    if args.test_only:
        test(args, model, loader_eval, args.test_checkpoint)
        return

    # optimizer, learning rate scheduler, mask_optimizer
    optimizer, lr_scheduler = get_optimizer_and_lr_scheduler(args, model)
    mask_optimizer = None
    if args.maskopt:
        adjust_flops = args.sift_scaling and args.sift_family == "sparse_wide"
        mask_optimizer = get_mask_optimizer(
            args, mask_optimizer, train_loader, model, optimizer, adjust_flops,
        )

    # Get global step if resuming
    global_step = 0
    if args.resume:
        global_step = load_model_opt_state_for_resume(
            args, model, optimizer, mask_optimizer, global_step
        )

    train_profiler = None
    # loss functions
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    validate_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.device.type == "cuda":
        train_loss_fn = train_loss_fn.cuda()
        validate_loss_fn = validate_loss_fn.cuda()

    best_acc = 0.0
    last_acc = 0.0
    for epoch in range(args.start_epoch, args.epochs * args.multiplier):
        print("-" * 60)
        t0 = time.time()
        global_step = train(
            args,
            model,
            train_loader,
            optimizer,
            train_loss_fn,
            epoch,
            global_step,
            mask_optimizer=mask_optimizer,
            train_profiler=train_profiler,
        )
        lr_scheduler.step(epoch)

        val_acc = evaluate(
            args, model, loader_eval, validate_loss_fn, epoch, False
        )

        if epoch % args.save_every_epoch == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "mask_optimizer": mask_optimizer.state_dict()
                    if mask_optimizer
                    else None,
                },
                filename=os.path.join(
                    args.output_dir,
                    "checkpoints",
                    "model_epoch_{}.pth".format(epoch),
                ),
            )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "mask_optimizer": mask_optimizer.state_dict()
                    if mask_optimizer
                    else None,
                },
                filename=os.path.join(
                    args.output_dir, "checkpoints", "model_best.pth"
                ),
            )

        last_acc = val_acc
        print(f"\nTime taken for epoch {epoch}: {time.time() - t0} seconds.")

    print("-" * 60)
    print(f"\n*** Last epoch accuracy metric: {last_acc} ***")
    print(f"*** Best epoch accuracy metric: {best_acc} ***")
    print("-" * 60)


if __name__ == '__main__':
    main()
