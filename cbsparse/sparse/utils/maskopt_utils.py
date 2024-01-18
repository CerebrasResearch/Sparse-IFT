import torch

from ..optimizers import (
    RigLMaskOptimizer,
    SETMaskOptimizer,
    StaticMaskOptimizer,
    optimizer_trackers,
)
from .should_sparsify_utils import get_should_sparsify


@torch.no_grad()
def get_required_args(model, should_sparsify, dense_train_profiler):
    _params = []
    _names = []
    _flops = []

    _total_params = 0.0
    for name, module in model.named_modules():
        params = module._parameters
        if not params:
            continue

        for k, v in params.items():
            full_name = name + "." + k
            if v is None:
                continue
            elif v is not None and not v.requires_grad:
                continue

            _total_params += v.numel()

            if should_sparsify(full_name, v):
                _params.append(v)
                _names.append(full_name)

                if dense_train_profiler is not None:
                    flops = 0.0
                    profile = dense_train_profiler.complexity_profile[
                        'profiler_dict'
                    ][module]
                    for op in profile.values():
                        if 'sparse' in op and op['sparse'] is not None:
                            op = op['sparse']
                        flops += (
                            op['adds']
                            + op['multiplies']
                            + op['logical']
                            + op['other']
                        )
                    _flops.append(flops)

    _total_flops = None
    if dense_train_profiler is not None:
        _total_flops = dense_train_profiler.complexity_profile['flops']

    return _params, _total_params, _names, _flops, _total_flops


def create_mask_optimizer(
    args,
    model,
    optimizer,
    num_updates,
    config=None,
    profiler=None,
    get_sparsify_fn=get_should_sparsify,
):
    sparse_args = {
        'sparsity': args.maskopt_sparsity,
        'mask_distribution': args.maskopt_mask_distribution,
        'mask_init_method': args.maskopt_mask_init_method,
        'erk_power_scale': args.maskopt_erk_power_scale,
        'num_updates': num_updates,
        'grad_accum_steps': args.gradient_accumulation_steps,
        'verbose': args.maskopt_verbose,
    }

    if args.maskopt in ['rigl', 'set']:
        dynamic_sparse_args = {
            'begin_iteration': args.maskopt_begin_iteration,
            'end_iteration': args.maskopt_end_iteration,
            'frequency': args.maskopt_frequency,
            'drop_fraction': args.maskopt_drop_fraction,
            'drop_fraction_anneal': args.maskopt_drop_fraction_anneal,
            'mask_metrics': args.maskopt_mask_metrics,
        }
        sparse_args.update(dynamic_sparse_args)

    should_sparsify = get_sparsify_fn(args, model)

    if args.maskopt == 'rigl':
        MaskOptimizerClass = RigLMaskOptimizer
    elif args.maskopt == 'set':
        MaskOptimizerClass = SETMaskOptimizer
    elif args.maskopt == 'static':
        MaskOptimizerClass = StaticMaskOptimizer
    else:
        raise ValueError(
            f"{args.maskopt} not implemented currently, either"
            + f" implement it and include here, or use one of rigl, set,"
            + f" static."
        )

    _params, _total_params, _names, _flops, _total_flops = get_required_args(
        model, should_sparsify, dense_train_profiler=None
    )
    sparse_args.update({'total_params': _total_params})

    mask_optimizer = MaskOptimizerClass(
        params=_params,
        names=_names,
        flops=_flops,
        total_flops=_total_flops,
        optimizers=(optimizer, optimizer_trackers(optimizer)),
        config=config,
        **sparse_args,
    )

    return mask_optimizer
