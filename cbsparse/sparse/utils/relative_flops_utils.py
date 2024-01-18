"""
From output of train_profiler, get the actual number of FLOPs used for

- dense training
- static sparse training
- rigl training

- relative reduction in FLOPs for sparse compared to dense training
"""


def _rigl_per_step(sparse_flops, dense_flops, delta_t):
    with_update = (
        (3 * sparse_flops * delta_t) + (2 * sparse_flops + dense_flops)
    ) / (delta_t + 1)
    without_update = 3 * sparse_flops

    return with_update, without_update


def _static_per_step(sparse_flops):
    return 3 * sparse_flops


def _dense_per_step(dense_flops):
    return 3 * dense_flops


def _static_total_flops(sparse_flops, batch_size, total_iterations):
    per_step_flops = _static_per_step(sparse_flops)
    return per_step_flops * batch_size * total_iterations


def _rigl_total_flops(
    sparse_flops,
    dense_flops,
    delta_t,
    batch_size,
    total_iterations,
    rigl_end_fraction=0.8,
):
    update_iters = total_iterations * float(rigl_end_fraction)
    non_update_iters = total_iterations * float(1.0 - rigl_end_fraction)

    with_update, without_update = _rigl_per_step(
        sparse_flops, dense_flops, delta_t
    )

    return batch_size * (
        (with_update * update_iters) + (without_update * non_update_iters)
    )


def _dense_total_flops(dense_flops, batch_size, total_iterations):
    per_step_flops = _dense_per_step(dense_flops)
    return per_step_flops * batch_size * total_iterations


def print_statistics(
    sparsity_levels,
    sparsity_flops,
    dense_flops,
    batch_size,
    total_iterations,
    sparsity_distribution=None,
):
    total_dense_flops = _dense_total_flops(
        dense_flops, batch_size, total_iterations
    )
    print(f'\nTotal dense FLOPs {total_dense_flops}')

    print(f'\nFor {sparsity_distribution} distribution')
    print('=' * 50)

    for s, _flops in zip(sparsity_levels, sparsity_flops):
        total_static_flops = _static_total_flops(
            _flops, batch_size, total_iterations
        )
        total_rigl_flops = _rigl_total_flops(
            _flops,
            dense_flops,
            delta_t=100,
            batch_size=batch_size,
            total_iterations=total_iterations,
            rigl_end_fraction=0.8,
        )

        print(f'\n@ sparsity {s}')
        print(f'~~~~~~~~~~~~~~~~~~~')
        print(f'Static FLOPs: {total_static_flops}')
        print(
            f'Relative reduction in FLOPs for static:'
            + f' {(total_dense_flops / total_static_flops)}'
        )
        print('')
        print(f'RigL FLOPs: {total_rigl_flops}')
        print(
            f'Relative reduction in FLOPs for rigl:'
            + f' {(total_dense_flops/ total_rigl_flops)}'
        )
        