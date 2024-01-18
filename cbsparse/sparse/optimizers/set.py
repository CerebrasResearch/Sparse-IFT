"""
Released under BSD 3-Clause License,
Copyright (c) 2021 Cerebras Systems Inc.
All rights reserved.
"""
import torch
from torch.optim.optimizer import required

from .dynamic import DynamicMaskOptimizer


class SETMaskOptimizer(DynamicMaskOptimizer):
    r"""Implements SET.

    `Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science`__.

    Args:
        params (iterable): iterable of parameters to sparsify or dicts defining
            parameter groups to sparsify
        names (iterable): iterable of names to sparsify or dicts defining
            name groups to sparsify
        flops (iterable): iterable of flops to sparsify or dicts defining
            flop groups to sparsify
        total_flops (int): total flops of model to sparsify
        optimizers (tuple, list(tuple)): a tuple or list of tuple where the
            where the first element of the tuple is the optimizer and the second
            is a list of trackers to mask
        sparsity (float): target sparsity
        begin_iteration (int): when to begin sparsity updates
        end_iteration (float): when to end sparsity updates
        num_updates (int): number of update steps to run
        frequency (int): how frequently to update sparsity
        drop_fraction (float): how much to drop weights during update
        drop_fraction_anneal (str): which schedule to anneal drop fraction with
        mask_distribution (str): description of how to distribute saprsity
        mask_init_method (str): method by which masks are initialized
        erk_power_scale (float): erk power scale if mask_distribution is erk
        mask_metrics (bool): whether to enable metrics for dynamic runs
        grad_accum_steps (int): number of gradient accumulation steps
        config (class): class for extra configs to pass in for models
            can be model dependent. Currently, accepts BertConfig.
        verbose (bool): print extra information during run, defaults to False

    Example:
        >>> optimizer = torch.optim.SGD(
                model.parameters(), lr=0.1, momentum=0.9
            )
        >>> mask_opt = SETMaskOptimizer(
                [p for n,p in model.named_parameters() if should_sparsify(n,p)],
                sparsity=0.5,
                frequency=100
            )
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
        >>> mask_opt.step()

    __ https://www.nature.com/articles/s41467-018-04316-3

    """

    def __init__(
        self,
        params,
        names,
        flops,
        total_flops,
        optimizers=[],
        sparsity=required,
        begin_iteration=0,
        end_iteration=None,
        num_updates=None,
        frequency=100,
        drop_fraction=0.3,
        drop_fraction_anneal='cosine',
        mask_distribution='uniform',
        mask_init_method='random',
        erk_power_scale=1.0,
        mask_metrics=False,
        grad_accum_steps=1,
        config=None,
        verbose=False,
        **kwargs,
    ):
        super().__init__(
            params=params,
            names=names,
            flops=flops,
            total_flops=total_flops,
            optimizers=optimizers,
            sparsity=sparsity,
            begin_iteration=begin_iteration,
            end_iteration=end_iteration,
            num_updates=num_updates,
            frequency=frequency,
            drop_fraction=drop_fraction,
            drop_fraction_anneal=drop_fraction_anneal,
            mask_distribution=mask_distribution,
            mask_init_method=mask_init_method,
            erk_power_scale=erk_power_scale,
            mask_metrics=mask_metrics,
            grad_accum_steps=grad_accum_steps,
            config=config,
            verbose=verbose,
            **kwargs,
        )

    @torch.no_grad()
    def update_mask(self, p, fused_p, drop_fraction, layer_idx=None):
        layer_name = self.names[layer_idx]
        if self.mask_init_method in ['random_balanced', 'topk_balanced']:
            head_layer_masks = []
            head_new_masks = []
            head_dim = int(p.shape[0] / self.config.num_attention_heads)

            # For now, we don't have dense output projection in the RigL drop/regrow.
            layer_type = layer_name.split('.')[-2]
            if layer_type in ['query', 'key', 'value']:
                # attention_head_size = hidden_size / num_attention_heads (e.g., 768 / 12 = 64)
                # attention matrix (q, k, v) shape = [hidden_size,
                #               num_attention_heads * attention_head_size] (e.g., [768, 12*64])

                for num_head in range(self.config.num_attention_heads):
                    _start = head_dim * num_head
                    _end = head_dim * (num_head + 1)
                    head = p[_start:_end, :]

                    head_score = torch.abs(head)
                    head_layer_mask_dropped, n_prune = self.drop_minimum(
                        head_score,
                        self.backward_masks[layer_idx][_start:_end, :],
                        drop_fraction,
                    )

                    # grow number of params to target sparsity
                    density_t = 1 - self.state[p]['sparsity']
                    n_ones_t = (
                        density_t
                        * self.state[p]['mask'][_start:_end, :].numel()
                    )
                    n_ones_c = head_layer_mask_dropped.count_nonzero()
                    n_grow = int(n_ones_t - n_ones_c)
                    n_grow = max(n_grow, 0)

                    # grow drop_fraction ind with largest grad
                    # Randomly revive n_grow many connections from non-existing
                    # connections.
                    head_score_grow = head.new_empty(head.shape).uniform_()
                    head_score_grow *= ~head_layer_mask_dropped
                    head_layer_mask, head_new_mask = self.grow_maximum(
                        head_score_grow, head_layer_mask_dropped, n_grow
                    )

                    head_layer_masks.append(head_layer_mask)
                    head_new_masks.append(head_new_mask)

                layer_mask = torch.concat(head_layer_masks, dim=0).contiguous()
                new_mask = torch.concat(head_new_masks, dim=0).contiguous()

            else:
                # drop lowest drop_fraction
                score = torch.abs(p)
                layer_mask_dropped, n_prune = self.drop_minimum(
                    score, self.backward_masks[layer_idx], drop_fraction,
                )

                # grow number of params to target sparsity
                density_t = 1 - self.state[p]['sparsity']
                n_ones_t = density_t * self.state[p]['mask'].numel()
                n_ones_c = layer_mask_dropped.count_nonzero()
                n_grow = int(n_ones_t - n_ones_c)
                n_grow = max(n_grow, 0)

                # grow drop_fraction ind with largest grad
                # Randomly revive n_grow many connections from non-existing
                # connections.
                score_grow = p.new_empty(p.shape).uniform_()
                score_grow *= ~layer_mask_dropped
                layer_mask, new_mask = self.grow_maximum(
                    score_grow, layer_mask_dropped, n_grow
                )

        else:
            # drop lowest drop_fraction
            score = None
            if torch.is_tensor(fused_p):
                score = torch.abs(fused_p)
            else:
                score = torch.abs(p)

            layer_mask_dropped, n_prune = self.drop_minimum(
                score, self.backward_masks[layer_idx], drop_fraction,
            )

            assert torch.is_tensor(
                score
            ), 'Did not get a tensor for the dropped weights, recheck implemntation.'

            # grow number of params to target sparsity
            density_t = 1 - self.state[p]['sparsity']
            n_ones_t = density_t * self.state[p]['mask'].numel()
            n_ones_c = layer_mask_dropped.count_nonzero()
            n_grow = int(n_ones_t - n_ones_c)
            n_grow = max(n_grow, 0)

            # grow drop_fraction ind with largest grad
            # Randomly revive n_grow many connections from non-existing
            # connections.
            score_grow = p.new_empty(p.shape).uniform_()
            score_grow *= ~layer_mask_dropped
            layer_mask, new_mask = self.grow_maximum(
                score_grow, layer_mask_dropped, n_grow
            )

        if torch.distributed.is_initialized():
            torch.distributed.broadcast(layer_mask, src=0)
            torch.distributed.broadcast(new_mask, src=0)

        self.state[p]['mask'] = layer_mask
        if self.defaults['mask_metrics']:
            self.update_fired_mask(p)
            self.update_accumulated_mask(p)

        return new_mask
