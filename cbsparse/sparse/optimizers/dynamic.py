"""
Released under BSD 3-Clause License,
Copyright (c) 2021 Cerebras Systems Inc.
All rights reserved.
"""
from abc import ABC, abstractmethod
from math import cos, exp, pi

import torch
from torch.optim.optimizer import required

from .static import StaticMaskOptimizer
from .utils import IndexMaskHook, extract_number, get_named_parameters


class DynamicMaskOptimizer(StaticMaskOptimizer, ABC):
    r"""Abstract base class for a dynamic mask optimizer. Subclasses must
        implement the update_mask function.

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
        >>> mask_opt = DynamicMaskOptimizer(
                [p for n,p in model.named_parameters() if should_sparsify(n,p)],
                sparsity=0.5,
                frequency=100
            )
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
        >>> mask_opt.step()

    Notes:
        - implementation based on PyTorch optimizers.
        - whenever randomness is generated, use torch.distributed.broadcast to
            guarantee synchronized masks between devices and matching updates.
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
        if sparsity is not required and not (0.0 < sparsity < 1.0):
            raise ValueError(f'Invalid sparsity level: {sparsity}')
        if not (0.0 <= drop_fraction < 1.0):
            raise ValueError(f'Invalid drop_fraction: {drop_fraction}')
        if end_iteration is not None and not (0.0 < end_iteration <= 1.0):
            raise ValueError(f'Invalid end_iteration: {end_iteration}')
        if frequency <= 0:
            raise ValueError(f'Invalid frequency value: {frequency}')

        if end_iteration is None:
            end_iteration = 0.75  # from paper
        if isinstance(end_iteration, float):
            end_iteration = int(end_iteration * num_updates)

        defaults = dict(
            begin_iteration=begin_iteration,
            end_iteration=end_iteration,
            frequency=frequency,
            drop_fraction=drop_fraction,
            drop_fraction_init_value=drop_fraction,
            drop_fraction_anneal=drop_fraction_anneal,
            mask_metrics=mask_metrics,
        )
        # set this here for triggering correct `init_masks` function
        self.frequency = frequency

        super().__init__(
            params=params,
            names=names,
            flops=flops,
            total_flops=total_flops,
            defaults=defaults,
            optimizers=optimizers,
            sparsity=sparsity,
            mask_distribution=mask_distribution,
            mask_init_method=mask_init_method,
            erk_power_scale=erk_power_scale,
            num_updates=num_updates,
            grad_accum_steps=grad_accum_steps,
            config=config,
            verbose=verbose,
            **kwargs,
        )

        self.end_iteration = end_iteration
        self.mask_metrics = mask_metrics
        self.grad_accum_steps = grad_accum_steps
        self.sparsity_updated = False

        # start metrics computation
        if self.mask_metrics:
            self.init_metrics()

    def state_dict(self):
        state_dict = super(DynamicMaskOptimizer, self).state_dict()
        if self.defaults['mask_metrics']:
            state_dict['_current_total_itop'] = self._current_total_itop
            state_dict['_current_total_doc'] = self._current_total_doc
        return state_dict

    def load_state_dict(self, state_dict):
        super(DynamicMaskOptimizer, self).load_state_dict(state_dict)
        if self.defaults['mask_metrics']:
            self._current_total_itop = state_dict['_current_total_itop']
            self._current_total_doc = state_dict['_current_total_doc']

        for group in self.param_groups:
            for p in group['params']:
                if self.defaults['mask_metrics']:
                    self.state[p]['fired_mask'] = self.state[p][
                        'fired_mask'
                    ].type(dtype=torch.bool)
                    self.state[p]['accumulated_mask'] = self.state[p][
                        'accumulated_mask'
                    ].type(dtype=torch.bool)
        self.apply_all_masks()

    def init_metrics(self):
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['fired_mask'] = (
                    self.state[p]['mask'].detach().clone()
                )
                self.state[p]['accumulated_mask'] = (
                    self.state[p]['mask'].detach().clone()
                )
        self.compute_current_total_itop()  # in-time-over-parameterization
        self.compute_current_total_doc()  # degrees of constraint

    def compute_current_total_itop(self):
        total_dense_el, total_num_el = 0.0, 0.0
        for group in self.param_groups:
            for p in group['params']:
                dense_el = self.state[p]['fired_mask'].count_nonzero().item()
                num_el = self.state[p]['fired_mask'].numel()
                self.state[p]['r_itop'] = dense_el / num_el
                total_dense_el += dense_el
                total_num_el += num_el
        self._current_total_itop = total_dense_el / total_num_el

    @property
    def current_total_itop(self):
        return self._current_total_itop

    def compute_current_total_doc(self):
        total_dense_el, total_num_el = 0.0, 0.0
        for group in self.param_groups:
            for p in group['params']:
                dense_el = (
                    self.state[p]['accumulated_mask'].count_nonzero().item()
                )
                num_el = self.state[p]['accumulated_mask'].numel()
                self.state[p]['r_doc'] = dense_el / num_el
                total_dense_el += dense_el
                total_num_el += num_el
        self._current_total_doc = total_dense_el / total_num_el
        return self._current_total_doc

    @property
    def current_total_doc(self):
        return self._current_total_doc

    def is_update_step(self, frequency, begin_iteration, end_iteration):
        if begin_iteration < self.step_number < end_iteration:
            return (self.step_number % frequency) == 0
        return False

    def update_group_drop_fraction(self, group):
        cur_iter = self.step_number

        begin_iteration = group['begin_iteration']
        end_iteration = group['end_iteration']

        drop_fraction = group['drop_fraction']
        drop_fraction_init_value = group['drop_fraction_init_value']
        drop_fraction_anneal = group['drop_fraction_anneal']

        if drop_fraction_anneal == 'constant':
            m = 1

        elif drop_fraction_anneal == 'cosine':
            decay_iteratons = end_iteration - begin_iteration
            m = cos(pi * (cur_iter - begin_iteration) / decay_iteratons)
            m = m * 0.5 + 0.5

        elif drop_fraction_anneal.startswith('exponential'):
            decay_iteratons = end_iteration - begin_iteration
            exponent = extract_number(drop_fraction_anneal)
            m = exp(-exponent * (cur_iter - begin_iteration) / decay_iteratons)

        else:
            raise KeyError(f'{drop_fraction_anneal} is not a valid annealer')

        group['drop_fraction'] = drop_fraction_init_value * m
        return group['drop_fraction']

    def init_masks(self):
        # also, register backward hook so sparse elements
        # cannot be recovered during normal training
        self.backward_hook_objects = []
        self.backward_masks = []

        for group in self.param_groups:

            mask_init_method = group['mask_init_method']
            for idx, p in enumerate(group['params']):
                self.state[p]['mask'] = self.init_mask(
                    p,
                    self.state[p]['sparsity'],
                    self.names[idx],
                    mask_init_method,
                )
                self.backward_masks.append(self.state[p]['mask'])

                if getattr(p, '_has_rigl_backward_hook', False):
                    raise Exception(
                        'This model already has been registered to a RigL Optimizer.'
                    )

                hook = IndexMaskHook(idx, self)
                p.register_hook(hook)
                setattr(p, '_has_rigl_backward_hook', True)
                self.backward_hook_objects.append(hook)

            d = self.compute_current_group_density(group)

        self.compute_current_total_density()
        assert (
            self.grad_accum_steps > 0 and self.grad_accum_steps < self.frequency
        )

    def check_if_backward_hook_should_accumulate_grad(self):
        """
        Used by the backward hooks. Basically just checks how far away the next
        rigl step is, if it's within `self.grad_accum_steps` steps, return True.
        """
        if self.step_number >= self.end_iteration:
            return False

        steps_til_next_rigl_step = self.frequency - (
            self.step_number % self.frequency
        )
        return steps_til_next_rigl_step <= 1

    @torch.no_grad()
    def drop_minimum(self, score, mask, drop_fraction):
        """
        drop the weights with minimum score (weight or the gradient of weight)
        """
        n_ones = int(mask.count_nonzero())
        n_prune = int(drop_fraction * n_ones)
        n_keep = n_ones - n_prune

        _, indices = score.view([-1]).sort(descending=True)
        indices = indices[:n_keep]

        layer_mask_drop = torch.zeros_like(mask)

        mask_vec = layer_mask_drop.view([-1])
        mask_vec[indices] = True

        return layer_mask_drop, n_prune

    @torch.no_grad()
    def grow_maximum(self, score, mask, n_grow):
        """
        grow the weights with maximum score (weight or the gradient of weight)
        """
        _, indices = score.view([-1]).sort(descending=True)
        indices = indices[:n_grow]

        mask_vec = mask.view([-1])
        mask_vec[indices] = True

        new_mask = torch.zeros_like(mask)
        new_mask.view([-1])[indices] = True

        return mask, new_mask

    @torch.no_grad()
    def update_group_masks(self, group, fused_model=None):
        p_d = self.current_group_density(group)
        drop_fraction = self.update_group_drop_fraction(group)
        new_masks = []

        if fused_model is not None:
            named_params = get_named_parameters(fused_model)

        for idx, p in enumerate(group['params']):

            fused_param = None
            if fused_model is not None:
                name = group['names'][idx]
                fused_param = named_params[name]
                assert (
                    fused_param.shape == p.shape
                ), "param shapes are not same, mismatch in re-growth phase"

            new_masks += [
                self.update_mask(
                    p=p,
                    fused_p=fused_param,
                    drop_fraction=drop_fraction,
                    layer_idx=idx,
                )
            ]
            current_mask = self.backward_masks[idx]
            current_mask.data = self.state[p]['mask']

        # computes and sets group density
        c_d = self.compute_current_group_density(group)

        if self.verbose:
            print(
                f'Updated group density from {p_d:.4} to '
                f'{c_d:.4} using a drop fraction of {drop_fraction:.4}'
            )

        return new_masks

    @abstractmethod
    @torch.no_grad()
    def update_mask(self, p, fused_p, drop_fraction, layer_idx=None):
        # Note: when using a sparsity schedule, drop_fraction should dictate
        #       the rate at which params are dropped (ie sparsity is increased).
        #       Dropping large quantities of params too quickly can result in a
        #       "shock" to the training trajectory; having this "shock" buffered
        #       by drop_fraction might just save training.
        #       When growing the number of parameters (ie decreasing sparsity),
        #       since the params are zeroed (along with their parameter state)
        #       no harm will come from automagically increasing density at will.
        pass

    @torch.no_grad()
    def update_fired_mask(self, p):
        self.state[p]['fired_mask'] = torch.logical_or(
            self.state[p]['fired_mask'], self.state[p]['mask']
        )

    @torch.no_grad()
    def update_accumulated_mask(self, p):
        self.state[p]['accumulated_mask'] = torch.logical_and(
            self.state[p]['accumulated_mask'], self.state[p]['mask']
        )

    @torch.no_grad()
    def step(self, closure=None, fused_model=None):
        # NOTE: since the mask is first updated then applied, one opt step
        #       is applied to the parameters and one update to the momentum is
        #       made
        # NOTE: closure is especially useful when using RigL since it allows you
        #       to compute gradient with a holdout set or a larger set of data or
        #       using a larger gradient accumulation resulting in a better
        #       estimate of the grad for the rigl updt.
        self.update_step_count()
        loss = None

        if closure is not None:
            for group in self.param_groups:
                # get gradients if using holdout set if any group is due for update
                if self.is_update_step(
                    group['frequency'],
                    group['begin_iteration'],
                    group['end_iteration'],
                ):
                    with torch.enable_grad():
                        loss = closure()
                        break  # only run closure once

        self.sparsity_updated = False
        for group in self.param_groups:

            if self.is_update_step(
                group['frequency'],
                group['begin_iteration'],
                group['end_iteration'],
            ):
                new_masks = self.update_group_masks(group, fused_model)
                # NOTE: new_mask could potentially be used to init params
                #       or optimizer state
                self.sparsity_updated = True

        if self.sparsity_updated:
            self.compute_current_total_density()
            if self.defaults['mask_metrics']:
                self.compute_current_total_itop()
                self.compute_current_total_doc()

        self.apply_all_masks()

        return loss

    def get_logging_dict(self):
        mask_dict = {}
        for idx, group in enumerate(self.param_groups):
            mask_dict[f'drop_frac_g{idx}'] = group['drop_fraction']
        for idx, density in enumerate(self.current_group_densities):
            mask_dict[f'density_g{idx}'] = density
        if self.defaults["mask_metrics"]:
            mask_dict[f'total_itop'] = self._current_total_itop
            mask_dict[f'total_doc'] = self._current_total_doc

        return mask_dict
