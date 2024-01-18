import warnings
from math import floor

import torch
from torch.optim.optimizer import Optimizer, required

from .utils import (
    IndexMaskHook,
    erdos_renyi_distribution,
    get_random_mask,
    get_topk_mask,
)
import pdb

class StaticMaskOptimizer(Optimizer):
    r"""Implements a static mask optimizer.

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
        mask_distribution (str): description of how to distribute saprsity
        mask_init_method (str): method by which masks are initialized
        erk_power_scale (float): erk power scale if mask_distribution is erk
        num_updates (int): number of update steps to run
        grad_accum_steps (int): number of gradient accumulation steps
        config (class): class for extra configs to pass in for models
            can be model dependent. Currently, accepts BertConfig.
        verbose (bool): print extra information during run, defaults to False

    Example:
        >>> optimizer = torch.optim.SGD(
                model.parameters(), lr=0.1, momentum=0.9
            )
        >>> mask_opt = StaticMaskOptimizer(
                [p for n,p in model.named_parameters() if should_sparsify(n,p)],
                sparsity=0.5,
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
        defaults={},
        optimizers=[],
        sparsity=required,
        mask_distribution='uniform',
        mask_init_method='random',
        erk_power_scale=1.0,
        num_updates=0,
        grad_accum_steps=1,
        config=None,
        verbose=False,
        **kwargs,
    ):
        if sparsity is not required and not (0.0 < sparsity < 1.0):
            raise ValueError(f'Invalid sparsity level: {sparsity}')

        defaults['sparsity'] = sparsity
        defaults['mask_distribution'] = mask_distribution
        defaults['mask_init_method'] = mask_init_method
        defaults['erk_power_scale'] = erk_power_scale
        defaults['names'] = names
        defaults['flops'] = flops
        defaults['total_flops'] = total_flops

        super().__init__(params, defaults)

        if optimizers:
            if not isinstance(optimizers, list):
                optimizers = [optimizers]
            for opt in optimizers:
                assert isinstance(
                    opt, tuple
                ), f'Expected type: tuple or list of tuples'
                o, s = opt
                assert isinstance(o, Optimizer)
                for _s in s:
                    assert isinstance(_s, str)
            self.optimizers = optimizers
        else:
            warnings.warn(f'mask will not be applied to optimizer state')
            self.optimizers = []

        self.verbose = verbose
        self.step_number = 0
        self._current_total_density = None
        self.backward_masks = None
        self.end_iteration = num_updates
        self.grad_accum_steps = grad_accum_steps

        for k, v in kwargs.items():
            setattr(self, k, v)

        # needed for block sparse attention.
        self.names = names
        self.config = config
        self.mask_init_method = mask_init_method

        self.reset_masks_and_sparsity()

    def reset_masks_and_sparsity(self):
        self.init_sparsities()
        self.init_masks()
        self.apply_masks_params()

    @property
    def current_total_density(self):
        return self._current_total_density

    def compute_current_total_density(self):
        sparse_parameters, dense_parameters = 0, 0
        for s_i in self.state.values():
            sparse_parameters += s_i['mask'].count_nonzero().item()
            dense_parameters += s_i['mask'].numel()
        self._current_total_density = sparse_parameters / dense_parameters
        return self._current_total_density

    def current_group_density(self, group):
        return group['density']

    def compute_current_group_density(self, group):
        dense_el, num_el = 0.0, 0.0
        for p in group['params']:
            dense_el += self.state[p]['mask'].count_nonzero().item()
            num_el += self.state[p]['mask'].numel()
        group['density'] = dense_el / num_el
        return group['density']

    @property
    def current_group_densities(self):
        densities = []
        for group in self.param_groups:
            densities += [self.current_group_density(group)]
        return densities

    def state_dict(self):
        state_dict = super(StaticMaskOptimizer, self).state_dict()
        state_dict['step_number'] = self.step_number
        state_dict['_current_total_density'] = self._current_total_density
        return state_dict

    def load_state_dict(self, state_dict):
        super(StaticMaskOptimizer, self).load_state_dict(state_dict)
        # torch optimizer load_state_dict casts all state dict values to the dtypes of params. We need to undo the casting for some params here.
        # https://github.com/pytorch/pytorch/blob/58eb23378f2a376565a66ac32c93a316c45b6131/torch/optim/optimizer.py#L153
        self.step_number = state_dict['step_number']
        self._current_total_density = state_dict['_current_total_density']

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = self.state[p]['mask'].type(
                    dtype=torch.bool
                )
        self.apply_all_masks()

    def init_sparsities(self, component='sparsity'):
        # set per param desired sparsity level
        for group in self.param_groups:
            sparsity = group[component]
            mask_distribution = group['mask_distribution']

            if mask_distribution in ['uniform', 'uniform+']:
                for i, p in enumerate(group['params']):
                    self.state[p][component] = sparsity

                    # cap at 80% sparsity for output layer
                    if (
                        mask_distribution == 'uniform+'
                        and self.num_classes in p.shape
                    ):
                        if sparsity >= 0.8:
                            self.state[p][component] = 0.8

            elif mask_distribution in ['er', 'erk', 'er+', 'erk+']:
                is_kernel = mask_distribution in ['erk', 'erk+']

                (
                    epsilon,
                    dense_layers,
                    raw_probabilities,
                ) = erdos_renyi_distribution(group, sparsity, is_kernel)

                for i, p in enumerate(group['params']):
                    self.state[p][component] = 0.0
                    if p not in dense_layers:
                        p_density = epsilon * raw_probabilities[p]
                        self.state[p][component] = 1.0 - p_density

                        # cap at 80% sparsity for output layer
                        if (
                            mask_distribution in ['er+', 'erk+']
                            and self.num_classes in p.shape
                        ):
                            if self.state[p][component] >= 0.8:
                                self.state[p][component] = 0.8
            else:
                raise KeyError(
                    f'{mask_distribution} not implemented or valid '
                    f'mask distribution'
                )

            if self.verbose:
                log_str = 'final sparsities:\n'
                for i, p in enumerate(group['params']):
                    s_l = self.state[p][component]
                    s_n = group['names'][i]
                    log_str += s_n
                    log_str += '\t'
                    log_str += '{:0.7f}'.format(s_l)
                    log_str += '\n'
                    assert (
                        0.0 <= s_l < 1.0
                    ), f'Expected sparsity range in [0, 1), got {s_l} for layer {s_n}'

                print(log_str)

        if torch.distributed.is_initialized():
            # guarentee all opts have same sparsity list
            sparsities_list = [self.state[p][component] for p in self.state]
            torch.distributed.broadcast_object_list(sparsities_list, src=0)
            for p, s in zip(self.state, sparsities_list):
                self.state[p][component] = s

    def init_mask(self, p, sparsity, layer_name, mask_init_method):
        if mask_init_method == 'random':
            mask = get_random_mask(p, sparsity)
            mask = mask.contiguous()

        elif mask_init_method == 'dense':
            # ie all ones
            mask = torch.ones_like(p, dtype=torch.bool)

        elif mask_init_method == 'zeros':
            # Note: cant actually be zeros; using 99.99% sparse
            sparsity = 0.999999
            mask = get_random_mask(p, sparsity)
            mask = mask.contiguous()

        elif mask_init_method == 'topk':
            mask = get_topk_mask(p, sparsity)

        else:
            raise KeyError(
                f'Mask init method not implemented or valid: {mask_init_method}'
            )

        if torch.distributed.is_initialized():
            torch.distributed.broadcast(mask, src=0)
        return mask

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

                # if getattr(p, '_has_rigl_backward_hook', False):
                #     raise Exception(
                #         'This model already registered to RigL Optimizer.'
                #     )

                hook = IndexMaskHook(idx, self)
                p.register_hook(hook)
                setattr(p, '_has_rigl_backward_hook', True)
                self.backward_hook_objects.append(hook)

            d = self.compute_current_group_density(group)
        self.compute_current_total_density()

        assert (
            self.grad_accum_steps > 0
            and self.grad_accum_steps < self.end_iteration
        )

    def update_step_count(self):
        self.step_number += 1

    def check_if_backward_hook_should_accumulate_grad(self):
        """
        Used by the backward hooks. Basically just checks how far away the next
        rigl step is, if it's within `self.grad_accum_steps` steps, return True.
        """

        if self.step_number >= self.end_iteration:
            return False

        steps_til_next_rigl_step = self.end_iteration - (
            self.step_number % self.end_iteration
        )
        return steps_til_next_rigl_step <= self.grad_accum_steps

    @torch.no_grad()
    def apply_masks_opt_state(self):
        if self.optimizers:
            for group in self.param_groups:
                for p in group['params']:

                    p_in_opts = False
                    for opt, opt_states_to_mask in self.optimizers:
                        if p in opt.state:
                            p_in_opts = True
                            for s_name in opt_states_to_mask:
                                if s_name in opt.state[p]:
                                    opt.state[p][s_name].mul_(
                                        self.state[p]['mask']
                                    )
                                elif self.step_number:
                                    warnings.warn(
                                        f'{s_name} not in params opt state'
                                    )

                    if self.step_number and not p_in_opts:
                        # Note: opt state not set till after the first step
                        warnings.warn(
                            f'mask not applied to optimizers state for {p.shape}'
                        )

    @torch.no_grad()
    def apply_masks_params(self):
        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
                p.mul_(self.backward_masks[idx])

    @torch.no_grad()
    def apply_masks_param_grads(self):
        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
                if torch.is_tensor(p.grad):
                    p.grad.mul_(self.backward_masks[idx])

    @torch.no_grad()
    def apply_all_masks(self):
        self.apply_masks_params()
        self.apply_masks_opt_state()
        self.apply_masks_param_grads()

    @torch.no_grad()
    def step(self, closure=None):
        self.update_step_count()
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.compute_current_total_density()
        self.apply_all_masks()
        self.sparsity_updated = True

        return loss

    def get_logging_dict(self):
        mask_dict = {}
        for idx, density in enumerate(self.current_group_densities):
            mask_dict[f'density_g{idx}'] = density

        return mask_dict

    def calc_density(self, weight):
        """Computes the density of a tensor.
        Density is the fraction of non-zero elements in a tensor.
        If a tensor has a density of 1.0, then it has no zero elements.
        Args:
            weight: the tensor for which we compute the density
        Returns:
            density (float)
        """

        state = self.state.get(weight)
        if state:
            mask = state['mask']
            return torch.count_nonzero(mask).item() / torch.numel(mask)
        return 1.0

    def get_mask_for_weight(self, weight):
        """Returns the mask of a weight, given the mask optimizer
        If a tensor does not have a mask associated with it, return a ones tensor
        with the weight shape, else return the mask
        Args:
            weight: the tensor for which we compute the mask
        Returns:
            mask (torch.Tensor)
        """

        state = self.state.get(weight)
        if state:
            return state['mask']
        return torch.ones(weight.shape)

    def get_metrics_for_itop(self, weight):
        """Returns the metrics to measure in-time-over-paramterization
        If a tensor does not have a mask associated with it, returns 1 for the itop
        measure, and a ones tensor with the weight shape.
        Args:
            weight: the tensor for which we get the itop metrics
        Returns:
            itop (float)
            fired_mask (torch.Tensor)
        """

        state = self.state.get(weight)
        if state:
            return (
                state['r_itop'],
                state['fired_mask'],
            )
        return 1.0, torch.ones(weight.shape)

    def get_metrics_for_doc(self, weight):
        """Returns the metrics to measure degrees of constraints
        If a tensor does not have a mask associated with it, returns 1 for the doc
        measure, and a ones tensor with the weight shape.
        Args:
            weight: the tensor for which we get the doc metrics
        Returns:
            doc (float)
            accumulated_mask (torch.Tensor)
        """

        state = self.state.get(weight)
        if state:
            return (
                state['r_doc'],
                state['accumulated_mask'],
            )
        return 1.0, torch.ones(weight.shape)
