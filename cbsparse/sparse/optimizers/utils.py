"""
Released under BSD 3-Clause License,
Copyright (c) 2021 Cerebras Systems Inc.
All rights reserved.
"""
import re
import sys
import warnings

import numpy as np
import torch

try:
    from apex.optimizers import FusedAdam, FusedLAMB, FusedSGD
except:
    pass

try:
    from huggingface.pytorch_transformers.pytorch_transformers import (
        optimization,
    )
except:
    optimization = None


def extract_number(token):
    """Strips the number from the end of the token if it exists.

    Args:
        token: str, s or s_d where d is a number: a float or int.
        `foo_.5`, `foo_foo.5`, `foo_0.5`, `foo_4` are all valid strings.

    Returns:
        float, d if exists otherwise 1.
    """
    regexp = re.compile(r'.*_(\d*\.?\d*)$')
    if regexp.search(token):
        return float(regexp.search(token).group(1))
    else:
        return 1.0


def calc_sparsity(tensor):
    """Computes the sparsity of a tensor.
    Sparsity is the fraction of zero elements in a tensor.
    If a tensor has a density of 0.0, then it has all zero elements.
    Sparsity and density are complementary.

    Args:
        tensor: the tensor for which we compute the density.
    Returns:
        sparsity (float)
    """
    nonzero = tensor.abs().gt(0).sum()
    val = float(nonzero.item()) / torch.numel(tensor)
    return 1.0 - val


def get_named_parameters(model):
    named_params = dict()

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            named_params[name] = param

    return named_params


def optimizer_trackers(optimizer):
    adam_variants = [torch.optim.Adam, torch.optim.AdamW]
    if optimization is not None:
        adam_variants += [optimization.AdamW]
    adam_variants = tuple(adam_variants)
    if isinstance(optimizer, torch.optim.Adadelta):
        return ['square_avg', 'acc_delta']
    if isinstance(optimizer, torch.optim.Adagrad):
        return ['sum']
    elif isinstance(optimizer, adam_variants):
        return ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']
    elif isinstance(optimizer, torch.optim.Adamax):
        return ['exp_avg', 'exp_inf']
    elif isinstance(optimizer, torch.optim.RMSprop):
        return ['square_avg', 'momentum_buffer', 'grad_avg']
    elif isinstance(optimizer, torch.optim.SGD):
        return ['momentum_buffer']
    elif 'apex.optimizers.fused_adam' in sys.modules and isinstance(
        optimizer, FusedAdam
    ):
        return ['exp_avg', 'exp_avg_sq']
    elif 'apex.optimizers.fused_sgd' in sys.modules and isinstance(
        optimizer, FusedSGD
    ):
        return ['momentum_buffer']
    elif 'apex.optimizers.fused_lamb' in sys.modules and isinstance(
        optimizer, FusedLAMB
    ):
        return ['exp_avg', 'exp_avg_sq']
    else:
        raise NotImplementedError


class IndexMaskHook:
    def __init__(self, layer, scheduler):
        self.layer = layer
        self.scheduler = scheduler
        self.dense_grad = None

    def __name__(self):
        return 'IndexMaskHook'

    @torch.no_grad()
    def __call__(self, grad):
        mask = self.scheduler.backward_masks[self.layer]

        # only calculate dense_grads when necessary
        if self.scheduler.check_if_backward_hook_should_accumulate_grad():
            if self.dense_grad is None:
                # initialize as all 0s so we can do a rolling average
                self.dense_grad = torch.zeros_like(grad)
            self.dense_grad += grad / self.scheduler.grad_accum_steps
        else:
            self.dense_grad = None

        return grad * mask


def erdos_renyi_distribution(group, sparsity, is_kernel=True):
    warnings.warn(f'erk dist done per param group')

    erk_power_scale = group['erk_power_scale']
    is_epsilon_valid = False
    dense_layers = set()
    density = 1.0 - sparsity

    while not is_epsilon_valid:
        # We will start with all layers and try to find right epsilon. However if
        # any probablity exceeds 1, we will make that layer dense and repeat the
        # process (finding epsilon) with the non-dense layers.

        # We want the total number of connections to be the same. Let say we have
        # four layers with N_1, N_2, N_3, N_4 parameters each. Lets say after some
        # iterations, probability of some dense layers (3, 4) exceeded 1 and
        # therefore we added them to the dense_layers set. Those layers will not
        # scale with erdos_renyi, however we need to count them so that target
        # paratemeter count is achieved. Hence, we solve for this:
        # --------------------------------------------------------------------------
        # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
        #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
        # --------------------------------------------------------------------------
        # eps * (p_1 * N_1 + p_2 * N_2) =
        #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
        # --------------------------------------------------------------------------
        # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for p in group['params']:
            n_param = p.numel()
            n_zeros = n_param * sparsity
            n_ones = n_param * density

            if p in dense_layers:
                # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                rhs -= n_zeros
            else:
                # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                # equation above.
                rhs += n_ones
                if is_kernel:
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    raw_probabilities[p] = (
                        sum(p.shape) / n_param
                    ) ** erk_power_scale
                else:
                    n_in, n_out = p.shape[:2]
                    raw_probabilities[p] = (n_in + n_out) / (n_in * n_out)

                # Note that raw_probabilities[mask] * n_param gives the individual
                # elements of the divisor.
                divisor += raw_probabilities[p] * n_param

        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        epsilon = rhs / divisor

        # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob = max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for p_id, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    dense_layers.add(p_id)
        else:
            is_epsilon_valid = True

    # With the valid epsilon, we can set sparsities of the remaining layers.
    return epsilon, dense_layers, raw_probabilities


def get_random_mask(param, sparsity):
    mask = param.new_empty(param.shape).uniform_() < (1.0 - sparsity)
    mask = mask.type(dtype=torch.bool)
    return mask


def get_topk_mask(param, sparsity):
    shape = param.shape

    num_sparse_elem = int(sparsity * param.numel())
    num_dense_elem = param.numel() - num_sparse_elem

    param = param.clone().detach().abs().view(-1)
    _, indices = torch.topk(param, num_dense_elem)

    mask = torch.zeros_like(param, dtype=torch.bool)
    mask = mask.scatter(0, indices, True)
    return mask.view(shape).contiguous()
