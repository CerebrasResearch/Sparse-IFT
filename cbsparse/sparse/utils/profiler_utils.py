"""
Calculate FLOPs used by model fwd pass

Copyright 2021 Cerebras Systems, Inc.
"""
import sys
from functools import partial
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


default_dtype = torch.float64

unit_dict = OrderedDict()
unit_dict['Z'] = 21.0
unit_dict['E'] = 18.0
unit_dict['P'] = 15.0
unit_dict['T'] = 12.0
unit_dict['G'] = 9.0
unit_dict['M'] = 6.0
unit_dict['K'] = 3.0


def count_to_string(count, units=None, precision=4):
    if units is None:
        for k, v in unit_dict.items():
            if count // 10 ** v > 0:
                units = k
                break

    if units not in unit_dict:
        return str(count)

    return str(round(count / 10.0 ** v, precision)) + units


def get_model_complexity_info(
    model,
    input_res,
    as_strings=False,
    mask_optimizer=None,
    input_constructor=None,
    ostd=sys.stdout,
    print_per_layer_stat=False,
    verbose=False,
    ignore_modules=[],
    custom_modules_hooks={},
    global_verbose=True,
    drop_connect=False,
):
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    assert isinstance(model, nn.Module)
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks

    global maskopt
    global dropconnect
    global is_sparse
    global is_training
    global ost

    is_sparse = mask_optimizer is not None or drop_connect
    maskopt = mask_optimizer
    prev_training_status = model.training
    is_training = prev_training_status
    ost = ostd
    dropconnect = drop_connect

    flops_model = add_flop_counting_methods(model)
    flops_model.eval()
    flops_model.start_flop_count(
        ost=ost, verbose=verbose, ignore_list=ignore_modules
    )

    if input_constructor:
        input = input_constructor(input_res)
        with torch.no_grad():
            _ = flops_model(**input)
    else:
        try:
            batch = torch.ones(()).new_empty(
                (1, *input_res),
                dtype=next(flops_model.parameters()).dtype,
                device=next(flops_model.parameters()).device,
            )
        except StopIteration:
            batch = torch.ones(()).new_empty((1, *input_res))

        with torch.no_grad():
            _ = flops_model(batch)

    profile = flops_model.compute_total_flop_cost()
    if print_per_layer_stat:
        print_model_with_flops(flops_model, profile, ost=ost)

    flops_model.stop_flop_count()
    CUSTOM_MODULES_MAPPING = {}

    model.train(prev_training_status)

    if global_verbose:
        print(
            f'Dense model FLOP complexity: {profile["flops"]}, '
            f'Dense model MAC complexity: {profile["multiplies"]}, '
            f'model parameter complexity: {profile["parameters"]}',
            file=ost,
        )
        if is_sparse:
            flop_density = profile["sparse"]["flops"] / profile["flops"]
            mac_denstiy = profile["sparse"]["multiplies"] / profile["multiplies"]
            param_density = profile["sparse"]["parameters"] / profile["parameters"]
            print(
                f'Sparse model FLOP complexity: {profile["sparse"]["flops"]} (density = {flop_density:.4}), '
                f'Sparse model MAC complexity: {profile["sparse"]["multiplies"]} (density = {mac_denstiy:.4}), '
                f'model parameter complexity: {profile["sparse"]["parameters"]} (density = {param_density:.4})',
                file=ost,
            )

    if as_strings:
        string_keys = ['adds', 'multiplies', 'logical', 'other', 'parameters', 'flops']
        for k in ['adds', 'multiplies', 'logical', 'other', 'parameters', 'flops']:
            profile[k] = count_to_string(profile[k], units=None, precision=4)
        if is_sparse:
            for k in ['adds', 'multiplies', 'logical', 'other', 'parameters', 'flops']:
                profile['sparse'][k] = count_to_string(profile['sparse'][k], units=None, precision=4)

    return profile


def print_model_with_flops(
    model,
    profile,
    units='',
    precision=3,
    ost=sys.stdout,
):

    def accumulate_parameters(self):
        if is_supported_instance(self):
            parameters = 0
            for op in self.op_profile.values():
                if 'sparse' in op and op['sparse'] is not None:
                    op = op['sparse']
                parameters += op['parameters']
            return parameters
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_parameters()
            return sum

    def accumulate_flops(self):
        if is_supported_instance(self):
            flops = 0
            for op in self.op_profile.values():
                if 'sparse' in op and op['sparse'] is not None:
                    op = op['sparse']
                flops += op['adds'] + op['multiplies'] + op['logical'] + op['other']

            return flops
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_flops()
            return sum

    def profile_repr(self):
        accumulated_parameters = self.accumulate_parameters()
        accumulated_flops = self.accumulate_flops()

        selfparams = get_module_parameters_number(self)

        # total network params
        params, flops = profile['parameters'], profile['flops']
        if is_sparse:
            params, flops = profile['sparse']['parameters'], profile['sparse']['flops']

        output = [
            f'{count_to_string(accumulated_flops, precision=precision)}'
            f' ({accumulated_flops / flops:.2%})'
            ' FLOPs',
            f'{count_to_string(accumulated_parameters, precision=precision)}'
            f' ({accumulated_parameters / params:.2%})'
            ' Parameters',
            f'{count_to_string(selfparams, precision=precision)}'
            f' self.parameters',
        ]
        if selfparams:
            output += [f'param density: {accumulated_parameters / selfparams:.2%}']
        else:
            output += [f'param density: N/A']

        return ', '.join(output)

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_parameters = accumulate_parameters.__get__(m)
        flops_extra_repr = profile_repr.__get__(m)

        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops
        if hasattr(m, 'accumulate_parameters'):
            del m.accumulate_parameters

    model.apply(add_extra_repr)
    print(model, file=ost)
    model.apply(del_extra_repr)


def get_module_parameters_number(module):
    params_num = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return params_num


def add_flop_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flop_count = start_flop_count.__get__(
        net_main_module
    )
    net_main_module.stop_flop_count = stop_flop_count.__get__(net_main_module)
    net_main_module.reset_flop_count = reset_flop_count.__get__(
        net_main_module
    )
    net_main_module.compute_total_flop_cost = compute_total_flop_cost.__get__(
        net_main_module
    )

    net_main_module.reset_flop_count()

    return net_main_module


def compute_total_flop_cost(self):
    """
    A method that will be available after add_flop_counting_methods() is called
    on a desired net object.

    Returns current flop consumption per iteration forward propagagtion.
    """

    profile = {
        'adds': 0, 'multiplies': 0, 'logical': 0, 'other': 0,
        'parameters': 0, 'flops': None,
        'sparse': None, 'profiler_dict': None,
    }
    if is_sparse:
        profile['sparse'] = {
            'adds': 0, 'multiplies': 0, 'logical': 0, 'other': 0,
            'parameters': 0, 'flops': None, 'density': 0,
        }

    profiler_dict = {}
    for module in self.modules():
        if is_supported_instance(module):
            profiler_dict[module] = module.op_profile
            for op in module.op_profile.values():
                profile['adds'] += op['adds']
                profile['multiplies'] += op['multiplies']
                profile['logical'] += op['logical']
                profile['other'] += op['other']
                profile['parameters'] += op['parameters']

                if is_sparse:
                    if 'sparse' in op: op = op['sparse']
                    profile['sparse']['adds'] += op['adds']
                    profile['sparse']['multiplies'] += op['multiplies']
                    profile['sparse']['logical'] += op['logical']
                    profile['sparse']['other'] += op['other']
                    profile['sparse']['parameters'] += op['parameters']

    model_parameters = get_module_parameters_number(self)
    if profile['parameters'] != model_parameters:
        p_params = profile['parameters']
        print(
            f'Warning: number of parameters in model ({model_parameters}) '
            f'not equal number of parameters in profiled modules ({p_params})',
            file=ost,
        )

    profile['flops'] = profile['adds'] + profile['multiplies'] + profile['logical'] + profile['other']
    if is_sparse:
        profile['sparse']['flops'] = (
            profile['sparse']['adds'] + profile['sparse']['multiplies'] +
            profile['sparse']['logical'] + profile['sparse']['other']
        )
        profile['sparse']['density'] = profile['sparse']['parameters'] / profile['parameters']

    profile['profiler_dict'] = profiler_dict

    return profile


def start_flop_count(self, **kwargs):
    """
    A method that will be available after add_flop_counting_methods() is called
    on a desired net object.

    Activates the computation of flops used by input tensor.
    Call it before you run the network.
    """
    seen_types = set()

    def add_flop_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
        elif is_supported_instance(module):
            if hasattr(module, "__op_handle__"):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                    CUSTOM_MODULES_MAPPING[type(module)]
                )
            else:
                handle = module.register_forward_hook(
                    MODULES_MAPPING[type(module)]
                )
            module.__op_handle__ = handle
            seen_types.add(type(module))
        else:
            if (
                verbose
                and not type(module) in (nn.Sequential, nn.ModuleList)
                and not type(module) in seen_types
            ):
                print(
                    "Warning: module "
                    + type(module).__name__
                    + " is treated as a zero-op.",
                    file=ost,
                )
            seen_types.add(type(module))

    self.apply(partial(add_flop_counter_hook_function, **kwargs))


def stop_flop_count(self):
    """
    A method that will be available after add_flop_counting_methods() is called
    on a desired net object.

    remove all utils from net used for counting flops
    """
    self.apply(remove_flop_counter_hook_function)


def reset_flop_count(self):
    """
    A method that will be available after add_flop_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    self.apply(add_flop_counter_variable_or_reset)


# ---- Internal functions
def combine_density(old_param_count, old_sparse_param_count, density, param_count):
    total_params = old_param_count + param_count
    total_sparse_params = old_sparse_param_count + int(param_count * density)
    return total_sparse_params / total_params


def empty_flop_counter_hook(module, input, output):
    if isinstance(input, (list, tuple)):
        input = input[0]
    in_numel = input.numel()

    if isinstance(output, (list, tuple)):
        output = output[0]
    out_numel = output.numel()

    module.op_profile['ops']['adds'] += 0
    module.op_profile['ops']['multiplies'] += 0
    module.op_profile['ops']['logical'] += 0
    module.op_profile['ops']['other'] += 0
    module.op_profile['ops']['parameters'] += 0


def conv_flop_counter_hook(module, input, output):
    if isinstance(input, (list, tuple)):
        input = input[0]
    in_numel = input.numel()

    if isinstance(output, (list, tuple)):
        output = output[0]
    out_numel = output.numel()

    outN = output.size(0)
    outD = output[0, 0].numel()

    # add conv op with sparse masking to op profiler dict
    if 'conv' not in module.op_profile:
        module.op_profile['conv'] = {
            'adds': 0,
            'multiplies': 0,
            'logical': 0,
            'other': 0,
            'parameters': 0,
        }
    module.op_profile['conv']['adds'] += outN * (module.weight.numel() - 1) * outD
    module.op_profile['conv']['multiplies'] += outN * module.weight.numel() * outD
    module.op_profile['conv']['parameters'] += module.weight.numel()
    if (maskopt and module.weight in maskopt.state) or hasattr(module, 'dc_rate'):
        if 'sparse' not in module.op_profile['conv']:
            module.op_profile['conv']['sparse'] = {
                'density': 0,
                'adds': 0,
                'multiplies': 0,
                'logical': 0,
                'other': 0,
                'parameters': 0,
            }
        if maskopt and module.weight in maskopt.state:
            density = maskopt.calc_density(module.weight)
        elif hasattr(module, 'dc_rate'):
            density = 1. - module.dc_rate
        combined_density = combine_density(
            module.op_profile['conv']['parameters'],
            module.op_profile['conv']['sparse']['parameters'],
            density,
            module.weight.numel()
        )
        module.op_profile['conv']['sparse']['density'] = combined_density
        module.op_profile['conv']['sparse']['adds'] += int(module.op_profile['conv']['adds'] * density)
        module.op_profile['conv']['sparse']['multiplies'] += int(module.op_profile['conv']['multiplies'] * density)
        module.op_profile['conv']['sparse']['parameters'] += int(module.op_profile['conv']['parameters'] * density)

    # add bias op to op profiler dict
    if module.bias is not None:
        if 'bias' not in module.op_profile:
            module.op_profile['bias'] = {
                'adds': 0,
                'multiplies': 0,
                'logical': 0,
                'other': 0,
                'parameters': 0,
            }
        module.op_profile['bias']['adds'] += out_numel
        module.op_profile['bias']['parameters'] += module.bias.numel()


def linear_flop_counter_hook(module, input, output):
    if isinstance(input, (list, tuple)):
        input = input[0]
    in_numel = input.numel()

    if isinstance(output, (list, tuple)):
        output = output[0]
    out_numel = output.numel()

    # add matmul op with sparse masking to op profiler dict
    in_features = module.in_features
    out_features = module.out_features
    k = in_numel // in_features  # number of tensors being put thru matmul

    if 'matmul' not in module.op_profile:
        module.op_profile['matmul'] = {
            'adds': 0,
            'multiplies': 0,
            'logical': 0,
            'other': 0,
            'parameters': 0,
        }
    module.op_profile['matmul']['adds'] += k * (in_features - 1) * out_features
    module.op_profile['matmul']['multiplies'] += k * in_features * out_features
    module.op_profile['matmul']['parameters'] += module.weight.numel()
    if (maskopt and module.weight in maskopt.state) or hasattr(module, 'dc_rate'):
        if 'sparse' not in module.op_profile['matmul']:
            module.op_profile['matmul']['sparse'] = {
                'density': 0,
                'adds': 0,
                'multiplies': 0,
                'logical': 0,
                'other': 0,
                'parameters': 0,
            }

        if maskopt and module.weight in maskopt.state:
            density = maskopt.calc_density(module.weight)
        elif hasattr(module, 'dc_rate'):
            density = 1. - module.dc_rate
        combined_density = combine_density(
            module.op_profile['matmul']['parameters'],
            module.op_profile['matmul']['sparse']['parameters'],
            density,
            module.weight.numel()
        )
        module.op_profile['matmul']['sparse']['density'] = combined_density
        module.op_profile['matmul']['sparse']['adds'] += int(module.op_profile['matmul']['adds'] * density)
        module.op_profile['matmul']['sparse']['multiplies'] += int(module.op_profile['matmul']['multiplies'] * density)
        module.op_profile['matmul']['sparse']['parameters'] += int(module.op_profile['matmul']['parameters'] * density)

    # add bias op to op profiler dict
    if module.bias is not None:
        if 'bias' not in module.op_profile:
            module.op_profile['bias'] = {
                'adds': 0,
                'multiplies': 0,
                'logical': 0,
                'other': 0,
                'parameters': 0,
            }
        module.op_profile['bias']['adds'] += out_numel
        module.op_profile['bias']['parameters'] += module.bias.numel()


def bn_flop_counter_hook(module, input, output):
    if isinstance(input, (list, tuple)):
        input = input[0]
    in_numel = input.numel()

    inC = input.size(1)
    inND = in_numel // inC

    m_reduce_add = inC * (inND - 1)  # reduce first moment
    sigma_mult = in_numel  # square
    sigma_reduce_add = inC * (inND - 1)  # reduce squares (ie second momentu)
    sigma_sqrt = inC
    m_sub = sigma_div = in_numel

    m_mult = bias_add = 0
    if module.affine:
        m_mult = bias_add = in_numel

        module.op_profile['ops']['parameters'] += module.weight.numel()
        module.op_profile['ops']['parameters'] += module.bias.numel()

    module.op_profile['ops']['adds'] += m_reduce_add + sigma_reduce_add + m_sub + bias_add
    module.op_profile['ops']['multiplies'] += sigma_mult + sigma_div + m_mult
    module.op_profile['ops']['other'] += sigma_sqrt


def ln_flop_counter_hook(module, input, output):
    if isinstance(input, (list, tuple)):
        input = input[0]
    in_numel = input.numel()

    inN = input.size(0)
    inCD = input[0].numel()

    m_reduce_add = inN * (inCD - 1)  # reduce first moment
    sigma_mult = in_numel  # square
    sigma_reduce_add = inN * (inCD - 1)  # reduce squares (ie second momentu)
    sigma_sqrt = inN
    m_sub = sigma_div = in_numel

    m_mult = bias_add = 0
    if module.elementwise_affine:
        m_mult = bias_add = in_numel

        module.op_profile['ops']['parameters'] += module.weight.numel()
        module.op_profile['ops']['parameters'] += module.bias.numel()

    module.op_profile['ops']['adds'] += m_reduce_add + sigma_reduce_add + m_sub + bias_add
    module.op_profile['ops']['multiplies'] += sigma_mult + sigma_div + m_mult
    module.op_profile['ops']['other'] += sigma_sqrt


def relu_flop_counter_hook(module, input, output):
    if isinstance(output, (list, tuple)):
        output = output[0]
    out_numel = output.numel()

    module.op_profile['ops']['logical'] += out_numel


def lrelu_flop_counter_hook(module, input, output):
    if isinstance(output, (list, tuple)):
        output = output[0]
    out_numel = output.numel()

    module.op_profile['ops']['logical'] += out_numel
    # assuming on average half of activations are < 0
    module.op_profile['ops']['multiplies'] += out_numel / 2


def relu6_flop_counter_hook(module, input, output):
    if isinstance(output, (list, tuple)):
        output = output[0]
    out_numel = output.numel()

    module.op_profile['ops']['logical'] += 2 * out_numel


def tanh_flop_counter_hook(module, input, output):
    if isinstance(output, (list, tuple)):
        output = output[0]
    out_numel = output.numel()

    adds = total_exps = 2 * out_numel
    total_divs = out_numel

    module.op_profile['ops']['adds'] += adds
    module.op_profile['ops']['multiplies'] += total_divs
    module.op_profile['ops']['other'] += total_exps


def swish_flop_counter_hook(module, input, output):
    if isinstance(output, (list, tuple)):
        output = output[0]
    out_numel = output.numel()

    adds = total_exps = total_divs = out_numel

    module.op_profile['ops']['adds'] += adds
    module.op_profile['ops']['multiplies'] += total_divs
    module.op_profile['ops']['other'] += total_exps


def gelu_flop_counter_hook(module, input, output):
    if isinstance(output, (list, tuple)):
        output = output[0]
    out_numel = output.numel()

    adds = 2 * out_numel
    total_muls = 4 * out_numel
    total_tanh_muls = 3 * out_numel
    total_tanh_adds = 2 * out_numel

    module.op_profile['ops']['adds'] += adds + total_tanh_adds
    module.op_profile['ops']['multiplies'] += total_muls + total_tanh_muls


def mpool_flop_counter_hook(module, input, output):
    if isinstance(output, (list, tuple)):
        output = output[0]
    out_numel = output.numel()

    ops = out_numel
    if '1d' in module.__class__.__name__:
        ops *= (module.kernel_size - 1)
    elif '2d' in module.__class__.__name__:
        ops *= (module.kernel_size ** 2 - 1)
    elif '3d' in module.__class__.__name__:
        ops *= (module.kernel_size ** 3 - 1)
    else:
        raise NotImplementedError

    module.op_profile['ops']['logical'] += int(ops)


def adaptive_mpool_flop_counter_hook(module, input, output):
    if isinstance(input, (list, tuple)):
        input = input[0]
    in_numel = input.numel()

    if isinstance(output, (list, tuple)):
        output = output[0]
    out_numel = output.numel()

    ops_per_out = in_numel // out_numel  - 1

    module.op_profile['ops']['logical'] += out_numel * ops_per_out


def avgpool_flop_counter_hook(module, input, output):
    if isinstance(output, (list, tuple)):
        output = output[0]
    out_numel = output.numel()

    add_ops = out_numel
    if '1d' in module.__class__.__name__:
        add_ops *= (module.kernel_size - 1)
    elif '2d' in module.__class__.__name__:
        add_ops *= (module.kernel_size ** 2 - 1)
    elif '3d' in module.__class__.__name__:
        add_ops *= (module.kernel_size ** 3 - 1)
    else:
        raise NotImplementedError

    module.op_profile['ops']['adds'] += int(add_ops)
    module.op_profile['ops']['multiplies'] += out_numel


def adaptive_avgpool_flop_counter_hook(module, input, output):
    if isinstance(input, (list, tuple)):
        input = input[0]
    in_numel = input.numel()

    if isinstance(output, (list, tuple)):
        output = output[0]
    out_numel = output.numel()

    adds_per_out = in_numel // out_numel - 1

    module.op_profile['ops']['adds'] += out_numel * adds_per_out
    module.op_profile['ops']['multiplies'] += out_numel


def upsample_flop_counter_hook(module, input, output):
    # NOTE: probably needs updating
    if isinstance(output, (list, tuple)):
        output = output[0]
    out_numel = output.numel()

    module.op_profile['ops']['multiplies'] += out_numel


def softmax_flop_counter_hook(module, input, output):
    if isinstance(input, (list, tuple)):
        input = input[0]
    in_numel = input.numel()

    k = input.size(module.dim)
    n = in_numel // k

    total_exp = total_div = n
    total_add = n - 1

    module.op_profile['ops']['adds'] += total_add
    module.op_profile['ops']['multiplies'] += total_div
    module.op_profile['ops']['other'] += total_exp


def hardswish_flop_counter_hook(module, input, output):
    if isinstance(output, (list, tuple)):
        output = output[0]
    out_numel = output.numel()

    logical = 2 * out_numel
    adds = divs = multiplies = out_numel

    module.op_profile['ops']['adds'] += adds
    module.op_profile['ops']['multiplies'] += divs + multiplies
    module.op_profile['ops']['logical'] += logical


def hardsigmoid_flop_counter_hook(module, input, output):
    if isinstance(output, (list, tuple)):
        output = output[0]
    out_numel = output.numel()

    logical = 2 * out_numel
    adds = divs = out_numel

    module.op_profile['ops']['adds'] += adds
    module.op_profile['ops']['multiplies'] += divs
    module.op_profile['ops']['logical'] += logical


def residual_flop_counter_hook(module, input, output):
    if isinstance(output, (list, tuple)):
        output = output[0]
    module.op_profile['ops']['adds'] += output.numel()


ost = None
is_sparse = False
is_training = False
maskopt = None

CUSTOM_MODULES_MAPPING = {}
MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_flop_counter_hook,
    nn.Conv2d: conv_flop_counter_hook,
    nn.Conv3d: conv_flop_counter_hook,
    # fc
    nn.Linear: linear_flop_counter_hook,
    # bn
    nn.BatchNorm1d: bn_flop_counter_hook,
    nn.BatchNorm2d: bn_flop_counter_hook,
    nn.BatchNorm3d: bn_flop_counter_hook,
    # ln
    nn.LayerNorm: ln_flop_counter_hook,
    # activations
    nn.ReLU: relu_flop_counter_hook,
    nn.LeakyReLU: lrelu_flop_counter_hook,
    nn.PReLU: lrelu_flop_counter_hook,
    nn.ReLU6: relu6_flop_counter_hook,
    nn.Tanh: tanh_flop_counter_hook,
    nn.ELU: relu_flop_counter_hook,  # TODO
    nn.Hardswish: hardswish_flop_counter_hook,
    nn.Hardsigmoid: hardsigmoid_flop_counter_hook,
    # softmax
    nn.Softmax: softmax_flop_counter_hook,
    # pooling modules
    nn.MaxPool1d: mpool_flop_counter_hook,
    nn.MaxPool2d: mpool_flop_counter_hook,
    nn.MaxPool3d: mpool_flop_counter_hook,
    nn.AvgPool1d: avgpool_flop_counter_hook,
    nn.AvgPool2d: avgpool_flop_counter_hook,
    nn.AvgPool3d: avgpool_flop_counter_hook,
    nn.AdaptiveMaxPool1d: adaptive_mpool_flop_counter_hook,
    nn.AdaptiveMaxPool2d: adaptive_mpool_flop_counter_hook,
    nn.AdaptiveMaxPool3d: adaptive_mpool_flop_counter_hook,
    nn.AdaptiveAvgPool1d: adaptive_avgpool_flop_counter_hook,
    nn.AdaptiveAvgPool2d: adaptive_avgpool_flop_counter_hook,
    nn.AdaptiveAvgPool3d: adaptive_avgpool_flop_counter_hook,
    # upscale
    nn.Upsample: upsample_flop_counter_hook,
    # other
    nn.Dropout: empty_flop_counter_hook,
}


def is_supported_instance(module):
    if (
        type(module) in MODULES_MAPPING
        or type(module) in CUSTOM_MODULES_MAPPING
    ):
        return True
    return False


def add_flop_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, 'op_profile'):
            # reset
            for op in module.op_profile:
                op['adds'] = 0
                op['multiplies'] = 0
                op['logical'] = 0
                op['other'] = 0
                op['parameters'] = 0

                # remove sparse counter if it exists
                sparse = op.pop('sparse', None)
        else:
            # set
            module.op_profile = {
                'ops': {
                    'adds': 0,
                    'multiplies': 0,
                    'logical': 0,
                    'other': 0,
                    'parameters': 0,
                },
            }


def remove_flop_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, "__op_handle__"):
            module.__op_handle__.remove()
            del module.__op_handle__

        if hasattr(module, "op_profile"):
            delattr(module, 'op_profile')
