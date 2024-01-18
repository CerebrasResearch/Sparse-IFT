"""
Track FLOPs used in training

Copyright 2021 Cerebras Systems, Inc.
"""
import sys

from .profiler_utils import get_model_complexity_info


import sys

from .profiler_utils import get_model_complexity_info


class TrainProfiler:
    def __init__(
        self,
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
        self.verbose = verbose
        self.complexity_profile = get_model_complexity_info(
            model=model,
            input_res=input_res,
            as_strings=as_strings,
            mask_optimizer=mask_optimizer,
            input_constructor=input_constructor,
            ostd=ostd,
            print_per_layer_stat=print_per_layer_stat,
            verbose=verbose,
            ignore_modules=ignore_modules,
            custom_modules_hooks=custom_modules_hooks,
            global_verbose=global_verbose,
            drop_connect=drop_connect,
        )

        self.profile = {
            'adds': 0, 'multiplies': 0, 'logical': 0, 'other': 0,
            'flops': 0, 'parameters': 0,
        }

        self.sparse_profile = None
        if self.complexity_profile['sparse']:
            self.sparse_profile = {
                'adds': 0, 'multiplies': 0, 'logical': 0, 'other': 0,
                'flops': 0, 'parameters': 0,
            }
        self.samples_processed = 0

    def set_dense_profile(self):
        profile = {
            'adds': 0, 'multiplies': 0, 'logical': 0, 'other': 0,
            'parameters': 0,
        }

        for op_profile in self.complexity_profile['profiler_dict'].values():
            for op in op_profile.values():
                profile['adds'] += op['adds']
                profile['multiplies'] += op['multiplies']
                profile['logical'] += op['logical']
                profile['other'] += op['other']
                profile['parameters'] += op['parameters']

        self.complexity_profile['adds'] = profile['adds']
        self.complexity_profile['multiplies'] = profile['multiplies']
        self.complexity_profile['logical'] = profile['logical']
        self.complexity_profile['other'] = profile['other']
        self.complexity_profile['parameters'] = profile['parameters']

        self.complexity_profile['flops'] = profile['adds'] + profile['multiplies'] + profile['logical'] + profile['other']

        return profile

    def set_sparse_profile(self):
        profile = {
            'adds': 0,
            'multiplies': 0,
            'logical': 0,
            'other': 0,
            'parameters': 0,
        }

        for op_profile in self.complexity_profile['profiler_dict'].values():
            for op in op_profile.values():
                if 'sparse' in op: op = op['sparse']
                profile['adds'] += op['adds']
                profile['multiplies'] += op['multiplies']
                profile['logical'] += op['logical']
                profile['other'] += op['other']
                profile['parameters'] += op['parameters']

        self.complexity_profile['sparse']['adds'] = profile['adds']
        self.complexity_profile['sparse']['multiplies'] = profile['multiplies']
        self.complexity_profile['sparse']['logical'] = profile['logical']
        self.complexity_profile['sparse']['other'] = profile['other']
        self.complexity_profile['sparse']['parameters'] = profile['parameters']

        self.complexity_profile['sparse']['density'] = self.complexity_profile['sparse']['parameters'] / self.complexity_profile['parameters']

        self.complexity_profile['sparse']['flops'] = profile['adds'] + profile['multiplies'] + profile['logical'] + profile['other']

        return profile

    def update_profiler_dict_sparsity(self, maskopt):
        if self.verbose:
            v_str = f'Warning: updating spare flop complexity.\n'
            v_str += f'Old density: {self.complexity_profile["sparse"]["density"]:.4}\n'
        for module, op_profile in self.complexity_profile['profiler_dict'].items():
            for op_key, op in op_profile.items():
                if 'sparse' in op:
                    if maskopt is not None:
                        mask = maskopt.state.get(module.weight)['mask']
                        density = mask.count_nonzero().item() / mask.numel()
                    elif hasattr(module, 'dc_rate'):
                        density = 1. - module.dc_rate

                    op['sparse']['density'] = density
                    op['sparse']['adds'] = int(op['adds'] * density)
                    op['sparse']['multiplies'] = int(op['multiplies'] * density)
                    op['sparse']['logical'] = int(op['logical'] * density)
                    op['sparse']['other'] = int(op['other'] * density)
                    op['sparse']['parameters'] = int(op['parameters'] * density)

        profile = self.set_sparse_profile()

        if self.verbose:
            v_str += f'New density: {self.complexity_profile["sparse"]["density"]:.4}'
            print(v_str)
        return profile

    def state_dict(self):
        state_dict = {
            'profile': self.profile,
            'sparse_profile': self.sparse_profile,
            'samples_processed': self.samples_processed,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.profile = state_dict['profile']
        self.sparse_profile = state_dict['sparse_profile']
        self.samples_processed = state_dict['samples_processed']

    def update_profile(self, samples):
        self.samples_processed += samples

        self.profile['adds'] += samples * self.complexity_profile['adds']
        self.profile['multiplies'] += samples * self.complexity_profile['multiplies']
        self.profile['logical'] += samples * self.complexity_profile['logical']
        self.profile['other'] += samples * self.complexity_profile['other']
        self.profile['flops'] += samples * self.complexity_profile['flops']
        self.profile['parameters'] += samples * self.complexity_profile['parameters']

        if self.sparse_profile:
            self.sparse_profile['adds'] += samples * self.complexity_profile['sparse']['adds']
            self.sparse_profile['multiplies'] += samples * self.complexity_profile['sparse']['multiplies']
            self.sparse_profile['logical'] += samples * self.complexity_profile['sparse']['logical']
            self.sparse_profile['other'] += samples * self.complexity_profile['sparse']['other']
            self.sparse_profile['flops'] += samples * self.complexity_profile['sparse']['flops']
            self.sparse_profile['parameters'] += samples * self.complexity_profile['sparse']['parameters']

    def dense_flops_used(self):
        return self.profile['flops']

    def avg_dense_flops_used(self):
        return self.profile['flops'] / self.samples_processed

    def sparse_flops_used(self):
        return self.sparse_profile['flops']

    def avg_sparse_flops_used(self):
        return self.sparse_profile['flops'] / self.samples_processed

    def sparse_parameters_used(self):
        return self.sparse_profile['parameters']

    def avg_sparse_parameters_used(self):
        return self.sparse_profile['parameters'] / self.samples_processed

    def dense_macs_used(self):
        return self.profile['multiplies']

    def sparse_macs_used(self):
        return self.sparse_profile['multiplies']

    def avg_sparse_macs_used(self):
        return self.sparse_profile['multiplies'] / self.samples_processed
