import argparse
import hashlib
import itertools
import os
import shutil
import tarfile
import uuid
from collections import OrderedDict

import yaml
import copy


def create_argparser():
    # fmt: off
    parser = argparse.ArgumentParser(description='Prepare experiments and their folders before launching')
    parser.add_argument('--job-name', type=str, required=True,
                        help='Get job name to run')
    parser.add_argument('--base-dir', type=str, default='/cb/home/vithu/extra-storage/want/sift/pretraining',
                        help='base dir to set up launch files and logs')
    parser.add_argument('--overwrite', default=False, const=True, action='store_const',
                        help='Overwrite the experiment-directories')
    parser.add_argument('--run-cifar', default=False, action='store_true',
                        help='Run CIFAR or ImageNet')
    parser.add_argument('--base-cfg', type=str, required=True,
                        help='which base config to run')
    parser.add_argument('--exp-cfg', type=str, required=True,
                        help='which configs to run for the experiment')
    # fmt: on
    return parser


def get_run_name(exp_params):
    _exp_name = []

    for key, value in exp_params.items():
        if key == 'pretrain_artifact_dir':
            main_val = value.split('/')[-1]
            # 5 chars since 32 chars is way too long for life
            hex_string = hashlib.md5(main_val.encode()).hexdigest()[:5]
            _exp_name.append(f'checkpoint_{hex_string}')
        elif key == 'model':
            continue
        elif key == 'sift_scaling':
            continue
        elif key == 'sift_family':
            _exp_name.append(f'sift_family_{value}')
        elif key == 'block_name':
            continue
        elif key == 'maskopt':
            _exp_name.append(f'{value}')
        elif key == 'maskopt_sparsity':
            _exp_name.append(f'sparse_{value}')
        elif key == 'maskopt_mask_distribution':
            _exp_name.append(f'distri_{value}')
        elif key == 'maskopt_mask_init_method':
            _exp_name.append(f'init_{value}')
        elif key == 'maskopt_end_iteration':
            _exp_name.append(f'end_{value}')
        elif key == 'maskopt_frequency':
            _exp_name.append(f'freq_{value}')
        elif key == 'maskopt_drop_fraction':
            _exp_name.append(f'df_{value}')
        elif key == 'maskopt_drop_fraction_anneal':
            _exp_name.append(f'anneal_{value}')
        else:
            _exp_name.append(f'{key}_{value}')

    _exp_name = '_'.join(_exp_name)
    return _exp_name


def create_run_command(args, code_source_dir, exp_dir):
    # Command common to hyper-param experiments
    setup_exp_command = f'cd {code_source_dir}\n'
    if args.run_cifar:
        setup_exp_command += 'cd CIFAR\n'
    else:
        setup_exp_command += 'cd ImageNet\n'
    setup_exp_command += f'env $(cat {exp_dir}/experiment_params.env | xargs) bash ./scripts/launch_pretraining.sh\n'
    return setup_exp_command


def main():
    parser = create_argparser()
    args = parser.parse_args()

    # This is the directory which will be archived
    # base shell would be launched from here, post decompression
    code_source_dir = os.environ.get('CODE_SOURCE_DIR', None)
    assert code_source_dir is not None and code_source_dir.strip() != '', \
        "Please specify the source code directory using the environment variable CODE_SOURCE_DIR."

    # Stores launch.txt and logs files for your jobs
    dir_launch_files = os.path.join(args.base_dir, 'launch_utils')
    efs_storage_dir = os.path.join(args.base_dir, 'logs', args.job_name)
    launch_file_prefix = args.job_name

    # Create necessary folders and files
    if not os.path.exists(dir_launch_files):
        os.makedirs(dir_launch_files)

    launch_file = f'{dir_launch_files}/launch_{launch_file_prefix}.txt'
    if not args.overwrite:
        assert (
            os.path.exists(launch_file) is False
        ), f'Launch file already present at: {launch_file}'
    else:
        if os.path.exists(launch_file):
            os.remove(launch_file)

    with open(args.base_cfg, 'r') as stream:
        exp_base_params = yaml.safe_load(stream)

    with open(args.exp_cfg, 'r') as stream:
        exp_cfg = yaml.safe_load(stream)

    # get all elements as list to gridify them
    for k, v in exp_cfg.items():
        if not isinstance(v, list):
            exp_cfg[k] = [v]

    # Generate all the possible combinations
    keys, values = zip(*exp_cfg.items())
    exp_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f'Preparing scripts to launch {len(exp_params)} experiments')

    for idx in range(len(exp_params)):
        _exp_params = OrderedDict(exp_params[idx])
        _exp_name = get_run_name(_exp_params)

        # Construct the full command to be launched, and name of experiment-dir
        # (to store code and artifacts)
        _exp_dir = f'{efs_storage_dir}/{_exp_name}'

        # Write the path to this folder in launch_file
        with open(launch_file, 'a+') as _file:
            _file.write(f'{_exp_dir}\n')

        # Make the root dir
        if os.path.exists(_exp_dir):
            if args.overwrite:
                print(f'Deleting experiment-dir: {_exp_dir}')
                shutil.rmtree(_exp_dir, ignore_errors=True)
            else:
                raise OSError(
                    f'Old artifacts found at: {_exp_dir}\n.'
                    + f' Pass in the --overwrite flag to overwrite the content'
                )
        os.makedirs(_exp_dir)

        # Copy, update and dump the experiment_params.env
        exp_base_params_copy = copy.deepcopy(exp_base_params)
        exp_base_params_copy.update(_exp_params)
        with open(f'{_exp_dir}/experiment_params.env', 'w') as export_file:
            # Write out base and exp parameters for experiment
            for key in exp_base_params_copy.keys():
                export_file.write(f'{key}={exp_base_params_copy[key]}\n')

            # Write result directory
            export_file.write(f'output_dir={_exp_dir}\n')
            export_file.write(f'job_name={_exp_name}\n')

        # Dump the run.sh
        with open(f'{_exp_dir}/run.sh', 'w') as _file:
            _file.write('#!/bin/bash \n\n')
            setup_exp_command = create_run_command(args, code_source_dir, _exp_dir)
            _file.write(f'{setup_exp_command}')
        
        print(f'bash {_exp_dir}/run.sh')


if __name__ == '__main__':
    main()
