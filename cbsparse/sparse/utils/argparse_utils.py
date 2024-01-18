"""
Released under BSD 3-Clause License,
Copyright (c) 2021 Cerebras Systems Inc.
All rights reserved.
"""
import argparse


def add_mask_optimizer_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument(
        '--maskopt_sparsity',
        default=0.0,
        help='Final sparsity level. Either a float between (0,1) or str("from_weights")',
    )
    parser.add_argument(
        '--maskopt_start_sparsity',
        type=float,
        default=None,
        help='Starting sparsity level. 0=None',
    )
    parser.add_argument(
        '--maskopt_sparsity_sched',
        type=str,
        default='constant',
        help='Mask sparsity schedule. '
        '(default: constant) '
        '(options: constant, linear, cosine)',
    )
    parser.add_argument(
        '--maskopt_sparsity_sched_start_itr',
        type=int,
        default=0,
        metavar='N',
        help='Itr at which to start sparsity sched. (default: 0)',
    )
    parser.add_argument(
        '--maskopt_sparsity_sched_end_itr',
        type=int,
        default=None,
        metavar='N',
        help='Itr at which to end sparsity sched. '
        '(default: None = maskopt_end_iteration)',
    )
    parser.add_argument(
        '--maskopt_begin_iteration',
        type=int,
        default=0,
        metavar='N',
        help='Itr at which to start mask updates. (default: 0)',
    )
    parser.add_argument(
        '--maskopt_end_iteration',
        type=float,
        default=None,
        metavar='N',
        help='Itr at which to end mask updates. (default: None = 0.8 * epochs)',
    )
    parser.add_argument(
        '--maskopt_frequency',
        type=int,
        default=100,
        metavar='N',
        help='frequency at which mask updates are made. (default: 100)',
    )
    parser.add_argument(
        '--maskopt_drop_fraction',
        type=float,
        default=0.3,
        help='fraction of params to drop during mask update. (default: 0.3)',
    )
    parser.add_argument(
        '--maskopt_drop_fraction_anneal',
        type=str,
        default='cosine',
        help='aneal method for maskopt_drop_fraction (default: cosine). '
        'Options are: constant, cosine, exponential_EX)',
    )
    parser.add_argument(
        '--maskopt_mask_distribution',
        type=str,
        default='uniform',
        help='sparsity mask distribution. '
        '(default: erk) (options: uniform, erk, er, from_weights)',
    )
    parser.add_argument(
        '--maskopt_mask_init_method',
        type=str,
        default='random',
        help='mask initialization method. '
        '(default: random) '
        '(options: random, dense, zeros, lamp, topk, from_weights)',
    )
    parser.add_argument(
        '--maskopt_erk_power_scale',
        type=float,
        default=1.0,
        help='maskopt erk power scale. (default: 1.0)',
    )
    parser.add_argument(
        '--maskopt_mask_metrics',
        action='store_true',
        default=False,
        help='compute mask opt running metrics. (default: False)',
    )
    parser.add_argument(
        '--maskopt_verbose',
        action='store_true',
        default=False,
        help='print maskopt running information. (default: False)',
    )
    parser.add_argument(
        '--no_resume_mask_opt',
        action='store_true',
        default=False,
        help='prevent resume of mask optimizer state',
    )
    parser.add_argument(
        '--maskopt_adjust_sparsity',
        action='store_true',
        default=False,
        help='adjust sparsity according to skipped layers',
    )
    parser.add_argument(
        '--maskopt_fused_bn_updates',
        action='store_true',
        default=False,
        help='use fused batch norm during updates',
    )

    return parser


def add_mask_optimizer_specific_args_cv(parent_parser):
    parser = add_mask_optimizer_specific_args(parent_parser)
    parser = argparse.ArgumentParser(parents=[parser], add_help=False)

    # select sparsity subset
    parser.add_argument(
        '--maskopt_sparsity_head',
        action='store_true',
        default=False,
        help='enable head sparsification in should_sparsify',
    )
    parser.add_argument(
        '--maskopt_sparsity_stem',
        action='store_true',
        default=False,
        help='enable stem sparsification in should_sparsify',
    )
    parser.add_argument(
        '--maskopt_sparsity_norm',
        action='store_true',
        default=False,
        help='enable norm sparsification in should_sparsify',
    )

    # ViT specific flags
    parser.add_argument(
        '--maskopt_vit_just_att',
        action='store_true',
        default=False,
        help='enable vit_just_att in vit_special_should_sparsify',
    )
    parser.add_argument(
        '--maskopt_vit_just_ffn',
        action='store_true',
        default=False,
        help='enable vit_just_ffn in vit_special_should_sparsify',
    )

    # mlp-mixer specific flags
    parser.add_argument(
        '--maskopt_mixer_just_mlp_channels',
        action='store_true',
        default=False,
        help='enable mixer_just_mlp_channels in mixer_special_should_sparsify',
    )
    parser.add_argument(
        '--maskopt_mixer_just_mlp_tokens',
        action='store_true',
        default=False,
        help='enable mixer_just_mlp_tokens in mixer_special_should_sparsify',
    )

    return parser
