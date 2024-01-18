"""
Released under BSD 3-Clause License,
Copyright (c) 2021 Cerebras Systems Inc.
All rights reserved.
"""
import argparse

import torch


def generic_cv_should_dropconnect(n, m, args, desc={torch.nn.Linear: 'weight', torch.nn.Conv2d: 'weight',}):
    if type(m) not in desc.keys():
        return False
    
    p = getattr(m, desc[type(m)])

    # remove first layer
    if len(p.shape) > 1 and p.size(1) == args.input_size[0]:
        return False

    # remove last layer
    if args.num_classes in p.shape:
        return False

    # remove norm parameters
    if 'norm' in n:
        return False

    # misc
    # for ViT and ViT type models
    if 'cls_token' in n or 'pos_embed' in n:
        return False

    # removes all biases and other vectors; return the rest
    return len(p.shape) > 1


def add_drop_connect_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument(
        '--dc_rate',
        type=float,
        default=0.5,
        help='drop connect rate (final dc_rate if dc_sched used). 0=None (default: 0.5)',
    )
    parser.add_argument(
        '--dc_init_rate',
        type=float,
        default=None,
        help='starting dc_rate if dc_sched used.',
    )
    parser.add_argument(
        '--dc_sched',
        type=str,
        default='constant',
        help='mask sparsity schedule. '
        '(default: constant) '
        '(options: constant, cosine)',
    )
    parser.add_argument(
        '--dc_begin_iteration',
        type=int,
        default=0,
        metavar='N',
        help='Itr at which to start dc sched. (default: 0)',
    )
    parser.add_argument(
        '--dc_end_iteration',
        type=int,
        default=None,
        metavar='N',
        help='Itr at which to end dc sched. '
        '(default: None = 0.8 * total train time)',
    )
    parser.add_argument(
        '--dc_unscale',
        action='store_true',
        default=False,
        help='use unscale in dropconnect',
    )
    parser.add_argument(
        '--dc_inter_iteration',
        type=int,
        default=None,
        metavar='N',
        help='Itr at which we switch from one schedule to another. '
        '(default: None = 0.2 * total train time)',
    )
    parser.add_argument(
        '--dc_inter_rate',
        type=float,
        default=None,
        help='intermediate dc_rate if bilinear dc_sched used.',
    )
    parser.add_argument(
        '--dc_drop_std',
        type=float,
        default=None,
        help='std to top-k version of mask generation.',
    )
    parser.add_argument(
        '--dc_binomialsamp',
        action='store_true',
        default=False,
        help='use binomial sampling to generate mask.',
    )

    return parser
