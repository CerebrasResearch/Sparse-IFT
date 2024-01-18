from functools import partial
from warnings import warn
import re


def generic_cv_should_sparsify(n, p, args):
    is_sparse = False
    if args.maskopt:
        is_sparse = True

    # remove first layer
    if 'stem' in n or (len(p.shape) > 1 and p.size(1) == args.input_size[0]):
        if is_sparse and args.maskopt_sparsity_stem:
            return True
        return False

    # remove last layer
    if 'head' in n or args.num_classes in p.shape:
        if is_sparse and args.maskopt_sparsity_head:
            return True
        return False

    # remove norm parameters
    if 'norm' in n:
        if is_sparse and args.maskopt_sparsity_norm:
            return True
        return False

    # misc
    # for ViT and ViT type models
    if 'cls_token' in n or 'pos_embed' in n:
        return False

    # removes all biases and other vectors; return the rest
    return len(p.shape) > 1


def resnet_should_sparsify(n, p, args):
    is_sparse = False
    if args.maskopt:
        is_sparse = True

    # remove first layer
    if n.startswith('conv1'):
        if is_sparse and args.maskopt_sparsity_stem:
            return True
        return False

    # remove last layer
    if n.startswith('fc'):
        if is_sparse and args.maskopt_sparsity_head:
            return True
        return False

    # remove norm parameters
    if re.search(f'bn\d+.', n):
        if is_sparse and args.maskopt_sparsity_norm:
            return True
        return False

    # removes all biases and other vectors; return the rest
    return len(p.shape) > 1


def get_should_sparsify(args, model):
    if callable(getattr(model, 'should_sparsify', None)):
        fn = model.should_sparsify
    else:
        if re.fullmatch(f'resnet\d+', args.model.lower()):
            fn = resnet_should_sparsify
        else:
            warn(f'model {type(model)} does not implement should_sparsify '
                 'and there is model-specific implementation of should_sparsify '
                 'in sparse/utils/should_sparsify_utils.py. '
                 'Falling back to generic_cv_should_sparsify which may not be correct!')
            fn = generic_cv_should_sparsify

    return partial(fn, args=args)
