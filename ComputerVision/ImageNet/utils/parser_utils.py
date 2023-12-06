import argparse


def get_base_parser():
    parser = argparse.ArgumentParser(
        description='PyTorch Classification Training', add_help=True
    )

    # fmt: off
    # Data parameters
    parser.add_argument('--data_path', default='/cb/datasets/cv/imagenet/imagenet1k_ilsvrc2012/',
            type=str, help='dataset path')
    parser.add_argument('--interpolation', default='bilinear', type=str,
            help='the interpolation method (default: bilinear)')
    parser.add_argument('--val_resize_size', default=256, type=int,
            help='the resize size used for validation (default: 256)')
    parser.add_argument('--val_crop_size', default=224, type=int,
            help='the central crop size used for validation (default: 224)')
    parser.add_argument('--train_crop_size', default=224, type=int,
            help='the random crop size used for training (default: 224)')
    parser.add_argument('--batch_size', default=32, type=int,
            help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--workers', default=16, type=int,
            help='number of data loading workers (default: 16)')
    parser.add_argument('--input_size', default=[3, 224, 224], nargs=3, type=int,
            help=(
                'Input all image dimensions (d h w, e.g. --input_size 3 224 224),'
                ' uses model default if empty',
            )
    )

    # Model parameters
    parser.add_argument('--model', default='resnet_18', type=str,
            help='model to train')
    parser.add_argument('--block_name', type=str, default='Block', choices=['Block', 'PreActBlock', 'BlockISOFlopParallel'],
            help=(
                'Choose the block in ResNet architecture. Default is the original'
                'block.'
            )
    )
    parser.add_argument('--zero_init_residual', action='store_false', default=True,
            help='Initialize layer before residuals with zero weights')
    parser.add_argument('--mbnet_width_mult', type=float, default=1.0,
            help='which width multiplier to use for ImageNet')

    # Optimizer parameters
    parser.add_argument('--optimizer', default='sgd', type=str,
            help='optimizer')
    parser.add_argument('--lr', default=0.1, type=float,
            help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
            help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--nesterov', action='store_true', default=False,
            help='apply nesterov for sgd optimizer')
    parser.add_argument('--filter_norm_and_bias', action='store_false', default=True,
            help='filter parameters based on weight decays')
    parser.add_argument('--label_smoothing', default=0.0, type=float,
            help='label smoothing (default: 0.0)', dest='label_smoothing')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
            help='gradient accumulation (default: 1)')
    parser.add_argument('--epochs', default=90, type=int,
            help='number of total epochs to run')

    # Learning Rate Parameters
    parser.add_argument('--lr_scheduler', default='cosineannealing', type=str,
            help='the lr scheduler (default: cosineannealing)')
    parser.add_argument('--lr_step_size', default=30, type=int,
            help='decrease lr every step-size epochs')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr_min', default=0.0, type=float,
            help='minimum lr of lr schedule (default: 0.0)')
    parser.add_argument('--lr_milestones', nargs='+', type=int,
            help='milestones for multisteplr')
    parser.add_argument('--lr_warmup_epochs', default=0, type=int,
        help='the number of epochs to warmup (default: 0)')
    parser.add_argument('--lr_warmup_method', default='constant', type=str,
            help='the warmup method (default: constant)')
    parser.add_argument('--lr_warmup_decay', default=0.01, type=float,
            help='the decay for lr')

    # Misc. parameters
    parser.add_argument('--print_freq', default=500, type=int,
            help='print frequency')
    parser.add_argument('--output_dir', default='.', type=str,
            help='path to save outputs')
    parser.add_argument('--resume_ckpt', default='', type=str,
            help='path of checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int,
            help='start epoch')
    parser.add_argument('--seed', type=int, default=0,
            help='random seed (default: 0)')
    parser.add_argument('--device', default='cuda', type=str,
            help='device (Use cuda or cpu Default: cuda)')

    # Distributed parameters
    parser.add_argument('--local_rank', default=0, type=int,
            help='rank for running distributed processes')
    parser.add_argument('--sync_bn', dest='sync_bn', action='store_true',
            help='Use sync batch norm')

    # Mixed precision training parameters
    parser.add_argument('--amp', action='store_true',
            help='Use mixed precision for training')
    parser.add_argument('--clip_grad_norm', default=None, type=float,
            help='the maximum gradient norm (default None)')
    parser.add_argument('--flop_budget', type=float, default=None)

    # Sparsity args
    parser.add_argument('--maskopt', type=str, default=None, choices=['rigl', 'set', 'static'],
            help='enable dynamic mask optimization (default: None)')

    # Sift args
    parser.add_argument('--sift_scaling', action='store_true', default=False,
            help='Run sift for imagenet models')
    parser.add_argument('--sift_family', type=str, default=None,
            help='Which sift family to run')
    # fmt: on

    return parser
