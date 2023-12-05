# Copyright 2023 Cerebras Systems Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


class BlockISOFlopParallel(nn.Module):
    """ Block where RepVGG style parallel blocks are trained for each conv.
        Each block is sparse such that overall FLOPs remain same.
    """

    expansion: int = 1

    def __init__(
        self, f_in: int, f_out: int, downsample=False, num_parallel_branch=1
    ):
        '''
        Initializes the BlockISOFlopParallel block with the specified parameters.

        Args:
            f_in (int): Number of input features.
            f_out (int): Number of output features.
            downsample (bool, optional): Whether to apply downsampling (default is False).
            num_parallel_branch (int, optional): Number of parallel sparse conv blocks (default is 1).
        '''

        super(BlockISOFlopParallel, self).__init__()
        stride = 2 if downsample else 1
        self.num_parallel_branch = num_parallel_branch

        # Replicate block of Conv and BN
        self.conv1_list, self.bn1_list = nn.ModuleList(), nn.ModuleList()
        for _ in range(self.num_parallel_branch):
            self.conv1_list.append(
                nn.Conv2d(
                    f_in,
                    f_out,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                )
            )
            self.bn1_list.append(nn.BatchNorm2d(f_out))

        # Replicate block of Conv and BN
        self.conv2_list, self.bn2_list = nn.ModuleList(), nn.ModuleList()
        for _ in range(self.num_parallel_branch):
            self.conv2_list.append(
                nn.Conv2d(
                    f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False
                )
            )
            self.bn2_list.append(nn.BatchNorm2d(f_out))

        # No parameters for shortcut connections.
        if downsample or f_in != f_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    f_in, f_out, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(f_out),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # 1st Conv + BN
        out_first = 0
        for idx in range(self.num_parallel_branch):
            # additional relu in branch
            out_first = out_first + F.relu(
                self.bn1_list[idx](self.conv1_list[idx](x))
            )
        out_first = F.relu(out_first)

        # 2nd Conv + BN
        out_second = 0
        for idx in range(self.num_parallel_branch):
            # additional relu in branch
            out_second = out_second + F.relu(
                self.bn2_list[idx](self.conv2_list[idx](out_first))
            )

        out_second += self.shortcut(x)
        out = F.relu(out_second)
        return out


class Block(nn.Module):
    """A ResNet block (original without pre-activation)."""

    expansion: int = 1

    def __init__(self, f_in: int, f_out: int, downsample=False):
        """
        Initializes the Block with the specified parameters.

        Args:
            f_in (int): Number of input features.
            f_out (int): Number of output features.
            downsample (bool, optional): Whether to apply downsampling (default is False).
        """

        super(Block, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(
            f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(f_out)
        self.conv2 = nn.Conv2d(
            f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(f_out)

        # No parameters for shortcut connections.
        if downsample or f_in != f_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(f_out),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    """A ResNet PreAct block."""

    expansion: int = 1

    def __init__(self, f_in: int, f_out: int, downsample=False):
        """
        Initializes the PreActBlock with the specified parameters.

        Args:
            f_in (int): Number of input features.
            f_out (int): Number of output features.
            downsample (bool, optional): Whether to apply downsampling (default is False).
        """

        super(PreActBlock, self).__init__()

        stride = 2 if downsample else 1
        self.bn1 = nn.BatchNorm2d(f_in)
        self.conv1 = nn.Conv2d(
            f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(f_out)
        self.conv2 = nn.Conv2d(
            f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False
        )

        # No parameters for shortcut connections.
        if downsample or f_in != f_out:
            self.shortcut = (
                nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        x = F.relu(self.bn1(x))
        out = self.conv1(x)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class Model(nn.Module):
    """A residual neural network as originally designed for CIFAR-10/100."""

    def __init__(self, plan, num_classes=1000, block_args={}):
        """
        Initializes the Model with the specified parameters.

        Args:
            plan (list): Contains width and num blocks per ResNet block.
            num_classes (int, optional): Number of classes for the output layer (default is 1000).
            block_args (dict, optional): Arguments for blocks dictionary (default is {}).
        """

        super(Model, self).__init__()
        self.block_name = block_args['block_name']
        self.block = str_to_class(self.block_name)

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv_stem = nn.Conv2d(
            3, current_filters, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn = nn.BatchNorm2d(current_filters)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                if self.block == BlockISOFlopParallel:
                    blocks.append(
                        self.block(
                            current_filters,
                            filters,
                            downsample,
                            block_args['num_parallel_branch'],
                        )
                    )
                else:
                    blocks.append(
                        self.block(current_filters, filters, downsample)
                    )
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)
        if self.block == PreActBlock:
            self.norm = nn.BatchNorm2d(filters)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0] * self.block.expansion, num_classes)

        # Initialize
        self.init_weights(zero_init_last=block_args['zero_init_residual'])

    def init_weights(self, zero_init_last=False):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_last:
            for m in self.modules():
                if isinstance(m, Block):
                    nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        out = F.relu(self.bn(self.conv_stem(x)))
        out = self.maxpool(out)
        out = self.blocks(out)
        if self.block == PreActBlock:
            out = F.relu(self.norm(out))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    @staticmethod
    def get_model_from_name(args):
        """Get model given args, including scaling for width or parallelism

        Args:
            block_args (dict): arguments for the resnet block.
        """

        # num_classes
        num_classes = args.num_classes

        # block arguments
        resnet_block_args = {}
        resnet_block_args['block_name'] = args.block_name
        resnet_block_args['zero_init_residual'] = args.zero_init_residual

        num_branches = 1
        if args.sift_scaling and args.sift_family == 'sparse_parallel':
            num_branches = args.scaling_factor
        resnet_block_args['num_parallel_branch'] = num_branches

        # base model arguments
        model_name = args.model
        name = model_name.split('_')

        base_width = 64
        if args.sift_scaling and args.sift_family == 'sparse_wide':
            base_width = int(base_width * args.scaling_factor)
        W = base_width

        base_depth = int(name[1])
        if base_depth == 18:
            plan = [(W, 2), (2 * W, 2), (4 * W, 2), (8 * W, 2)]
        elif base_depth == 34:
            plan = [(W, 3), (2 * W, 4), (4 * W, 6), (8 * W, 3)]

        return Model(plan, num_classes, resnet_block_args)
