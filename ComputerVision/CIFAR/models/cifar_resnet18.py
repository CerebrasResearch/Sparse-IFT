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
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


class DopedConv3x3(nn.Module):
    """Block where W=UV decomposed blocks are trained for each conv.
       Each block is sparse such that overall FLOPs remain same.
    """

    def __init__(self, f_in: int, f_out: int, stride=1, sparsity=0.01):
        """
        Args:
            f_in (int): Number of input features.
            f_out (int): Number of output features.
            stride (int, optional): Stride value for the operation (default is 1).
            sparsity (float, optional): Sparsity parameter (default is 0.01).
        """

        super(DopedConv3x3, self).__init__()

        self.sparse_conv = nn.Conv2d(
            f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False,
        )

        rank = self.isoflop_width_control(f_in, f_out, sparsity=sparsity)
        self.U_conv3x3 = nn.Conv2d(
            f_in, rank, kernel_size=3, stride=stride, padding=1, bias=False,
        )

        self.V_conv1x1 = nn.Conv2d(
            rank, f_out, kernel_size=1, stride=1, bias=False
        )

        # Placeholder for storing weight-decay (when U, V are sampled)
        self.dynamic_weight_norm = None

        self.apply_init()

    def isoflop_width_control(self, f_in, f_out, sparsity):
        numerator = (sparsity) * ((3 * 3) * (f_in * f_out))
        denominator = (3 * 3 * f_in) + f_out
        rank = numerator / denominator
        rank = int(math.floor(rank))

        return rank

    def apply_init(self):
        # Random init using Kaiming He's init
        nn.init.kaiming_normal_(
            self.U_conv3x3.weight, mode='fan_out', nonlinearity='relu'
        )
        nn.init.kaiming_normal_(
            self.V_conv1x1.weight, mode='fan_out', nonlinearity='relu'
        )

    def sample_UV(self):
        # There is no stochastic sampling in this case, we return both U, V.
        _U, _V = self.U_conv3x3.weight, self.V_conv1x1.weight

        return _U, _V

    def compute_weight_norm(self):
        """Compute l2 norm on folded U, V.
        This compute is dynamic when we sample U, V.
        """

        U_conv3x3, V_conv1x1 = self.sample_UV()
        _low_channel, _in_channel, _, _ = U_conv3x3.shape
        _U = U_conv3x3.reshape(_low_channel, -1)  # (lowrank, in_channel*3*3)
        _V = V_conv1x1.squeeze(-1).squeeze(-1)  # (out_channel, lowrank)
        self.dynamic_weight_norm = (torch.mm(_V, _U) ** 2).sum()

    def compute_weight_norm_fullW(self):
        """Compute l2 norm on folded U, V.
        This compute is dynamic when we sample U, V.
        """

        U_conv3x3, V_conv1x1 = self.sample_UV()
        _low_channel, _in_channel, _, _ = U_conv3x3.shape
        _U = U_conv3x3.reshape(_low_channel, -1)  # (lowrank, in_channel*3*3)
        _V = V_conv1x1.squeeze(-1).squeeze(-1)  # (out_channel, lowrank)

        _sparse_W = self.sparse_conv.weight
        _sparse_W = _sparse_W.view(_sparse_W.size(0), -1)

        self.dynamic_weight_norm = ((_sparse_W + torch.mm(_V, _U)) ** 2).sum()

    def forward(self, x):

        sparse_out = self.sparse_conv(x)

        # Sample U, V
        lowrank_out = self.U_conv3x3(x)
        lowrank_out = self.V_conv1x1(lowrank_out)

        out = sparse_out + lowrank_out

        # Dynamically compute value of weight-decay based on sampling
        self.compute_weight_norm()

        return out


class BlockISOFlopDoping(nn.Module):
    """Block where W=Sparse(W)+UV decomposed blocks are trained for each conv.
       Each block is sparse such that overall FLOPs remain same.

    """

    expansion: int = 1

    def __init__(
        self, f_in: int, f_out: int, downsample=False, sparsity_level=0.0
    ):
        """
        Args:
            f_in (int): Number of input features.
            f_out (int): Number of output features.
            downsample (bool, optional): Whether to apply downsampling (default is False).
            sparsity_level (float, optional): Sparsity level for the block (default is 0.0).
        """

        super(BlockISOFlopDoping, self).__init__()
        stride = 2 if downsample else 1

        self.conv1 = DopedConv3x3(
            f_in, f_out, stride=stride, sparsity=sparsity_level
        )
        self.bn1 = nn.BatchNorm2d(f_out)

        self.conv2 = DopedConv3x3(
            f_out, f_out, stride=1, sparsity=sparsity_level
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


class LowRankConv3x3(nn.Module):
    """
    Block where W=UV decomposed blocks are trained for each conv.
    Each block is sparse such that overall FLOPs remain the same.

    """

    def __init__(
        self,
        f_in: int,
        f_out: int,
        stride: int = 1,
        sparsity: float = 0.0,
        sparse_u_only: bool = False,
        uv_layer: str = 'linear',
    ):
        """
        Args:
            f_in (int): Number of input features.
            f_out (int): Number of output features.
            stride (int, optional): Stride value for the operation (default is 1).
            sparsity (float, optional): Sparsity parameter (default is 0.0).
            sparse_u_only (bool, optional): Whether to use only sparse U in decomposition (default is False).
            uv_layer (str, optional): Type of UV layer ('linear' by default).
        """

        super(LowRankConv3x3, self).__init__()

        self.uv_layer = uv_layer
        expanded_planes = self.isoflop_width_control(
            f_in, f_out, sparsity=sparsity, sparse_u_only=sparse_u_only
        )

        self.U_conv3x3 = nn.Conv2d(
            f_in,
            expanded_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        if self.uv_layer == 'nonlinear_norelu':
            self.U_conv3x3_bn = nn.BatchNorm2d(expanded_planes)
        elif self.uv_layer == 'nonlinear_relu':
            self.U_conv3x3_bn = nn.BatchNorm2d(expanded_planes)
            self.relu = nn.ReLU(inplace=True)

        self.V_conv1x1 = nn.Conv2d(
            expanded_planes, f_out, kernel_size=1, stride=1, bias=False
        )

        # Placeholder for storing weight-decay (when U, V are sampled)
        self.dynamic_weight_norm = None

        self.apply_init()

    def isoflop_width_control(self, f_in, f_out, sparsity, sparse_u_only=False):
        if sparse_u_only:
            expanded_planes = ((3 * 3) * f_in * f_out) / (
                ((1 - sparsity) * (3 * 3) * f_in) + f_out
            )
        else:
            expanded_planes = ((3 * 3) * f_in * f_out) / (
                (1 - sparsity) * ((3 * 3) * f_in + f_out)
            )

        expanded_planes = int(math.ceil(expanded_planes))

        return expanded_planes

    def apply_init(self):
        # Random init using Kaiming He's init
        nn.init.kaiming_normal_(
            self.U_conv3x3.weight, mode='fan_out', nonlinearity='relu'
        )
        nn.init.kaiming_normal_(
            self.V_conv1x1.weight, mode='fan_out', nonlinearity='relu'
        )

    def sample_UV(self):
        # There is no stochastic sampling in this case, we return both U, V.
        _U, _V = self.U_conv3x3.weight, self.V_conv1x1.weight

        return _U, _V

    def compute_weight_norm(self):
        """Compute l2 norm on folded U, V.
        This compute is dynamic when we sample U, V.
        """

        U_conv3x3, V_conv1x1 = self.sample_UV()
        _low_channel, _in_channel, _, _ = U_conv3x3.shape
        _U = U_conv3x3.reshape(_low_channel, -1)  # (lowrank, in_channel*3*3)
        _V = V_conv1x1.squeeze(-1).squeeze(-1)  # (out_channel, lowrank)

        self.dynamic_weight_norm = (torch.mm(_V, _U) ** 2).sum()

    def forward(self, x):
        # Sample U, V
        out = self.U_conv3x3(x)

        if self.uv_layer == 'nonlinear_norelu':
            out = self.U_conv3x3_bn(out)
        elif self.uv_layer == 'nonlinear_relu':
            out = self.relu(self.U_conv3x3_bn(out))

        out = self.V_conv1x1(out)

        # Dynamically compute value of weight-decay based on sampling
        if self.uv_layer == 'linear':
            self.compute_weight_norm()

        return out


class WideLowRankConv3x3(nn.Module):
    """Block where W=UV decomposed blocks are trained for each conv.
       Each block is sparse such that overall FLOPs remain same.
    """

    def __init__(
        self,
        f_in: int,
        f_out: int,
        stride=1,
        width_scaling_factor=0.0,
        uv_layer='linear',
    ):
        """
        Args:
            f_in (int): Number of input features.
            f_out (int): Number of output features.
            stride (int, optional): Stride value for the operation (default is 1).
            width_scaling_factor (float, optional): Scaling factor for width (default is 0.0).
            uv_layer (str, optional): Type of UV layer ('linear' by default).
        """

        super(WideLowRankConv3x3, self).__init__()

        self.uv_layer = uv_layer
        expanded_planes = self.isoflop_width_control(
            f_in, f_out, width_scaling_factor=width_scaling_factor
        )

        self.U_conv3x3 = nn.Conv2d(
            f_in,
            expanded_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        if self.uv_layer == 'nonlinear_norelu':
            self.U_conv3x3_bn = nn.BatchNorm2d(expanded_planes)
        elif self.uv_layer == 'nonlinear_relu':
            self.U_conv3x3_bn = nn.BatchNorm2d(expanded_planes)
            self.relu = nn.ReLU(inplace=True)

        self.V_conv1x1 = nn.Conv2d(
            expanded_planes, f_out, kernel_size=1, stride=1, bias=False
        )

        # Placeholder for storing weight-decay (when U, V are sampled)
        self.dynamic_weight_norm = None

        self.apply_init()

    def isoflop_width_control(self, f_in, f_out, width_scaling_factor):

        old_f_in = int(f_in / width_scaling_factor)
        old_f_out = int(f_out / width_scaling_factor)

        expanded_planes = (((3 * 3) * old_f_in * old_f_out)) / (
            ((3 * 3) * f_in + f_out)
        )

        expanded_planes = int(math.ceil(expanded_planes))

        return expanded_planes

    def apply_init(self):
        # Random init using Kaiming He's init
        nn.init.kaiming_normal_(
            self.U_conv3x3.weight, mode='fan_out', nonlinearity='relu'
        )
        nn.init.kaiming_normal_(
            self.V_conv1x1.weight, mode='fan_out', nonlinearity='relu'
        )

    def sample_UV(self):
        # There is no stochastic sampling in this case, we return both U, V.
        _U, _V = self.U_conv3x3.weight, self.V_conv1x1.weight

        return _U, _V

    def compute_weight_norm(self):
        """Compute l2 norm on folded U, V.
        This compute is dynamic when we sample U, V.
        """

        U_conv3x3, V_conv1x1 = self.sample_UV()
        _low_channel, _in_channel, _, _ = U_conv3x3.shape
        _U = U_conv3x3.reshape(_low_channel, -1)  # (lowrank, in_channel*3*3)
        _V = V_conv1x1.squeeze(-1).squeeze(-1)  # (out_channel, lowrank)

        self.dynamic_weight_norm = (torch.mm(_V, _U) ** 2).sum()

    def forward(self, x):
        # Sample U, V
        out = self.U_conv3x3(x)

        if self.uv_layer == 'nonlinear_norelu':
            out = self.U_conv3x3_bn(out)
        elif self.uv_layer == 'nonlinear_relu':
            out = self.relu(self.U_conv3x3_bn(out))

        out = self.V_conv1x1(out)

        # Dynamically compute value of weight-decay based on sampling
        if self.uv_layer == 'linear':
            self.compute_weight_norm()

        return out


class BlockISOFlopFactorized(nn.Module):
    """Block where W=UV decomposed blocks are trained for each conv.
       Each block is sparse such that overall FLOPs remain same.
    """

    expansion: int = 1

    def __init__(
        self,
        f_in: int,
        f_out: int,
        downsample=False,
        sparsity_level=0.0,
        sparse_u_only=False,
        uv_layer='linear',
    ):
        """
        Args:
            f_in (int): Number of input features.
            f_out (int): Number of output features.
            downsample (bool, optional): Whether to apply downsampling (default is False).
            sparsity_level (float, optional): Sparsity level for the block (default is 0.0).
            sparse_u_only (bool, optional): Whether to use only sparse U in decomposition (default is False).
            uv_layer (str, optional): Type of UV layer ('linear' by default).
        """

        super(BlockISOFlopFactorized, self).__init__()
        stride = 2 if downsample else 1

        self.conv1 = LowRankConv3x3(
            f_in, f_out, stride, sparsity_level, sparse_u_only, uv_layer
        )
        self.bn1 = nn.BatchNorm2d(f_out)

        self.conv2 = LowRankConv3x3(
            f_out, f_out, 1, sparsity_level, sparse_u_only, uv_layer
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


class BlockISOFlopParallel(nn.Module):
    """Block where RepVGG style parallel blocks are trained for each conv.
       Each block is sparse such that overall FLOPs remain same.
    """

    expansion: int = 1

    def __init__(
        self, f_in: int, f_out: int, downsample=False, num_parallel_branch=1
    ):
        """
        Args:
            f_in (int): Number of input features.
            f_out (int): Number of output features.
            downsample (bool, optional): Whether to apply downsampling (default is False).
            num_parallel_branch (int, optional): Number of parallel sparse conv blocks (default is 1).
        """

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

        self.bn1_psum = nn.BatchNorm2d(f_out)

        # Replicate block of Conv and BN
        self.conv2_list, self.bn2_list = nn.ModuleList(), nn.ModuleList()
        for _ in range(self.num_parallel_branch):
            self.conv2_list.append(
                nn.Conv2d(
                    f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False
                )
            )
            self.bn2_list.append(nn.BatchNorm2d(f_out))

        self.bn2_psum = nn.BatchNorm2d(f_out)

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

        self.dynamic_weight_norm1 = None
        self.dynamic_weight_norm2 = None

    def compute_weight_norm(self):
        ''' Compute l2 norm on folded U, V. This compute is dynamic when we sample U, V. '''

        conv_weights_conv1_list = 0
        for conv in self.conv1_list:
            conv_weights_conv1_list += conv.weight.view(conv.weight.size(0), -1)
        self.dynamic_weight_norm1 = (conv_weights_conv1_list**2).sum()

        conv_weights_conv2_list = 0
        for conv in self.conv2_list:
            conv_weights_conv2_list += conv.weight.view(conv.weight.size(0), -1)
        self.dynamic_weight_norm2 = (conv_weights_conv2_list**2).sum()

    def forward(self, x):
        # 1st Conv + BN
        out_first = 0
        for idx in range(self.num_parallel_branch):
            # only convs + BN
            out_first = out_first + \
                self.bn1_list[idx](self.conv1_list[idx](x))

        out_first = F.relu(out_first)

        # 2nd Conv + BN
        out_second = 0
        for idx in range(self.num_parallel_branch):
            # only convs + BN
            out_second = out_second + \
                self.bn2_list[idx](self.conv2_list[idx](out_first))

        out_second += self.shortcut(x)
        out = F.relu(out_second)
        return out


class BlockISOFlopWideFactorized(nn.Module):
    """Block where Wide(W)=Wide(UV) decomposed blocks are trained for each conv.
       Each block is sparse such that overall FLOPs remain same.
    """

    expansion: int = 1

    def __init__(
        self,
        f_in: int,
        f_out: int,
        downsample=False,
        width_scaling_factor=0.0,
        uv_layer='linear',
    ):
        """
        Args:
            f_in (int): Number of input features.
            f_out (int): Number of output features.
            downsample (bool, optional): Whether to apply downsampling (default is False).
            width_scaling_factor (float, optional): Scaling factor for width (default is 0.0).
            uv_layer (str, optional): Type of UV layer ('linear' by default).
        """

        super(BlockISOFlopWideFactorized, self).__init__()
        stride = 2 if downsample else 1

        self.conv1 = WideLowRankConv3x3(
            f_in, f_out, stride, width_scaling_factor, uv_layer
        )
        self.bn1 = nn.BatchNorm2d(f_out)

        self.conv2 = WideLowRankConv3x3(
            f_out, f_out, 1, width_scaling_factor, uv_layer
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


class Block(nn.Module):
    """A ResNet block (original without pre-activation).
    """

    expansion: int = 1

    def __init__(self, f_in: int, f_out: int, downsample=False):
        """
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
    """A ResNet PreAct block.
    """

    expansion: int = 1

    def __init__(self, f_in: int, f_out: int, downsample=False):
        """
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
    """A residual neural network as originally designed for CIFAR-100.
    """
    
    def __init__(self, plan, num_classes=100, block_args={}):
        """
        Args:
            plan (list): Contains width and num blocks per ResNet block.
            num_classes (int, optional): Number of classes for the output layer (default is 100).
            block_args (dict, optional): Arguments for blocks dictionary (default is {}).
        """
        
        super(Model, self).__init__()
        self.block_name = block_args["block_name"]
        self.block = str_to_class(self.block_name)

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv_stem = nn.Conv2d(
            3, current_filters, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(current_filters)

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
                            block_args["num_parallel_branch"],
                        )
                    )
                elif self.block == BlockISOFlopFactorized:
                    blocks.append(
                        self.block(
                            current_filters,
                            filters,
                            downsample,
                            block_args["sparsity_level"],
                            block_args["sparse_u"],
                            block_args["uv_layer"],
                        )
                    )
                elif self.block == BlockISOFlopWideFactorized:
                    blocks.append(
                        self.block(
                            current_filters,
                            filters,
                            downsample,
                            block_args["width_scaling_factor"],
                            block_args["uv_layer"],
                        )
                    )
                elif self.block == BlockISOFlopDoping:
                    blocks.append(
                        self.block(
                            current_filters,
                            filters,
                            downsample,
                            block_args["sparsity_level"],
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

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0] * self.block.expansion, num_classes)

        # Initialize
        self.init_weights(zero_init_last=block_args["zero_init_residual"])

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
        out = self.blocks(out)
        if self.block == PreActBlock:
            out = F.relu(self.norm(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    @staticmethod
    def get_model_from_name(args):
        """Get model given args, including scaling for:
            - width
            - parallelism
            - doping
            - factorization

        Args:
            args (namespace): arguments for training run.
        """

        # num_classes
        num_classes = args.num_classes

        # block arguments
        resnet_block_args = {}
        resnet_block_args["block_name"] = args.block_name
        resnet_block_args["zero_init_residual"] = args.zero_init_residual

        num_branches = 1
        if args.sift_scaling and args.sift_family == "sparse_parallel":
            num_branches = args.base_scaling
        resnet_block_args["num_parallel_branch"] = num_branches

        if args.sift_scaling and args.sift_family in ["sparse_factorized"]:
            resnet_block_args["sparsity_level"] = args.maskopt_sparsity
            resnet_block_args["sparse_u"] = args.sparse_u_block
            resnet_block_args["uv_layer"] = args.uv_layer

        if args.sift_scaling and args.sift_family == "sparse_doped":
            resnet_block_args["sparsity_level"] = args.maskopt_sparsity

        # base model arguments
        model_name = args.model
        name = model_name.split("_")

        base_width = 64
        if args.sift_scaling and args.sift_family == "sparse_wide":
            width_scaling_factor = math.sqrt(args.base_scaling)
            base_width = int(base_width * width_scaling_factor)

        if args.sift_scaling and args.sift_family == "sparse_wide_factorized":
            width_scaling_factor = math.sqrt(args.base_scaling)
            base_width = int(base_width * width_scaling_factor)
            
            resnet_block_args["width_scaling_factor"] = width_scaling_factor
            resnet_block_args["uv_layer"] = args.uv_layer

        W = base_width

        base_depth = int(name[1])
        if base_depth == 18:
            plan = [(W, 2), (2 * W, 2), (4 * W, 2), (8 * W, 2)]
        elif base_depth == 34:
            plan = [(W, 3), (2 * W, 4), (4 * W, 6), (8 * W, 3)]

        return Model(plan, num_classes, resnet_block_args)
