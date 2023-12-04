import copy

import torch
import torch.nn as nn


def _fuse_conv_bn(conv: nn.Module, bn: nn.Module):
    """Fuse conv and bn into one module.

    Args:
        conv (nn.Module): Conv to be fused.
        bn (nn.Module): BN to be fused.

    Returns:
        nn.Module: Fused module.
    """
    conv_w = conv.weight
    conv_b = (
        conv.bias
        if conv.bias is not None
        else torch.zeros_like(bn.running_mean)
    )

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(
        conv_w * factor.reshape([conv.out_channels, 1, 1, 1])
    )
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv


def fuse_conv_bn(module: nn.Module):
    """Recursively fuse conv and bn in a module.
    During inference, the functionary of batch norm layers is turned off
    but only the mean and var alone channels are used, which exposes the
    chance to fuse it with the preceding conv layers to save computations and
    simplify network structures.

    Args:
        module (nn.Module): Module to be fused.

    Returns:
        nn.Module: Fused module.
    """
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(
            child, (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)
        ):
            if last_conv is None:  # only fuse BN that is after Conv
                continue

            # To reduce changes, set BN as Identity instead of deleting it.
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)

    return module


def get_fused_model(model):
    fused_model = None

    # get the copy and the fusion without any autograd changes
    with torch.no_grad():
        fused_model = copy.deepcopy(model)
        fuse_conv_bn(fused_model)

    return model, fused_model


def unit_test_fusion():
    from torchvision.models import ResNet18_Weights, resnet18

    torch.manual_seed(42)

    model = resnet18(weights=ResNet18_Weights.DEFAULT, progress=True)
    model, fused_model = get_fused_model(model)

    model = model.eval()
    fused_model = fused_model.eval()

    norm_diff = []
    num_iters = 25
    for _ in range(num_iters):
        input_sample = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model.forward(input_sample)
            fused_out = fused_model.forward(input_sample)

        norm_diff.append((out - fused_out).norm())

    norm_diff = torch.FloatTensor(norm_diff)
    print(f"Averge diff over {num_iters} iterations is {norm_diff.mean()}")


if __name__ == "__main__":
    unit_test_fusion()
