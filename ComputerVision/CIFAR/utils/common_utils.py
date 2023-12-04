import os
import random
from collections import OrderedDict

import numpy as np
import torch


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    result = None
    with torch.no_grad():
        maxk = min(max(topk), output.size()[1])
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        result = []
        for k in topk:
            correct_k = correct[: min(k, maxk)].reshape(-1).float().sum(0)
            result.append(correct_k.mul_(100.0 / batch_size))

    return result


def set_seed(seed=42, rank=0):
    """Set seed to ensure deterministic runs.
    Note: Setting torch to be deterministic can lead to slow down in training.
    """
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def load_checkpoint(model, checkpoint_path):
    """Load checkpoint from given path to model state_dict
    """
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device('cpu')
        )
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)


def num_model_parameters(model):
    """Returns the total number of trainable parameters used by `model` (only
    counting shared parameters once)
    """
    parameters = list(model.parameters())
    parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)
