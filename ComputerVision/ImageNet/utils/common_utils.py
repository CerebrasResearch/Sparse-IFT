import os
import random
from collections import OrderedDict

import numpy as np
import torch


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
