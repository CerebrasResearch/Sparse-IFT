import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms


class DatasetSplitter(torch.utils.data.Dataset):
    """This splitter makes sure that we always use the same training/validation split"""

    def __init__(self, parent_dataset, split_start=-1, split_end=-1):
        split_start = split_start if split_start != -1 else 0
        split_end = split_end if split_end != -1 else len(parent_dataset)
        assert (
            split_start <= len(parent_dataset) - 1
            and split_end <= len(parent_dataset)
            and split_start < split_end
        ), "invalid dataset split"

        self.parent_dataset = parent_dataset
        self.split_start = split_start
        self.split_end = split_end

    def __len__(self):
        return self.split_end - self.split_start

    def __getitem__(self, index):
        assert index < len(self), "index out of bounds in split_datset"
        return self.parent_dataset[index + self.split_start]


def get_threads(args):
    max_threads = args.max_threads

    train_threads = 0
    valid_threads = 0
    test_threads = 0

    if max_threads <= 4:
        max_threads = 4
        train_threads = 2
        if args.validation_split > 0.0:
            test_threads = 1
            valid_threads = 1
        else:
            test_threads = 2
    elif max_threads == 5:
        test_threads = 2
        if args.validation_split > 0.0:
            valid_threads = 1
            train_threads = 2
        else:
            train_threads = 3
    else:
        test_threads = 2
        if args.validation_split > 0.0:
            valid_threads = 2
            train_threads = max_threads - 4
        else:
            train_threads = max_threads - 2

    return train_threads, valid_threads, test_threads


def get_loaders(args, full_dataset, test_dataset):
    train_threads, valid_threads, test_threads = get_threads(args)

    valid_loader = None
    if args.validation_split > 0.0:
        split = int(np.floor((1.0 - args.validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset, split_end=split)
        val_dataset = DatasetSplitter(full_dataset, split_start=split)

        # for cleaner printing
        args.train_data_len = train_dataset.__len__()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            shuffle=True,
            num_workers=train_threads,
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=valid_threads,
            pin_memory=True,
        )
    else:
        # for cleaner printing
        args.train_data_len = full_dataset.__len__()
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            shuffle=True,
            num_workers=train_threads,
            pin_memory=True,
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=test_threads,
        pin_memory=True,
    )

    return train_loader, valid_loader, test_loader


def get_cifar100_dataloaders(args):
    """Creates augmented train, validation, and test data loaders."""
    cifar_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cifar_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(cifar_mean, cifar_std),]
    )

    full_dataset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=transform_test
    )
    return get_loaders(args, full_dataset, test_dataset)


def get_cifar10_dataloaders(args):
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: F.pad(
                    x.unsqueeze(0), (4, 4, 4, 4), mode='reflect'
                ).squeeze()
            ),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(cifar_mean, cifar_std)]
    )

    full_dataset = datasets.CIFAR10(
        root=args.data_dir, train=True, transform=transform_train, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform_test
    )
    return get_loaders(args, full_dataset, test_dataset)


def get_data_loaders_for_runtime(args):
    print('-' * 60)

    fn = None
    if args.data == 'cifar100':
        print('Creating loaders for CiFAR100')
        fn = get_cifar100_dataloaders
    elif args.data == 'cfiar10':
        print('Creating loaders for CiFAR10')
        fn = get_cifar10_dataloaders
    else:
        raise ValueError('Dataset not supported currently')
    return fn(args)
