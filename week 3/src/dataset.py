import math
import os
import random
from math import floor

import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import default_collate, DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100, FashionMNIST
from torchvision.datasets.samplers import DistributedSampler

_dataset_dict = {
    'ImageFolder': ImageFolder,
    'CIFAR10': CIFAR10,
    'CIFAR100': CIFAR100,
    'FashionMNIST': FashionMNIST,
}

class TrainTransform:
    def __init__(self, size, pad, hflip, mean, std):
        transform_list = [transforms.RandomCrop(size, padding=pad)] # 랜덤하게 자르기
    # 오토 어그도 넣어주면 성능 향상
        if hflip:
            transform_list.append(transforms.RandomHorizontalFlip(hflip)) # 수평으로 뒤집기
            transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)) # 많은 어그멘테이션중에 cifar에 어울리는거 넣어줌

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.transform_fn = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transform_fn(x)


class ValTransform:
    def __init__(self, size, mean, std):

        transform_list = [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        self.transform_fn = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transform_fn(x)


def get_dataloader(train_dataset, val_dataset, args):
    # 1. create sampler
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    # 2. create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=args.pin_memory, persistent_workers=True)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
                                num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    args.iter_per_epoch = len(train_dataloader)

    return train_dataloader, val_dataloader


def get_dataset(args):
    dataset_class = _dataset_dict[args.dataset_type]
    train_transform = TrainTransform(args.train_size, args.random_crop_pad, args.hflip, args.mean, args.std)
    val_transform = ValTransform(args.test_size, args.mean, args.std)
    if args.dataset_type in _dataset_dict.keys():
        train_dataset = dataset_class(root=args.data_dir, train=True, download=True, transform=train_transform)
        val_dataset = dataset_class(root=args.data_dir, train=False, download=True, transform=val_transform)
        args.num_classes = len(train_dataset.classes)
    else:
        assert f"{args.dataset_type} is not supported yet. Just make your own code for it"

    return train_dataset, val_dataset