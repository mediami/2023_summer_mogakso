import math

from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100, FashionMNIST
from torchvision.transforms import InterpolationMode

from .imagenet_c import ImageNetC

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

_DATASET_DICT = {
    'ImageFolder': ImageFolder,
    'CIFAR10': CIFAR10,
    'CIFAR100': CIFAR100,
    'FashionMNIST': FashionMNIST,
}

_CORRUPTION_DATASET_DICT = {
    'ImageNetC': ImageNetC,
}


def get_transform(img_size=224, mean=None, std=None):
    mean = IMAGENET_DEFAULT_MEAN if mean is None else mean
    std = IMAGENET_DEFAULT_STD if std is None else std
    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            scale_size = int(math.floor(img_size[0] / DEFAULT_CROP_PCT))
        else:
            scale_size = tuple([int(x / DEFAULT_CROP_PCT) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / DEFAULT_CROP_PCT))

    transform = transforms.Compose([
        transforms.Resize(scale_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform


def get_dataloader(dataset_type, data_dir, img_size, batch_size=128, num_workers=4, **kwargs):
    dataset_class = _DATASET_DICT.get(dataset_type, None)
    transform = get_transform(img_size)
    if dataset_class:
        dataset = dataset_class(root=data_dir, transform=transform, **kwargs)
    else:
        raise NotImplementedError(f"{dataset_type} is not supported yet. Just make your own code for it")

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=SequentialSampler(dataset),
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
    return data_loader


def get_dataloader_c(dataset_type, data_dir, ctype, intensity, img_size, batch_size=128, num_workers=4, **kwargs):
    dataset_class = _CORRUPTION_DATASET_DICT.get(dataset_type, None)
    transform = get_transform(img_size)
    if dataset_class:
        dataset = dataset_class(root=data_dir, ctype=ctype, intensity=intensity, transform=transform,
                                **kwargs)
    else:
        raise NotImplementedError(f"{dataset_type} is not supported yet. Just make your own code for it")

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=SequentialSampler(dataset),
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
    return data_loader
