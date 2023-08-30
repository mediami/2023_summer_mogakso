import gc
import math

import torch
from torchvision import transforms
from torchvision.models.feature_extraction import get_graph_node_names as torch_graph_node_name
from torchvision.transforms import InterpolationMode

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def transform(img, img_size=224):
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
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    return transform(img).unsqueeze(0)


def list_graph_node_names(model, verbose=True):
    graph_list = torch_graph_node_name(model)
    if verbose:
        from pprint import pprint
        pprint(graph_list)
    return graph_list


def clear():
    torch.cuda.empty_cache()
    gc.collect()
