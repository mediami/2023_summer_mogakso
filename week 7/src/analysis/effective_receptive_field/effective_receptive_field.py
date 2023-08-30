from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from captum.attr import visualization as v
from termcolor import colored
from timm import create_model
from timm.utils import AverageMeter
from torch import optim

from src.analysis.utils import to_dict, set_matplotlib, extract_sub_features
from src.data.dataloader import get_dataloader
from src.utils import register_method

set_matplotlib()


@register_method
def effective_receptive_field(model: torch.nn.Module,
                              data_dir: str,
                              img_size: int,
                              device: str,
                              return_nodes: Union[dict, List[str], str],
                              n_sample: int = 5,
                              batch_size: int = 32,
                              save: str = None,
                              verbose: bool = False,
                              plot_kwargs: dict = None,
                              **kwargs):
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    if verbose:
        print(f'Analysis: {colored("Effective Receptive Field Visualization", "blue")}')
    returns = list()
    return_nodes = to_dict(return_nodes)

    loader = get_dataloader('ImageFolder', data_dir, img_size, batch_size=batch_size)
    grad_map = get_grad_map(model, loader, return_nodes, n_sample, device)

    grad_map = np.log10(grad_map + 1)  # the scores differ in magnitude. take the logarithm for better readability
    grad_map = grad_map / np.max(grad_map)  # rescale to [0,1] for the comparability among models
    grad_map = np.expand_dims(grad_map, axis=-1)

    plt_fig, plt_axis = v.visualize_image_attr(grad_map, cmap='YlGn', **plot_kwargs)
    returns.append((plt_fig, plt_axis))

    if verbose:
        for thresh in [0.2, 0.3, 0.5, 0.99]:
            side_length, area_ratio = get_rectangle(grad_map, thresh)
            print('thresh, rectangle side length, area ratio: ', thresh, side_length, area_ratio)
    plt_fig.savefig(f'{save}/ERF.svg', dpi=500) if save else None

    return returns if len(returns) > 1 else returns[0]


@register_method
def erf(model: torch.nn.Module, return_nodes: str, data: torch.Tensor, save: bool = False, verbose: bool = False):
    return effective_receptive_field(model, return_nodes, data, save, verbose)


def get_rectangle(data, thresh):
    h, w = data.shape
    all_sum = np.sum(data)
    for i in range(1, h // 2):
        selected_area = data[h // 2 - i:h // 2 + 1 + i, w // 2 - i:w // 2 + 1 + i]
        area_sum = np.sum(selected_area)
        if area_sum / all_sum > thresh:
            return i * 2 + 1, (i * 2 + 1) / h * (i * 2 + 1) / w
    return None


def get_input_grad(model, data, return_nodes):
    output = extract_sub_features(model, data, return_nodes)
    output = list(output.values())[0]

    if output.dim() == 3:  # for Transformer
        output = output.permute(0, 2, 1)[:, :, 1:]
        patch_size = int(output.size(-1) ** 0.5)
        output = output.reshape(output.size(0), output.size(1), patch_size, patch_size)

    out_size = output.size()
    central_point = torch.nn.functional.relu(output[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, data, allow_unused=True)[0]
    grad = torch.nn.functional.relu(grad)
    grad_map = grad.sum((0, 1)).cpu().numpy()
    return grad_map


def get_grad_map(model, loader, return_nodes, n_sample, device):
    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)
    meter = AverageMeter()
    optimizer.zero_grad()

    for i, data in enumerate(loader):
        if i == n_sample:
            break
        data = data[0].to(device)
        data.requires_grad = True
        optimizer.zero_grad()
        grad_map = get_input_grad(model, data, return_nodes)
        meter.update(grad_map)

    return meter.avg


# Previous implemented
def heatmap(data, camp='RdYlGn', fig_size=(10, 10), ax=None):
    plt_fig, plt_axis = plt.subplots(figsize=fig_size)
    sns.heatmap(data,
                xticklabels=False,
                yticklabels=False, cmap=camp,
                center=0, annot=False, ax=ax, cbar=True, annot_kws={"size": 24}, fmt='.2f')
    return plt_fig, plt_axis


if __name__ == '__main__':
    model = create_model('resnet18', pretrained=True)
    methods = effective_receptive_field(model, torch.rand(1, 3, 224, 224), {'layer4.1.bn2': 'layer4'})
