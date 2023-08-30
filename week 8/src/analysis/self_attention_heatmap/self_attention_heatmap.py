from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from captum.attr import visualization as v
from termcolor import colored
from timm import create_model

from src.analysis.utils import extract_sub_features, to_dict, set_matplotlib
from src.utils import register_method, transform

set_matplotlib()


@register_method
def self_attention_heatmap(model: torch.nn.Module,
                           data: torch.Tensor,
                           attn_return_nodes: Union[dict, List[str], str],
                           patch_size: int = 1,
                           save: str = None,
                           verbose: bool = False,
                           plot_kwargs: dict = None,
                           **kwargs):
    if verbose:
        print(f'Analysis: {colored("Self Attention Heatmap Visualization", "blue")}')
    if patch_size == 1:
        print(colored(f'CNN is not supported self attention heatmap analysis', 'red'))
        return

    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    returns = list()
    attn_return_nodes = to_dict(attn_return_nodes)

    attentions = extract_attention(model, data, attn_return_nodes, patch_size)

    for k, attention in attentions.items():
        mean_attn = np.mean(attention, axis=0, keepdims=True).transpose(1, 2, 0)
        plt_fig, plt_axis = v.visualize_image_attr(mean_attn, cmap='inferno', **plot_kwargs)
        returns.append((plt_fig, plt_axis))

        plt_fig.savefig(f'{save}/SelfAttentionHeatmap_{k}.svg', dpi=500) if save else None

    return returns if len(returns) > 1 else returns[0]


@register_method
def sah(model: torch.nn.Module,
        data: torch.Tensor,
        attn_return_nodes: Union[dict, List[str], str],
        patch_size: int,
        save: bool = False,
        verbose: bool = False,
        plot_kwargs: dict = None,
        **kwargs):
    return self_attention_heatmap(model, data, attn_return_nodes, patch_size, save, verbose, plot_kwargs, **kwargs)


def extract_attention(model, data, attn_return_nodes, patch_size):
    # make the image divisible by the patch size
    w, h = data.shape[2] - data.shape[2] % patch_size, data.shape[3] - data.shape[3] % patch_size
    data = data[:, :, :w, :h]

    w_featmap = data.shape[-2] // patch_size
    h_featmap = data.shape[-1] // patch_size

    attentions_dict = extract_sub_features(model, data, attn_return_nodes)

    attentions = dict()

    for k, _attentions in attentions_dict.items():
        nh = _attentions.shape[1]  # number of head
        # keep only the output patch attention
        _attentions = _attentions[0, :, 0, 1:].reshape(nh, -1)
        _attentions = _attentions.reshape(nh, w_featmap, h_featmap)
        _attentions = F.interpolate(_attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
            0].detach().cpu().numpy()
        attentions.update({k: _attentions})

    return attentions


if __name__ == '__main__':
    model = create_model('dino_vitsmall_patch8', pretrained=True)
    img = Image.open('../../data/sample_images/corgi.jpg')
    methods = self_attention_heatmap(model, transform(img, 224), {'blocks.11.attn.softmax': 'blocks11'}, 8)
