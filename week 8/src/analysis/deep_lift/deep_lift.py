import os

import torch
from captum.attr import DeepLift
from captum.attr import visualization as v
from termcolor import colored
from timm import create_model

from src.analysis.utils import set_matplotlib
from src.utils import register_method

set_matplotlib()


@register_method
def deeplift(model: torch.nn.Module,
             data: torch.Tensor,
             label: int,
             save: str = None,
             verbose: bool = False,
             plot_kwargs: dict = None,
             **kwargs):
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    if verbose:
        print(f'Analysis: {colored("DeepLift Visualization", "blue")}')
    if data.dim() == 3:
        data = data.unsqueeze(0)
    data.requires_grad = True

    deeplift_engine = DeepLift(model)
    attribute = deeplift_engine.attribute(data, target=label, baselines=data * 0)
    attribute = attribute.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()

    plt_fig, plt_axis = v.visualize_image_attr(attribute, **plot_kwargs)

    plt_fig.savefig(f'{save}/DeepLift.svg', dpi=500) if save else None

    return plt_fig, plt_axis


if __name__ == '__main__':
    model = create_model('resnet18', pretrained=True)
    methods = deeplift(model, torch.rand(1, 3, 224, 224), 5, save='exp-res')

    model = create_model('dino_vitsmall_patch8', pretrained=True)
    methods = deeplift(model, torch.rand(1, 3, 224, 224), 5, save='exp-dino')
