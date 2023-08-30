from typing import List, Union

import torch
from captum.attr import visualization as v, LayerGradCam
from termcolor import colored
from timm import create_model

from src.analysis.utils import to_dict, set_matplotlib
from src.utils import register_method

set_matplotlib()


@register_method
def gradcam(model: torch.nn.Module,
            data: torch.Tensor,
            return_nodes: Union[dict, List[str], str],
            label: Union[int, List[int]],
            save: str = None,
            verbose: bool = False,
            plot_kwargs: dict = None,
            **kwargs):
    if verbose:
        print(f'Analysis: {colored("GradCAM Visualization", "blue")}')

    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    return_nodes = to_dict(return_nodes)
    returns = list()

    for return_node_name, save_name in return_nodes.items():
        gradcam_engine = LayerGradCam(model, get_submodule(model, return_node_name))
        try:
            attribute = gradcam_engine.attribute(data, target=label)
            if attribute.dim() == 3:
                warning_architecture()
                return
        except AssertionError:
            warning_architecture()
            return
        attribute = attribute.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()

        plt_fig, plt_axis = v.visualize_image_attr(attribute, **plot_kwargs)
        returns.append((plt_fig, plt_axis))

        plt_fig.savefig(f"{save}/GradCAM_{save_name}.svg", dpi=500) if save else None

    return returns if len(returns) > 1 else returns[0]


def warning_architecture():
    print(colored(f'GradCAM is not supported Transformer.', 'red'), ' Skip GradCAM.')


def get_submodule(model, return_node_name):
    try:
        return model.get_submodule(return_node_name)
    except AttributeError:
        print(
            f'Use {colored("from torchvision.models.feature_extraction import get_graph_node_names", "green")} to find return node name.')
        raise


if __name__ == '__main__':
    model = create_model('dino_vitsmall_patch8', pretrained=True)
    methods = gradcam(model, torch.rand(1, 3, 224, 224), 'blocks.11', 5,
                      original_image=None,
                      show_colorbar=True,
                      sign='absolute',
                      method='heat_map',
                      save='gradcam.svg'
                      )
