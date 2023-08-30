import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored
from torchvision.models.feature_extraction import create_feature_extractor


def to_dict(instance):
    if isinstance(instance, dict):
        return instance
    elif isinstance(instance, str):
        return {instance: instance}
    elif isinstance(instance, list):
        return {x: x for x in instance}
    else:
        raise ValueError(f'{instance} is not supported instance. Use dict, str, or list[str]')


def to_list(instance):
    if isinstance(instance, list):
        return instance
    elif isinstance(instance, (str, int)):
        return [instance]
    else:
        raise ValueError(f'{instance} is not supported instance. Use dict, str, or list[str]')


def set_matplotlib():
    # plt.rcParams["font.family"] = "Arial"
    large = 24
    med = 18
    small = 10
    params = {'axes.titlesize': large,
              'legend.fontsize': med,
              'figure.figsize': (16, 10),
              'axes.labelsize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    plt.rcParams['axes.unicode_minus'] = False


def plot_features(img, features):
    n_heads = features.shape[0]

    plt.figure(figsize=(10, 10))
    text = ["Original Image", "Head Mean"]
    for i, fig in enumerate([img, np.mean(features, 0)]):
        plt.subplot(1, 2, i + 1)
        plt.imshow(fig, cmap='inferno')
        plt.title(text[i])
    plt.show()

    plt.figure(figsize=(10, 10))
    for i in range(n_heads):
        plt.subplot(n_heads // 3, 3, i + 1)
        plt.imshow(features[i], cmap='inferno')
        plt.title(f"Head n: {i + 1}")
    plt.tight_layout()
    plt.show()


def extract_sub_features(model, data, return_nodes):
    if data.dim() == 3:
        data = data.unsqueeze(0)
    try:
        _model = create_feature_extractor(model, return_nodes=return_nodes)
        _model.eval()
        outputs = _model(data)
        return outputs

    except AttributeError:
        print(
            f'Use {colored("from torchvision.models.feature_extraction import get_graph_node_names", "green")} to find return node name.')
        raise
