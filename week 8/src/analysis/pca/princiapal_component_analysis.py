import os
from glob import glob
from typing import List, Union

import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from termcolor import colored
from timm import create_model

from src.analysis.utils import extract_sub_features, to_dict, set_matplotlib
from src.utils import register_method, transform

set_matplotlib()


@register_method
def pca(model: torch.nn.Module,
        pca_data_dir: str,
        return_nodes: Union[dict, List[str], str],
        img_size: int,
        n_components: int,
        device: str,
        save: str = None,
        verbose: bool = False,
        plot_kwargs: dict = None,
        **kwargs):
    if verbose:
        print(f'Analysis: {colored("Principal Component Analysis Visualization", "blue")}')

    return_nodes = to_dict(return_nodes)

    features = extract_features(model, pca_data_dir, return_nodes, img_size, device, verbose)
    pca_features = extract_principal_component(features, n_components)

    n_img = pca_features.shape[0]
    for i in range(n_img):
        plt.subplot(n_img // 4, 4, i + 1)
        plt.imshow(pca_features[i])

        plt.savefig(f'{save}/PCA_{i}.svg', dpi=500) if save else None

    plt.show()
    return pca_features


@torch.no_grad()
def extract_features(model, data_dir, return_nodes, img_size, device, verbose):
    images = glob(os.path.join(data_dir, '*'))
    features = list()
    if verbose:
        print(f"Images: {images}")
    for img in images:
        img = Image.open(img).convert('RGB')
        feature = list(extract_sub_features(model, transform(img, img_size).to(device), return_nodes).values())[0]

        if feature.dim() == 4:  # For CNN
            feature = torch.flatten(feature, 2, 3).permute(0, 2, 1)
            features.append(feature)
        else:
            features.append(feature[:, 1:, :])

    features = torch.cat(features, dim=0)
    return features.detach().cpu()


def extract_principal_component(features, n_components):
    n_img, n_patch, n_channels = features.shape
    n_patch = int(n_patch ** 0.5)
    features = features.view(-1, n_channels)

    sklearn_pca = PCA(n_components)
    sklearn_pca.fit(features)
    pca_features = sklearn_pca.transform(features)

    pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
                         (pca_features[:, 0].max() - pca_features[:, 0].min())
    pca_features_bg = pca_features[:, 0] > 0.35  # from first histogram
    pca_features_fg = ~pca_features_bg

    sklearn_pca.fit(features[pca_features_fg])
    pca_features_left = sklearn_pca.transform(features[pca_features_fg])

    for i in range(n_components):
        pca_features_left[:, i] = (pca_features_left[:, i] - pca_features_left[:, i].min()) / (
                pca_features_left[:, i].max() - pca_features_left[:, i].min())

    pca_features_rgb = pca_features.copy()
    pca_features_rgb[pca_features_bg] = 0
    pca_features_rgb[pca_features_fg] = pca_features_left

    # reshaping to numpy image format
    pca_features_rgb = pca_features_rgb.reshape(n_img, n_patch, n_patch, n_components)
    return pca_features_rgb


if __name__ == '__main__':
    model = create_model('resnet50', pretrained=True)
    img_path = '../../data/harryported_giffins'
    methods = pca(model, img_path, {'layer4': 'feature_map'}, 224, 3)
