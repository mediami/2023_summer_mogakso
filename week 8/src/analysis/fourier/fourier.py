import math

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from matplotlib.collections import LineCollection
from termcolor import colored
from timm import create_model

from src.analysis.fourier.split_blocks import get_blocks
from src.data.dataloader import get_dataloader
from src.utils import register_method


@register_method
def frequency(model: torch.nn.Module,
              data_dir: str,
              model_name: str,
              img_size: int,
              device: str,
              n_sample: int = 5,
              save: str = None,
              verbose: bool = False,
              loader_kwargs: dict = None,
              **kwargs):
    loader_kwargs = {} if loader_kwargs is None else loader_kwargs
    if verbose:
        print(f'Analysis: {colored("Consistency", "blue")}')

    blocks = get_blocks(model, model_name)

    loader = get_dataloader('ImageFolder', data_dir, img_size, batch_size=n_sample, **loader_kwargs)
    data = next(iter(loader))
    xs, ys = map(lambda x: x.to(device), data)

    latents = make_latent(blocks, xs, model_name)
    fourier_latents = make_fourier_latent(latents)
    fig = make_plot(fourier_latents)

    fig.tight_layout()
    fig.savefig(f'{save}/FourierFrequency.svg', dpi=500) if save else None

    fig.show()
    return fig


def fourier(x):  # 2D Fourier transform
    f = torch.fft.fft2(x)
    f = f.abs() + 1e-6
    f = f.log()
    return f


def shift(x):  # shift Fourier transformed feature map
    b, c, h, w = x.shape
    return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))


def make_segments(x, y):  # make segment for `plot_segment`
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_segment(ax, xs, ys, cmap_name="plasma", marker='o'):  # plot with cmap segments
    z = np.linspace(0.0, 1.0, len(ys))
    z = np.asarray(z)

    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(0.0, 1.0)
    segments = make_segments(xs, ys)
    lc = LineCollection(segments, array=z, cmap=cmap_name, norm=norm,
                        linewidth=2.5, alpha=1.0)
    ax.add_collection(lc)

    colors = [cmap(x) for x in xs]
    ax.scatter(xs, ys, color=colors, marker=marker, zorder=100)


def make_latent(blocks, xs, model_name):
    latents = []

    for block in blocks:
        xs = block(xs)
        latents.append(xs)

    if 'vit' in model_name or 'deit' in model_name:
        latents = [latent[:, 1:] for latent in latents]
    latents = latents[:-1]  # drop logit (output)

    return latents


def make_fourier_latent(latents):
    fourier_latents = []
    for latent in latents:  # `latents` is a list of hidden feature maps in latent spaces
        latent = latent.cpu()

        if len(latent.shape) == 3:  # for ViT
            latent = latent[:, 1:, :]
            b, n, c = latent.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
        elif len(latent.shape) == 4:  # for CNN
            b, c, h, w = latent.shape
        else:
            raise Exception("shape: %s" % str(latent.shape))
        latent = fourier(latent)
        latent = shift(latent).mean(dim=(0, 1))
        latent = latent.diag()[int(h / 2):]  # only use the half-diagonal components
        latent = latent - latent[0]  # visualize 'relative' log amplitudes
        # (i.e., low-freq amp - high freq amp)
        fourier_latents.append(latent)
    return fourier_latents


def make_plot(fourier_latents):
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 7), dpi=500)
    for i, latent in enumerate(reversed(fourier_latents[:-1])):
        freq = np.linspace(0, 1, len(latent))
        ax1.plot(freq, latent.detach().cpu().numpy(), color=cm.plasma_r(i / len(fourier_latents)))

    ax1.set_xlim(left=0, right=1)

    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("$\Delta$ Log amplitude")

    from matplotlib.ticker import FormatStrFormatter
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1fπ'))

    return fig


if __name__ == '__main__':
    model = create_model('resnet18', pretrained=True)
    method = frequency(model, '/home/seungmin/shared/hdd_ext/nvme1/classification/imageNet', 'resnet', 224, 'cpu')
