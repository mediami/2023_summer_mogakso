from typing import Union, List

import pandas as pd
import torch
from termcolor import colored
from timm import create_model

from src.analysis.utils import to_list
from src.data import get_dataloader_c
from src.utils import register_method, TestEngine


@register_method
def robustness(model: torch.nn.Module,
               data_dir: str,
               ctype: Union[List[str], str],
               intensity: Union[List[int], int],
               img_size: int,
               num_class: int,
               device: str,
               task: str = 'multiclass',
               save: str = None,
               verbose: bool = False,
               loader_kwargs: dict = None,
               **kwargs):
    loader_kwargs = {} if loader_kwargs is None else loader_kwargs
    ctype = to_list(ctype)
    intensity = to_list(intensity)
    if verbose:
        print(f'Analysis: {colored("Robustness", "blue")}')
    engine = TestEngine(model, device, num_class, 'Robustness', task, verbose)

    result = pd.DataFrame()

    for c in ctype:
        for i in intensity:
            if verbose:
                print("Corruption type: %s, Intensity: %d, " % (ctype, intensity), end="")
            loader = get_dataloader_c('ImageNetC', data_dir, c, i, img_size, **loader_kwargs)
            metrics = engine(loader=loader)
            metrics = pd.DataFrame([metrics]).astype("float")
            metrics.insert(0, 'intensity', i)
            metrics.insert(0, 'ctype', c)

            result = pd.concat([result, metrics], ignore_index=True)

    result.to_csv(f'{save}/robustness.csv') if save else None

    print(result)
    return result


if __name__ == '__main__':
    model = create_model('resnet18', pretrained=True)
    method = robustness(model, '/home/seungmin/shared/hdd_ext/nvme1/classification/tiny-imageNet',
                        ['contrast', 'defocus_blur'], [1, 2], 224, 1000, 'cpu')
