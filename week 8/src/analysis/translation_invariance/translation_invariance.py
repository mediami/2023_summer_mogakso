import pandas as pd
import torch
from termcolor import colored
from timm import create_model

from src.data.dataloader import get_dataloader
from src.utils import register_method, ConsistencyEngine


@register_method
def consistency(model: torch.nn.Module,
                data_dir: str,
                img_size: int,
                num_class: int,
                device: str,
                task: str = 'multiclass',
                save: str = None,
                verbose: bool = False,
                loader_kwargs: dict = None,
                **kwargs):
    loader_kwargs = {} if loader_kwargs is None else loader_kwargs
    if verbose:
        print(f'Analysis: {colored("Consistency", "blue")}')
    engine = ConsistencyEngine(model, device, num_class, 'Robustness', task, verbose)

    loader = get_dataloader('ImageFolder', data_dir, img_size, **loader_kwargs)
    metrics = engine(loader=loader)
    metrics = pd.DataFrame([metrics]).astype("float")
    metrics = metrics.rename(columns={'Accuracy': 'Consistency'})

    metrics.to_csv(f'{save}/consistency.csv') if save else None

    print(metrics)
    return metrics


if __name__ == '__main__':
    model = create_model('resnet18', pretrained=True)
    method = consistency(model, '/home/seungmin/shared/hdd_ext/nvme1/classification/imageNet', 224, 1000, 'cpu')
