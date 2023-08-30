import os
import os.path
from typing import Callable, Optional

import torchvision.datasets as tdatasets
from torchvision.transforms import transforms


class ImageNetC(tdatasets.ImageFolder):
    """
    `ImageNet-C <https://github.com/hendrycks/robustness>`_ Dataset.

    Args:
        root (string): Root directory of dataset.
        ctype (string): Corruption type.
        intensity (int): Corruption intensity from 1 to 5
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    base_folder = "distort"

    def __init__(
            self,
            root: str,
            ctype: str,
            intensity: int,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        path = os.path.join(root, self.base_folder, ctype, str(intensity))
        self.classes = 1000

        super().__init__(path,
                         transform=transform,
                         target_transform=target_transform)


def get_imagenet_c(ctype, intensity,  mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                   root="./data", **kwargs):

    return ImageNetC(root, ctype, intensity, transform)
