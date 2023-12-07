from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, partial
from glob import glob
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
import torch
from albumentations import BaseCompose
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, image_folder: str, transform: BaseCompose = None) -> None:
        self.image_paths = glob(f"{image_folder}/**")
        self.transform = transform

    @property
    def image_names(self) -> List[str]:
        return [image_path.rsplit("/", 1)[1].rsplit(".", 1)[0] for image_path in self.image_paths]

    @cached_property
    def mean(self) -> np.ndarray:
        with ThreadPoolExecutor() as executor:
            means = []
            for mean in executor.map(self._calc_mean, self.image_paths):
                means.append(mean)
            return np.mean(np.concatenate(means, axis=0), axis=0)

    @staticmethod
    def _calc_mean(image_path: str) -> np.ndarray:
        image = cv2.imread(image_path, flags=cv2.IMREAD_ANYCOLOR)
        return np.mean(image, axis=0)

    @cached_property
    def std(self) -> np.ndarray:
        with ThreadPoolExecutor() as executor:
            variances = []
            for variance in executor.map(
                partial(self._calc_variance, mean=self.mean), self.image_paths
            ):
                variances.append(variance)
            return np.sqrt(np.mean(np.concatenate(variances, axis=0), axis=0))

    @staticmethod
    def _calc_variance(image_path: str, mean: np.ndarray) -> np.ndarray:
        image = cv2.imread(image_path, flags=cv2.IMREAD_ANYCOLOR)
        return np.mean((image - mean) ** 2, axis=0)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, torch.Tensor], Any]:
        image = cv2.imread(self.image_paths[idx], flags=cv2.IMREAD_ANYCOLOR)
        if self.transform:
            image = self.transform(image=image)["image"]

        return image
