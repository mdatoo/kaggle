"""Image classification PyTorch dataset."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import cached_property, partial
from glob import glob
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
import numpy.typing as npt
import torch
from albumentations import BaseCompose
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

T = TypeVar("T")


class ClassificationDataset(Dataset[Tuple[Union[npt.NDArray[np.uint8], torch.Tensor], T]]):
    """Image classification PyTorch dataset.

    PyTorch dataset reading from a folder of images.
    """

    def __init__(self, image_folder: str, labels: Dict[str, T], transform: Optional[BaseCompose] = None) -> None:
        """Initialise object.

        Args:
            image_folder: Path to folder of images
            labels: Mapping from image name (filename without extension) to label
            transform: Image transformations to apply
        """
        self._image_folder = image_folder
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[Union[npt.NDArray[np.uint8], torch.Tensor], T]:
        """Return image and label given index.

        Args:
            idx: Data index
        """
        image = cv2.imread(self.image_paths[idx], flags=cv2.IMREAD_ANYCOLOR)
        label = self.labels[self.image_names[idx]]

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self.image_paths)

    def split_train_test(
        self, test_size: float, random_state: int
    ) -> Tuple[ClassificationDataset[T], ClassificationDataset[T]]:
        """Split dataset into train and test.

        Args:
            test_size: Relative size of test dataset
            random_state: Random seed
        """
        train_indices, test_indices = train_test_split(
            list(self.labels.keys()),
            test_size=test_size,
            stratify=list(self.labels.values()),
            random_state=random_state,
        )

        train_dataset = ClassificationDataset(
            self.image_folder, {key: self.labels[key] for key in train_indices}, self.transform
        )
        test_dataset = ClassificationDataset(
            self.image_folder, {key: self.labels[key] for key in test_indices}, self.transform
        )

        return train_dataset, test_dataset

    @cached_property
    def mean(self) -> npt.NDArray[np.uint8]:
        """Calculate mean of all images in dataset."""
        with ProcessPoolExecutor() as executor:
            means = []
            for mean in tqdm(
                executor.map(self._calc_mean, self.image_paths),
                desc="Calculating mean of images",
                total=len(self.image_paths),
            ):
                means.append(mean)
            return np.mean(np.concatenate(means, axis=0), axis=0)  # type: ignore[no-any-return]

    @staticmethod
    def _calc_mean(image_path: str) -> npt.NDArray[np.uint8]:
        image = cv2.imread(image_path, flags=cv2.IMREAD_ANYCOLOR)
        return np.mean(image, axis=0)  # type: ignore[no-any-return]

    @cached_property
    def std(self) -> npt.NDArray[np.uint8]:
        """Calculate standard deviation of all images in dataset."""
        with ProcessPoolExecutor() as executor:
            variances = []
            for variance in tqdm(
                executor.map(partial(self._calc_variance, mean=self.mean), self.image_paths),
                desc="Calculating std of images",
                total=len(self.image_paths),
            ):
                variances.append(variance)
            return np.sqrt(np.mean(np.concatenate(variances, axis=0), axis=0))  # type: ignore[no-any-return]

    @staticmethod
    def _calc_variance(image_path: str, mean: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        image = cv2.imread(image_path, flags=cv2.IMREAD_ANYCOLOR)
        return np.mean((image - mean) ** 2, axis=0)  # type: ignore[no-any-return]

    @cached_property
    def image_names(self) -> List[str]:
        """All image names in dataset."""
        return [image_path.rsplit("/", 1)[1].rsplit(".", 1)[0] for image_path in self.image_paths]

    @cached_property
    def image_paths(self) -> List[str]:
        """All image paths in dataset."""
        return glob(f"{self.image_folder}/**")

    @property
    def image_folder(self) -> str:
        """Image folder."""
        return self._image_folder

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return len(set(self.labels.values()))
