"""Image classification PyTorch dataset."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, partial
from glob import glob
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
import numpy.typing as npt
import torch
from albumentations import BaseCompose
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

T = TypeVar("T")


class ClassificationDataset(
    Dataset[Tuple[Union[npt.NDArray[np.uint8], torch.Tensor], npt.NDArray[np.uint8]]], Generic[T]
):
    """Image classification PyTorch dataset.

    PyTorch dataset reading from a folder of images.
    """

    def __init__(
        self, image_folder: str, image_name_to_label: Dict[str, T], transform: Optional[BaseCompose] = None
    ) -> None:
        """Initialise object.

        Args:
            image_folder: Path to folder of images
            image_name_to_label: Mapping from image name (filename without extension) to label
            transform: Image transformations to apply
        """
        self._image_folder = image_folder
        self._image_name_to_label = image_name_to_label
        self._label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[Union[npt.NDArray[np.uint8], torch.Tensor], npt.NDArray[np.uint8]]:
        """Return image and target given index.

        Args:
            idx: Data index
        """
        image = cv2.imread(self.image_paths[idx], flags=cv2.IMREAD_ANYCOLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self._image_name_to_label[self.image_names[idx]]
        target = np.eye(self.num_classes)[self._label_to_idx[label]]

        if self.transform:
            image, target = list(self.transform(image=image, global_label=target).values())[:2]

        return image, target

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
        train_image_names, test_image_names = train_test_split(
            self.image_names,
            test_size=test_size,
            stratify=self.labels,
            random_state=random_state,
        )

        train_dataset = ClassificationDataset(
            self.image_folder,
            {train_image_name: self._image_name_to_label[train_image_name] for train_image_name in train_image_names},
            self.transform,
        )
        test_dataset = ClassificationDataset(
            self.image_folder,
            {test_image_name: self._image_name_to_label[test_image_name] for test_image_name in test_image_names},
            self.transform,
        )

        return train_dataset, test_dataset

    @cached_property
    def mean(self) -> npt.NDArray[np.uint8]:
        """Calculate mean of all images in dataset."""
        with ThreadPoolExecutor() as executor:
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
        with ThreadPoolExecutor() as executor:
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

    @property
    def labels(self) -> List[T]:
        """Labels for each datum in dataset."""
        return [self._image_name_to_label[image_name] for image_name in self.image_names]

    @property
    def image_names(self) -> List[str]:
        """Image names for each datum in dataset."""
        return [self._get_image_name(image_path) for image_path in self.image_paths]

    @cached_property
    def image_paths(self) -> List[str]:
        """Image paths for each datum in dataset."""
        return [
            image_path
            for image_path in glob(f"{self.image_folder}/**")
            if self._get_image_name(image_path) in self._image_name_to_label
        ]

    def _get_image_name(self, image_path: str) -> str:
        """Get image name from path."""
        return image_path.rsplit("/", 1)[1].rsplit(".", 1)[0]

    @property
    def image_folder(self) -> str:
        """Path to folder of images."""
        return self._image_folder

    @property
    def image_name_to_label(self) -> Dict[str, T]:
        """Mapping from image name (filename without extension) to label."""
        return self._image_name_to_label

    @property
    def label_to_idx(self) -> Dict[T, int]:
        """Mapping from label to index."""
        return self._label_to_idx

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return len(set(self.labels))
