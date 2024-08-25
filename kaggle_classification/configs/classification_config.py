"""Image classification config."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Generic, List, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import torch
from albumentations import BaseCompose
from lightning.pytorch import Callback
from torch import nn, optim
from torch.utils.data import DataLoader

from ..batch_augmentations import BatchAugmentation
from ..datasets import ClassificationDataset

T = TypeVar("T")


class ClassificationConfig(ABC, Generic[T]):
    """Image classification config.

    Config object for an image classification task.
    """

    # pylint: disable=too-many-public-methods
    @property
    def train_dataloader(self) -> DataLoader[Tuple[Union[npt.NDArray[np.uint8], torch.Tensor], npt.NDArray[np.uint8]]]:
        """Dataloader for train dataset."""

        def _collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
            images, targets = zip(*batch)
            images = torch.stack(images)
            targets = torch.from_numpy(np.array(targets))

            return self.train_batch_augmentations.apply((images, targets))

        return DataLoader(
            dataset=self._train_dataset_with_augmentations,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            shuffle=True,
            collate_fn=_collate,
        )

    @property
    def val_dataloader(self) -> DataLoader[Tuple[Union[npt.NDArray[np.uint8], torch.Tensor], npt.NDArray[np.uint8]]]:
        """Dataloader for val dataset."""
        return DataLoader(
            dataset=self._val_dataset_with_augmentations,
            batch_size=self.val_batch_size,
            num_workers=self.val_num_workers,
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=True,
            shuffle=False,
        )

    @property
    def _train_dataset_with_augmentations(self) -> ClassificationDataset[T]:
        dataset = self.train_dataset
        dataset.transform = self.train_augmentations
        return dataset

    @property
    def _val_dataset_with_augmentations(self) -> ClassificationDataset[T]:
        dataset = self.val_dataset
        dataset.transform = self.val_augmentations
        return dataset

    @property
    def train_dataset(self) -> ClassificationDataset[T]:
        """Train dataset."""
        return self._train_val_datasets[0]

    @property
    def val_dataset(self) -> ClassificationDataset[T]:
        """Val dataset."""
        return self._train_val_datasets[1]

    @cached_property
    def _train_val_datasets(
        self,
    ) -> Tuple[ClassificationDataset[T], ClassificationDataset[T]]:
        return self.dataset.split_train_test(self.train_val_split, self.seed)

    @property
    @abstractmethod
    def experiment_name(self) -> str:
        """Name of experiment."""

    @property
    @abstractmethod
    def work_dir(self) -> str:
        """Working directory of experiment."""

    @property
    @abstractmethod
    def dataset(self) -> ClassificationDataset[T]:
        """Train and val dataset."""

    @property
    @abstractmethod
    def train_val_split(self) -> float:
        """Portion of data that should be in val (0.0-1.0)."""

    @property
    @abstractmethod
    def seed(self) -> int:
        """Random seed for dataset splitting."""

    @property
    @abstractmethod
    def epochs(self) -> int:
        """No of epochs."""

    @property
    @abstractmethod
    def precision(self) -> str:
        """Floating point precision."""

    @property
    @abstractmethod
    def gradient_max_magnitude(self) -> float:
        """Clip gradients' max magnitude."""

    @property
    @abstractmethod
    def train_augmentations(self) -> BaseCompose:
        """Augmentations for train dataset."""

    @property
    @abstractmethod
    def train_batch_augmentations(self) -> BatchAugmentation:
        """Batch augmentations for train dataset."""

    @property
    @abstractmethod
    def val_augmentations(self) -> BaseCompose:
        """Augmentations for val dataset."""

    @property
    @abstractmethod
    def train_batch_size(self) -> int:
        """Batch size used for training."""

    @property
    @abstractmethod
    def train_num_workers(self) -> int:
        """No of workers used in train dataloader."""

    @property
    @abstractmethod
    def val_batch_size(self) -> int:
        """Batch size used for validation."""

    @property
    @abstractmethod
    def val_num_workers(self) -> int:
        """No of workers used in val dataloader."""

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        """PyTorch model."""

    @property
    @abstractmethod
    def loss(self) -> nn.Module:
        """PyTorch loss."""

    @property
    @abstractmethod
    def optimiser(self) -> optim.Optimizer:
        """PyTorch optimiser."""

    @property
    @abstractmethod
    def optimiser_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        """PyTorch LR scheduler."""

    @property
    @abstractmethod
    def callbacks(self) -> List[Callback]:
        """PyTorch Lightning trainer callbacks."""
