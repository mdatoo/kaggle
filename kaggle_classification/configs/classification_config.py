"""Image classification config."""

from abc import abstractmethod
from functools import cached_property
from typing import List, Optional, Tuple

from albumentations import BaseCompose
from lightning.pytorch import Callback
from torch.utils.data import DataLoader, Subset

from ..datasets import ClassificationDataset


class ClassificationConfig:
    """Image classification config.

    Config object for an image classification task.
    """

    @property
    def train_dataloader(self) -> DataLoader:
        """Dataloader for train dataset."""
        return DataLoader(
            dataset=self._train_dataset_with_augmentations,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            pin_memory=True,
            shuffle=True,
        )

    @property
    def val_dataloader(self) -> DataLoader:
        """Dataloader for val dataset."""
        return DataLoader(
            dataset=self._val_dataset_with_augmentations,
            batch_size=self.val_batch_size,
            num_workers=self.val_num_workers,
            pin_memory=True,
            shuffle=False,
        )

    @property
    def _train_dataset_with_augmentations(self) -> Subset:
        dataset = self.train_dataset
        dataset.dataset.transform = self.train_augmentations
        return dataset

    @property
    def _val_dataset_with_augmentations(self) -> Subset:
        dataset = self.val_dataset
        dataset.dataset.transform = self.val_augmentations
        return dataset

    @property
    def train_dataset(self) -> Subset:
        """Train dataset."""
        return self._train_val_datasets[0]

    @property
    def val_dataset(self) -> Subset:
        """Val dataset."""
        return self._train_val_datasets[1]

    @cached_property
    def _train_val_datasets(self) -> Tuple[Subset, Subset]:
        return self.dataset.split_train_test(self.train_val_split, self.seed)

    @abstractmethod
    @property
    def dataset(self) -> ClassificationDataset:
        """Train and val dataset."""

    @abstractmethod
    @property
    def train_val_split(self) -> float:
        """Portion of data that should be in val (0.0-1.0)."""

    @abstractmethod
    @property
    def seed(self) -> int:
        """Random seed for dataset splitting."""

    @abstractmethod
    @property
    def train_augmentations(self) -> Optional[BaseCompose]:
        """Augmentations for train dataset."""

    @abstractmethod
    @property
    def val_augmentations(self) -> Optional[BaseCompose]:
        """Augmentations for val dataset."""

    @abstractmethod
    @property
    def train_batch_size(self) -> int:
        """Batch size used for training."""

    @abstractmethod
    @property
    def train_num_workers(self) -> int:
        """No of workers used in train dataloader."""

    @abstractmethod
    @property
    def val_batch_size(self) -> int:
        """Batch size used for validation."""

    @abstractmethod
    @property
    def val_num_workers(self) -> int:
        """No of workers used in val dataloader."""

    @abstractmethod
    @property
    def model_name(self) -> str:
        """Name of backbone."""

    @abstractmethod
    @property
    def callbacks(self) -> List[Callback]:
        """PyTorch Lightning trainer callbacks."""
