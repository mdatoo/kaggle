from abc import abstractproperty
from functools import cached_property
from typing import List, Optional, Tuple

from albumentations import BaseCompose
from lightning.pytorch import Callback
from torch.utils.data import DataLoader, Subset

from ..datasets import ClassificationDataset


class ClassificationConfig:
    @property
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._train_dataset_with_augmentations,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            pin_memory=True,
            shuffle=True,
        )

    @property
    def val_dataloader(self) -> DataLoader:
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
        return self._train_val_datasets[0]

    @property
    def val_dataset(self) -> Subset:
        return self._train_val_datasets[1]

    @cached_property
    def _train_val_datasets(self) -> Tuple[Subset, Subset]:
        return self.dataset.split_train_test(self.train_val_split, self.seed)

    @abstractproperty
    def dataset(self) -> ClassificationDataset:
        pass

    @abstractproperty
    def train_val_split(self) -> float:
        pass

    @abstractproperty
    def seed(self) -> int:
        pass

    @abstractproperty
    def train_augmentations(self) -> Optional[BaseCompose]:
        pass

    @abstractproperty
    def val_augmentations(self) -> Optional[BaseCompose]:
        pass

    @abstractproperty
    def train_batch_size(self) -> int:
        pass

    @abstractproperty
    def train_num_workers(self) -> int:
        pass

    @abstractproperty
    def val_batch_size(self) -> int:
        pass

    @abstractproperty
    def val_num_workers(self) -> int:
        pass

    @abstractproperty
    def model_name(self) -> str:
        pass

    @abstractproperty
    def callbacks(self) -> List[Callback]:
        pass
