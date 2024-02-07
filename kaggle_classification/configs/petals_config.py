from __future__ import annotations

from typing import List, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from ..datasets import PetalsDataset
from .classification_config import ClassificationConfig


class PetalsConfig(ClassificationConfig):
    dataset = PetalsDataset("kaggle_classification/data/petals/train/", "kaggle_classification/data/petals/labels.csv")
    train_val_split = 0.2
    seed = 0
    train_batch_size = 16
    train_num_workers = 8
    val_batch_size = 16
    val_num_workers = 8
    model_name = "resnet50"

    @property
    def train_augmentations(self) -> Optional[A.BaseCompose]:
        return A.Compose(
            [
                A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.2),
                A.AdvancedBlur(p=0.2),
                A.ShiftScaleRotate(p=0.5),
                A.Normalize(self.train_dataset.dataset.mean, self.train_dataset.dataset.std, 1),
                ToTensorV2(),
            ]
        )

    @property
    def val_augmentations(self) -> Optional[A.BaseCompose]:
        return A.Compose(
            [
                A.Normalize(self.train_dataset.dataset.mean, self.train_dataset.dataset.std, 1),
                ToTensorV2(),
            ]
        )

    @property
    def callbacks(self) -> List[Callback]:
        return [
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                dirpath="checkpoints/",
                filename="{epoch}-{val_loss:.2f}-{val_accuracy:.2f}",
                monitor="val_loss",
            ),
            EarlyStopping(monitor="val_loss", patience=20),
        ]
