"""Petals config."""

from __future__ import annotations

import albumentations as A
import numpy as np
import timm
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch import nn, optim

from ..datasets import PetalsDataset
from ..lr_schedulers import ReduceLROnPlateauWrapper
from .classification_config import ClassificationConfig


class PetalsConfig(ClassificationConfig[str]):
    """Petals config.

    Config object for petals classification task.
    """

    experiment_name = "resnet50"
    dataset = PetalsDataset("kaggle_classification/data/petals/train/", "kaggle_classification/data/petals/labels.csv")
    train_val_split = 0.2
    seed = 0
    train_batch_size = 16
    train_num_workers = 8
    val_batch_size = 16
    val_num_workers = 8
    model = timm.create_model("resnet50", pretrained=True, in_chans=3, num_classes=dataset.num_classes)
    loss = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    optimiser_scheduler = ReduceLROnPlateauWrapper(  # type: ignore[assignment]
        optim.lr_scheduler.LambdaLR(
            optimiser,
            lambda epoch: min(1, 0.1 + 0.9 * (np.exp(epoch / 5) - 1) / (np.e - 1)),
        ),
        optimiser=optimiser,
        patience=10,
    )
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath="checkpoints/",
            filename="{epoch}-{val_loss:.2f}-{val_accuracy:.2f}",
            monitor="val_loss",
        ),
        EarlyStopping(monitor="val_loss", patience=20),
    ]

    @property
    def train_augmentations(self) -> A.BaseCompose:
        """Augmentations for train dataset."""
        return A.Compose(
            [
                A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.2),
                A.AdvancedBlur(p=0.2),
                A.ShiftScaleRotate(p=0.5),
                A.Normalize(self.train_dataset.mean, self.train_dataset.std, 1),
                ToTensorV2(),
            ]
        )

    @property
    def val_augmentations(self) -> A.BaseCompose:
        """Augmentations for val dataset."""
        return A.Compose(
            [
                A.Normalize(self.train_dataset.mean, self.train_dataset.std, 1),
                ToTensorV2(),
            ]
        )
