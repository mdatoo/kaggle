"""Petals config."""

from __future__ import annotations

import albumentations as A
import cv2
import timm
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from torch import nn, optim

from ..batch_augmentations import (
    BatchAugmentation,
    CompositeBatchAugmentation,
    CutMixBatchAugmentation,
    MixUpBatchAugmentation,
)
from ..datasets import PetalsDataset
from .classification_config import ClassificationConfig


class PetalsConfig(ClassificationConfig[int]):
    """Petals config.

    Config object for petals classification task.
    """

    experiment_name = "petals"
    work_dir = "logs"
    dataset = PetalsDataset("kaggle_classification/data/petals/train/", "kaggle_classification/data/petals/labels.csv")
    train_val_split = 0.2
    seed = 0
    epochs = 75
    precision = "bf16-mixed"
    gradient_max_magnitude = 0.5
    train_batch_size = 256
    train_num_workers = 8
    val_batch_size = 256
    val_num_workers = 8
    model = timm.create_model("resnet50", pretrained=True, in_chans=3, num_classes=dataset.num_classes)
    loss = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
    optimiser_scheduler = optim.lr_scheduler.SequentialLR(
        optimiser,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimiser, 0.1, 1.0, 5),
            optim.lr_scheduler.ExponentialLR(optimiser, 0.95),
        ],
        milestones=[5],
    )
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=f"{work_dir}/{experiment_name}/checkpoints",
            filename="{epoch}-{val_loss:.2f}-{val_accuracy:.2f}",
            monitor="val_loss",
        ),
        StochasticWeightAveraging(swa_lrs=0.0001, swa_epoch_start=60, annealing_epochs=5),
    ]

    @property
    def train_augmentations(self) -> A.BaseCompose:
        """Augmentations for train dataset."""
        return A.Compose(
            [
                A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.5),
                A.Transpose(p=0.2),
                A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.2),
                A.Resize(height=256, width=256),
                A.RandomCrop(height=224, width=224),
                A.RandomBrightnessContrast(p=0.2),
                A.Blur(p=0.2),
                A.CoarseDropout(p=0.5),
                A.Normalize(mean=self.train_dataset.mean, std=self.train_dataset.std, max_pixel_value=1),
                ToTensorV2(),
            ]
        )

    @property
    def train_batch_augmentations(self) -> BatchAugmentation:
        """Batch augmentations for train dataset."""
        return CompositeBatchAugmentation(
            [CutMixBatchAugmentation(probability=0.5), MixUpBatchAugmentation(probability=0.5)], probability=0.2
        )

    @property
    def val_augmentations(self) -> A.BaseCompose:
        """Augmentations for val dataset."""
        return A.Compose(
            [
                A.Resize(height=256, width=256),
                A.Normalize(mean=self.train_dataset.mean, std=self.train_dataset.std, max_pixel_value=1),
                ToTensorV2(),
            ]
        )
