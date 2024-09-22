"""Catsdogs config."""

# pylint: disable=duplicate-code

from __future__ import annotations

from datetime import datetime
from os import path

import albumentations as A
import cv2
import timm
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import (
    BackboneFinetuning,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch import nn, optim

from ..batch_augmentations import (
    BatchAugmentation,
    CompositeBatchAugmentation,
    CutMixBatchAugmentation,
    MixUpBatchAugmentation,
)
from ..datasets import CatsdogsDataset
from ..losses import ClassBalancedFocalLoss
from .classification_config import ClassificationConfig


class CatsdogsConfig(ClassificationConfig[str]):
    """Catsdogs config.

    Config object for catsdogs classification task.
    """

    experiment_name = "catsdogs"
    work_dir = "logs"
    version = str(datetime.now())
    dataset = CatsdogsDataset("kaggle_classification/data/catsdogs/")
    train_val_split = 0.2
    seed = 0
    epochs = 75
    precision = "bf16-mixed"
    gradient_max_magnitude = 0.5
    train_batch_size = 64
    train_num_workers = 8
    val_batch_size = 64
    val_num_workers = 8
    model = timm.create_model(
        "vit_small_patch14_reg4_dinov2.lvd142m", pretrained=True, in_chans=3, num_classes=dataset.num_classes
    )
    optimiser = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)
    optimiser_scheduler = optim.lr_scheduler.SequentialLR(
        optimiser,
        schedulers=[
            optim.lr_scheduler.LambdaLR(optimiser, lambda epoch: 2 * (epoch + 1)),
            optim.lr_scheduler.LambdaLR(optimiser, lambda _: 10.0),
            optim.lr_scheduler.LambdaLR(optimiser, lambda _: 1.0),
            optim.lr_scheduler.ExponentialLR(optimiser, 0.95),
        ],
        milestones=[5, 10, 15],
    )
    callbacks = [
        BackboneFinetuning(unfreeze_backbone_at_epoch=10, backbone_initial_ratio_lr=1.0),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=path.join(work_dir, experiment_name, version),
            filename="{epoch}-{val_loss:.2f}-{val_accuracy:.2f}",
            monitor="val_loss",
        ),
    ]

    @property
    def loss(self) -> nn.Module:
        """Loss function."""
        return ClassBalancedFocalLoss(self.train_dataset.class_counts, 0.9999, 1.0)

    @property
    def train_augmentations(self) -> A.BaseCompose:
        """Augmentations for train dataset."""
        return A.Compose(
            [
                A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.5),
                A.Transpose(p=0.2),
                A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.2),
                A.RandomResizedCrop(height=518, width=518, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
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
                A.Resize(height=518, width=518),
                A.Normalize(mean=self.train_dataset.mean, std=self.train_dataset.std, max_pixel_value=1),
                ToTensorV2(),
            ]
        )
