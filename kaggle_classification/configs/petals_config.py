"""Petals config."""

from __future__ import annotations

from typing import Any, Dict

import albumentations as A
import cv2
import numpy as np
import timm
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from torch import nn, optim

from ..datasets import PetalsDataset
from .classification_config import ClassificationConfig


class PetalsConfig(ClassificationConfig[str]):
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

        reference_data = [
            {"image_path": image_path, "class_id": self.train_dataset.labels[image_name]}
            for image_path, image_name in zip(self.train_dataset.image_paths, self.train_dataset.image_names)
        ]

        def read_fn(item: Dict[str, Any]) -> Dict[str, Any]:
            image = cv2.imread(item["image_path"], flags=cv2.IMREAD_ANYCOLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = np.eye(self.dataset.num_classes)[item["class_id"]]

            return {"image": image, "global_label": label}

        return A.Compose(
            [
                A.MixUp(reference_data=reference_data, read_fn=read_fn, p=0.5),
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
    def val_augmentations(self) -> A.BaseCompose:
        """Augmentations for val dataset."""
        return A.Compose(
            [
                A.Resize(height=256, width=256),
                A.Normalize(mean=self.train_dataset.mean, std=self.train_dataset.std, max_pixel_value=1),
                ToTensorV2(),
            ]
        )
