import random

import albumentations as A
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dataset import ClassificationDataset
from model.classification_model import ClassificationModel

SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def run() -> None:
    train_val_df = pd.read_csv("data/labels.csv")
    train_val_labels = train_val_df.set_index("id").to_dict()["label"]
    train_val_dataset = ClassificationDataset("data/train/", train_val_labels)

    train_dataset, val_dataset = train_val_dataset.split_train_test(0.2, SEED)
    train_dataset.dataset.transform = A.Compose(
        [
            A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.2),
            A.AdvancedBlur(p=0.2),
            A.ShiftScaleRotate(p=0.5),
            A.Normalize(train_dataset.dataset.mean, train_dataset.dataset.std, 1),
            ToTensorV2(),
        ]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=12, shuffle=True)
    val_dataset.dataset.transform = A.Compose(
        [
            A.Normalize(train_dataset.dataset.mean, train_dataset.dataset.std, 1),
            ToTensorV2(),
        ]
    )
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=12, shuffle=False)

    classification_model = ClassificationModel("resnet50", num_classes=104)
    trainer = pl.Trainer(
        logger=TensorBoardLogger("logs/", name="resnet50"),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                dirpath="checkpoints/",
                filename="{epoch}-{val_loss:.2f}-{val_accuracy:.2f}",
                monitor="val_loss",
            ),
            EarlyStopping(monitor="val_loss", patience=20),
        ],
    )
    trainer.fit(
        model=classification_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    run()
