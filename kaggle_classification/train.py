"""Train model script."""

import random
from argparse import ArgumentParser

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from kaggle_classification.configs import ClassificationConfig, config_argparse
from kaggle_classification.models.classification_model import ClassificationModel

SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def run() -> None:
    """Run training."""
    parser = ArgumentParser(description="Train a classification model")
    parser.add_argument("config", type=config_argparse)
    args = parser.parse_args()

    config: ClassificationConfig = args.config

    classification_model = ClassificationModel(config.model_name, config.dataset.num_classes)
    train_dataloader = config.train_dataloader
    val_dataloader = config.val_dataloader

    trainer = pl.Trainer(
        logger=TensorBoardLogger("logs/", name=config.model_name),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        callbacks=config.callbacks,
    )
    trainer.fit(
        model=classification_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    run()
