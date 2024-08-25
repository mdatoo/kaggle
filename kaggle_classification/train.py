"""Train model script."""

import random
from argparse import ArgumentParser
from typing import Any

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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def run() -> None:
    """Run training."""
    parser = ArgumentParser(description="Train a classification model")
    parser.add_argument("config", type=config_argparse)
    args = parser.parse_args()

    config: ClassificationConfig[Any] = args.config

    classification_model = ClassificationModel(
        config.model,
        config.dataset.idx_to_label,
        config.loss,
        config.optimiser,
        config.optimiser_scheduler,
    )
    train_dataloader = config.train_dataloader
    val_dataloader = config.val_dataloader

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        precision=config.precision,  # type: ignore[arg-type]
        gradient_clip_val=config.gradient_max_magnitude,
        logger=TensorBoardLogger(config.work_dir, name=config.experiment_name),
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
