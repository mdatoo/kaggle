"""Image classification PyTorch model."""

from typing import Any, Dict

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import timm
import torch
from torch import nn, optim
from torchmetrics import (
    Accuracy,
    CatMetric,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
)
from torchvision.utils import make_grid

from ..lr_schedulers import ReduceLROnPlateauWrapper


class ClassificationModel(pl.LightningModule):
    """Image classification PyTorch model.

    PyTorch model to perform image classification.
    """

    def __init__(self, model_name: str, num_classes: int) -> None:
        """Initialise object.

        Args:
            model_name: Name of backbone to use
            num_classes: Number of classification classes
        """
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=True, in_chans=3, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.01)

        self.metrics = {
            "train": {
                "acc": Accuracy(task="multiclass", num_classes=num_classes),
                "pre": Precision(task="multiclass", num_classes=num_classes),
                "rec": Recall(task="multiclass", num_classes=num_classes),
                "f1": F1Score(task="multiclass", num_classes=num_classes),
                "confusion": ConfusionMatrix(task="multiclass", num_classes=num_classes),
                "outputs": CatMetric(),
                "labels": CatMetric(),
            },
            "val": {
                "acc": Accuracy(task="multiclass", num_classes=num_classes),
                "pre": Precision(task="multiclass", num_classes=num_classes),
                "rec": Recall(task="multiclass", num_classes=num_classes),
                "f1": F1Score(task="multiclass", num_classes=num_classes),
                "confusion": ConfusionMatrix(task="multiclass", num_classes=num_classes),
                "outputs": CatMetric(),
                "labels": CatMetric(),
            },
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Calculate and log metrics/loss to tensorboard.

        Args:
            batch: Current dataloader batch (images/labels)
            batch_idx: Index of current batch
        """
        images, labels = batch

        if batch_idx == 0 and self.logger:
            grid = make_grid(images, nrow=64)
            self.logger.experiment.add_image("first_batch", grid, 0)  # type: ignore[attr-defined]
            self.logger.experiment.add_graph(self.model, images)  # type: ignore[attr-defined]

        outputs = self.model(images)
        _, predictions = torch.max(outputs.data, 1)

        loss: torch.Tensor = self.criterion(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=False)

        self.metrics["train"]["acc"].update(predictions, labels)
        self.log("train_accuracy", self.metrics["train"]["acc"], on_step=False, on_epoch=True)

        self.metrics["train"]["pre"].update(predictions, labels)
        self.log("train_precision", self.metrics["train"]["pre"], on_step=False, on_epoch=True)

        self.metrics["train"]["rec"].update(predictions, labels)
        self.log("train_recall", self.metrics["train"]["rec"], on_step=False, on_epoch=True)

        self.metrics["train"]["f1"].update(predictions, labels)
        self.log("train_f1", self.metrics["train"]["f1"], on_step=False, on_epoch=True)

        self.metrics["train"]["confusion"].update(predictions, labels)
        self.metrics["train"]["outputs"].update(outputs)
        self.metrics["train"]["labels"].update(labels)

        return loss

    def on_train_epoch_end(self) -> None:
        """Log confusion matrix and PR curve."""
        if self.logger:
            confusion_matrix = self.metrics["train"]["confusion"].compute().detach().cpu().numpy().astype(int)

            plt.figure(figsize=(10, 7))
            figure = sn.heatmap(pd.DataFrame(confusion_matrix), cmap="mako").get_figure()
            plt.close(figure)
            self.logger.experiment.add_figure("train_confusion", figure, self.current_epoch)  # type: ignore[attr-defined]

            train_probs = torch.softmax(self.metrics["train"]["outputs"].compute(), 1)
            train_labels = self.metrics["train"]["labels"].compute()

            for image_class in range(train_probs.shape[1]):
                self.logger.experiment.add_pr_curve(  # type: ignore[attr-defined]
                    f"train_{image_class}",
                    train_labels == image_class,
                    train_probs[:, image_class],
                    self.current_epoch,
                )

        self.metrics["train"]["confusion"].reset()
        self.metrics["train"]["outputs"].reset()
        self.metrics["train"]["labels"].reset()

    def validation_step(self, batch: torch.Tensor, _: int) -> None:  # pylint: disable=arguments-differ
        """Calculate and log metrics/loss to tensorboard.

        Args:
            batch: Current dataloader batch (images/labels)
            _: Unused (index of current batch)
        """
        images, labels = batch

        outputs = self.model(images)
        _, predictions = torch.max(outputs.data, 1)

        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        self.metrics["val"]["acc"].update(predictions, labels)
        self.log("val_accuracy", self.metrics["val"]["acc"], on_step=False, on_epoch=True)

        self.metrics["val"]["pre"].update(predictions, labels)
        self.log("val_precision", self.metrics["val"]["pre"], on_step=False, on_epoch=True)

        self.metrics["val"]["rec"].update(predictions, labels)
        self.log("val_recall", self.metrics["val"]["rec"], on_step=False, on_epoch=True)

        self.metrics["val"]["f1"].update(predictions, labels)
        self.log("val_f1", self.metrics["val"]["f1"], on_step=False, on_epoch=True)

        self.metrics["val"]["confusion"].update(predictions, labels)
        self.metrics["val"]["outputs"].update(outputs)
        self.metrics["val"]["labels"].update(labels)

    def on_validation_epoch_end(self) -> None:
        """Log confusion matrix and PR curve."""
        if self.logger:
            confusion_matrix = self.metrics["val"]["confusion"].compute().detach().cpu().numpy().astype(int)

            plt.figure(figsize=(10, 7))
            figure = sn.heatmap(pd.DataFrame(confusion_matrix), cmap="mako").get_figure()
            plt.close(figure)
            self.logger.experiment.add_figure("val_confusion", figure, self.current_epoch)  # type: ignore[attr-defined]

            val_probs = torch.softmax(self.metrics["val"]["outputs"].compute(), 1)
            val_labels = self.metrics["val"]["labels"].compute()

            for image_class in range(val_probs.shape[1]):
                self.logger.experiment.add_pr_curve(  # type: ignore[attr-defined]
                    f"val_{image_class}",
                    val_labels == image_class,
                    val_probs[:, image_class],
                    self.current_epoch,
                )

        self.metrics["val"]["confusion"].reset()
        self.metrics["val"]["outputs"].reset()
        self.metrics["val"]["labels"].reset()

    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> Dict[str, Any]:
        """Return optimiser with schedules."""
        return {
            "optimizer": self.optimiser,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateauWrapper(
                    optim.lr_scheduler.LambdaLR(
                        self.optimiser,
                        lambda epoch: min(1, 0.1 + 0.9 * (np.exp(epoch / 5) - 1) / (np.e - 1)),
                    ),
                    optimiser=self.optimiser,
                    patience=10,
                ),
                "monitor": "val_loss",
            },
        }
