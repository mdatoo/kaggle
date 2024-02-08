"""Image classification PyTorch model."""

from typing import Any, Dict

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
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


class ClassificationModel(pl.LightningModule):
    """Image classification PyTorch model.

    PyTorch model to perform image classification.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        criterion: nn.Module,
        optimiser: optim.Optimizer,
        optimiser_scheduler: optim.lr_scheduler.LRScheduler,
        optimiser_scheduler_monitor: str,
    ) -> None:
        """Initialise object.

        Args:
            model_name: Backbone to use
            num_classes: Number of classification classes
            loss: Loss to use
            optimiser: Optimiser to use
            optimiser_scheduler: LR scheduler to use
        """
        super().__init__()

        self.model = model
        self.criterion = criterion
        self.optimiser = optimiser
        self.optimiser_scheduler = optimiser_scheduler
        self.optimiser_scheduler_monitor = optimiser_scheduler_monitor

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_pre = Precision(task="multiclass", num_classes=num_classes)
        self.train_rec = Recall(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.train_confusion = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.train_outputs = CatMetric()
        self.train_labels = CatMetric()

        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_pre = Precision(task="multiclass", num_classes=num_classes)
        self.val_rec = Recall(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_confusion = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.val_outputs = CatMetric()
        self.val_labels = CatMetric()

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

        self.train_acc.update(predictions, labels)
        self.log("train_accuracy", self.train_acc, on_step=False, on_epoch=True)

        self.train_pre.update(predictions, labels)
        self.log("train_precision", self.train_pre, on_step=False, on_epoch=True)

        self.train_rec.update(predictions, labels)
        self.log("train_recall", self.train_rec, on_step=False, on_epoch=True)

        self.train_f1.update(predictions, labels)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)

        self.train_confusion.update(predictions, labels)
        self.train_outputs.update(outputs)
        self.train_labels.update(labels)

        return loss

    def on_train_epoch_end(self) -> None:
        """Log confusion matrix and PR curve."""
        if self.logger:
            confusion_matrix = self.train_confusion.compute().detach().cpu().numpy().astype(int)  # type: ignore[func-returns-value]

            plt.figure(figsize=(10, 7))
            figure = sn.heatmap(pd.DataFrame(confusion_matrix), cmap="mako").get_figure()
            plt.close(figure)
            self.logger.experiment.add_figure("train_confusion", figure, self.current_epoch)  # type: ignore[attr-defined]

            train_probs = torch.softmax(self.train_outputs.compute(), 1)
            train_labels = self.train_labels.compute()

            for image_class in range(train_probs.shape[1]):
                self.logger.experiment.add_pr_curve(  # type: ignore[attr-defined]
                    f"train_{image_class}",
                    train_labels == image_class,
                    train_probs[:, image_class],
                    self.current_epoch,
                )
            self.logger.experiment.add_pr_curve(
                "train_micro",
                torch.concat([train_labels == image_class for image_class in range(train_probs.shape[1])]),
                train_probs.transpose(0, 1).flatten(),
                self.current_epoch,
            )

        self.train_confusion.reset()
        self.train_outputs.reset()
        self.train_labels.reset()

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

        self.val_acc.update(predictions, labels)
        self.log("val_accuracy", self.val_acc, on_step=False, on_epoch=True)

        self.val_pre.update(predictions, labels)
        self.log("val_precision", self.val_pre, on_step=False, on_epoch=True)

        self.val_rec.update(predictions, labels)
        self.log("val_recall", self.val_rec, on_step=False, on_epoch=True)

        self.val_f1.update(predictions, labels)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)

        self.val_confusion.update(predictions, labels)
        self.val_outputs.update(outputs)
        self.val_labels.update(labels)

    def on_validation_epoch_end(self) -> None:
        """Log confusion matrix and PR curve."""
        if self.logger:
            confusion_matrix = self.val_confusion.compute().detach().cpu().numpy().astype(int)  # type: ignore[func-returns-value]

            plt.figure(figsize=(10, 7))
            figure = sn.heatmap(pd.DataFrame(confusion_matrix), cmap="mako").get_figure()
            plt.close(figure)
            self.logger.experiment.add_figure("val_confusion", figure, self.current_epoch)  # type: ignore[attr-defined]

            val_probs = torch.softmax(self.val_outputs.compute(), 1)
            val_labels = self.val_labels.compute()

            for image_class in range(val_probs.shape[1]):
                self.logger.experiment.add_pr_curve(  # type: ignore[attr-defined]
                    f"val_{image_class}",
                    val_labels == image_class,
                    val_probs[:, image_class],
                    self.current_epoch,
                )
            self.logger.experiment.add_pr_curve(
                "val_micro",
                torch.concat([val_labels == image_class for image_class in range(val_probs.shape[1])]),
                val_probs.transpose(0, 1).flatten(),
                self.current_epoch,
            )

        self.val_confusion.reset()
        self.val_outputs.reset()
        self.val_labels.reset()

    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> Dict[str, Any]:
        """Return optimiser with schedules."""
        return {
            "optimizer": self.optimiser,
            "lr_scheduler": {
                "scheduler": self.optimiser_scheduler,
                "monitor": self.optimiser_scheduler_monitor,
            },
        }
