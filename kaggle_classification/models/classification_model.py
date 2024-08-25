"""Image classification PyTorch model."""

from typing import Any, Dict

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    PrecisionRecallCurve,
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
    ) -> None:
        """Initialise object.

        Args:
            model: Backbone to use
            num_classes: Number of classification classes
            criterion: Loss function to use
            optimiser: Optimiser to use
            optimiser_scheduler: LR scheduler to use
        """
        super().__init__()

        self.model = model
        self.criterion = criterion
        self.optimiser = optimiser
        self.optimiser_scheduler = optimiser_scheduler

        task = "multiclass" if num_classes > 2 else "binary"
        self.val_acc = Accuracy(task=task, num_classes=num_classes)  # type: ignore[arg-type]
        self.val_pre = Precision(task=task, num_classes=num_classes)  # type: ignore[arg-type]
        self.val_rec = Recall(task=task, num_classes=num_classes)  # type: ignore[arg-type]
        self.val_f1 = F1Score(task=task, num_classes=num_classes)  # type: ignore[arg-type]
        self.val_confusion = ConfusionMatrix(task=task, num_classes=num_classes)  # type: ignore[arg-type]
        self.val_pr_curve = PrecisionRecallCurve(task=task, num_classes=num_classes)  # type: ignore[arg-type]

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Calculate and log metrics/loss to tensorboard.

        Args:
            batch: Current dataloader batch (images/targets)
            batch_idx: Index of current batch
        """
        images, targets = batch

        if batch_idx == 0 and self.logger:
            grid = make_grid(images, nrow=64)
            self.logger.experiment.add_image("first_batch", grid, 0)  # type: ignore[attr-defined]

        outputs = self.model(images)

        loss: torch.Tensor = self.criterion(outputs, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: torch.Tensor, _: int) -> None:  # pylint: disable=arguments-differ
        """Calculate and log metrics/loss to tensorboard.

        Args:
            batch: Current dataloader batch (images/targets)
            _: Unused (index of current batch)
        """
        images, targets = batch
        _, labels = torch.max(targets, 1)

        outputs = self.model(images)

        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        self.val_acc.update(outputs, labels)
        self.log("val_accuracy", self.val_acc, on_step=False, on_epoch=True)

        self.val_pre.update(outputs, labels)
        self.log("val_precision", self.val_pre, on_step=False, on_epoch=True)

        self.val_rec.update(outputs, labels)
        self.log("val_recall", self.val_rec, on_step=False, on_epoch=True)

        self.val_f1.update(outputs, labels)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)

        self.val_confusion.update(outputs, labels)
        self.val_pr_curve.update(outputs, labels)

    def on_validation_epoch_end(self) -> None:
        """Log confusion matrix and PR curve."""
        if self.logger:
            plt.figure(figsize=(10, 7))
            figure, _ = self.val_confusion.plot(add_text=False)
            plt.close(figure)
            self.logger.experiment.add_figure("val_confusion", figure, self.current_epoch)  # type: ignore[attr-defined]

            plt.figure(figsize=(10, 7))
            figure, _ = self.val_pr_curve.plot()
            plt.close(figure)
            self.logger.experiment.add_figure("val_pr_curve", figure, self.current_epoch)  # type: ignore[attr-defined]

        self.val_confusion.reset()
        self.val_pr_curve.reset()

    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> Dict[str, Any]:
        """Return optimiser with schedules."""
        return {"optimizer": self.optimiser, "lr_scheduler": self.optimiser_scheduler}
