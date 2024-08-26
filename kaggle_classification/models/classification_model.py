"""Image classification PyTorch model."""

from typing import Any, Dict, Generic, TypeVar

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sn
import torch
from torch import nn, optim
from torchmetrics import Accuracy, CatMetric, ConfusionMatrix

from ..datasets import ClassificationDataset

T = TypeVar("T")


class ClassificationModel(pl.LightningModule, Generic[T]):  # pylint: disable=too-many-ancestors
    """Image classification PyTorch model.

    PyTorch model to perform image classification.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        model: nn.Module,
        train_dataset: ClassificationDataset[T],
        criterion: nn.Module,
        optimiser: optim.Optimizer,
        optimiser_scheduler: optim.lr_scheduler.LRScheduler,
    ) -> None:
        """Initialise object.

        Args:
            model: Backbone to use
            train_dataset: Train dataset to use
            criterion: Loss function to use
            optimiser: Optimiser to use
            optimiser_scheduler: LR scheduler to use
        """
        super().__init__()

        self.model = model
        self.train_dataset = train_dataset
        self.criterion = criterion
        self.optimiser = optimiser
        self.optimiser_scheduler = optimiser_scheduler

        self.val_acc = Accuracy(task="multiclass", num_classes=len(self.idx_to_label))
        self.val_confusion = ConfusionMatrix(task="multiclass", num_classes=len(self.idx_to_label))
        self.val_outputs = CatMetric()
        self.val_labels = CatMetric()

    @property
    def idx_to_label(self) -> Dict[int, T]:
        """Mapping from index to label."""
        return self.train_dataset.idx_to_label

    @property
    def mean(self) -> npt.NDArray[np.uint8]:
        """Mean of all images in dataset."""
        return self.train_dataset.mean

    @property
    def std(self) -> npt.NDArray[np.uint8]:
        """Standard deviation of all images in dataset."""
        return self.train_dataset.std

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Calculate and log metrics/loss to tensorboard.

        Args:
            batch: Current dataloader batch (images/targets)
            batch_idx: Index of current batch
        """
        images, targets = batch

        outputs = self.model(images)

        if batch_idx == 0 and self.logger:
            with torch.no_grad():
                figure = plt.figure(figsize=(20, 20))
                probabilities = torch.softmax(outputs, 1)
                _, predictions = torch.max(outputs, 1)

                for idx in range(64):
                    ax = figure.add_subplot(8, 8, idx + 1)
                    ax.axis("off")
                    ax.imshow((np.transpose(images[idx].cpu(), (1, 2, 0)) * self.std + self.mean).int())  # type: ignore[attr-defined]
                    ax.set_title(
                        f"Predicted: {predictions[idx]} ({probabilities[idx, predictions[idx]] * 100:.2f}%)",
                        fontsize=8,
                    )
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.tight_layout()
                plt.close(figure)
                self.logger.experiment.add_figure("train_first_batch", figure, self.current_epoch)  # type: ignore[attr-defined]

        loss: torch.Tensor = self.criterion(outputs, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:  # pylint: disable=arguments-differ
        """Calculate and log metrics/loss to tensorboard.

        Args:
            batch: Current dataloader batch (images/targets)
            batch_idx: Index of current batch
        """
        images, targets = batch
        _, labels = torch.max(targets, 1)

        outputs = self.model(images)

        if batch_idx == 0 and self.logger:
            figure = plt.figure(figsize=(20, 20))
            probabilities = torch.softmax(outputs, 1)
            _, predictions = torch.max(outputs, 1)

            for idx in range(64):
                ax = figure.add_subplot(8, 8, idx + 1)
                ax.axis("off")
                ax.imshow((np.transpose(images[idx].cpu(), (1, 2, 0)) * self.std + self.mean).int())  # type: ignore[attr-defined]
                ax.set_title(
                    f"Predicted: {predictions[idx]} ({probabilities[idx, predictions[idx]] * 100:.2f}%), Actual: {labels[idx]}",
                    fontsize=8,
                )
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.tight_layout()
            plt.close(figure)
            self.logger.experiment.add_figure("val_first_batch", figure, self.current_epoch)  # type: ignore[attr-defined]

        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        self.val_acc.update(outputs, labels)
        self.log("val_accuracy", self.val_acc, on_step=False, on_epoch=True)

        self.val_confusion.update(outputs, labels)
        self.val_outputs.update(outputs)
        self.val_labels.update(labels)

    def on_validation_epoch_end(self) -> None:
        """Log confusion matrix and PR curve."""
        if self.logger:
            confusion_matrix = self.val_confusion.compute().detach().cpu().numpy().astype(int)  # type: ignore[func-returns-value]

            plt.figure(figsize=(20, 20))
            figure = sn.heatmap(
                pd.DataFrame(confusion_matrix, index=self.idx_to_label.values(), columns=self.idx_to_label.values()),
                cmap="mako",
            ).get_figure()
            plt.close(figure)
            self.logger.experiment.add_figure("val_confusion", figure, self.current_epoch)  # type: ignore[attr-defined]

            val_probs = torch.softmax(self.val_outputs.compute(), 1)
            val_labels = self.val_labels.compute()

            for class_idx, class_name in self.idx_to_label.items():
                self.logger.experiment.add_pr_curve(  # type: ignore[attr-defined]
                    f"val_{class_name}",
                    val_labels == class_idx,
                    val_probs[:, class_idx],
                    self.current_epoch,
                )

            self.logger.experiment.add_pr_curve(  # type: ignore[attr-defined]
                "val_micro",
                torch.concat([val_labels == class_idx for class_idx in self.idx_to_label.keys()]),  # type: ignore[misc]
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
        return {"optimizer": self.optimiser, "lr_scheduler": self.optimiser_scheduler}
