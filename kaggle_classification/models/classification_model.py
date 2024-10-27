"""Image classification PyTorch model."""

from typing import Any, Dict, Generic, TypeVar

import lightning.pytorch as pl
import numpy as np
import numpy.typing as npt
import torch
from torch import nn, optim
from torchmetrics import Accuracy

from ..datasets import ClassificationDataset
from ..metrics import ConfusionMatrix, Embeddings, IncorrectPredictionsGrid, PRCurve

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

        self.backbone = model
        self.classifier = model.get_classifier()
        self.backbone.reset_classifier(0)

        self.train_dataset = train_dataset
        self.criterion = criterion
        self.optimiser = optimiser
        self.optimiser_scheduler = optimiser_scheduler

        self.val_acc = Accuracy(task="multiclass", num_classes=len(self.idx_to_label))
        self.val_confusion = ConfusionMatrix(self.idx_to_label, "val")
        self.val_incorrect_predictions = IncorrectPredictionsGrid(self.mean, self.std, self.idx_to_label, 8, 8, "val")
        self.val_embeddings = Embeddings(self.mean, self.std, self.idx_to_label, 64, "val")
        self.val_pr_curve = PRCurve(self.idx_to_label, "val")

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

        outputs = self.classifier(self.backbone((images)))

        loss: torch.Tensor = self.criterion(outputs, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=False)

        if self.current_epoch == 0 and batch_idx == 0 and self.logger:
            self.logger.experiment.add_graph(self.backbone, images[:1])  # type: ignore[attr-defined]

        return loss

    def validation_step(self, batch: torch.Tensor, _: int) -> None:  # pylint: disable=arguments-differ
        """Calculate and log metrics/loss to tensorboard.

        Args:
            batch: Current dataloader batch (images/targets)
            batch_idx: Index of current batch
        """
        images, targets = batch
        _, labels = torch.max(targets, 1)

        features = self.backbone(images)
        outputs = self.classifier(features)

        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        self.val_acc.update(outputs, labels)
        self.log("val_accuracy", self.val_acc, on_step=False, on_epoch=True)

        self.val_confusion.update(outputs, labels)
        self.val_incorrect_predictions.update(outputs, labels, images)
        self.val_embeddings.update(features, labels, images)
        self.val_pr_curve.update(outputs, labels)

    def on_validation_epoch_end(self) -> None:
        """Log confusion matrix and PR curve."""
        if self.logger:
            self.val_confusion.plot(self.logger, self.current_epoch)  # type: ignore[arg-type]
            self.val_incorrect_predictions.plot(self.logger, self.current_epoch)  # type: ignore[arg-type]
            self.val_embeddings.plot(self.logger, self.current_epoch)  # type: ignore[arg-type]
            self.val_pr_curve.plot(self.logger, self.current_epoch)  # type: ignore[arg-type]

        self.val_confusion.reset()
        self.val_incorrect_predictions.reset()
        self.val_embeddings.reset()
        self.val_pr_curve.reset()

    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> Dict[str, Any]:
        """Return optimiser with schedules."""
        return {"optimizer": self.optimiser, "lr_scheduler": self.optimiser_scheduler}
