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
    PrecisionRecallCurve,
    Recall,
)
from torchvision.utils import make_grid

from ..lr_schedulers import ReduceLROnPlateauWrapper


def get_last_lr() -> float:
    return 0.0001


class ClassificationModel(pl.LightningModule):
    def __init__(self, model_name: str, num_classes: int) -> None:
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=True, in_chans=3, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.01)

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
        self.val_pr_curve = PrecisionRecallCurve(task="multiclass", num_classes=num_classes)
        self.val_outputs = CatMetric()
        self.val_labels = CatMetric()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
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

        self.train_confusion.reset()
        self.train_outputs.reset()
        self.train_labels.reset()

    def validation_step(self, batch: torch.Tensor, _: int) -> None:
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

        self.val_confusion.reset()
        self.val_outputs.reset()
        self.val_labels.reset()

    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> Dict[str, Any]:
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
