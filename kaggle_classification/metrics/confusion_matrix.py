"""Confusion matrix tensorboard logger."""

# pylint: disable=duplicate-code

from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch
from lightning.pytorch.loggers import Logger
from torchmetrics import Metric
from torchmetrics.functional import confusion_matrix


class ConfusionMatrix(Metric):
    """Confusion matrix tensorboard logger."""

    def __init__(self, idx_to_label: Dict[int, Any], stage: str, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)

        self.idx_to_label = idx_to_label
        self.stage = stage

        self.add_state("outputs", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:  # pylint: disable=arguments-differ
        """Update metric.

        Args:
            outputs: Model logits
            labels: Associated labels
        """
        self.outputs.extend(outputs)
        self.labels.extend(labels)

    def compute(self) -> None:
        """Compute metric."""
        raise NotImplementedError("Compute not supported")

    def plot(self, logger: Logger, epoch: int) -> None:  # pylint: disable=arguments-differ
        """Plot metric.

        Args:
            logger: Tensorboard logger object
            epoch: Current epoch
        """
        matrix = (
            confusion_matrix(
                torch.stack(self.outputs),
                torch.stack(self.labels),
                task="multiclass",
                num_classes=len(self.idx_to_label),
            )
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        plt.figure(figsize=(10, 10))
        figure = sn.heatmap(
            pd.DataFrame(matrix, index=self.idx_to_label.values(), columns=self.idx_to_label.values()),
            cmap="mako",
        ).get_figure()
        plt.close(figure)
        logger.experiment.add_figure(f"{self.stage}_confusion", figure, epoch)  # type: ignore[attr-defined]
