"""Incorrect predictions grid tensorboard logger."""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from lightning.pytorch.loggers import Logger
from torchmetrics import Metric


class IncorrectPredictionsGrid(Metric):
    """Incorrect predictions grid tensorboard logger."""

    def __init__(  # type: ignore[no-untyped-def]
        self,
        mean: npt.NDArray[np.uint8],
        std: npt.NDArray[np.uint8],
        idx_to_label: Dict[int, Any],
        width: int,
        height: int,
        stage: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.mean = mean
        self.std = std
        self.idx_to_label = idx_to_label
        self.width = width
        self.height = height
        self.stage = stage

        self.add_state("outputs", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        self.add_state("images", default=[], dist_reduce_fx="cat")

    @property
    def sample_size(self) -> int:
        """Get number of logged samples."""
        return self.width * self.height

    def update(  # pylint: disable=arguments-differ
        self, outputs: torch.Tensor, labels: torch.Tensor, images: torch.Tensor
    ) -> None:
        """Update metric.

        Args:
            outputs: Model logits
            labels: Associated labels
            images: Associated images
        """
        _, predictions = torch.max(outputs, 1)
        for output, label, image, prediction in zip(outputs, labels, images, predictions):
            if len(self.outputs) < self.sample_size and prediction != label:
                self.outputs.append(output)
                self.labels.append(label)
                self.images.append(image)

    def compute(self) -> None:
        """Compute metric."""
        raise NotImplementedError("Compute not supported")

    def plot(self, logger: Logger, epoch: int) -> None:  # pylint: disable=arguments-differ
        """Plot metric.

        Args:
            logger: Tensorboard logger object
            epoch: Current epoch
        """
        figure = plt.figure(figsize=(10, 10))
        probabilities = torch.softmax(torch.stack(self.outputs), 1)
        _, predictions = torch.max(torch.stack(self.outputs), 1)

        for idx, image in enumerate(self.images):
            ax = figure.add_subplot(self.height, self.width, idx + 1)
            ax.axis("off")
            ax.imshow((np.transpose(image.cpu(), (1, 2, 0)) * self.std + self.mean).int())  # type: ignore[attr-defined]
            ax.set_title(
                f"Predicted: {self.idx_to_label[predictions[idx].item()]} ({probabilities[idx, predictions[idx]] * 100:.2f}%)\nActual: {self.idx_to_label[self.labels[idx].item()]}",
                fontsize=6,
            )
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        plt.close(figure)

        logger.experiment.add_figure(f"{self.stage}_incorrect_predictions", figure, epoch)  # type: ignore[attr-defined]
