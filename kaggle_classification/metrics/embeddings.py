"""Embeddings tensorboard logger."""

from typing import Any, Dict

import numpy as np
import numpy.typing as npt
import torch
from lightning.pytorch.loggers import Logger
from torchmetrics import Metric


class Embeddings(Metric):
    """Embeddings tensorboard logger."""

    def __init__(  # type: ignore[no-untyped-def]
        self,
        mean: npt.NDArray[np.uint8],
        std: npt.NDArray[np.uint8],
        idx_to_label: Dict[int, Any],
        sample_size: int,
        stage: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.mean = mean
        self.std = std
        self.idx_to_label = idx_to_label
        self.sample_size = sample_size
        self.stage = stage

        self.add_state("features", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        self.add_state("images", default=[], dist_reduce_fx="cat")

    def update(  # pylint: disable=arguments-differ
        self, features: torch.Tensor, labels: torch.Tensor, images: torch.Tensor
    ) -> None:
        """Update metric.

        Args:
            features: Model features
            labels: Associated labels
            images: Associated images
        """
        for feature, label, image in zip(features, labels, images):
            if len(self.features) < self.sample_size:
                self.features.append(feature)
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
        logger.experiment.add_embedding(  # type: ignore[attr-defined]
            torch.flatten(
                torch.stack(self.features).cpu().float(),
                start_dim=1,
            ),
            [self.idx_to_label[label.item()] for label in self.labels],
            (
                torch.stack(self.images).cpu() * torch.unsqueeze(torch.unsqueeze(torch.tensor(self.std), -1), -1)
                + torch.unsqueeze(torch.unsqueeze(torch.tensor(self.mean), -1), -1)
            )
            / 255.0,
            epoch,
            f"{self.stage}_embedding",
        )
