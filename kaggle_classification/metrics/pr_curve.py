"""PR Curve tensorboard logger."""

# pylint: disable=duplicate-code

from typing import Any, Dict

import torch
from lightning.pytorch.loggers import Logger
from torchmetrics import Metric


class PRCurve(Metric):
    """PR Curve tensorboard logger."""

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
        probs = torch.softmax(torch.stack(self.outputs), 1)

        for class_idx, class_name in self.idx_to_label.items():
            logger.experiment.add_pr_curve(  # type: ignore[attr-defined]
                f"{self.stage}_{class_name}",
                torch.stack(self.labels) == class_idx,
                probs[:, class_idx].float(),
                epoch,
            )

        logger.experiment.add_pr_curve(  # type: ignore[attr-defined]
            f"{self.stage}_micro",
            torch.concat([torch.stack(self.labels) == class_idx for class_idx in self.idx_to_label.keys()]),  # type: ignore[misc]
            probs.transpose(0, 1).flatten().float(),
            epoch,
        )
