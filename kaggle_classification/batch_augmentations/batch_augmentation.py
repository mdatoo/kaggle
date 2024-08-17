"""Batch level PyTorch augmentation."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch


class BatchAugmentation(ABC):
    """Batch level PyTorch augmentation.

    PyTorch augmentation that works on batches.
    """

    def __init__(self, probability: float = 1.0) -> None:
        """Initialise object.

        Args:
            probability: Probability of applying augmentation
        """
        self.probability = probability

    def apply(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Possibly apply augmentation to batch.

        Args:
            batch: Batch to possibly apply augmentation to
        """
        if np.random.rand() < self.probability:
            batch = self.always_apply(batch)
        return batch

    @abstractmethod
    def always_apply(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation to batch.

        Args:
            batch: Batch to apply augmentation to
        """
