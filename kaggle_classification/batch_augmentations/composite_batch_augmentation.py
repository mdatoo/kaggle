"""Composite batch level PyTorch augmentation."""

from typing import List, Tuple

import torch

from .batch_augmentation import BatchAugmentation


class CompositeBatchAugmentation(BatchAugmentation):
    """Composite batch level PyTorch augmentation.

    Collection of PyTorch augmentations that work on batches.
    """

    def __init__(self, augmentations: List[BatchAugmentation], probability: float = 1.0) -> None:
        """Initialise object.

        Args:
            augmentations: List of augmentations to apply in order
            probability: Probability of applying augmentation
        """
        super().__init__(probability)
        self.augmentations = augmentations

    def always_apply(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations to batch.

        Args:
            batch: Batch to apply augmentations to
        """
        for augmentation in self.augmentations:
            batch = augmentation.apply(batch)

        return batch
