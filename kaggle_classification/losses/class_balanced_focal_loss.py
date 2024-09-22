"""Class balanced focal loss."""

from typing import List

from .focal_loss import FocalLoss


class ClassBalancedFocalLoss(FocalLoss):
    """Class balanced focal loss."""

    def __init__(self, class_counts: List[int], beta: float = 0.0, gamma: float = 0.0) -> None:
        alpha = [(1 - beta) / (1 - beta**class_count) for class_count in class_counts]
        super().__init__(gamma, alpha)
