"""PyTorch losses."""

from .class_balanced_focal_loss import ClassBalancedFocalLoss
from .focal_loss import FocalLoss

__all__ = ["ClassBalancedFocalLoss", "FocalLoss"]
