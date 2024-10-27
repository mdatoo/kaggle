"""Focal loss."""

from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Focal loss."""

    def __init__(self, gamma: float = 0.0, alpha: Optional[List[float]] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = nn.Parameter(torch.Tensor(alpha)) if alpha else None

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss.

        Args:
            output: Model logits
            target: Associated labels
        """
        logpt = F.log_softmax(output, dim=1)
        focal = (1 - logpt.exp() * target) ** self.gamma
        loss = -logpt * target * focal * self.alpha

        return loss.sum()  # type: ignore[no-any-return]
