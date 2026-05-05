"""Binary focal loss on logits for imbalanced fraud classification."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Focal loss (Lin et al.) for binary classification.
    targets are float in {0, 1}; logits are pre-sigmoid.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    with torch.no_grad():
        p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    focal_weight = (1.0 - p_t) ** gamma
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * focal_weight * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
