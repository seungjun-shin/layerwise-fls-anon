from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy as a fraction."""
    return float((logits.argmax(dim=1) == targets).float().mean().item())

