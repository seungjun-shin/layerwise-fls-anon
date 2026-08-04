from __future__ import annotations

from torch import nn


def build_loss() -> nn.Module:
    """Build the default classification loss."""
    return nn.CrossEntropyLoss()

