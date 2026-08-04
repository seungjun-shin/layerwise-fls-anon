from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, step: int, metrics: dict) -> None:
    """Save a training checkpoint."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
        },
        path,
    )

