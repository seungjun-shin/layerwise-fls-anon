from __future__ import annotations

import torch


def _flatten_features(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().reshape(x.shape[0], -1)


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute linear CKA between two feature matrices."""
    x = _flatten_features(x)
    y = _flatten_features(y)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    numerator = torch.linalg.matrix_norm(xy, ord="fro") ** 2
    denominator = torch.linalg.matrix_norm(xx, ord="fro") * torch.linalg.matrix_norm(yy, ord="fro")
    return float((numerator / (denominator + eps)).clamp(0.0, 1.0).item())


def representation_drift(initial: torch.Tensor, current: torch.Tensor) -> float:
    """Return 1 - linear CKA as representation drift."""
    return 1.0 - linear_cka(initial, current)

