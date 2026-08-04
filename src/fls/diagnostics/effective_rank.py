from __future__ import annotations

import torch


def effective_rank(features: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute entropy-based effective rank of centered features."""
    x = features.detach().float().reshape(features.shape[0], -1)
    x = x - x.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(x)
    probs = singular_values / (singular_values.sum() + eps)
    entropy = -(probs * torch.log(probs + eps)).sum()
    return float(torch.exp(entropy).item())

