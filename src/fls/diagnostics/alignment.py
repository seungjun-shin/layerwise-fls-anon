from __future__ import annotations

import torch
import torch.nn.functional as F


def class_means(features: torch.Tensor, labels: torch.Tensor) -> dict[int, torch.Tensor]:
    """Compute class mean feature vectors."""
    x = features.detach().float().reshape(features.shape[0], -1)
    return {int(label): x[labels == label].mean(dim=0) for label in labels.unique() if (labels == label).any()}


def mean_class_alignment(features: torch.Tensor, labels: torch.Tensor) -> float:
    """Average cosine similarity between samples and their class mean."""
    x = features.detach().float().reshape(features.shape[0], -1)
    means = class_means(x, labels)
    scores = []
    for idx, label in enumerate(labels):
        scores.append(F.cosine_similarity(x[idx], means[int(label)], dim=0))
    return float(torch.stack(scores).mean().item()) if scores else 0.0


def within_between_separation(features: torch.Tensor, labels: torch.Tensor) -> float:
    """Return average between-class distance minus within-class distance."""
    x = features.detach().float().reshape(features.shape[0], -1)
    means = class_means(x, labels)
    if len(means) < 2:
        return 0.0
    within = torch.stack([(x[labels == label] - mean).norm(dim=1).mean() for label, mean in means.items()]).mean()
    mean_values = list(means.values())
    pairs = []
    for i in range(len(mean_values)):
        for j in range(i + 1, len(mean_values)):
            pairs.append((mean_values[i] - mean_values[j]).norm())
    between = torch.stack(pairs).mean()
    return float((between - within).item())

