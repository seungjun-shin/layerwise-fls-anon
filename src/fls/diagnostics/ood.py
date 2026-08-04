"""Out-of-distribution (OOD) detection scores from trained classifiers.

This module computes post-hoc OOD-detection scores from a trained model and
two data loaders: an in-distribution (ID) loader (e.g. CIFAR-100 test) and an
OOD loader (e.g. SVHN test).  Three scores are implemented:

* **MSP** (maximum softmax probability) -- the classic baseline of Hendrycks &
  Gimpel (2017).  Higher max-softmax => more confident => more "in-distribution".
* **Energy** -- the free-energy score ``-logsumexp(logits)`` of Liu et al.
  (2020).  Lower energy => more in-distribution.  We negate it so that, like MSP,
  a *higher* returned score means *more in-distribution*.
* **Mahalanobis** -- the feature-space score of Lee et al. (2018).  We fit a
  class-conditional Gaussian with a shared covariance on the ID penultimate
  features and score a sample by the maximum (over classes) negative
  Mahalanobis distance.  Higher => closer to an ID class mean => more
  in-distribution.

For every score, "in-distribution" samples are assigned the positive label
(``1``) and OOD samples the negative label (``0``).  Because every score is
oriented so that larger means more-in-distribution, ``roc_auc_score(labels,
scores)`` directly yields the AUROC of the ID-vs-OOD detector, where ~1.0 is a
perfect detector and ~0.5 is chance.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


# Which penultimate feature to use for the Mahalanobis score.  The CIFAR models
# expose a ``late`` region feature map (B, C, H, W); we global-average-pool it to
# obtain a (B, C) pre-head pooled feature, matching the head's input.
_PENULTIMATE_FEATURE = "late"


def _pool_feature(feature: torch.Tensor) -> torch.Tensor:
    """Reduce a feature tensor to a 2D (batch, dim) representation.

    Convolutional feature maps (B, C, H, W) are global-average-pooled over the
    spatial dimensions to mirror the model's own ``AdaptiveAvgPool2d`` before the
    head.  Already-flat features (B, D) are returned unchanged.
    """
    feature = feature.detach().float()
    if feature.dim() > 2:
        feature = feature.mean(dim=tuple(range(2, feature.dim())))
    return feature.reshape(feature.shape[0], -1)


@torch.no_grad()
def collect_logits_and_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    feature_key: str = _PENULTIMATE_FEATURE,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run ``model`` over ``loader`` and return logits, pooled features, labels.

    Returns CPU tensors ``(logits [N, num_classes], features [N, D], labels
    [N])``.  Labels come from the dataset's clean label (``y_clean`` if present,
    else ``y``); they are only used to fit the ID Mahalanobis statistics.
    """
    model.eval()
    logits_list: list[torch.Tensor] = []
    features_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    seen = 0
    for batch in loader:
        x = batch["x"].to(device)
        logits, feats = model.forward_with_features(x)
        if feature_key not in feats:
            raise KeyError(
                f"Model features do not contain '{feature_key}'; available: {sorted(feats)}"
            )
        pooled = _pool_feature(feats[feature_key])
        logits_list.append(logits.detach().float().cpu())
        features_list.append(pooled.cpu())
        labels = batch.get("y_clean", batch["y"])
        labels_list.append(torch.as_tensor(labels, dtype=torch.long).cpu())
        seen += x.shape[0]
        if max_samples is not None and seen >= max_samples:
            break
    logits = torch.cat(logits_list, dim=0)
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    if max_samples is not None:
        logits, features, labels = logits[:max_samples], features[:max_samples], labels[:max_samples]
    return logits, features, labels


def msp_score(logits: torch.Tensor) -> torch.Tensor:
    """Maximum softmax probability score (higher => more in-distribution)."""
    probs = F.softmax(logits.float(), dim=1)
    return probs.max(dim=1).values


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Negated free-energy score (higher => more in-distribution).

    The energy is ``-T * logsumexp(logits / T)`` (lower energy => more
    in-distribution).  We return ``-energy = T * logsumexp(logits / T)`` so the
    orientation matches the other scores.
    """
    logits = logits.float()
    return temperature * torch.logsumexp(logits / temperature, dim=1)


def fit_mahalanobis(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit class-conditional means and a shared precision matrix on ID features.

    Returns ``(means [K, D], precision [D, D])`` where ``means[k]`` is the mean
    of class ``k`` and ``precision`` is the inverse of the shared (tied)
    covariance estimated from within-class deviations.  Classes that are absent
    in ``labels`` receive a mean equal to the global mean so they never become
    the closest centroid spuriously.
    """
    features = features.detach().float()
    labels = labels.detach().long()
    dim = features.shape[1]
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1
    global_mean = features.mean(dim=0)
    means = global_mean.unsqueeze(0).repeat(num_classes, 1).clone()
    centered_parts: list[torch.Tensor] = []
    for k in range(num_classes):
        mask = labels == k
        if int(mask.sum().item()) == 0:
            continue
        class_feats = features[mask]
        class_mean = class_feats.mean(dim=0)
        means[k] = class_mean
        centered_parts.append(class_feats - class_mean)
    centered = torch.cat(centered_parts, dim=0) if centered_parts else features - global_mean
    cov = (centered.T @ centered) / max(1, centered.shape[0])
    cov = cov + eps * torch.eye(dim, dtype=cov.dtype)
    precision = torch.linalg.inv(cov)
    return means, precision


def mahalanobis_score(
    features: torch.Tensor,
    means: torch.Tensor,
    precision: torch.Tensor,
) -> torch.Tensor:
    """Max (over classes) negative Mahalanobis distance (higher => more ID).

    For each sample we compute the squared Mahalanobis distance to every class
    mean and take the smallest distance (closest class); the returned score is
    the negation of that minimum, so a sample close to an ID class mean scores
    high.
    """
    features = features.detach().float()
    means = means.detach().float()
    precision = precision.detach().float()
    # diff[n, k, d] = features[n, d] - means[k, d]
    diff = features.unsqueeze(1) - means.unsqueeze(0)  # (N, K, D)
    # squared Mahalanobis distance per (sample, class)
    left = diff @ precision  # (N, K, D)
    dist_sq = (left * diff).sum(dim=2)  # (N, K)
    min_dist_sq = dist_sq.min(dim=1).values  # (N,)
    return -min_dist_sq


def _auroc(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> float:
    """AUROC for separating ID (positive) from OOD (negative) samples.

    ``id_scores`` and ``ood_scores`` must be oriented so that larger values
    indicate more-in-distribution.
    """
    scores = torch.cat([id_scores, ood_scores]).cpu().numpy()
    labels = np.concatenate(
        [np.ones(id_scores.numel()), np.zeros(ood_scores.numel())]
    )
    return float(roc_auc_score(labels, scores))


def compute_ood_auroc(
    model: torch.nn.Module,
    id_loader: DataLoader,
    ood_loader: DataLoader,
    device: torch.device | str = "cpu",
    num_classes: int | None = None,
    include_mahalanobis: bool = True,
    feature_key: str = _PENULTIMATE_FEATURE,
    max_samples: int | None = None,
) -> dict[str, float]:
    """Compute MSP/energy/Mahalanobis OOD-detection AUROCs for a trained model.

    Parameters
    ----------
    model:
        A trained model exposing ``forward_with_features``.
    id_loader, ood_loader:
        DataLoaders yielding dict batches with key ``"x"`` (and ``"y"`` /
        ``"y_clean"`` for the Mahalanobis fit).  ``id_loader`` is the
        in-distribution set (CIFAR-100 test); ``ood_loader`` is the OOD set
        (SVHN test).
    device:
        Device to run inference on.
    num_classes:
        Number of ID classes (inferred from ID labels if omitted).
    include_mahalanobis:
        When ``False``, the Mahalanobis branch is skipped and its AUROC is
        omitted from the result.

    Returns
    -------
    dict
        ``{"msp_auroc": ..., "energy_auroc": ..., "mahalanobis_auroc": ...}``
        (the last key only when ``include_mahalanobis`` is true).
    """
    device = torch.device(device)
    id_logits, id_feats, id_labels = collect_logits_and_features(
        model, id_loader, device, feature_key=feature_key, max_samples=max_samples
    )
    ood_logits, ood_feats, _ = collect_logits_and_features(
        model, ood_loader, device, feature_key=feature_key, max_samples=max_samples
    )

    results: dict[str, float] = {
        "msp_auroc": _auroc(msp_score(id_logits), msp_score(ood_logits)),
        "energy_auroc": _auroc(energy_score(id_logits), energy_score(ood_logits)),
    }

    if include_mahalanobis:
        means, precision = fit_mahalanobis(id_feats, id_labels, num_classes=num_classes)
        results["mahalanobis_auroc"] = _auroc(
            mahalanobis_score(id_feats, means, precision),
            mahalanobis_score(ood_feats, means, precision),
        )

    return results


def ood_auroc_from_scores(
    id_scores: Iterable[float] | torch.Tensor,
    ood_scores: Iterable[float] | torch.Tensor,
) -> float:
    """Convenience AUROC for pre-computed (ID-oriented) scores.

    Useful for testing and for callers that have already extracted scores.
    """
    return _auroc(torch.as_tensor(list(id_scores) if not torch.is_tensor(id_scores) else id_scores, dtype=torch.float),
                  torch.as_tensor(list(ood_scores) if not torch.is_tensor(ood_scores) else ood_scores, dtype=torch.float))
