"""Tests for OOD-detection score functions (no data download required).

These exercise the pure scoring/AUROC logic on synthetic logits and features so
they run fast on CPU and never touch the network or disk.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from fls.diagnostics.ood import (
    compute_ood_auroc,
    energy_score,
    fit_mahalanobis,
    mahalanobis_score,
    msp_score,
    ood_auroc_from_scores,
)


def _make_logits(num: int, num_classes: int, peak: float, seed: int) -> torch.Tensor:
    """Logits whose argmax-class gets a ``peak`` boost; larger peak => confident."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(num, num_classes, generator=g)
    logits[torch.arange(num), 0] += peak
    return logits


def test_msp_separates_confident_from_uniform() -> None:
    # ID = very confident (large peak), OOD = near-uniform (no peak).
    id_logits = _make_logits(200, 10, peak=12.0, seed=0)
    ood_logits = _make_logits(200, 10, peak=0.0, seed=1)
    auroc = ood_auroc_from_scores(msp_score(id_logits), msp_score(ood_logits))
    assert auroc > 0.99


def test_energy_separates_high_from_low_magnitude_logits() -> None:
    # ID = large-magnitude logits (low energy), OOD = small-magnitude logits.
    g = torch.Generator().manual_seed(2)
    id_logits = torch.randn(200, 10, generator=g) * 6.0
    ood_logits = torch.randn(200, 10, generator=g) * 0.2
    auroc = ood_auroc_from_scores(energy_score(id_logits), energy_score(ood_logits))
    assert auroc > 0.99


def test_msp_and_energy_chance_on_identical_distributions() -> None:
    id_logits = _make_logits(300, 10, peak=3.0, seed=10)
    ood_logits = _make_logits(300, 10, peak=3.0, seed=11)
    msp_auroc = ood_auroc_from_scores(msp_score(id_logits), msp_score(ood_logits))
    energy_auroc = ood_auroc_from_scores(energy_score(id_logits), energy_score(ood_logits))
    assert 0.4 < msp_auroc < 0.6
    assert 0.4 < energy_auroc < 0.6


def test_mahalanobis_separates_far_features() -> None:
    g = torch.Generator().manual_seed(3)
    # ID features: two tight class-conditional clusters near the origin.
    feats0 = torch.randn(100, 8, generator=g) * 0.3
    feats1 = torch.randn(100, 8, generator=g) * 0.3 + 3.0
    id_feats = torch.cat([feats0, feats1], dim=0)
    id_labels = torch.cat([torch.zeros(100), torch.ones(100)]).long()
    # OOD features: far away from both ID class means.
    ood_feats = torch.randn(200, 8, generator=g) * 0.3 + 30.0

    means, precision = fit_mahalanobis(id_feats, id_labels, num_classes=2)
    id_scores = mahalanobis_score(id_feats, means, precision)
    ood_scores = mahalanobis_score(ood_feats, means, precision)
    auroc = ood_auroc_from_scores(id_scores, ood_scores)
    assert auroc > 0.99


class _DictTensorDataset(Dataset):
    """Minimal dict-style dataset matching the repo's loader contract."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {"x": self.x[idx], "y": int(self.y[idx]), "y_clean": int(self.y[idx])}


class _ToyModel(torch.nn.Module):
    """Tiny model exposing forward_with_features with a 'late' feature map."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 3, padding=1)
        self.head = torch.nn.Linear(6, num_classes)

    def forward_with_features(self, x: torch.Tensor):
        feat = torch.relu(self.conv(x))  # (B, 6, H, W)
        pooled = feat.mean(dim=(2, 3))
        logits = self.head(pooled)
        return logits, {"late": feat}

    def forward(self, x: torch.Tensor):
        return self.forward_with_features(x)[0]


def test_compute_ood_auroc_end_to_end() -> None:
    torch.manual_seed(0)
    model = _ToyModel(num_classes=4)
    # ID inputs small-valued, OOD inputs large-valued -> different feature stats.
    id_x = torch.randn(40, 3, 8, 8) * 0.5
    id_y = torch.randint(0, 4, (40,))
    ood_x = torch.randn(40, 3, 8, 8) * 5.0 + 5.0
    ood_y = torch.randint(0, 4, (40,))

    id_loader = DataLoader(_DictTensorDataset(id_x, id_y), batch_size=16)
    ood_loader = DataLoader(_DictTensorDataset(ood_x, ood_y), batch_size=16)

    result = compute_ood_auroc(model, id_loader, ood_loader, device="cpu", num_classes=4)
    assert set(result) == {"msp_auroc", "energy_auroc", "mahalanobis_auroc"}
    for value in result.values():
        assert 0.0 <= value <= 1.0
