import torch

from fls.diagnostics.alignment import mean_class_alignment, within_between_separation
from fls.diagnostics.cka import linear_cka
from fls.diagnostics.effective_rank import effective_rank


def test_cka_identical_features() -> None:
    x = torch.randn(8, 4)
    assert linear_cka(x, x) > 0.99


def test_effective_rank_positive() -> None:
    rank = effective_rank(torch.randn(8, 4))
    assert rank > 0


def test_alignment_metrics_run() -> None:
    features = torch.randn(8, 4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    assert isinstance(mean_class_alignment(features, labels), float)
    assert isinstance(within_between_separation(features, labels), float)

