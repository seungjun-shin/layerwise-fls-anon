from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper_reproduction" / "scripts" / "mechanism_analysis.py"


def load_module():
    """Load the standalone mechanism-analysis script for unit testing."""
    spec = importlib.util.spec_from_file_location("paper_mechanism_analysis", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_class_geometry_matches_a_collapsed_two_class_example() -> None:
    module = load_module()
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    classifier_weight = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

    metrics = module.class_geometry(features, labels, classifier_weight)

    assert metrics["within_class_scatter_trace"] == pytest.approx(0.0)
    assert metrics["between_class_scatter_trace"] == pytest.approx(1.0)
    assert metrics["nc1_within_between_trace_ratio"] == pytest.approx(0.0)
    assert metrics["nc3_classifier_prototype_alignment"] == pytest.approx(1.0)
    assert metrics["observed_class_count"] == pytest.approx(2.0)


def test_effective_rank_is_two_for_equal_singular_values() -> None:
    module = load_module()
    assert module.effective_rank_from_singular_values(torch.tensor([1.0, 1.0])) == pytest.approx(
        2.0
    )


def test_operational_cosine_weight_uses_normalized_rows_and_fixed_scale() -> None:
    module = load_module()

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([[3.0, 4.0], [0.0, 2.0]]))
            self.logit_scale = 2.0

    weight, representation = module.operational_classifier_weight(Head())

    assert representation == "row_normalized_fixed_scale"
    assert torch.allclose(weight.norm(dim=1), torch.tensor([2.0, 2.0]))


def test_diagnostic_subset_rejects_a_changed_validation_split(monkeypatch) -> None:
    module = load_module()
    monkeypatch.setattr(
        module,
        "build_train_val_datasets",
        lambda _config, _fraction, _seed: ([], list(range(10)), "regenerated"),
    )
    config = {
        "data": {
            "val_fraction": 0.1,
            "val_seed": 0,
            "val_indices_sha256": "training-time",
        }
    }

    with pytest.raises(RuntimeError, match="training-time frozen index hash"):
        module.diagnostic_subset(config, 5)


def test_seed_table_must_contain_the_complete_frozen_design(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module()
    workspace = tmp_path / "workspace"
    seed_table = workspace / "seed_tables" / "head.csv"
    completion = workspace / "test_once" / "head_norm" / "evaluation_complete.json"
    seed_table.parent.mkdir(parents=True)
    completion.parent.mkdir(parents=True)
    seed_table.write_text(
        "condition,seed,run_id\nHN-COSINE-ACT,20,one\n",
        encoding="utf-8",
    )
    completion.write_text(
        json.dumps(
            {
                "family": "head_norm",
                "seed_table_sha256": module.sha256(seed_table),
                "evaluated_jobs": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "WORKSPACE", workspace)

    with pytest.raises(RuntimeError, match="complete frozen condition/seed design"):
        module.validate_seed_table(seed_table, "head_norm")
