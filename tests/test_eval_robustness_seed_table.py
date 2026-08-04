from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.eval_robustness import find_seed_runs_from_table
from scripts.summarize_ivc_auxiliary import (
    CORRUPTION_FAMILIES,
    exact_sign_flip_p,
    holm_adjust,
    summarize_robustness,
)


def write_seed_table(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "seed", "run_dir"])
        writer.writeheader()
        writer.writerows(rows)


def test_find_seed_runs_from_frozen_table(tmp_path: Path) -> None:
    run_5 = tmp_path / "run5"
    run_6 = tmp_path / "run6"
    run_5.mkdir()
    run_6.mkdir()
    (run_5 / "checkpoint_best.pt").touch()
    (run_6 / "checkpoint_best.pt").touch()
    table = tmp_path / "seeds.csv"
    write_seed_table(
        table,
        [
            {"condition": "ACT", "seed": "5", "run_dir": str(run_5)},
            {"condition": "ACT", "seed": "6", "run_dir": str(run_6)},
            {"condition": "LR", "seed": "5", "run_dir": str(run_5)},
        ],
    )

    assert find_seed_runs_from_table(table, "ACT", [5, 6]) == {
        5: run_5,
        6: run_6,
    }


def test_find_seed_runs_from_table_rejects_missing_seed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "checkpoint_best.pt").touch()
    table = tmp_path / "seeds.csv"
    write_seed_table(
        table, [{"condition": "ACT", "seed": "5", "run_dir": str(run)}]
    )

    with pytest.raises(ValueError, match="Missing ACT rows for seeds: \\[6\\]"):
        find_seed_runs_from_table(table, "ACT", [5, 6])


def robustness_payload(seed: int, offset: float) -> dict:
    per_corruption = {
        name: 0.4 + offset
        for names in CORRUPTION_FAMILIES.values()
        for name in names
    }
    return {
        "seed": seed,
        "corruption": {
            "mCA": 0.4 + offset,
            "per_corruption_mean": per_corruption,
            "per_severity_mean": {str(index): 0.4 + offset for index in range(1, 6)},
        },
        "ood": {
            "msp_auroc": 0.6 + offset,
            "energy_auroc": 0.7 + offset,
            "mahalanobis_auroc": 0.8 + offset,
        },
    }


def test_summarize_robustness_merges_seed_shards(tmp_path: Path) -> None:
    root = tmp_path / "robustness"
    output = tmp_path / "summary"
    for experiment, seed, offset in (
        ("act_a", 5, 0.02),
        ("act_b", 6, 0.03),
        ("lr_a", 5, 0.00),
        ("lr_b", 6, 0.01),
    ):
        directory = root / experiment
        directory.mkdir(parents=True)
        (directory / f"seed_{seed}.json").write_text(
            json.dumps(robustness_payload(seed, offset)),
            encoding="utf-8",
        )

    summary = summarize_robustness(
        root, output, ["act_a", "act_b"], ["lr_a", "lr_b"], [5, 6]
    )

    assert summary["num_paired_seeds"] == 2
    assert summary["mCA"]["mean"] == pytest.approx(0.02)
    assert summary["family_noise_accuracy"]["mean"] == pytest.approx(0.02)
    assert (output / "robustness_seed_level.csv").exists()


def test_exact_sign_flip_and_holm_adjustment() -> None:
    assert exact_sign_flip_p([1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.125)
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.2})
