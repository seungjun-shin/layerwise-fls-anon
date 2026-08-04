from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper_reproduction" / "scripts" / "assemble_accuracy_results.py"
SPEC = importlib.util.spec_from_file_location("assemble_accuracy_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
assembly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assembly)


RAW_RUNS_AVAILABLE = (
    ROOT
    / "paper_reproduction"
    / "test_once"
    / "practical_main_confirmatory"
    / "evaluation_complete.json"
).exists()
LEGACY_DEPTH_METRICS_AVAILABLE = (
    ROOT / "outputs" / "_summaries" / "vgg_depth_paired_20260731" / "seed_level.csv"
).exists()


def test_exact_sign_flip_and_holm_reference_values() -> None:
    assert assembly.exact_sign_flip_pvalue([1.0, 1.0]) == 0.5
    assert assembly.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_complete_role_validation_rejects_duplicate_seed() -> None:
    rows = [
        {"evidence_family": "example", "role": role, "seed": seed}
        for role in ("act", "lr")
        for seed in range(5)
    ]
    rows.append({"evidence_family": "example", "role": "act", "seed": 0})
    with pytest.raises(RuntimeError, match="duplicate"):
        assembly.validate_complete_roles(rows, {"example"})


def test_completed_family_ledger_audit_matches_frozen_design() -> None:
    if not RAW_RUNS_AVAILABLE:
        pytest.skip("raw-run ledgers are required for this integration audit")
    rows = assembly.completed_family_rows("practical_main_confirmatory")
    assert len(rows) == 30
    assert {
        (row["family"], row["pair_role"], row["condition"], int(row["seed"])) for row in rows
    } == assembly.expected_completed_keys("practical_main_confirmatory")


def test_frozen_artifact_hash_check_rejects_mutation(tmp_path: Path) -> None:
    filenames = (
        "checkpoint_best.pt",
        "resolved_config.yaml",
        "metrics.csv",
        "protocol_sanity.json",
    )
    for filename in filenames:
        (tmp_path / filename).write_bytes(f"frozen:{filename}".encode())
    record = {
        "run_id": "example",
        "checkpoint_sha256": assembly.sha256(tmp_path / "checkpoint_best.pt"),
        "resolved_config_sha256": assembly.sha256(tmp_path / "resolved_config.yaml"),
        "metrics_sha256": assembly.sha256(tmp_path / "metrics.csv"),
        "protocol_sanity_sha256": assembly.sha256(tmp_path / "protocol_sanity.json"),
    }
    assembly.verify_record_artifact_hashes(record, tmp_path)
    (tmp_path / "metrics.csv").write_text("mutated", encoding="utf-8")
    with pytest.raises(RuntimeError, match="metrics.csv"):
        assembly.verify_record_artifact_hashes(record, tmp_path)


def test_merged_depth_and_scope_tables_are_exact_cartesian_designs() -> None:
    if not (RAW_RUNS_AVAILABLE and LEGACY_DEPTH_METRICS_AVAILABLE):
        pytest.skip("raw and legacy metrics are required for this integration audit")
    vgg = assembly.vgg_depth_rows()
    scope = assembly.scope_rows()
    assert len(vgg) == 8 * 2 * 5
    assert len(scope) == 7 * 3 * 5
    assert len({(row["evidence_family"], row["role"], row["seed"]) for row in vgg}) == len(vgg)
    assert len({(row["evidence_family"], row["role"], row["seed"]) for row in scope}) == len(scope)
