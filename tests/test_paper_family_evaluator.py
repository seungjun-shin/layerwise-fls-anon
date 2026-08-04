from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper_reproduction" / "scripts" / "evaluate_families_once.py"


def load_module():
    """Load the standalone family evaluator for unit testing."""
    spec = importlib.util.spec_from_file_location("paper_family_evaluator", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exclusive_json_cannot_overwrite_a_ledger(tmp_path: Path) -> None:
    module = load_module()
    target = tmp_path / "ledger.json"
    module.exclusive_json(target, {"status": "started"})

    with pytest.raises(FileExistsError):
        module.exclusive_json(target, {"status": "replaced"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "started"}


def test_manifest_state_identity_rejects_a_frozen_field_change() -> None:
    module = load_module()
    frozen = [{"id": "job-1", "condition": "ACT", "seed": 5, "overrides": ["a=1"]}]
    actual = [{"id": "job-1", "condition": "ACT", "seed": 5, "overrides": ["a=2"]}]
    top = {"project": "frozen"}

    with pytest.raises(RuntimeError, match="Frozen field mismatch"):
        module.verify_manifest_state_identity(top, top, frozen, actual)


def test_manifest_state_identity_rejects_a_top_level_change() -> None:
    module = load_module()
    jobs = [{"id": "job-1", "condition": "ACT", "seed": 5}]

    with pytest.raises(RuntimeError, match="Frozen top-level field mismatch"):
        module.verify_manifest_state_identity(
            {"project": "frozen", "max_parallel": 4},
            {"project": "frozen", "max_parallel": 2},
            jobs,
            jobs,
        )


def test_pair_roles_cannot_disappear_from_queue_state() -> None:
    module = load_module()
    manifest = [
        {"family": "pair", "seed": 3, "pair_role": "act"},
        {"family": "pair", "seed": 3, "pair_role": "lr"},
    ]
    state = [manifest[0]]

    with pytest.raises(RuntimeError, match="pair roles mismatch"):
        module.validate_pair_roles_and_sanities(manifest, state, primary_queue=False)


def test_prior_test_artifact_is_rejected(tmp_path: Path) -> None:
    module = load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "test_evaluation_started.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Prior per-run test artifacts"):
        module.ensure_no_prior_test_access([{"run_dir": str(run_dir)}])


def test_completed_family_rechecks_every_per_run_hash(tmp_path: Path) -> None:
    module = load_module()
    manifest = tmp_path / "manifest.json"
    state = tmp_path / "state.json"
    start = tmp_path / "start.json"
    complete = tmp_path / "complete.json"
    seed_table = tmp_path / "seeds.csv"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = run_dir / "test_metrics_once.json"
    manifest.write_text('{"jobs": []}\n', encoding="utf-8")
    state.write_text('{"jobs": []}\n', encoding="utf-8")
    seed_table.write_text("seed,accuracy\n3,50\n", encoding="utf-8")
    result.write_text('{"checkpoint_sha256": "checkpoint"}\n', encoding="utf-8")
    start.write_text(
        json.dumps(
            {
                "family": "family",
                "manifest_sha256": module.sha256(manifest),
                "state_sha256_before_test": module.sha256(state),
                "checkpoint_records": [
                    {
                        "run_id": "run-3",
                        "run_dir": str(run_dir),
                        "checkpoint_sha256": "checkpoint",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    complete.write_text(
        json.dumps(
            {
                "family": "family",
                "seed_table_sha256": module.sha256(seed_table),
                "per_run_result_sha256": {"run-3": module.sha256(result)},
            }
        ),
        encoding="utf-8",
    )

    module.audit_completed_family(
        "family", manifest, state, start, complete, seed_table
    )
    result.write_text('{"checkpoint_sha256": "changed"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed or is missing"):
        module.audit_completed_family(
            "family", manifest, state, start, complete, seed_table
        )
