from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch
import yaml

from fls.models.registry import build_model
from fls.scaling.global_fls import GlobalOutputMultiplier
from fls.scaling.position_fls import apply_position_scaling
from fls.training.protocol_audit import _boundary_observation

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "paper_reproduction" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


queue_module = load_script("local_queue")
generic_queue_module = load_script("generic_queue")
statistics_module = load_script("statistics")


def make_valid_run(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {"training": {"max_epochs": 1}}
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    fields = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "val_loss",
        "val_accuracy",
    ]
    with (tmp_path / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 0,
                "train_loss": 2.0,
                "train_accuracy": 0.2,
                "test_loss": "",
                "test_accuracy": "",
                "val_loss": 2.1,
                "val_accuracy": 0.1,
            }
        )
    checkpoint_metrics = {"val_accuracy": 0.1, "test_accuracy": None}
    torch.save(
        {"model": {"weight": torch.ones(2)}, "epoch": 0, "metrics": checkpoint_metrics},
        tmp_path / "checkpoint_best.pt",
    )
    sanity = {
        "test_evaluated_during_training": False,
        "validation_indices_sha256": "val",
        "initial_parameter_sha256": "params",
        "audit_batch_indices_sha256": "indices",
        "audit_batch_inputs_sha256": "inputs",
        "optimizer_group_checksum": "groups",
        "boundary_observation": {"observed_scale_ratio": 1.0},
    }
    (tmp_path / "protocol_sanity.json").write_text(
        json.dumps(sanity), encoding="utf-8"
    )
    return tmp_path


def test_queue_accepts_a_run_only_after_frozen_checks(tmp_path: Path) -> None:
    valid, errors, metadata = queue_module.validate_run(make_valid_run(tmp_path), "P-LR-FINAL")
    assert valid
    assert errors == []
    assert metadata["checkpoint_epoch"] == 0


def test_queue_rejects_training_time_test_metrics(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path)
    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8")
    (run_dir / "metrics.csv").write_text(metrics.replace(",,,2.1", ",1.0,0.5,2.1"), encoding="utf-8")
    valid, errors, _ = queue_module.validate_run(run_dir, "P-LR-FINAL")
    assert not valid
    assert any("test endpoint present" in error for error in errors)


def test_crash_recovery_can_detect_a_surviving_process() -> None:
    assert queue_module.pid_alive(os.getpid())


def test_recovery_never_reruns_a_finished_protocol_failure(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path / "run")
    sanity_path = run_dir / "protocol_sanity.json"
    sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
    sanity["boundary_observation"]["observed_scale_ratio"] = 9.0
    sanity_path.write_text(json.dumps(sanity), encoding="utf-8")
    log_path = tmp_path / "finished.log"
    log_path.write_text(str(run_dir) + "\n", encoding="utf-8")
    state = {
        "oom_retry": {"max_attempts": 2},
        "jobs": [
            {
                "id": "bad_finished",
                "condition": "P-LR-FINAL",
                "seed": 5,
                "status": "running",
                "pid": 999_999_999,
                "attempts": 1,
                "log_path": str(log_path),
            }
        ],
    }
    queue_module.recover_jobs(state)
    assert state["jobs"][0]["status"] == "failed_protocol"


def test_statistics_reject_nonfinite_differences() -> None:
    try:
        statistics_module.exact_sign_flip_pvalue([float("nan")])
    except ValueError:
        pass
    else:
        raise AssertionError("NaN input must not produce a p-value.")


def test_generic_queue_rejects_mismatched_act_lr_initialization(tmp_path: Path) -> None:
    act_dir = make_valid_run(tmp_path / "act")
    lr_dir = make_valid_run(tmp_path / "lr")
    lr_sanity_path = lr_dir / "protocol_sanity.json"
    lr_sanity = json.loads(lr_sanity_path.read_text(encoding="utf-8"))
    lr_sanity["initial_parameter_sha256"] = "different"
    lr_sanity_path.write_text(json.dumps(lr_sanity), encoding="utf-8")
    state = {
        "jobs": [
            {
                "id": "act",
                "block": "scope",
                "family": "example",
                "pair_role": "act",
                "condition": "ACT",
                "seed": 0,
                "status": "completed",
                "attempts": 1,
                "run_dir": str(act_dir),
            },
            {
                "id": "lr",
                "block": "scope",
                "family": "example",
                "pair_role": "lr",
                "condition": "LR",
                "seed": 0,
                "status": "completed",
                "attempts": 1,
                "run_dir": str(lr_dir),
            },
        ]
    }

    generic_queue_module.validate_pair_integrity(state, "scope")

    assert {job["status"] for job in state["jobs"]} == {"failed_protocol"}
    assert all("initial_parameter_sha256 mismatch" in job["notes"] for job in state["jobs"])


def test_position_protocol_audit_observes_the_configured_cut() -> None:
    config = {
        "data": {"num_classes": 10},
        "model": {
            "name": "resnet18_cifar",
            "width": 4,
            "num_classes": 10,
            "use_bn": True,
        },
    }
    model = apply_position_scaling(build_model(config), position=1, multiplier=0.25)

    observation = _boundary_observation(model, torch.randn(2, 3, 32, 32))

    assert observation["boundary"] == "position_1"
    assert observation["expected_scale_ratio"] == 0.25
    assert observation["observed_scale_ratio"] == pytest.approx(0.25)


def test_global_protocol_audit_observes_post_classifier_scale() -> None:
    config = {
        "data": {"num_classes": 10},
        "model": {
            "name": "small_cnn",
            "width": 4,
            "num_classes": 10,
            "use_bn": True,
        },
    }
    model = GlobalOutputMultiplier(build_model(config), 0.0625)

    observation = _boundary_observation(model, torch.randn(2, 3, 32, 32))

    assert observation["boundary"] == "after_classifier"
    assert observation["expected_scale_ratio"] == 0.0625
    assert observation["observed_scale_ratio"] == pytest.approx(0.0625)


def test_generic_queue_rejects_wrong_position_boundary(tmp_path: Path) -> None:
    run_dir = make_valid_run(tmp_path / "position")
    sanity_path = run_dir / "protocol_sanity.json"
    sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
    sanity["boundary_observation"]["boundary"] = "position_3"
    sanity_path.write_text(json.dumps(sanity), encoding="utf-8")
    state = {
        "jobs": [
            {
                "id": "wrong_cut",
                "block": "vgg_depth_extension",
                "condition": "VD-CUT1-ACT",
                "seed": 3,
                "status": "completed",
                "attempts": 1,
                "run_dir": str(run_dir),
                "expected_boundary_kind": "position",
                "expected_boundary_index": 1,
            }
        ]
    }

    generic_queue_module.validate_boundary_identity(state, "vgg_depth_extension")

    assert state["jobs"][0]["status"] == "failed_protocol"
    assert "expected position_1" in state["jobs"][0]["notes"]


def test_generic_queue_validates_recovered_completed_state_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = make_valid_run(tmp_path / "recovered")
    sanity_path = run_dir / "protocol_sanity.json"
    sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
    sanity["boundary_observation"]["boundary"] = "position_3"
    sanity_path.write_text(json.dumps(sanity), encoding="utf-8")
    job = {
        "id": "recovered_wrong_cut",
        "work_package": "WP3",
        "block": "vgg_depth_extension",
        "family": "vgg_depth_cut1",
        "pair_role": "act",
        "condition": "VD-CUT1-ACT",
        "architecture": "vgg19_bn_cifar",
        "dataset": "cifar100",
        "recipe": "controlled",
        "seed": 3,
        "status": "completed",
        "attempts": 1,
        "run_dir": str(run_dir),
        "config": "config.yaml",
        "expected_boundary_kind": "position",
        "expected_boundary_index": 1,
        "expected_boundary_ratio": 1.0,
    }
    payload = {
        "project": "recovery_test",
        "oom_retry": {"max_attempts": 1},
        "jobs": [job],
    }
    manifest_path = tmp_path / "manifest.json"
    state_path = tmp_path / "state.json"
    run_manifest_path = tmp_path / "runs.csv"
    failed_manifest_path = tmp_path / "failures.csv"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generic_queue.py",
            "--manifest",
            str(manifest_path),
            "--state",
            str(state_path),
            "--run-manifest",
            str(run_manifest_path),
            "--failed-manifest",
            str(failed_manifest_path),
            "--gpus",
            "0",
        ],
    )

    generic_queue_module.main()

    final_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert final_state["jobs"][0]["status"] == "failed_protocol"
    assert "expected position_1" in final_state["jobs"][0]["notes"]
