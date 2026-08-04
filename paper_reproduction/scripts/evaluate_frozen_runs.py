#!/usr/bin/env python
"""One-time test evaluation of validation-frozen checkpoints."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fls.data.datasets import build_dataset
from fls.training.model_factory import build_scaled_model

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import local_queue as queue_module  # noqa: E402

validate_run = queue_module.validate_run


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    """Create and fsync a JSON artifact without permitting replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return "unavailable"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _resolved_run_dir(path: str | Path) -> Path:
    """Resolve a recorded run path against the repository root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def _family_checkpoint_record(
    family_start: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Return the unique frozen checkpoint record for one family job."""
    records = [
        record for record in family_start.get("checkpoint_records", [])
        if record.get("run_id") == run_id
    ]
    if len(records) != 1:
        raise RuntimeError(
            f"Family start ledger must contain exactly one checkpoint record for {run_id}."
        )
    return records[0]


def evaluate_once(
    run_dir: Path,
    condition: str,
    device: torch.device,
    *,
    family_start_path: Path,
    run_id: str,
) -> dict[str, Any]:
    """Evaluate one run only under an already-created immutable family ledger."""
    if not family_start_path.exists():
        raise RuntimeError("Family start ledger must exist before any per-run test access.")
    family_start = json.loads(family_start_path.read_text(encoding="utf-8"))
    if family_start.get("status") != "started":
        raise RuntimeError("Family start ledger is not in the started state.")
    checkpoint_record = _family_checkpoint_record(family_start, run_id)
    run_dir = _resolved_run_dir(run_dir)
    if _resolved_run_dir(checkpoint_record["run_dir"]) != run_dir:
        raise RuntimeError(f"Family ledger run directory mismatch for {run_id}.")
    if checkpoint_record.get("condition") != condition:
        raise RuntimeError(f"Family ledger condition mismatch for {run_id}.")
    artifact_names = {
        "checkpoint_sha256": "checkpoint_best.pt",
        "resolved_config_sha256": "resolved_config.yaml",
        "metrics_sha256": "metrics.csv",
        "protocol_sanity_sha256": "protocol_sanity.json",
    }
    for digest_field, filename in artifact_names.items():
        artifact = run_dir / filename
        if not artifact.exists() or sha256(artifact) != checkpoint_record.get(digest_field):
            raise RuntimeError(f"Frozen {filename} digest mismatch for {run_id}.")

    output = run_dir / "test_metrics_once.json"
    if output.exists():
        marker = run_dir / "test_evaluation_started.json"
        if not marker.exists():
            raise RuntimeError(f"Per-run result has no start marker: {run_dir}.")
        result = json.loads(output.read_text(encoding="utf-8"))
        if (
            result.get("run_id") != run_id
            or result.get("condition") != condition
            or result.get("checkpoint_sha256") != checkpoint_record["checkpoint_sha256"]
        ):
            raise RuntimeError(f"Existing per-run result provenance mismatch for {run_id}.")
        return result
    marker = run_dir / "test_evaluation_started.json"
    if marker.exists():
        raise RuntimeError(
            f"A prior test evaluation started but did not write a result: {run_dir}. "
            "Manual audit is required; automatic re-evaluation is prohibited."
        )
    valid, errors, metadata = validate_run(run_dir, condition)
    if not valid:
        raise RuntimeError(f"Run failed frozen pre-test validation: {run_dir}: {'; '.join(errors)}")
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if bool(config["training"].get("evaluate_test_each_epoch", True)):
        raise RuntimeError(f"Run was not trained under the no-test protocol: {run_dir}")
    sanity = json.loads((run_dir / "protocol_sanity.json").read_text(encoding="utf-8"))
    if sanity["test_evaluated_during_training"]:
        raise RuntimeError(f"Protocol sanity reports training-time test evaluation: {run_dir}")
    checkpoint_path = run_dir / "checkpoint_best.pt"
    if metadata["checkpoint_sha256"] != checkpoint_record["checkpoint_sha256"]:
        raise RuntimeError(f"Validated checkpoint digest differs from family ledger for {run_id}.")
    family_start_sha256 = sha256(family_start_path)
    exclusive_json(
        marker,
        {
            "status": "started",
            "started_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "evaluation_family": family_start["family"],
            "family_start_ledger": display_path(family_start_path),
            "family_start_sha256": family_start_sha256,
            "run_id": run_id,
            "condition": condition,
            "checkpoint_sha256": metadata["checkpoint_sha256"],
        },
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_metrics = checkpoint.get("metrics") or {}
    test_checkpoint_fields = {
        key: value
        for key, value in checkpoint_metrics.items()
        if str(key).startswith("test") and value not in {None, ""}
    }
    if test_checkpoint_fields:
        raise RuntimeError(f"Selected checkpoint contains a test endpoint: {checkpoint_path}")
    model = build_scaled_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = build_dataset(config, train=False)
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
    )
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = torch.as_tensor(batch["y"], device=device, dtype=torch.long)
            logits = model(x)
            total_loss += float(F.cross_entropy(logits, y, reduction="sum").item())
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
    if total <= 0 or not math.isfinite(total_loss):
        raise RuntimeError(f"Test evaluation produced an invalid aggregate for {run_id}.")
    result = {
        "run_id": run_id,
        "evaluation_family": family_start["family"],
        "condition": condition,
        "run_dir": display_path(run_dir),
        "experiment_name": config["experiment"]["name"],
        "seed": int(config["training"]["seed"]),
        "checkpoint": "checkpoint_best.pt",
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection_split": "validation",
        "selection_metric": "validation_top1_accuracy",
        "validation_accuracy": float(checkpoint_metrics["val_accuracy"]),
        "test_evaluation_count": 1,
        "test_num_examples": total,
        "test_loss": total_loss / total,
        "test_accuracy": total_correct / total,
        "test_accuracy_percent": 100.0 * total_correct / total,
        "finite": True,
        "evaluated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "family_start_sha256": family_start_sha256,
    }
    # Exclusive creation prevents an accidental second evaluation from replacing
    # the original one-time report. Subsequent invocations only read this file.
    exclusive_json(output, result)
    return result


def main() -> None:
    raise SystemExit(
        "Direct per-block test evaluation is disabled. "
        "Use evaluate_families_once.py so the family ledger is created first."
    )


if __name__ == "__main__":
    main()
