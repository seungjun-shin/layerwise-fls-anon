#!/usr/bin/env python
"""Crash-safe queue for frozen experiment-family manifests."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import queue
import threading
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SPEC = importlib.util.spec_from_file_location(
    "paper_reproduction_local_queue",
    SCRIPT_DIR / "local_queue.py",
)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise ImportError("Could not load local_queue.py")
base = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base)


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "paper_reproduction"


def parse_args() -> argparse.Namespace:
    """Parse one isolated queue invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--failed-manifest", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--only-block")
    return parser.parse_args()


def run_manifest_row(job: dict[str, Any]) -> dict[str, Any]:
    """Serialize one experiment run without architecture-specific assumptions."""
    run_dir = job.get("run_dir", "")
    return {
        "run_id": job["id"],
        "work_package": job["work_package"],
        "block": job["block"],
        "family": job.get("family", ""),
        "pair_role": job.get("pair_role", ""),
        "condition": job["condition"],
        "architecture": job.get("architecture", ""),
        "dataset": job.get("dataset", ""),
        "recipe": job.get("recipe", ""),
        "seed": job["seed"],
        "status": job["status"],
        "config_path": job["config"],
        "run_dir": run_dir,
        "checkpoint_path": f"{run_dir}/checkpoint_best.pt" if run_dir else "",
        "checkpoint_sha256": job.get("checkpoint_sha256", ""),
        "test_evaluated": False,
        "exit_code": job.get("return_code", ""),
        "start_time": job.get("start_time", ""),
        "end_time": job.get("end_time", ""),
        "hardware": f"local_gpu_{job.get('gpu', '')}",
        "notes": job.get("notes", ""),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a complete manifest snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifests(state: dict[str, Any]) -> None:
    """Write queue-specific run and failure manifests."""
    run_path = Path(state["run_manifest_path"])
    failed_path = Path(state["failed_manifest_path"])
    active = [job for job in state["jobs"] if job["status"] != "pending"]
    run_fields = list(run_manifest_row(state["jobs"][0]))
    write_csv(run_path, run_fields, [run_manifest_row(job) for job in active])

    failed_rows: list[dict[str, Any]] = []
    for job in state["jobs"]:
        for attempt in job.get("attempt_history", []):
            if attempt["status"] == "completed":
                continue
            failed_rows.append(
                {
                    "run_id": job["id"],
                    "family": job.get("family", ""),
                    "condition": job["condition"],
                    "seed": job["seed"],
                    "failure_type": attempt["status"],
                    "reason": attempt["reason"],
                    "rerun_allowed": attempt["status"] in {"failed_oom", "interrupted"},
                    "rerun_of": attempt["attempt"],
                    "recorded_at": attempt["recorded_at"],
                }
            )
    failed_fields = [
        "run_id",
        "family",
        "condition",
        "seed",
        "failure_type",
        "reason",
        "rerun_allowed",
        "rerun_of",
        "recorded_at",
    ]
    write_csv(failed_path, failed_fields, failed_rows)


def sanity(job: dict[str, Any]) -> dict[str, Any]:
    """Load one completed run's protocol record."""
    return json.loads(
        (ROOT / job["run_dir"] / "protocol_sanity.json").read_text(encoding="utf-8")
    )


def validate_pair_integrity(state: dict[str, Any], block: str | None) -> None:
    """Enforce data/init/LR identity for every completed ACT/LR pair."""
    selected = [
        job
        for job in state["jobs"]
        if block is None or job["block"] == block
    ]
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for job in selected:
        if job["status"] != "completed" or not job.get("pair_role"):
            continue
        key = (str(job.get("family", job["block"])), int(job["seed"]))
        grouped.setdefault(key, {})[str(job["pair_role"])] = job

    for _key, roles in grouped.items():
        if "act" not in roles or "lr" not in roles:
            continue
        sanities = {role: sanity(job) for role, job in roles.items()}
        errors: list[str] = []
        for field in (
            "validation_indices_sha256",
            "audit_batch_indices_sha256",
            "audit_batch_inputs_sha256",
        ):
            if len({record[field] for record in sanities.values()}) != 1:
                errors.append(f"paired {field} mismatch")
        for field in ("initial_parameter_sha256", "optimizer_group_checksum"):
            if sanities["act"][field] != sanities["lr"][field]:
                errors.append(f"ACT/LR-only {field} mismatch")
        if errors:
            reason = "; ".join(errors)
            for job in roles.values():
                job["status"] = "failed_protocol"
                job["notes"] = reason
                base.record_attempt(job, "failed_protocol", reason)


def validate_boundary_identity(state: dict[str, Any], block: str | None) -> None:
    """Verify that arbitrary-position jobs audited the prespecified cut."""
    for job in state["jobs"]:
        if block is not None and job["block"] != block:
            continue
        if job["status"] != "completed" or not job.get("expected_boundary_kind"):
            continue
        observation = sanity(job)["boundary_observation"]
        kind = str(job["expected_boundary_kind"])
        if kind == "position":
            expected = f"position_{int(job['expected_boundary_index'])}"
        else:
            expected = str(job.get("expected_boundary_name", kind))
        if observation.get("boundary") == expected:
            continue
        reason = (
            f"boundary identity mismatch: expected {expected}, "
            f"observed {observation.get('boundary')}"
        )
        job["status"] = "failed_protocol"
        job["notes"] = reason
        base.record_attempt(job, "failed_protocol", reason)


def main() -> None:
    """Execute the selected immutable manifest on local GPUs."""
    args = parse_args()
    manifest_path = Path(args.manifest)
    state_path = Path(args.state)
    state = base.load_state(manifest_path, state_path)
    state["run_manifest_path"] = args.run_manifest
    state["failed_manifest_path"] = args.failed_manifest

    expected_ratios = dict(state.get("expected_ratios", {}))
    for job in state["jobs"]:
        if "expected_boundary_ratio" in job:
            expected_ratios[job["condition"]] = float(job["expected_boundary_ratio"])
    missing = [job["id"] for job in state["jobs"] if job["condition"] not in expected_ratios]
    if missing:
        raise SystemExit(f"Jobs missing an expected boundary ratio: {missing}")
    base.EXPECTED_RATIOS = expected_ratios
    base.write_manifests = write_manifests

    busy_gpus = base.recover_jobs(state)
    validate_boundary_identity(state, args.only_block)
    validate_pair_integrity(state, args.only_block)
    pending: queue.Queue[int] = queue.Queue()
    for index, job in enumerate(state["jobs"]):
        if job["status"] == "pending" and (
            args.only_block is None or job["block"] == args.only_block
        ):
            pending.put(index)
    requested_gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    gpus = [gpu for gpu in requested_gpus if gpu not in busy_gpus]
    if pending.empty():
        state["updated_at"] = base.now()
        base.atomic_json(state_path, state)
        write_manifests(state)
        print("No pending jobs match the requested block.")
        return
    if not gpus:
        raise SystemExit("All requested GPUs belong to surviving orphan jobs; no duplicates launched.")

    lock = threading.Lock()
    workers = [
        threading.Thread(
            target=base.run_worker,
            args=(gpu, pending, state, state_path, args.python, lock),
            daemon=False,
        )
        for gpu in gpus
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    validate_boundary_identity(state, args.only_block)
    validate_pair_integrity(state, args.only_block)
    selected = [
        job
        for job in state["jobs"]
        if args.only_block is None or job["block"] == args.only_block
    ]
    if selected and all(job["status"] in base.TERMINAL for job in selected):
        marker = args.only_block or "selected"
        state[f"{marker}_finished_at"] = base.now()
    if all(job["status"] in base.TERMINAL for job in state["jobs"]):
        state["all_finished_at"] = base.now()
    state["updated_at"] = base.now()
    base.atomic_json(state_path, state)
    write_manifests(state)
    counts: dict[str, int] = {}
    for job in state["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
