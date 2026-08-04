#!/usr/bin/env python
"""Crash-safe local GPU queue for the frozen practical runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "paper_reproduction"
OOM_PATTERN = re.compile(r"(?:torch\.OutOfMemoryError|CUDA out of memory)", re.IGNORECASE)
TERMINAL = {"completed", "failed_oom", "failed_other", "failed_protocol"}
EXPECTED_RATIOS = {"P-REF": 1.0, "P-ACT-FINAL": 1.5, "P-LR-FINAL": 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state", default=str(WORKSPACE / "manifests" / "queue_state.json"))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--only-block", choices=["reconstruction", "confirmatory"])
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def last_output_path(log_path: Path) -> Path | None:
    if not log_path.exists():
        return None
    for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        candidate = Path(line.strip())
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.is_dir() and (candidate / "resolved_config.yaml").exists():
            return candidate
    return None


def validate_run(run_dir: Path, condition: str) -> tuple[bool, list[str], dict[str, Any]]:
    """Enforce the preregistered stability and no-test requirements."""
    errors: list[str] = []
    config_path = run_dir / "resolved_config.yaml"
    metrics_path = run_dir / "metrics.csv"
    checkpoint_path = run_dir / "checkpoint_best.pt"
    sanity_path = run_dir / "protocol_sanity.json"
    required = [config_path, metrics_path, checkpoint_path, sanity_path]
    missing = [str(path.name) for path in required if not path.exists()]
    if missing:
        return False, [f"missing required artifacts: {', '.join(missing)}"], {}

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_epochs = int(config["training"]["max_epochs"])
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_epochs:
        errors.append(f"expected {expected_epochs} metric rows, found {len(rows)}")
    epochs = [int(float(row["epoch"])) for row in rows]
    if epochs != list(range(expected_epochs)):
        errors.append("epoch sequence is incomplete or non-contiguous")
    numeric_fields = ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    parsed: dict[str, list[float]] = {field: [] for field in numeric_fields}
    for row in rows:
        for field in numeric_fields:
            try:
                value = float(row[field])
            except (TypeError, ValueError):
                errors.append(f"missing/non-numeric {field} at epoch {row.get('epoch')}")
                continue
            if not math.isfinite(value):
                errors.append(f"non-finite {field} at epoch {row.get('epoch')}")
            else:
                parsed[field].append(value)
        if row.get("test_loss") not in {None, ""} or row.get("test_accuracy") not in {None, ""}:
            errors.append(f"test endpoint present during training at epoch {row.get('epoch')}")
    losses = parsed["train_loss"] + parsed["val_loss"]
    if losses and max(losses) > 100.0:
        errors.append("recorded loss exceeds frozen divergence threshold 100")
    validation = parsed["val_accuracy"]
    if not validation or max(validation) <= 0.05:
        errors.append("validation accuracy never exceeded frozen 5% sanity floor")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:  # corrupted/incomplete artifact must become a recorded failure
        errors.append(f"checkpoint is unreadable: {type(error).__name__}: {error}")
        checkpoint = {}
    for name, tensor in checkpoint.get("model", {}).items():
        if torch.is_tensor(tensor) and not bool(torch.isfinite(tensor).all()):
            errors.append(f"non-finite checkpoint tensor: {name}")
            break
    if rows and len(validation) == len(rows):
        best_value = max(validation)
        earliest_best_epoch = next(
            int(float(row["epoch"]))
            for row in rows
            if float(row["val_accuracy"]) == best_value
        )
        if int(checkpoint.get("epoch", -1)) != earliest_best_epoch:
            errors.append("checkpoint_best is not the earliest maximum-validation epoch")
    checkpoint_metrics = checkpoint.get("metrics") or {}
    if checkpoint_metrics.get("test_accuracy") is not None:
        errors.append("checkpoint_best contains a test accuracy")

    try:
        sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        errors.append(f"protocol sanity is unreadable: {type(error).__name__}: {error}")
        sanity = {}
    if sanity.get("test_evaluated_during_training") is not False:
        errors.append("protocol sanity does not confirm no-test training")
    ratio = sanity.get("boundary_observation", {}).get("observed_scale_ratio")
    if ratio is None or not math.isclose(
        float(ratio), EXPECTED_RATIOS[condition], rel_tol=1e-5, abs_tol=1e-7
    ):
        errors.append(f"boundary scale ratio {ratio} does not match {EXPECTED_RATIOS[condition]}")
    required_sanity = [
        "validation_indices_sha256",
        "initial_parameter_sha256",
        "audit_batch_indices_sha256",
        "audit_batch_inputs_sha256",
        "optimizer_group_checksum",
    ]
    for field in required_sanity:
        if not sanity.get(field):
            errors.append(f"protocol sanity is missing {field}")
    metadata = {
        "expected_epochs": expected_epochs,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_validation_accuracy": max(validation) if validation else None,
        "sanity": sanity,
    }
    return not errors, errors, metadata


def record_attempt(job: dict[str, Any], status: str, reason: str) -> None:
    job.setdefault("attempt_history", []).append(
        {
            "attempt": job.get("attempts", 0),
            "status": status,
            "reason": reason,
            "log_path": job.get("log_path", ""),
            "run_dir": job.get("run_dir", ""),
            "recorded_at": now(),
        }
    )


def recover_jobs(state: dict[str, Any]) -> set[str]:
    busy_gpus: set[str] = set()
    for job in state["jobs"]:
        if job["status"] not in {"running", "orphan_running"}:
            continue
        if pid_alive(job.get("pid")):
            job["status"] = "orphan_running"
            job["notes"] = "Training child is still alive; duplicate launch is prohibited."
            busy_gpus.add(str(job.get("gpu")))
            continue
        log_path = ROOT / job["log_path"] if job.get("log_path") else None
        run_dir = last_output_path(log_path) if log_path else None
        if run_dir is not None:
            job["run_dir"] = display_path(run_dir)
            valid, errors, metadata = validate_run(run_dir, job["condition"])
        else:
            valid, errors, metadata = False, ["interrupted process has no complete run directory"], {}
        if valid:
            job["status"] = "completed"
            job.update({key: value for key, value in metadata.items() if key != "sanity"})
            job["notes"] = "Recovered a complete, valid child run."
        elif run_dir is not None:
            job["status"] = "failed_protocol"
            job["notes"] = "; ".join(errors)
            record_attempt(job, "failed_protocol", job["notes"])
        else:
            record_attempt(job, "interrupted", "; ".join(errors))
            if int(job.get("attempts", 0)) < int(state["oom_retry"]["max_attempts"]):
                job["status"] = "pending"
                job["notes"] = "Interrupted infrastructure attempt; safe same-seed rerun queued."
            else:
                job["status"] = "failed_other"
                job["notes"] = "; ".join(errors)
    return busy_gpus


def load_state(manifest_path: Path, state_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not state_path.exists():
        manifest["started_at"] = now()
        manifest["updated_at"] = now()
        atomic_json(state_path, manifest)
        return manifest
    return json.loads(state_path.read_text(encoding="utf-8"))


def run_manifest_row(job: dict[str, Any]) -> dict[str, Any]:
    run_dir = job.get("run_dir", "")
    return {
        "run_id": job["id"],
        "work_package": job["work_package"],
        "condition": job["condition"],
        "architecture": "resnet18_cifar",
        "dataset": "cifar100",
        "recipe": "practical_cosine",
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifests(state: dict[str, Any]) -> None:
    active = [job for job in state["jobs"] if job["status"] != "pending"]
    run_fields = list(run_manifest_row(state["jobs"][0]))
    write_csv(
        WORKSPACE / "manifests" / "run_manifest.csv",
        run_fields,
        [run_manifest_row(job) for job in active],
    )
    failed_rows: list[dict[str, Any]] = []
    for job in state["jobs"]:
        for attempt in job.get("attempt_history", []):
            if attempt["status"] == "completed":
                continue
            failed_rows.append(
                {
                    "run_id": job["id"],
                    "condition": job["condition"],
                    "seed": job["seed"],
                    "failure_type": attempt["status"],
                    "epoch_reached": "",
                    "first_failure_epoch": "",
                    "reason": attempt["reason"],
                    "rerun_allowed": attempt["status"] in {"failed_oom", "interrupted"},
                    "rerun_of": attempt["attempt"],
                    "recorded_at": attempt["recorded_at"],
                }
            )
    failed_fields = [
        "run_id",
        "condition",
        "seed",
        "failure_type",
        "epoch_reached",
        "first_failure_epoch",
        "reason",
        "rerun_allowed",
        "rerun_of",
        "recorded_at",
    ]
    write_csv(WORKSPACE / "manifests" / "failed_runs.csv", failed_fields, failed_rows)


def validate_pair_integrity(state: dict[str, Any], block: str) -> None:
    jobs = [job for job in state["jobs"] if job["block"] == block]
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for job in jobs:
        if job["status"] == "completed":
            by_seed.setdefault(int(job["seed"]), {})[job["condition"]] = job
    for _seed, conditions in by_seed.items():
        if set(conditions) != set(EXPECTED_RATIOS):
            continue
        sanities = {
            condition: json.loads(
                (ROOT / job["run_dir"] / "protocol_sanity.json").read_text(encoding="utf-8")
            )
            for condition, job in conditions.items()
        }
        errors: list[str] = []
        data_fields = [
            "validation_indices_sha256",
            "audit_batch_indices_sha256",
            "audit_batch_inputs_sha256",
        ]
        for field in data_fields:
            if len({sanity[field] for sanity in sanities.values()}) != 1:
                errors.append(f"paired {field} mismatch")
        act = sanities["P-ACT-FINAL"]
        lr_only = sanities["P-LR-FINAL"]
        for field in ("initial_parameter_sha256", "optimizer_group_checksum"):
            if act[field] != lr_only[field]:
                errors.append(f"ACT/LR-only {field} mismatch")
        if errors:
            reason = "; ".join(errors)
            for job in conditions.values():
                job["status"] = "failed_protocol"
                job["notes"] = reason
                record_attempt(job, "failed_protocol", reason)


def run_worker(
    gpu: str,
    pending: queue.Queue[int],
    state: dict[str, Any],
    state_path: Path,
    python: str,
    lock: threading.Lock,
) -> None:
    retry = state["oom_retry"]
    while True:
        try:
            index = pending.get_nowait()
        except queue.Empty:
            return
        with lock:
            job = state["jobs"][index]
            job["attempts"] += 1
            job["status"] = "running"
            job["gpu"] = gpu
            job["start_time"] = now()
            log_path = WORKSPACE / "logs" / f"{job['id']}__attempt{job['attempts']}.log"
            job["log_path"] = str(log_path.relative_to(ROOT))
            state["updated_at"] = now()
            atomic_json(state_path, state)
        command = [python, "scripts/train.py", "--config", job["config"], *job["overrides"]]
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with lock:
                job["pid"] = process.pid
                job["process_group_id"] = os.getpgid(process.pid)
                atomic_json(state_path, state)
            return_code = process.wait()
        run_dir = last_output_path(log_path)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        is_oom = bool(OOM_PATTERN.search(log_text))
        if run_dir is not None and return_code == 0:
            valid, errors, metadata = validate_run(run_dir, job["condition"])
        else:
            valid = False
            errors = [f"training process exited with code {return_code}"]
            metadata = {}
        with lock:
            job["return_code"] = return_code
            job["end_time"] = now()
            job.pop("pid", None)
            job.pop("process_group_id", None)
            if run_dir is not None:
                job["run_dir"] = display_path(run_dir)
            if valid:
                job["status"] = "completed"
                job.update({key: value for key, value in metadata.items() if key != "sanity"})
                job["notes"] = "Passed frozen stability, no-test, and scale checks."
                record_attempt(job, "completed", job["notes"])
            elif is_oom and job["attempts"] < int(retry["max_attempts"]):
                job["status"] = "pending"
                job["notes"] = "CUDA OOM; unchanged config and seed queued once more."
                record_attempt(job, "failed_oom", job["notes"])
                pending.put(index)
            else:
                if is_oom:
                    job["status"] = "failed_oom"
                elif return_code != 0:
                    job["status"] = "failed_other"
                else:
                    job["status"] = "failed_protocol"
                job["notes"] = "; ".join(errors)
                record_attempt(job, job["status"], job["notes"])
            state["updated_at"] = now()
            atomic_json(state_path, state)
            write_manifests(state)
            print(f"[gpu {gpu}] {job['id']}: {job['status']}", flush=True)
        if is_oom and job["status"] == "pending":
            time.sleep(int(retry["delay_seconds"]))
        pending.task_done()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    state_path = Path(args.state)
    state = load_state(manifest_path, state_path)
    busy_gpus = recover_jobs(state)
    pending: queue.Queue[int] = queue.Queue()
    for index, job in enumerate(state["jobs"]):
        if job["status"] == "pending" and (
            args.only_block is None or job["block"] == args.only_block
        ):
            pending.put(index)
    requested_gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    gpus = [gpu for gpu in requested_gpus if gpu not in busy_gpus]
    if pending.empty():
        write_manifests(state)
        print("No pending jobs match the requested block.")
        return
    if not gpus:
        raise SystemExit("All requested GPUs belong to surviving orphan jobs; no duplicates launched.")
    lock = threading.Lock()
    workers = [
        threading.Thread(
            target=run_worker,
            args=(gpu, pending, state, state_path, args.python, lock),
            daemon=False,
        )
        for gpu in gpus
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    block = args.only_block
    if block is not None:
        validate_pair_integrity(state, block)
        selected = [job for job in state["jobs"] if job["block"] == block]
        if all(job["status"] in TERMINAL for job in selected):
            state[f"{block}_finished_at"] = now()
    if all(job["status"] in TERMINAL for job in state["jobs"]):
        state["all_finished_at"] = now()
    state["updated_at"] = now()
    atomic_json(state_path, state)
    write_manifests(state)
    counts: dict[str, int] = {}
    for job in state["jobs"]:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
