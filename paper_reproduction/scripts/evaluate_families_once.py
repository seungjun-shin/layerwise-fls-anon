#!/usr/bin/env python
"""Immutable family-aware one-time test evaluation for frozen checkpoints."""
from __future__ import annotations

import argparse
import copy
import csv
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

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "paper_reproduction"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_frozen_runs as evaluator  # noqa: E402
import generic_queue  # noqa: E402

FAMILIES: dict[str, dict[str, Any]] = {
    "practical_main_reconstruction": {
        "manifest": "practical_main_jobs.json",
        "state": "queue_state.json",
        "block": "reconstruction",
        "expected_jobs": 15,
        "primary_queue": True,
    },
    "practical_main_confirmatory": {
        "manifest": "practical_main_jobs.json",
        "state": "queue_state.json",
        "block": "confirmatory",
        "expected_jobs": 30,
        "primary_queue": True,
    },
    "practical_transfer": {
        "manifest": "practical_transfer_jobs.json",
        "state": "practical_transfer_queue_state.json",
        "block": "transfer",
        "expected_jobs": 30,
    },
    "head_norm": {
        "manifest": "head_norm_jobs.json",
        "state": "head_norm_queue_state.json",
        "block": "head_norm",
        "expected_jobs": 20,
    },
    "vgg_depth_extension": {
        "manifest": "vgg_depth_extension_jobs.json",
        "state": "vgg_depth_extension_queue_state.json",
        "block": "vgg_depth_extension",
        "expected_jobs": 80,
    },
    "scope_preservation": {
        "manifest": "scope_preservation_jobs.json",
        "state": "scope_preservation_queue_state.json",
        "block": "scope_preservation",
        "expected_jobs": 105,
    },
}

IMMUTABLE_JOB_FIELDS = (
    "id",
    "work_package",
    "block",
    "family",
    "pair_role",
    "condition",
    "architecture",
    "dataset",
    "recipe",
    "seed",
    "config",
    "overrides",
    "expected_boundary_kind",
    "expected_boundary_index",
    "expected_boundary_ratio",
)
FROZEN_TOP_LEVEL_FIELDS = (
    "project",
    "created_before_test_evaluation",
    "max_parallel",
    "gpus",
    "oom_retry",
    "expected_conditions",
    "expected_ratios",
)


def now() -> str:
    """Return a timezone-aware local timestamp."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    """Return the exact evaluator source snapshot."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def resolve_run_dir(value: str | Path) -> Path:
    """Resolve external absolute paths and repository-relative run paths."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    """Create and fsync an immutable JSON ledger entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def configure_expected_ratios(state: dict[str, Any]) -> None:
    """Install manifest-specific ratios in both validator module instances."""
    ratios = dict(evaluator.queue_module.EXPECTED_RATIOS)
    ratios.update(state.get("expected_ratios", {}))
    for job in state["jobs"]:
        if "expected_boundary_ratio" in job:
            ratios[job["condition"]] = float(job["expected_boundary_ratio"])
    evaluator.queue_module.EXPECTED_RATIOS = ratios
    generic_queue.base.EXPECTED_RATIOS = ratios


def selected_jobs(payload: dict[str, Any], block: str) -> list[dict[str, Any]]:
    """Select one frozen block and sort it deterministically."""
    return sorted(
        (job for job in payload["jobs"] if job["block"] == block),
        key=lambda job: (int(job["seed"]), str(job["condition"])),
    )


def verify_manifest_state_identity(
    manifest: dict[str, Any],
    state: dict[str, Any],
    manifest_jobs: list[dict[str, Any]],
    state_jobs: list[dict[str, Any]],
) -> None:
    """Refuse a state that does not exactly match the frozen job definitions."""
    for field in FROZEN_TOP_LEVEL_FIELDS:
        if manifest.get(field) != state.get(field):
            raise RuntimeError(
                f"Frozen top-level field mismatch: {field}: "
                f"manifest={manifest.get(field)!r}, state={state.get(field)!r}"
            )
    by_id = {job["id"]: job for job in state_jobs}
    if len(by_id) != len(state_jobs):
        raise RuntimeError("Queue state contains duplicate job identifiers.")
    if {job["id"] for job in manifest_jobs} != set(by_id):
        raise RuntimeError("Queue-state job identifiers do not match the frozen manifest.")
    for frozen in manifest_jobs:
        actual = by_id[frozen["id"]]
        for field in IMMUTABLE_JOB_FIELDS:
            if frozen.get(field) != actual.get(field):
                raise RuntimeError(
                    f"Frozen field mismatch for {frozen['id']}: {field}: "
                    f"manifest={frozen.get(field)!r}, state={actual.get(field)!r}"
                )


def validate_all_families_terminal() -> None:
    """Require the entire frozen campaign to finish before any family sees test data."""
    for family, spec in FAMILIES.items():
        state_path = WORKSPACE / "manifests" / spec["state"]
        state = json.loads(state_path.read_text(encoding="utf-8"))
        jobs = selected_jobs(state, spec["block"])
        if len(jobs) != int(spec["expected_jobs"]):
            raise RuntimeError(
                f"Global gate expected {spec['expected_jobs']} {family} jobs, found {len(jobs)}."
            )
        bad = [(job["id"], job["status"]) for job in jobs if job["status"] != "completed"]
        if bad:
            raise RuntimeError(f"Global terminal gate failed for {family}: {bad}")


def validate_run_binding(job: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Bind a queue record to its immutable checkpoint and resolved protocol."""
    for field in ("checkpoint_sha256", "checkpoint_epoch"):
        if job.get(field) != metadata.get(field):
            raise RuntimeError(
                f"State/checkpoint mismatch for {job['id']}: {field}: "
                f"state={job.get(field)!r}, artifact={metadata.get(field)!r}"
            )
    if not math.isclose(
        float(job["best_validation_accuracy"]),
        float(metadata["best_validation_accuracy"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"State/checkpoint validation value mismatch for {job['id']}.")
    if int(job["expected_epochs"]) != int(metadata["expected_epochs"]):
        raise RuntimeError(f"State/resolved-config epoch mismatch for {job['id']}.")
    run_dir = resolve_run_dir(job["run_dir"])
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if int(config["training"]["seed"]) != int(job["seed"]):
        raise RuntimeError(f"Resolved-config seed mismatch for {job['id']}.")
    if bool(config["training"].get("evaluate_test_each_epoch", True)):
        raise RuntimeError(f"Resolved config permits training-time test access for {job['id']}.")
    if job.get("dataset") and str(config["data"]["name"]) != str(job["dataset"]):
        raise RuntimeError(f"Resolved-config dataset mismatch for {job['id']}.")
    if job.get("architecture") and str(config["model"]["name"]) != str(job["architecture"]):
        raise RuntimeError(f"Resolved-config architecture mismatch for {job['id']}.")
    observation = metadata["sanity"]["boundary_observation"]
    if job.get("expected_boundary_kind") == "position":
        expected_boundary = f"position_{int(job['expected_boundary_index'])}"
    else:
        expected_boundary = str(job.get("expected_boundary_name", "after_late"))
    if observation.get("boundary") != expected_boundary:
        raise RuntimeError(
            f"Boundary identity mismatch for {job['id']}: expected {expected_boundary}, "
            f"observed {observation.get('boundary')}"
        )


def validate_pair_roles_and_sanities(
    manifest_jobs: list[dict[str, Any]],
    state_jobs: list[dict[str, Any]],
    *,
    primary_queue: bool,
) -> None:
    """Require every frozen role and compare every available frozen pair directly."""
    def key(job: dict[str, Any]) -> tuple[str, int]:
        family = "primary" if primary_queue else str(job["family"])
        return family, int(job["seed"])

    def role(job: dict[str, Any]) -> str:
        field = "condition" if primary_queue else "pair_role"
        return str(job[field])
    expected: dict[tuple[str, int], set[str]] = {}
    actual: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for job in manifest_jobs:
        expected.setdefault(key(job), set()).add(role(job))
    for job in state_jobs:
        group = actual.setdefault(key(job), {})
        job_role = role(job)
        if job_role in group:
            raise RuntimeError(f"Duplicate frozen role {key(job)}, {job_role}.")
        group[job_role] = job
    if set(expected) != set(actual):
        raise RuntimeError("Frozen pair-group identities differ between manifest and state.")
    for group_key, expected_roles in expected.items():
        roles = actual[group_key]
        if set(roles) != expected_roles:
            raise RuntimeError(
                f"Frozen pair roles mismatch for {group_key}: "
                f"expected {sorted(expected_roles)}, found {sorted(roles)}."
            )
        sanities = {
            name: json.loads(
                (resolve_run_dir(job["run_dir"]) / "protocol_sanity.json").read_text(
                    encoding="utf-8"
                )
            )
            for name, job in roles.items()
        }
        if len(sanities) > 1:
            for field in (
                "validation_indices_sha256",
                "audit_batch_indices_sha256",
                "audit_batch_inputs_sha256",
            ):
                if len({record[field] for record in sanities.values()}) != 1:
                    raise RuntimeError(f"Paired {field} mismatch for {group_key}.")
        act_name = "P-ACT-FINAL" if primary_queue else "act"
        lr_name = "P-LR-FINAL" if primary_queue else "lr"
        if act_name in sanities and lr_name in sanities:
            for field in ("initial_parameter_sha256", "optimizer_group_checksum"):
                if sanities[act_name][field] != sanities[lr_name][field]:
                    raise RuntimeError(f"ACT/LR-only {field} mismatch for {group_key}.")


def validate_completed_jobs(
    state: dict[str, Any],
    manifest_jobs: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    block: str,
    *,
    primary_queue: bool,
) -> list[dict[str, Any]]:
    """Revalidate selection, no-test, checkpoint, boundary, and pair integrity."""
    if any(job["status"] != "completed" for job in jobs):
        bad = [(job["id"], job["status"]) for job in jobs if job["status"] != "completed"]
        raise RuntimeError(f"Family is not entirely completed: {bad}")
    records: list[dict[str, Any]] = []
    run_dirs = [str(resolve_run_dir(job["run_dir"]).resolve()) for job in jobs]
    if len(run_dirs) != len(set(run_dirs)):
        raise RuntimeError("Two frozen jobs resolve to the same run directory.")
    for job in jobs:
        run_dir = resolve_run_dir(job["run_dir"])
        valid, errors, metadata = evaluator.validate_run(run_dir, job["condition"])
        if not valid:
            raise RuntimeError(f"Pre-test validation failed for {job['id']}: {'; '.join(errors)}")
        validate_run_binding(job, metadata)
        records.append(
            {
                "run_id": job["id"],
                "run_dir": str(run_dir),
                "condition": job["condition"],
                "seed": int(job["seed"]),
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "resolved_config_sha256": sha256(run_dir / "resolved_config.yaml"),
                "metrics_sha256": sha256(run_dir / "metrics.csv"),
                "protocol_sanity_sha256": sha256(run_dir / "protocol_sanity.json"),
                "checkpoint_epoch": int(metadata["checkpoint_epoch"]),
                "best_validation_accuracy": float(metadata["best_validation_accuracy"]),
            }
        )
    audit_state = copy.deepcopy(state)
    if primary_queue:
        evaluator.queue_module.validate_pair_integrity(audit_state, block)
    else:
        generic_queue.validate_boundary_identity(audit_state, block)
        generic_queue.validate_pair_integrity(audit_state, block)
    audited = selected_jobs(audit_state, block)
    failures = [(job["id"], job["status"], job.get("notes", "")) for job in audited
                if job["status"] != "completed"]
    if failures:
        raise RuntimeError(f"Family integrity validation failed: {failures}")
    validate_pair_roles_and_sanities(
        manifest_jobs,
        jobs,
        primary_queue=primary_queue,
    )
    return records


def ensure_no_prior_test_access(jobs: list[dict[str, Any]]) -> None:
    """Refuse partial or unledgered prior expansion test evaluation."""
    prior: list[str] = []
    for job in jobs:
        run_dir = resolve_run_dir(job["run_dir"])
        for name in ("test_evaluation_started.json", "test_metrics_once.json"):
            if (run_dir / name).exists():
                prior.append(str(run_dir / name))
    if prior:
        raise RuntimeError(f"Prior per-run test artifacts exist without a family completion: {prior}")


def write_seed_table(path: Path, rows: list[dict[str, Any]]) -> None:
    """Create the immutable seed-level result table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def audit_completed_family(
    family: str,
    manifest_path: Path,
    state_path: Path,
    start_path: Path,
    complete_path: Path,
    output_path: Path,
) -> None:
    """Revalidate every immutable ledger dependency on an idempotent read."""
    if not start_path.exists() or not output_path.exists():
        raise RuntimeError("Completed family is missing its start ledger or seed table.")
    start = json.loads(start_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if start.get("family") != family or complete.get("family") != family:
        raise RuntimeError("Family identity mismatch in completed test ledger.")
    if sha256(manifest_path) != start.get("manifest_sha256"):
        raise RuntimeError("Frozen manifest changed after family test evaluation.")
    if sha256(state_path) != start.get("state_sha256_before_test"):
        raise RuntimeError("Queue state changed after family test evaluation.")
    if sha256(output_path) != complete.get("seed_table_sha256"):
        raise RuntimeError("Completed family ledger does not match its seed table.")
    records = {record["run_id"]: record for record in start.get("checkpoint_records", [])}
    result_hashes = complete.get("per_run_result_sha256", {})
    if set(records) != set(result_hashes):
        raise RuntimeError("Start and completion ledgers disagree on evaluated run identifiers.")
    for run_id, record in records.items():
        result_path = resolve_run_dir(record["run_dir"]) / "test_metrics_once.json"
        if not result_path.exists() or sha256(result_path) != result_hashes[run_id]:
            raise RuntimeError(f"Per-run result changed or is missing for {run_id}.")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("checkpoint_sha256") != record["checkpoint_sha256"]:
            raise RuntimeError(f"Evaluated checkpoint identity mismatch for {run_id}.")


def parse_args() -> argparse.Namespace:
    """Parse one family and one explicit evaluation device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run every no-test integrity check without creating a ledger or loading test data.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate exactly one frozen family with an immutable access ledger."""
    args = parse_args()
    spec = FAMILIES[args.family]
    manifest_path = WORKSPACE / "manifests" / spec["manifest"]
    state_path = WORKSPACE / "manifests" / spec["state"]
    ledger_dir = WORKSPACE / "test_once" / args.family
    start_path = ledger_dir / "evaluation_started.json"
    complete_path = ledger_dir / "evaluation_complete.json"
    output_path = WORKSPACE / "seed_tables" / f"{args.family}_seed_level.csv"
    if complete_path.exists():
        audit_completed_family(
            args.family,
            manifest_path,
            state_path,
            start_path,
            complete_path,
            output_path,
        )
        print(output_path)
        return
    if start_path.exists():
        raise RuntimeError(
            f"Family evaluation previously started without completion: {start_path}. "
            "Automatic retry is prohibited; manual audit is required."
        )
    if output_path.exists():
        raise RuntimeError(f"Unledgered seed table already exists: {output_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    validate_all_families_terminal()
    configure_expected_ratios(state)
    manifest_jobs = selected_jobs(manifest, spec["block"])
    state_jobs = selected_jobs(state, spec["block"])
    if len(state_jobs) != int(spec["expected_jobs"]):
        raise RuntimeError(
            f"Expected {spec['expected_jobs']} jobs for {args.family}, found {len(state_jobs)}."
        )
    verify_manifest_state_identity(manifest, state, manifest_jobs, state_jobs)
    checkpoint_records = validate_completed_jobs(
        state,
        manifest_jobs,
        state_jobs,
        spec["block"],
        primary_queue=bool(spec.get("primary_queue", False)),
    )
    ensure_no_prior_test_access(state_jobs)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "family": args.family,
                    "validated_jobs": len(checkpoint_records),
                    "manifest_sha256": sha256(manifest_path),
                    "state_sha256": sha256(state_path),
                },
                sort_keys=True,
            )
        )
        return
    exclusive_json(
        start_path,
        {
            "status": "started",
            "family": args.family,
            "started_at": now(),
            "evaluator_git_commit": git_commit(),
            "evaluator_source_sha256": sha256(Path(__file__)),
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256(manifest_path),
            "state_path": str(state_path.relative_to(ROOT)),
            "state_sha256_before_test": sha256(state_path),
            "expected_jobs": int(spec["expected_jobs"]),
            "checkpoint_records": checkpoint_records,
        },
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    result_digests: dict[str, str] = {}
    for job in state_jobs:
        run_dir = resolve_run_dir(job["run_dir"])
        result = evaluator.evaluate_once(
            run_dir,
            job["condition"],
            device,
            family_start_path=start_path,
            run_id=job["id"],
        )
        result_path = run_dir / "test_metrics_once.json"
        result_digests[job["id"]] = sha256(result_path)
        rows.append(
            {
                "evaluation_family": args.family,
                "block": spec["block"],
                "family": job.get("family", ""),
                "pair_role": job.get("pair_role", ""),
                "condition": job["condition"],
                "seed": int(job["seed"]),
                **result,
            }
        )
    write_seed_table(output_path, rows)
    exclusive_json(
        complete_path,
        {
            "status": "completed",
            "family": args.family,
            "completed_at": now(),
            "evaluated_jobs": len(rows),
            "seed_table": str(output_path.relative_to(ROOT)),
            "seed_table_sha256": sha256(output_path),
            "per_run_result_sha256": result_digests,
        },
    )
    print(output_path)


if __name__ == "__main__":
    main()
