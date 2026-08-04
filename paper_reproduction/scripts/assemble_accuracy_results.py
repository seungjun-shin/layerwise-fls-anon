#!/usr/bin/env python
"""Assemble validation-controlled accuracy tables and paired statistics."""

from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "paper_reproduction"
COMPLETED_FAMILY_SPECS: dict[str, dict[str, str | int]] = {
    "practical_main_reconstruction": {
        "manifest": "practical_main_jobs.json",
        "state": "queue_state.json",
        "block": "reconstruction",
        "expected_jobs": 15,
    },
    "practical_main_confirmatory": {
        "manifest": "practical_main_jobs.json",
        "state": "queue_state.json",
        "block": "confirmatory",
        "expected_jobs": 30,
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


def arithmetic_mean(values: list[float]) -> float:
    """Return the arithmetic mean without importing a shadowed stdlib module."""
    return math.fsum(values) / len(values)


def sample_stdev(values: list[float]) -> float:
    """Return the Bessel-corrected sample standard deviation."""
    center = arithmetic_mean(values)
    return math.sqrt(math.fsum((value - center) ** 2 for value in values) / (len(values) - 1))


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV artifact into dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write and fsync a new CSV without replacing an existing result."""
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    """Create and fsync a JSON artifact without replacing prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    """Persist directory-entry updates for a completed result transaction."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_accuracy_bundle(outputs: dict[Path, list[dict[str, Any]]]) -> Path:
    """Commit all derived tables as one completion-bound transaction."""
    bundle = WORKSPACE / "analysis/statistics/accuracy_bundle_complete.json"
    reservation = WORKSPACE / "analysis/statistics/.accuracy_bundle.reserved"
    targets = list(outputs)
    if bundle.exists() or reservation.exists() or any(path.exists() for path in targets):
        raise RuntimeError("Accuracy bundle or one of its authoritative outputs already exists.")
    exclusive_json(
        reservation,
        {
            "status": "reserved",
            "outputs": [str(path.relative_to(ROOT)) for path in targets],
            "script_sha256": sha256(Path(__file__)),
        },
    )
    temporary: dict[Path, Path] = {}
    committed = 0
    try:
        for target, rows in outputs.items():
            temporary[target] = target.with_name(f".{target.name}.tmp.{os.getpid()}")
            write_csv_exclusive(temporary[target], rows)
        for target, temp in temporary.items():
            os.replace(temp, target)
            fsync_directory(target.parent)
            committed += 1
        output_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in targets}
        exclusive_json(
            bundle,
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc)
                .astimezone()
                .isoformat(timespec="seconds"),
                "script": str(Path(__file__).relative_to(ROOT)),
                "script_sha256": sha256(Path(__file__)),
                "outputs_sha256": output_hashes,
                "multiplicity_plan": {
                    "head_norm": "Holm over 2 ACT-minus-LR exact sign-flip tests",
                    "vgg_depth": "Holm over 8 cut-wise ACT-minus-LR exact sign-flip tests",
                    "scope_act_lr": "Holm over 7 scope ACT-minus-LR exact sign-flip tests",
                    "scope_act_ref": "Holm over 7 scope ACT-minus-reference exact sign-flip tests",
                },
            },
        )
        reservation.unlink()
        fsync_directory(reservation.parent)
    except Exception:
        for temp in temporary.values():
            if temp.exists():
                temp.unlink()
        if committed == 0 and reservation.exists():
            reservation.unlink()
        raise
    return bundle


def resolve_source_path(value: str | Path) -> Path:
    """Resolve repository-relative and external absolute result paths."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def expected_completed_keys(family: str) -> set[tuple[str, str, str, int]]:
    """Return the exact frozen (subfamily, role, condition, seed) design."""
    if family == "practical_main_reconstruction":
        return {
            ("", "", condition, seed)
            for condition in ("P-REF", "P-ACT-FINAL", "P-LR-FINAL")
            for seed in range(5)
        }
    if family == "practical_main_confirmatory":
        return {
            ("", "", condition, seed)
            for condition in ("P-REF", "P-ACT-FINAL", "P-LR-FINAL")
            for seed in range(5, 15)
        }
    if family == "practical_transfer":
        return {
            ("vgg_practical", role, condition, seed)
            for role, condition in (
                ("ref", "T-REF"),
                ("act", "T-ACT-FINAL"),
                ("lr", "T-LR-FINAL"),
            )
            for seed in range(10, 20)
        }
    if family == "head_norm":
        return {
            (subfamily, role, condition, seed)
            for subfamily, role, condition in (
                ("head_norm_cosine", "act", "HN-COSINE-ACT"),
                ("head_norm_cosine", "lr", "HN-COSINE-LR"),
                ("head_norm_preln", "act", "HN-PRELN-ACT"),
                ("head_norm_preln", "lr", "HN-PRELN-LR"),
            )
            for seed in range(20, 25)
        }
    if family == "vgg_depth_extension":
        return {
            (f"vgg_depth_cut{cut}", role, f"VD-CUT{cut}-{role.upper()}", seed)
            for cut in range(1, 16, 2)
            for role in ("act", "lr")
            for seed in range(5)
        }
    if family == "scope_preservation":
        return {
            (subfamily, role, f"S-{subfamily.upper()}-{role.upper()}", seed)
            for subfamily in (
            "cifar10",
            "nobn",
            "noise20",
            "noise50",
            "persistence",
            "stl10",
            "svhn",
            )
            for role in ("ref", "act", "lr")
            for seed in range(5)
        }
    raise KeyError(f"No frozen design for completed family {family}")


def assert_equal_number(actual: Any, expected: Any, context: str) -> None:
    """Require two finite numeric ledger values to agree to serialization precision."""
    left = float(actual)
    right = float(expected)
    if (
        not math.isfinite(left)
        or not math.isfinite(right)
        or not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise RuntimeError(f"Numeric provenance mismatch for {context}: {left} != {right}")


def verify_record_artifact_hashes(record: dict[str, Any], run_dir: Path) -> None:
    """Bind every mutable frozen-run artifact to its start-ledger digest."""
    fields = {
        "checkpoint_best.pt": "checkpoint_sha256",
        "resolved_config.yaml": "resolved_config_sha256",
        "metrics.csv": "metrics_sha256",
        "protocol_sanity.json": "protocol_sanity_sha256",
    }
    for filename, digest_field in fields.items():
        path = run_dir / filename
        if not path.is_file() or sha256(path) != record.get(digest_field):
            raise RuntimeError(
                f"Frozen run artifact changed for {record.get('run_id')}: {filename}"
            )


def completed_family_rows(family: str) -> list[dict[str, str]]:
    """Audit every immutable ledger and result before reading a completed family."""
    spec = COMPLETED_FAMILY_SPECS[family]
    ledger_dir = WORKSPACE / "test_once" / family
    start_path = ledger_dir / "evaluation_started.json"
    completion_path = WORKSPACE / "test_once" / family / "evaluation_complete.json"
    if not start_path.exists() or not completion_path.exists():
        raise RuntimeError(f"Completed family is missing immutable ledgers: {family}")
    start = json.loads(start_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    table_path = WORKSPACE / "seed_tables" / f"{family}_seed_level.csv"
    expected_relative = str(table_path.relative_to(ROOT))
    if (
        start.get("status") != "started"
        or completion.get("status") != "completed"
        or start.get("family") != family
        or completion.get("family") != family
        or completion.get("seed_table") != expected_relative
        or sha256(table_path) != completion.get("seed_table_sha256")
    ):
        raise RuntimeError(f"Family seed table is not completion-bound: {family}")
    manifest_path = WORKSPACE / "manifests" / str(spec["manifest"])
    state_path = WORKSPACE / "manifests" / str(spec["state"])
    if (
        start.get("manifest_path") != str(manifest_path.relative_to(ROOT))
        or start.get("state_path") != str(state_path.relative_to(ROOT))
        or sha256(manifest_path) != start.get("manifest_sha256")
        or sha256(state_path) != start.get("state_sha256_before_test")
    ):
        raise RuntimeError(f"Frozen manifest/state changed after evaluation: {family}")
    rows = read_csv(table_path)
    expected_n = int(spec["expected_jobs"])
    if (
        len(rows) != expected_n
        or int(completion["evaluated_jobs"]) != expected_n
        or int(start["expected_jobs"]) != expected_n
    ):
        raise RuntimeError(f"Family seed count mismatch: {family}")
    row_ids = [row["run_id"] for row in rows]
    row_keys = [
        (row["family"], row["pair_role"], row["condition"], int(row["seed"])) for row in rows
    ]
    if len(row_ids) != len(set(row_ids)) or len(row_keys) != len(set(row_keys)):
        raise RuntimeError(f"Duplicate immutable result key in family {family}")
    if set(row_keys) != expected_completed_keys(family):
        raise RuntimeError(f"Completed rows do not match frozen design: {family}")
    checkpoint_records = start["checkpoint_records"]
    record_ids = [record["run_id"] for record in checkpoint_records]
    if len(checkpoint_records) != expected_n or len(record_ids) != len(set(record_ids)):
        raise RuntimeError(f"Duplicate or incomplete start-ledger records: {family}")
    records = {record["run_id"]: record for record in checkpoint_records}
    result_hashes = completion.get("per_run_result_sha256", {})
    if (
        len(records) != expected_n
        or set(records) != set(row_ids)
        or set(records) != set(result_hashes)
    ):
        raise RuntimeError(f"Start/result/seed-table run IDs disagree: {family}")
    start_sha = sha256(start_path)
    required_finite = (
        "checkpoint_epoch",
        "validation_accuracy",
        "test_loss",
        "test_accuracy",
        "test_accuracy_percent",
        "test_num_examples",
    )
    for row in rows:
        run_id = row["run_id"]
        record = records[run_id]
        run_dir = resolve_source_path(row["run_dir"])
        verify_record_artifact_hashes(record, run_dir)
        result_path = run_dir / "test_metrics_once.json"
        if sha256(result_path) != result_hashes[run_id]:
            raise RuntimeError(f"Per-run result hash changed for {run_id}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            row["evaluation_family"] != family
            or row["block"] != spec["block"]
            or row["selection_split"] != "validation"
            or row["selection_metric"] != "validation_top1_accuracy"
            or row["checkpoint"] != "checkpoint_best.pt"
            or int(row["test_evaluation_count"]) != 1
            or row["finite"] != "True"
            or result.get("finite") is not True
            or result.get("family_start_sha256") != start_sha
            or row["family_start_sha256"] != start_sha
            or result.get("run_id") != run_id
            or int(result.get("seed")) != int(row["seed"])
            or result.get("condition") != row["condition"]
            or result.get("evaluation_family") != family
            or result.get("selection_split") != "validation"
            or int(result.get("test_evaluation_count")) != 1
            or record["condition"] != row["condition"]
            or int(record["seed"]) != int(row["seed"])
            or resolve_source_path(record["run_dir"]).resolve() != run_dir.resolve()
            or record["checkpoint_sha256"] != row["checkpoint_sha256"]
            or result.get("checkpoint_sha256") != row["checkpoint_sha256"]
            or int(record["checkpoint_epoch"]) != int(row["checkpoint_epoch"])
            or int(result.get("checkpoint_epoch")) != int(row["checkpoint_epoch"])
        ):
            raise RuntimeError(f"Immutable row/result binding mismatch for {run_id}")
        for field in required_finite:
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"Non-finite {field} for {run_id}")
            assert_equal_number(row[field], result[field], f"{run_id}/{field}")
        assert_equal_number(
            row["validation_accuracy"], record["best_validation_accuracy"], f"{run_id}/validation"
        )
    return rows


def selected_validation_row(metrics_path: Path) -> dict[str, Any]:
    """Select the earliest maximum-validation epoch from one complete legacy run."""
    rows = read_csv(metrics_path)
    config = yaml.safe_load(
        (metrics_path.parent / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    expected_epochs = int(config["training"]["max_epochs"])
    epochs = [int(float(row["epoch"])) for row in rows]
    if epochs != list(range(expected_epochs)):
        raise RuntimeError(f"Incomplete legacy metrics: {metrics_path}")
    values = [float(row["val_accuracy"]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Non-finite validation history: {metrics_path}")
    best_value = max(values)
    best = next(row for row in rows if float(row["val_accuracy"]) == best_value)
    test_accuracy = float(best["test_accuracy"])
    if not math.isfinite(test_accuracy):
        raise RuntimeError(f"Non-finite report-only test endpoint: {metrics_path}")
    return {
        "seed": int(config["training"]["seed"]),
        "test_accuracy_percent": 100.0 * test_accuracy,
        "validation_accuracy_percent": 100.0 * best_value,
        "checkpoint_epoch": int(float(best["epoch"])),
        "source_path": str(metrics_path),
        "source_metrics_sha256": sha256(metrics_path),
        "source_config_sha256": sha256(metrics_path.parent / "resolved_config.yaml"),
        "selection_protocol": "validation-selected earliest maximum",
        "test_protocol": "legacy run recorded test trajectory; test did not select checkpoint",
    }


def legacy_paths(directory: str, expected_seeds: set[int]) -> dict[int, Path]:
    """Resolve exactly one completed metrics file for each requested legacy seed."""
    paths: dict[int, Path] = {}
    for path in sorted((ROOT / "outputs" / directory).glob("**/metrics.csv")):
        seed = int(path.parent.name.removeprefix("seed_"))
        if seed in paths:
            raise RuntimeError(f"Duplicate legacy metrics for {directory}, seed={seed}")
        paths[seed] = path.relative_to(ROOT)
    if set(paths) != expected_seeds:
        raise RuntimeError(
            f"Legacy seed mismatch for {directory}: expected {sorted(expected_seeds)}, "
            f"found {sorted(paths)}"
        )
    return paths


def normalized_new_row(
    row: dict[str, str],
    evidence_family: str,
    role: str,
    status: str,
) -> dict[str, Any]:
    """Normalize a one-time family row for aggregation."""
    run_dir = resolve_source_path(row["run_dir"])
    return {
        "evidence_family": evidence_family,
        "role": role,
        "condition": row["condition"],
        "seed": int(row["seed"]),
        "test_accuracy_percent": float(row["test_accuracy_percent"]),
        "validation_accuracy_percent": 100.0 * float(row["validation_accuracy"]),
        "checkpoint_epoch": int(row["checkpoint_epoch"]),
        "source_path": row["run_dir"],
        "source_metrics_sha256": sha256(run_dir / "metrics.csv"),
        "source_config_sha256": sha256(run_dir / "resolved_config.yaml"),
        "source_checkpoint_sha256": row["checkpoint_sha256"],
        "source_test_result_sha256": sha256(run_dir / "test_metrics_once.json"),
        "selection_protocol": "validation-selected earliest maximum",
        "test_protocol": "immutable one-time test after complete family freeze",
        "evidence_status": status,
    }


def resolved_config(source: str | Path) -> dict[str, Any]:
    """Read the resolved config beside a metrics file or from a run directory."""
    path = resolve_source_path(source)
    config_path = (
        path.parent / "resolved_config.yaml"
        if path.name == "metrics.csv"
        else path / "resolved_config.yaml"
    )
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def protocol_fingerprint(source: str | Path) -> str:
    """Hash training semantics while ignoring logging and audit-only additions."""
    config = copy.deepcopy(resolved_config(source))
    for key in ("experiment", "output", "diagnostics", "sweep"):
        config.pop(key, None)
    training = config.get("training", {})
    for key in (
        "seed",
        "save_checkpoints",
        "checkpoint_at_accs",
        "evaluate_test_each_epoch",
        "protocol_sanity",
    ):
        training.pop(key, None)
    data = config.get("data", {})
    for key in ("deterministic_validation", "val_indices_sha256"):
        data.pop(key, None)
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_vgg_config(source: str | Path, cut: int, role: str, seed: int) -> None:
    """Require the exact VGG architecture, cut, intervention, recipe, and seed."""
    config = resolved_config(source)
    fls = config["fls"]
    training = config["training"]
    expected_fls = (
        {"mode": "position", "position": cut, "output_multiplier": 0.0625}
        if role == "act"
        else {
            "mode": "position",
            "position": cut,
            "output_multiplier": 1.0,
            "lr_compensation_multiplier": 0.0625,
        }
    )
    if (
        config["model"]["name"] != "vgg19_bn_cifar"
        or config["data"]["name"] != "cifar100"
        or config["data"].get("augment") is not False
        or int(training["seed"]) != seed
        or int(training["max_epochs"]) != 80
        or not math.isclose(float(training["lr"]), 0.04096)
        or not math.isclose(float(training["momentum"]), 0.0)
        or not math.isclose(float(training["weight_decay"]), 0.0)
        or training.get("position_lr_compensation") is not True
        or fls != expected_fls
    ):
        raise RuntimeError(f"Unexpected VGG cut{cut}/{role}/seed{seed} config: {source}")


def verify_summary_row(summary: dict[str, str], metrics_path: Path) -> dict[str, Any]:
    """Recompute every selected legacy number instead of trusting a summary CSV."""
    record = selected_validation_row(metrics_path)
    if int(summary["seed"]) != int(record["seed"]) or int(summary["best_val_epoch"]) != int(
        record["checkpoint_epoch"]
    ):
        raise RuntimeError(f"Legacy summary seed/epoch mismatch: {metrics_path}")
    assert_equal_number(
        summary["best_val_accuracy_pct"],
        record["validation_accuracy_percent"],
        f"{metrics_path}/val",
    )
    assert_equal_number(
        summary["test_accuracy_at_best_val_pct"],
        record["test_accuracy_percent"],
        f"{metrics_path}/test",
    )
    return record


def vgg_depth_rows() -> list[dict[str, Any]]:
    """Normalize the complete five-seed frozen VGG depth family."""
    output: list[dict[str, Any]] = []
    for row in completed_family_rows("vgg_depth_extension"):
        cut = int(row["family"].removeprefix("vgg_depth_cut"))
        seed = int(row["seed"])
        validate_vgg_config(row["run_dir"], cut, row["pair_role"], seed)
        normalized = normalized_new_row(
            row,
            row["family"],
            row["pair_role"],
            "descriptive controlled depth profile",
        )
        normalized["protocol_fingerprint"] = protocol_fingerprint(row["run_dir"])
        output.append(normalized)
    validate_complete_roles(output, {f"vgg_depth_cut{cut}" for cut in range(1, 16, 2)})
    fingerprints: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in output:
        fingerprints[(row["evidence_family"], row["role"])].add(row["protocol_fingerprint"])
    mismatched = {key: values for key, values in fingerprints.items() if len(values) != 1}
    if mismatched:
        raise RuntimeError(f"VGG legacy/new protocol mismatch: {mismatched}")
    return sorted(
        output,
        key=lambda row: (
            int(row["evidence_family"].removeprefix("vgg_depth_cut")),
            row["seed"],
            row["role"],
        ),
    )


def scope_rows() -> list[dict[str, Any]]:
    """Normalize the complete five-seed frozen scope family."""
    output: list[dict[str, Any]] = []
    for row in completed_family_rows("scope_preservation"):
        normalized = normalized_new_row(
            row,
            row["family"],
            row["pair_role"],
            "descriptive validation-controlled preservation",
        )
        normalized["protocol_fingerprint"] = protocol_fingerprint(row["run_dir"])
        output.append(normalized)
    families = {"cifar10", "nobn", "noise20", "noise50", "persistence", "stl10", "svhn"}
    validate_complete_roles(output, families, roles={"ref", "act", "lr"})
    fingerprints: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in output:
        fingerprints[(row["evidence_family"], row["role"])].add(row["protocol_fingerprint"])
    mismatched = {key: values for key, values in fingerprints.items() if len(values) != 1}
    if mismatched:
        raise RuntimeError(f"Scope legacy/new protocol mismatch: {mismatched}")
    return sorted(output, key=lambda row: (row["evidence_family"], row["seed"], row["role"]))


def validate_complete_roles(
    rows: list[dict[str, Any]],
    families: set[str],
    *,
    roles: set[str] | None = None,
) -> None:
    """Require exactly five paired seeds for every requested family and role."""
    if roles is None:
        roles = {"act", "lr"}
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(row["evidence_family"], row["role"])].append(int(row["seed"]))
    expected_seeds = set(range(5))
    for family in families:
        for role in roles:
            seeds = grouped[(family, role)]
            if len(seeds) != 5 or set(seeds) != expected_seeds:
                raise RuntimeError(
                    f"Incomplete or duplicate {family}/{role}: found {sorted(seeds)}"
                )
    expected_groups = {(family, role) for family in families for role in roles}
    if set(grouped) != expected_groups:
        raise RuntimeError(
            f"Unexpected family/role groups: {sorted(set(grouped) - expected_groups)}"
        )


def exact_sign_flip_pvalue(differences: list[float]) -> float:
    """Compute an exact two-sided paired sign-flip p-value on the mean."""
    observed = abs(arithmetic_mean(differences))
    statistics = (
        abs(arithmetic_mean([sign * value for sign, value in zip(signs, differences, strict=True)]))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    )
    values = list(statistics)
    return sum(value >= observed - 1e-15 for value in values) / len(values)


def paired_summary(differences: list[float]) -> dict[str, Any]:
    """Return preregistered paired effect summaries without exclusions."""
    n = len(differences)
    average = arithmetic_mean(differences)
    sd = sample_stdev(differences) if n > 1 else float("nan")
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n) if n > 1 else float("nan")
    try:
        wilcoxon = float(stats.wilcoxon(differences, alternative="two-sided").pvalue)
    except ValueError:
        wilcoxon = 1.0
    return {
        "n": n,
        "mean_difference_pp": average,
        "median_difference_pp": float(np.median(differences)),
        "sd_difference_pp": sd,
        "ci95_low_pp": average - half,
        "ci95_high_pp": average + half,
        "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
        "wilcoxon_sensitivity_p": wilcoxon,
        "paired_standardized_effect": average / sd if sd > 0 else float("nan"),
    }


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Return monotone Holm-adjusted p-values."""
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted.tolist()


def paired_row(
    rows: list[dict[str, Any]],
    family: str,
    left: str,
    right: str,
    tier: str,
    multiplicity_family: str,
) -> dict[str, Any]:
    """Pair two roles by seed and summarize their accuracy difference."""
    selected = [row for row in rows if row["evidence_family"] == family]
    grouped: dict[str, dict[int, float]] = {left: {}, right: {}}
    for row in selected:
        role = row["role"]
        if role not in grouped:
            continue
        seed = int(row["seed"])
        if seed in grouped[role]:
            raise RuntimeError(f"Duplicate comparison row for {family}/{role}/seed{seed}")
        grouped[role][seed] = float(row["test_accuracy_percent"])
    if set(grouped[left]) != set(grouped[right]) or not grouped[left]:
        raise RuntimeError(f"Unpaired comparison for {family}: {left} - {right}")
    seeds = sorted(grouped[left])
    differences = [grouped[left][seed] - grouped[right][seed] for seed in seeds]
    return {
        "evidence_family": family,
        "tier": tier,
        "comparison": f"{left} - {right}",
        "metric": "test_accuracy_percentage_points",
        "seeds": ";".join(map(str, seeds)),
        "multiplicity_family": multiplicity_family,
        **paired_summary(differences),
    }


def standard_new_rows() -> list[dict[str, Any]]:
    """Normalize primary, transfer, and head/normalization one-time results."""
    output: list[dict[str, Any]] = []
    primary_roles = {
        "P-REF": "ref",
        "P-ACT-FINAL": "act",
        "P-LR-FINAL": "lr",
    }
    for family, status in (
        ("practical_main_reconstruction", "separate reconstruction block"),
        ("practical_main_confirmatory", "independent confirmatory block"),
    ):
        for row in completed_family_rows(family):
            output.append(normalized_new_row(row, family, primary_roles[row["condition"]], status))
    for row in completed_family_rows("practical_transfer"):
        output.append(
            normalized_new_row(
                row,
                "practical_transfer",
                row["pair_role"],
                "supportive architecture transfer",
            )
        )
    for row in completed_family_rows("head_norm"):
        output.append(
            normalized_new_row(
                row,
                row["family"],
                row["pair_role"],
                "exploratory mechanism ablation",
            )
        )
    expected = {
        ("practical_main_reconstruction", role, seed)
        for role in ("ref", "act", "lr")
        for seed in range(5)
    }
    expected.update(
        ("practical_main_confirmatory", role, seed)
        for role in ("ref", "act", "lr")
        for seed in range(5, 15)
    )
    expected.update(
        ("practical_transfer", role, seed)
        for role in ("ref", "act", "lr")
        for seed in range(10, 20)
    )
    expected.update(
        (family, role, seed)
        for family in ("head_norm_cosine", "head_norm_preln")
        for role in ("act", "lr")
        for seed in range(20, 25)
    )
    actual = [(row["evidence_family"], row["role"], int(row["seed"])) for row in output]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("Normalized standard rows do not match the exact planned designs.")
    return output


def condition_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize each condition with individual-seed dispersion and a t interval."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["evidence_family"], row["role"], row["evidence_status"])].append(row)
    output: list[dict[str, Any]] = []
    for (family, role, status), group in sorted(grouped.items()):
        values = [float(row["test_accuracy_percent"]) for row in group]
        average = arithmetic_mean(values)
        sd = sample_stdev(values) if len(values) > 1 else float("nan")
        half = (
            float(stats.t.ppf(0.975, len(values) - 1)) * sd / math.sqrt(len(values))
            if len(values) > 1
            else float("nan")
        )
        output.append(
            {
                "evidence_family": family,
                "role": role,
                "evidence_status": status,
                "n": len(values),
                "seeds": ";".join(
                    str(row["seed"]) for row in sorted(group, key=lambda row: row["seed"])
                ),
                "mean_test_accuracy_percent": average,
                "sd_test_accuracy_percent": sd,
                "ci95_low_percent": average - half,
                "ci95_high_percent": average + half,
                "min_percent": min(values),
                "max_percent": max(values),
            }
        )
    return output


def outlier_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag robust-MAD extremes for inspection only; never exclude them."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["evidence_family"], row["role"])].append(row)
    output: list[dict[str, Any]] = []
    for (family, role), group in sorted(grouped.items()):
        values = np.asarray([float(row["test_accuracy_percent"]) for row in group])
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        for row, value in zip(group, values, strict=True):
            robust_z = 0.6745 * (float(value) - center) / mad if mad > 0 else 0.0
            output.append(
                {
                    "evidence_family": family,
                    "role": role,
                    "seed": row["seed"],
                    "test_accuracy_percent": float(value),
                    "median_percent": center,
                    "mad_percent": mad,
                    "robust_z": robust_z,
                    "flag_abs_robust_z_gt_3p5": abs(robust_z) > 3.5,
                    "exclusion_applied": False,
                }
            )
    return output


def main() -> None:
    """Create joined seed tables and all accuracy summaries."""
    vgg = vgg_depth_rows()
    scope = scope_rows()
    standard = standard_new_rows()
    all_rows = standard + vgg + scope

    comparisons = [
        paired_row(
            standard, "practical_main_reconstruction", "act", "lr", "reconstruction", "none"
        ),
        paired_row(standard, "practical_main_confirmatory", "act", "lr", "H1", "none"),
        paired_row(standard, "practical_main_confirmatory", "act", "ref", "secondary", "none"),
        paired_row(standard, "practical_main_confirmatory", "lr", "ref", "secondary", "none"),
        paired_row(standard, "practical_transfer", "act", "lr", "supportive H3", "none"),
        paired_row(standard, "practical_transfer", "act", "ref", "supportive", "none"),
        paired_row(standard, "practical_transfer", "lr", "ref", "supportive", "none"),
        paired_row(standard, "head_norm_cosine", "act", "lr", "exploratory", "head_norm"),
        paired_row(standard, "head_norm_preln", "act", "lr", "exploratory", "head_norm"),
    ]
    comparisons += [
        paired_row(vgg, f"vgg_depth_cut{cut}", "act", "lr", "descriptive", "vgg_depth")
        for cut in range(1, 16, 2)
    ]
    for family in ("cifar10", "nobn", "noise20", "noise50", "persistence", "stl10", "svhn"):
        comparisons.append(paired_row(scope, family, "act", "lr", "descriptive", "scope_act_lr"))
        comparisons.append(paired_row(scope, family, "act", "ref", "descriptive", "scope_act_ref"))
    for multiplicity_family in ("head_norm", "vgg_depth", "scope_act_lr", "scope_act_ref"):
        selected = [row for row in comparisons if row["multiplicity_family"] == multiplicity_family]
        adjusted = holm_adjust([float(row["exact_sign_flip_p"]) for row in selected])
        for row, value in zip(selected, adjusted, strict=True):
            row["holm_adjusted_p"] = value
            row["multiplicity_method"] = "Holm adjustment of exact paired sign-flip p-values"
            row["multiplicity_test_count"] = len(selected)
    for row in comparisons:
        row.setdefault("holm_adjusted_p", "")
        row.setdefault("multiplicity_method", "none")
        row.setdefault("multiplicity_test_count", 1)

    outputs = {
        WORKSPACE / "seed_tables/vgg_depth_validation_controlled_seed_level.csv": vgg,
        WORKSPACE / "seed_tables/scope_validation_controlled_seed_level.csv": scope,
        WORKSPACE / "tables/accuracy_condition_summary.csv": condition_summaries(all_rows),
        WORKSPACE / "analysis/statistics/accuracy_paired_summary.csv": comparisons,
        WORKSPACE / "analysis/statistics/accuracy_outlier_audit.csv": outlier_audit(all_rows),
    }
    bundle = write_accuracy_bundle(outputs)
    print(WORKSPACE / "analysis/statistics/accuracy_paired_summary.csv")
    print(bundle)


if __name__ == "__main__":
    main()
