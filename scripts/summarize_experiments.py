#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize completed experiment runs.")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def final_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: int(float(row.get("epoch", -1) or -1)))


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: float(row.get("test_accuracy", 0.0) or 0.0))


def first_epoch_reaching(rows: list[dict[str, Any]], key: str, threshold: float) -> int | None:
    for row in sorted(rows, key=lambda item: int(float(item.get("epoch", -1) or -1))):
        if float(row.get(key, 0.0) or 0.0) >= threshold:
            return int(float(row.get("epoch", -1) or -1))
    return None


def group_values(config: dict[str, Any]) -> dict[str, float | None]:
    fls = config.get("fls", {})
    if fls.get("mode") == "global":
        value = fls.get("global", {}).get("output_multiplier")
        return {"global": value, "early": None, "middle": None, "late": None, "head": None}
    groups = fls.get("groups") or {}
    return {
        "global": None,
        "early": (groups.get("early") or {}).get("output_multiplier"),
        "middle": (groups.get("middle") or {}).get("output_multiplier"),
        "late": (groups.get("late") or {}).get("output_multiplier"),
        "head": (groups.get("head") or {}).get("output_multiplier"),
    }


def run_record(resolved_path: Path) -> dict[str, Any] | None:
    metrics_path = resolved_path.parent / "metrics.csv"
    if not metrics_path.exists():
        return None
    rows = read_metrics(metrics_path)
    if not rows:
        return None
    with resolved_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    final = final_row(rows)
    best = best_row(rows)
    groups = group_values(config)
    return {
        "experiment_name": config.get("experiment", {}).get("name"),
        "run_dir": str(resolved_path.parent),
        "seed": config.get("training", {}).get("seed"),
        "dataset": config.get("data", {}).get("name"),
        "subset": config.get("data", {}).get("train_subset_fraction"),
        "label_noise": (config.get("label_noise") or {}).get("rate"),
        "model": config.get("model", {}).get("name"),
        "use_bn": config.get("model", {}).get("use_bn"),
        "freeze_bn_affine": config.get("model", {}).get("freeze_bn_affine", False),
        "freeze_bn_stats": config.get("model", {}).get("freeze_bn_stats", False),
        "fls_mode": config.get("fls", {}).get("mode"),
        "profile_name": config.get("fls", {}).get("profile_name"),
        "lr": config.get("training", {}).get("lr"),
        "max_epoch": final.get("epoch"),
        "final_train_accuracy": final.get("train_accuracy"),
        "final_test_accuracy": final.get("test_accuracy"),
        "best_epoch": best.get("epoch"),
        "best_test_accuracy": best.get("test_accuracy"),
        "first_epoch_train_acc_99": first_epoch_reaching(rows, "train_accuracy", 0.99),
        "groups_json": json.dumps(groups, sort_keys=True),
        **groups,
    }


def matched_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["dataset"],
        row["subset"],
        row["label_noise"],
        row["model"],
        row["use_bn"],
        row["freeze_bn_affine"],
        row["freeze_bn_stats"],
        row["seed"],
    )


def add_global_gaps(rows: list[dict[str, Any]]) -> None:
    best_global: dict[tuple[Any, ...], float] = {}
    for row in rows:
        if row["fls_mode"] != "global":
            continue
        key = matched_key(row)
        value = float(row["final_test_accuracy"])
        best_global[key] = max(best_global.get(key, float("-inf")), value)
    for row in rows:
        key = matched_key(row)
        baseline = best_global.get(key)
        row["matched_global_final_accuracy"] = "" if baseline is None else baseline
        row["matched_global_gap"] = "" if baseline is None else float(row["final_test_accuracy"]) - baseline


def main() -> None:
    args = parse_args()
    outputs = Path(args.outputs)
    rows = []
    for resolved_path in outputs.glob("**/resolved_config.yaml"):
        record = run_record(resolved_path)
        if record is None:
            continue
        if args.prefix and not any(str(record["experiment_name"]).startswith(prefix) for prefix in args.prefix):
            continue
        rows.append(record)
    add_global_gaps(rows)
    rows.sort(key=lambda row: float(row["best_test_accuracy"]), reverse=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    for row in rows[: args.top_k]:
        gap = row["matched_global_gap"]
        gap_text = "" if gap == "" else f" gap={float(gap) * 100:.2f}pp"
        print(
            f"best={float(row['best_test_accuracy']) * 100:.2f}%"
            f" final={float(row['final_test_accuracy']) * 100:.2f}%"
            f"{gap_text} exp={row['experiment_name']} seed={row['seed']} lr={row['lr']}"
            f" mode={row['fls_mode']} c={row['global']} groups={row['groups_json']}"
        )


if __name__ == "__main__":
    main()
