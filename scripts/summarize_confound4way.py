#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from compute_update_pressure_diagnostics import condition_label, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the 4-way boundary/LR confound ablation.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--diagnostics-file", default="diagnostics_confound4way.csv")
    parser.add_argument("--output-dir", default="outputs/confound4way_summary")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SystemExit(f"No rows to write for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def numeric(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", "nan", "None", None} else float("nan")


def completed_runs(outputs_root: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    runs: list[tuple[str, Path, dict[str, Any]]] = []
    for config_path in sorted(outputs_root.glob("**/resolved_config.yaml")):
        run_dir = config_path.parent
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        config = load_yaml(config_path)
        label = condition_label(config, "confound4way")
        if label is not None:
            runs.append((label, run_dir, config))
    return runs


def run_metric_rows(runs: list[tuple[str, Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, run_dir, config in runs:
        metrics = read_csv(run_dir / "metrics.csv")
        if not metrics:
            continue
        best = max(metrics, key=lambda row: numeric(row, "test_accuracy"))
        final = metrics[-1]
        rows.append(
            {
                "condition": condition,
                "run_dir": str(run_dir),
                "experiment_name": config.get("experiment", {}).get("name"),
                "seed": config.get("training", {}).get("seed"),
                "best_epoch": best.get("epoch"),
                "best_test_accuracy": numeric(best, "test_accuracy"),
                "best_test_loss": numeric(best, "test_loss"),
                "final_epoch": final.get("epoch"),
                "final_train_accuracy": numeric(final, "train_accuracy"),
                "final_train_loss": numeric(final, "train_loss"),
                "final_test_accuracy": numeric(final, "test_accuracy"),
                "final_test_loss": numeric(final, "test_loss"),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], group_keys: list[str], metric_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        out = {name: value for name, value in zip(group_keys, key, strict=True)}
        out["num_runs"] = len(group_rows)
        for metric in metric_keys:
            values = [float(row[metric]) for row in group_rows]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        output.append(out)
    return output


def diagnostic_rows(runs: list[tuple[str, Path, dict[str, Any]]], diagnostics_file: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, run_dir, config in runs:
        path = run_dir / diagnostics_file
        if not path.exists():
            continue
        for row in read_csv(path):
            rows.append(
                {
                    "condition": condition,
                    "run_dir": str(run_dir),
                    "experiment_name": config.get("experiment", {}).get("name"),
                    "seed": config.get("training", {}).get("seed"),
                    "group": row.get("group"),
                    "cka_drift": numeric(row, "cka_drift"),
                    "effective_rank": numeric(row, "effective_rank"),
                    "clean_label_alignment": numeric(row, "clean_label_alignment"),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    runs = completed_runs(Path(args.outputs_root))
    expected = 20
    if len(runs) != expected:
        raise SystemExit(f"Expected {expected} confound4way runs, found {len(runs)}.")
    output_dir = Path(args.output_dir)
    metrics = run_metric_rows(runs)
    write_csv(output_dir / "confound4way_run_metrics.csv", metrics)
    write_csv(
        output_dir / "confound4way_metric_summary.csv",
        summarize(
            metrics,
            ["condition"],
            ["best_test_accuracy", "best_test_loss", "final_train_accuracy", "final_train_loss", "final_test_accuracy", "final_test_loss"],
        ),
    )
    diagnostics = diagnostic_rows(runs, args.diagnostics_file)
    if diagnostics:
        write_csv(output_dir / "confound4way_diagnostics.csv", diagnostics)
        write_csv(
            output_dir / "confound4way_diagnostics_summary.csv",
            summarize(diagnostics, ["condition", "group"], ["cka_drift", "effective_rank", "clean_label_alignment"]),
        )


if __name__ == "__main__":
    main()
