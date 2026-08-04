#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml


TARGET_PREFIXES = (
    "experiment_79_practical_boundary_location_c_lr_rescue",
    "experiment_81_practical_global_c_lr_reference",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize practical rescue calibration grids.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/practical_rescue_summary")
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
    return float(row[key])


def base_experiment(name: str) -> str | None:
    for prefix in TARGET_PREFIXES:
        if name.startswith(prefix):
            return prefix
    return None


def condition(config: dict[str, Any]) -> tuple[str, float, float, float]:
    fls = config.get("fls", {})
    lr = float(config["training"]["lr"])
    if fls.get("mode") == "global":
        c = float(fls.get("global", {}).get("output_multiplier", 1.0))
        return "global", c, lr, lr / c
    boundaries = fls.get("boundaries", {})
    for boundary in ["after_early", "after_middle", "after_late"]:
        value = float(boundaries.get(boundary, {}).get("output_multiplier", 1.0))
        if abs(value - 1.0) > 1e-12:
            return boundary, value, lr, lr / value
    return "identity_boundary", 1.0, lr, lr


def collect_rows(outputs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_path in sorted(outputs_root.glob("experiment_*/**/resolved_config.yaml")):
        run_dir = config_path.parent
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        exp_name = config.get("experiment", {}).get("name", "")
        base = base_experiment(str(exp_name))
        if base is None:
            continue
        metrics = read_csv(metrics_path)
        if not metrics:
            continue
        best = max(metrics, key=lambda row: numeric(row, "test_accuracy"))
        final = metrics[-1]
        family, c_value, base_lr, effective_lr = condition(config)
        rows.append(
            {
                "family": family,
                "base_experiment": base,
                "experiment_name": exp_name,
                "seed": config["training"]["seed"],
                "base_lr": base_lr,
                "output_multiplier": c_value,
                "effective_lr_reference": effective_lr,
                "best_epoch": best["epoch"],
                "best_test_accuracy": numeric(best, "test_accuracy"),
                "best_test_loss": numeric(best, "test_loss"),
                "final_epoch": final["epoch"],
                "final_train_accuracy": numeric(final, "train_accuracy"),
                "final_train_loss": numeric(final, "train_loss"),
                "final_test_accuracy": numeric(final, "test_accuracy"),
                "final_test_loss": numeric(final, "test_loss"),
                "run_dir": str(run_dir),
            }
        )
    return sorted(rows, key=lambda row: (row["family"], row["output_multiplier"], row["base_lr"], row["seed"]))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), float(row["output_multiplier"]), float(row["base_lr"]))].append(row)
    metrics = [
        "best_test_accuracy",
        "best_test_loss",
        "final_train_accuracy",
        "final_train_loss",
        "final_test_accuracy",
        "final_test_loss",
    ]
    output: list[dict[str, Any]] = []
    for (family, c_value, base_lr), group_rows in sorted(grouped.items()):
        row: dict[str, Any] = {
            "family": family,
            "output_multiplier": c_value,
            "base_lr": base_lr,
            "effective_lr_reference": base_lr / c_value,
            "num_runs": len(group_rows),
            "seeds": "|".join(str(item["seed"]) for item in sorted(group_rows, key=lambda item: int(item["seed"]))),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in group_rows]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        output.append(row)
    return sorted(output, key=lambda row: float(row["best_test_accuracy_mean"]), reverse=True)


def main() -> None:
    args = parse_args()
    rows = collect_rows(Path(args.outputs_root))
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "practical_rescue_run_metrics.csv", rows)
    write_csv(output_dir / "practical_rescue_metric_summary.csv", summarize(rows))


if __name__ == "__main__":
    main()
