#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml


TARGET_PREFIXES = (
    "experiment_82_practical_early_narrow_c_lr",
    "experiment_83_practical_global_narrow_c_lr",
    "experiment_84_practical_middle_late_identity_neighborhood",
    "experiment_85_practical_global_seed_expansion",
    "experiment_86_practical_late_seed_expansion",
    "experiment_87_practical_early_seed_expansion",
)

TARGET_CONDITIONS = {
    ("global", 1.0, 0.00384): "global_c_1_lr_0p00384",
    ("global", 1.25, 0.00384): "global_c_1p25_lr_0p00384",
    ("after_late", 1.25, 0.00512): "after_late_c_1p25_lr_0p00512",
    ("after_late", 1.5, 0.00512): "after_late_c_1p5_lr_0p00512",
    ("after_early", 0.375, 0.00384): "after_early_c_0p375_lr_0p00384",
}

BASELINES = (
    "global_c_1_lr_0p00384",
    "global_c_1p25_lr_0p00384",
)
TREATMENTS = (
    "after_late_c_1p25_lr_0p00512",
    "after_late_c_1p5_lr_0p00512",
    "after_early_c_0p375_lr_0p00384",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize practical seed expansion experiments.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/practical_seed_expansion_summary")
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


def raw_condition(config: dict[str, Any]) -> tuple[str, float, float, float]:
    fls = config.get("fls", {})
    lr = float(config["training"]["lr"])
    if fls.get("mode") == "global":
        c_value = float(fls.get("global", {}).get("output_multiplier", 1.0))
        return "global", c_value, lr, lr / c_value
    boundaries = fls.get("boundaries", {})
    for boundary in ["after_early", "after_middle", "after_late"]:
        c_value = float(boundaries.get(boundary, {}).get("output_multiplier", 1.0))
        if not math.isclose(c_value, 1.0, rel_tol=0.0, abs_tol=1e-12):
            return boundary, c_value, lr, lr / c_value
    return "identity_boundary", 1.0, lr, lr


def condition_label(family: str, c_value: float, base_lr: float) -> str | None:
    for (target_family, target_c, target_lr), label in TARGET_CONDITIONS.items():
        if (
            family == target_family
            and math.isclose(c_value, target_c, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(base_lr, target_lr, rel_tol=0.0, abs_tol=1e-12)
        ):
            return label
    return None


def collect_rows(outputs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_path in sorted(outputs_root.glob("experiment_*/**/resolved_config.yaml")):
        run_dir = config_path.parent
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        exp_name = str(config.get("experiment", {}).get("name", ""))
        base = base_experiment(exp_name)
        if base is None:
            continue
        family, c_value, base_lr, effective_lr = raw_condition(config)
        label = condition_label(family, c_value, base_lr)
        if label is None:
            continue
        metrics = read_csv(metrics_path)
        if not metrics:
            continue
        best = max(metrics, key=lambda row: numeric(row, "test_accuracy"))
        final = metrics[-1]
        rows.append(
            {
                "condition": label,
                "family": family,
                "base_experiment": base,
                "experiment_name": exp_name,
                "seed": int(config["training"]["seed"]),
                "base_lr": base_lr,
                "output_multiplier": c_value,
                "effective_lr_reference": effective_lr,
                "best_epoch": int(float(best["epoch"])),
                "best_test_accuracy": numeric(best, "test_accuracy"),
                "best_test_loss": numeric(best, "test_loss"),
                "final_epoch": int(float(final["epoch"])),
                "final_train_accuracy": numeric(final, "train_accuracy"),
                "final_train_loss": numeric(final, "train_loss"),
                "final_test_accuracy": numeric(final, "test_accuracy"),
                "final_test_loss": numeric(final, "test_loss"),
                "run_dir": str(run_dir),
            }
        )
    return dedupe_rows(rows)


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (str(item["base_experiment"]), str(item["run_dir"]))):
        key = (str(row["condition"]), int(row["seed"]))
        if key not in keyed:
            keyed[key] = row
            continue
        previous = keyed[key]
        if str(row["base_experiment"]).startswith("experiment_8") and not str(previous["base_experiment"]).startswith(
            "experiment_8"
        ):
            keyed[key] = row
    return sorted(keyed.values(), key=lambda row: (str(row["condition"]), int(row["seed"])))


def metric_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    metrics = [
        "best_test_accuracy",
        "best_test_loss",
        "final_train_accuracy",
        "final_train_loss",
        "final_test_accuracy",
        "final_test_loss",
    ]
    output: list[dict[str, Any]] = []
    for condition, group_rows in sorted(grouped.items()):
        first = group_rows[0]
        summary: dict[str, Any] = {
            "condition": condition,
            "family": first["family"],
            "output_multiplier": first["output_multiplier"],
            "base_lr": first["base_lr"],
            "effective_lr_reference": first["effective_lr_reference"],
            "num_runs": len(group_rows),
            "seeds": "|".join(str(item["seed"]) for item in sorted(group_rows, key=lambda item: int(item["seed"]))),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in group_rows]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        output.append(summary)
    return sorted(output, key=lambda row: float(row["best_test_accuracy_mean"]), reverse=True)


def scipy_paired_pvalue(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return None
    result = stats.ttest_1samp(values, popmean=0.0)
    return float(result.pvalue)


def paired_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_condition: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_condition[str(row["condition"])][int(row["seed"])] = row

    seed_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    for treatment in TREATMENTS:
        for baseline in BASELINES:
            shared_seeds = sorted(set(by_condition[treatment]) & set(by_condition[baseline]))
            diffs_by_metric: dict[str, list[float]] = {
                "best_accuracy_gain_pp": [],
                "final_accuracy_gain_pp": [],
                "best_loss_delta": [],
                "final_loss_delta": [],
            }
            for seed in shared_seeds:
                treated = by_condition[treatment][seed]
                base = by_condition[baseline][seed]
                values = {
                    "best_accuracy_gain_pp": 100.0
                    * (float(treated["best_test_accuracy"]) - float(base["best_test_accuracy"])),
                    "final_accuracy_gain_pp": 100.0
                    * (float(treated["final_test_accuracy"]) - float(base["final_test_accuracy"])),
                    "best_loss_delta": float(treated["best_test_loss"]) - float(base["best_test_loss"]),
                    "final_loss_delta": float(treated["final_test_loss"]) - float(base["final_test_loss"]),
                }
                seed_rows.append(
                    {
                        "treatment": treatment,
                        "baseline": baseline,
                        "seed": seed,
                        **values,
                    }
                )
                for key, value in values.items():
                    diffs_by_metric[key].append(value)
            stat: dict[str, Any] = {
                "treatment": treatment,
                "baseline": baseline,
                "num_paired_seeds": len(shared_seeds),
                "paired_seeds": "|".join(str(seed) for seed in shared_seeds),
            }
            for metric, values in diffs_by_metric.items():
                if values:
                    std = stdev(values) if len(values) > 1 else 0.0
                    sem = std / math.sqrt(len(values)) if values else 0.0
                    stat[f"{metric}_mean"] = mean(values)
                    stat[f"{metric}_std"] = std
                    stat[f"{metric}_sem"] = sem
                    stat[f"{metric}_ci95_halfwidth"] = 1.96 * sem if len(values) > 1 else 0.0
                    pvalue = scipy_paired_pvalue(values)
                    stat[f"{metric}_paired_t_pvalue"] = "" if pvalue is None else pvalue
                else:
                    stat[f"{metric}_mean"] = ""
                    stat[f"{metric}_std"] = ""
                    stat[f"{metric}_sem"] = ""
                    stat[f"{metric}_ci95_halfwidth"] = ""
                    stat[f"{metric}_paired_t_pvalue"] = ""
            stat_rows.append(stat)
    return seed_rows, stat_rows


def main() -> None:
    args = parse_args()
    rows = collect_rows(Path(args.outputs_root))
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "practical_seed_expansion_run_metrics.csv", rows)
    write_csv(output_dir / "practical_seed_expansion_metric_summary.csv", metric_summary(rows))
    seed_rows, stat_rows = paired_rows(rows)
    write_csv(output_dir / "practical_seed_expansion_paired_seed_summary.csv", seed_rows)
    write_csv(output_dir / "practical_seed_expansion_paired_statistics.csv", stat_rows)


if __name__ == "__main__":
    main()
