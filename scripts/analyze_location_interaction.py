#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import stats


GROUPS = ["early", "middle", "late", "head"]
BOUNDARIES = ["after_early", "after_middle", "after_late"]
METRICS = ["final_test_accuracy", "final_test_loss", "best_test_accuracy", "best_test_loss"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze location-by-output-multiplier interactions.")
    parser.add_argument("--outputs", default="outputs", help="Root directory containing experiment outputs.")
    parser.add_argument("--prefix", action="append", default=[], help="Experiment-name prefix to include.")
    parser.add_argument("--out-dir", default="paper/tables", help="Directory for analysis CSV outputs.")
    parser.add_argument("--min-rows", type=int, default=8, help="Minimum rows required for a metric test.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(rows, key=lambda row: float(row.get("test_accuracy", 0.0) or 0.0))


def min_loss_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda row: float(row.get("test_loss", float("inf")) or float("inf")))


def one_site_location_and_c(config: dict[str, Any]) -> tuple[str, float] | None:
    fls = config.get("fls", {})
    mode = fls.get("mode")
    if mode == "global":
        return "global", float(fls.get("global", {}).get("output_multiplier", 1.0))
    if mode == "location":
        boundaries = fls.get("boundaries") or {}
        varied: list[tuple[str, float]] = []
        for boundary in BOUNDARIES:
            values = boundaries.get(boundary) or {}
            c = float(values.get("output_multiplier", 1.0))
            if abs(c - 1.0) > 1e-12:
                varied.append((boundary, c))
        if len(varied) == 1:
            return varied[0]
        if not varied:
            return "identity", 1.0
        return None
    if mode != "blockwise":
        return None
    groups = fls.get("groups") or {}
    varied: list[tuple[str, float]] = []
    for group in GROUPS:
        values = groups.get(group) or {}
        c = float(values.get("output_multiplier", 1.0))
        if abs(c - 1.0) > 1e-12:
            varied.append((group, c))
    if len(varied) == 1:
        return varied[0]
    if not varied:
        return "identity", 1.0
    return None


def run_record(resolved_config: Path) -> dict[str, Any] | None:
    metrics_path = resolved_config.parent / "metrics.csv"
    if not metrics_path.exists():
        return None
    rows = read_metrics(metrics_path)
    if not rows:
        return None
    config = load_yaml(resolved_config)
    location_c = one_site_location_and_c(config)
    if location_c is None:
        return None
    location, c = location_c
    final = rows[-1]
    best = best_row(rows)
    min_loss = min_loss_row(rows)
    return {
        "run_dir": str(resolved_config.parent),
        "experiment_name": config.get("experiment", {}).get("name"),
        "dataset": config.get("data", {}).get("name"),
        "model": config.get("model", {}).get("name"),
        "seed": config.get("training", {}).get("seed"),
        "lr": config.get("training", {}).get("lr"),
        "lr_compensation": config.get("training", {}).get("blockwise_lr_compensation", ""),
        "location": location,
        "c": c,
        "final_test_accuracy": float(final.get("test_accuracy", "nan")),
        "final_test_loss": float(final.get("test_loss", "nan")),
        "best_test_accuracy": float(best.get("test_accuracy", "nan")),
        "best_test_loss": float(min_loss.get("test_loss", "nan")),
    }


def collect_records(outputs: Path, prefixes: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for resolved_config in outputs.glob("**/resolved_config.yaml"):
        record = run_record(resolved_config)
        if record is None:
            continue
        name = str(record["experiment_name"])
        if prefixes and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        records.append(record)
    return records


def design_matrix(rows: list[dict[str, Any]], include_interaction: bool) -> np.ndarray:
    locations = sorted({str(row["location"]) for row in rows})
    cs = sorted({float(row["c"]) for row in rows})
    columns = [np.ones(len(rows))]
    for location in locations[1:]:
        columns.append(np.array([1.0 if row["location"] == location else 0.0 for row in rows]))
    for c in cs[1:]:
        columns.append(np.array([1.0 if float(row["c"]) == c else 0.0 for row in rows]))
    if include_interaction:
        for location in locations[1:]:
            for c in cs[1:]:
                columns.append(
                    np.array([1.0 if row["location"] == location and float(row["c"]) == c else 0.0 for row in rows])
                )
    return np.column_stack(columns)


def residual_sum_squares(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    rank = int(np.linalg.matrix_rank(x))
    return float(np.dot(residuals, residuals)), rank


def interaction_test(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    clean_rows = [row for row in rows if np.isfinite(float(row[metric]))]
    if len(clean_rows) < 2:
        return None
    y = np.array([float(row[metric]) for row in clean_rows], dtype=float)
    x_additive = design_matrix(clean_rows, include_interaction=False)
    x_full = design_matrix(clean_rows, include_interaction=True)
    rss_additive, rank_additive = residual_sum_squares(x_additive, y)
    rss_full, rank_full = residual_sum_squares(x_full, y)
    df_num = rank_full - rank_additive
    df_den = len(clean_rows) - rank_full
    if df_num <= 0 or df_den <= 0:
        return None
    numerator = max(rss_additive - rss_full, 0.0) / df_num
    denominator = rss_full / df_den
    f_stat = float("inf") if denominator == 0 else numerator / denominator
    p_value = 0.0 if np.isinf(f_stat) else float(stats.f.sf(f_stat, df_num, df_den))
    return {
        "metric": metric,
        "n": len(clean_rows),
        "locations": "|".join(sorted({str(row["location"]) for row in clean_rows})),
        "c_values": "|".join(str(value) for value in sorted({float(row["c"]) for row in clean_rows})),
        "rss_additive": rss_additive,
        "rss_full": rss_full,
        "df_num": df_num,
        "df_den": df_den,
        "f_stat": f_stat,
        "p_value": p_value,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    records = collect_records(Path(args.outputs), args.prefix)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "location_c_curve_rows.csv", records)
    tests = [result for metric in METRICS if (result := interaction_test(records, metric)) is not None]
    tests = [row for row in tests if int(row["n"]) >= args.min_rows]
    write_csv(out_dir / "location_c_interaction_tests.csv", tests)
    for row in tests:
        print(
            f"{row['metric']}: n={row['n']} F={row['f_stat']:.4g} "
            f"p={row['p_value']:.4g} locations={row['locations']} c={row['c_values']}"
        )


if __name__ == "__main__":
    main()
