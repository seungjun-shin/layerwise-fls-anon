"""Build paper-facing evidence summaries from generated experiment tables.

The script consumes the CSV tables produced by ``scripts/make_plots.py`` and
writes compact paired summaries for the claims that need publication-facing
statistical support.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=Path("paper/tables"))
    return parser.parse_args()


def _close(series: pd.Series, value: float, atol: float = 1e-9) -> pd.Series:
    return series.astype(float).sub(value).abs() <= atol


def _format_seeds(seeds: list[int]) -> str:
    return ",".join(str(seed) for seed in sorted(seeds))


def _condition_rows(
    runs: pd.DataFrame,
    *,
    experiment_name: str,
    base_lr: float,
    mode: str,
    global_c: float | None = None,
    after_late_c: float | None = None,
) -> pd.DataFrame:
    rows = runs[
        (runs["experiment_name"] == experiment_name)
        & _close(runs["base_lr"], base_lr)
        & (runs["fls_mode"] == mode)
    ].copy()
    if global_c is not None:
        rows = rows[_close(rows["global_output_multiplier"], global_c)]
    if after_late_c is not None:
        rows = rows[_close(rows["after_late_multiplier"], after_late_c)]
    rows = rows.sort_values(["seed", "best_test_accuracy"], ascending=[True, False])
    return rows.drop_duplicates(subset=["seed"], keep="first")


def _paired_run_rows(
    baseline: pd.DataFrame,
    treatment: pd.DataFrame,
    *,
    baseline_label: str,
    treatment_label: str,
) -> pd.DataFrame:
    keep = [
        "seed",
        "base_lr",
        "learning_rate",
        "final_test_accuracy",
        "best_test_accuracy",
        "final_test_loss",
        "best_test_loss",
        "best_epoch",
    ]
    left = baseline[keep].rename(columns={col: f"{baseline_label}_{col}" for col in keep if col != "seed"})
    right = treatment[keep].rename(columns={col: f"{treatment_label}_{col}" for col in keep if col != "seed"})
    paired = left.merge(right, on="seed", how="inner").sort_values("seed")
    paired["best_accuracy_gain_pp"] = (
        paired[f"{treatment_label}_best_test_accuracy"]
        - paired[f"{baseline_label}_best_test_accuracy"]
    ) * 100.0
    paired["final_accuracy_gain_pp"] = (
        paired[f"{treatment_label}_final_test_accuracy"]
        - paired[f"{baseline_label}_final_test_accuracy"]
    ) * 100.0
    paired["best_loss_delta"] = (
        paired[f"{treatment_label}_best_test_loss"]
        - paired[f"{baseline_label}_best_test_loss"]
    )
    paired["final_loss_delta"] = (
        paired[f"{treatment_label}_final_test_loss"]
        - paired[f"{baseline_label}_final_test_loss"]
    )
    return paired


def _paired_stats(
    paired: pd.DataFrame,
    *,
    label: str,
    seed_subset: list[int],
    treatment_label: str,
    baseline_label: str,
    metrics: list[tuple[str, str, float]],
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    subset = paired[paired["seed"].isin(seed_subset)].copy()
    for metric_name, diff_col, scale in metrics:
        diffs = subset[diff_col].dropna().astype(float).to_numpy()
        n = int(diffs.size)
        if n == 0:
            continue
        mean = float(np.mean(diffs))
        std = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
        sem = std / float(np.sqrt(n)) if n > 1 else 0.0
        if n > 1:
            ci_low, ci_high = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
            t_stat, t_pvalue = stats.ttest_1samp(diffs, 0.0)
            try:
                _, wilcoxon_pvalue = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                wilcoxon_pvalue = np.nan
            cohen_dz = mean / std if std > 0 else np.nan
        else:
            ci_low = ci_high = mean
            t_stat = t_pvalue = wilcoxon_pvalue = cohen_dz = np.nan
        rows.append(
            {
                "comparison": f"{treatment_label}_minus_{baseline_label}",
                "subset": label,
                "metric": metric_name,
                "n": n,
                "seeds": _format_seeds(subset["seed"].astype(int).tolist()),
                "mean_diff": mean / scale,
                "std_diff": std / scale,
                "sem_diff": sem / scale,
                "ci95_low": float(ci_low) / scale,
                "ci95_high": float(ci_high) / scale,
                "paired_t": float(t_stat) if np.isfinite(t_stat) else np.nan,
                "paired_t_pvalue": float(t_pvalue) if np.isfinite(t_pvalue) else np.nan,
                "wilcoxon_pvalue": float(wilcoxon_pvalue) if np.isfinite(wilcoxon_pvalue) else np.nan,
                "cohen_dz": float(cohen_dz) if np.isfinite(cohen_dz) else np.nan,
            }
        )
    return rows


def _write_clean_matched(runs: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    global_rows = _condition_rows(
        runs,
        experiment_name="experiment_49_global_matched_seed_expansion",
        base_lr=0.04096,
        mode="global",
        global_c=0.25,
    )
    boundary_rows = _condition_rows(
        runs,
        experiment_name="experiment_40_boundary_late_seed_expansion_after_late",
        base_lr=0.04096,
        mode="location",
        after_late_c=0.0625,
    )
    paired = _paired_run_rows(global_rows, boundary_rows, baseline_label="global", treatment_label="late_boundary")
    paired.to_csv(tables_dir / "matched_clean_seed_summary.csv", index=False)

    metrics = [
        ("best_accuracy_gain_pp", "best_accuracy_gain_pp", 1.0),
        ("final_accuracy_gain_pp", "final_accuracy_gain_pp", 1.0),
        ("best_loss_delta", "best_loss_delta", 1.0),
        ("final_loss_delta", "final_loss_delta", 1.0),
    ]
    seeds = sorted(paired["seed"].astype(int).tolist())
    stats_rows = _paired_stats(
        paired,
        label="all_matched",
        seed_subset=seeds,
        treatment_label="late_boundary",
        baseline_label="global",
        metrics=metrics,
    )
    if 0 in seeds and len(seeds) > 1:
        stats_rows.extend(
            _paired_stats(
                paired,
                label="seed0_excluded_confirmation",
                seed_subset=[seed for seed in seeds if seed != 0],
                treatment_label="late_boundary",
                baseline_label="global",
                metrics=metrics,
            )
        )
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(tables_dir / "matched_clean_statistics.csv", index=False)
    return stats_df


def _write_diagnostic_matched(diagnostics: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    metrics = ["cka_drift", "effective_rank", "feature_norm", "clean_label_alignment"]
    base = diagnostics[
        (diagnostics["experiment_name"] == "experiment_49_global_matched_seed_expansion")
        & (diagnostics["group"].isin(["early", "middle", "late", "head"]))
    ][["seed", "group", *metrics]]
    late = diagnostics[
        (diagnostics["experiment_name"] == "experiment_40_boundary_late_seed_expansion_after_late")
        & (diagnostics["group"].isin(["early", "middle", "late", "head"]))
    ][["seed", "group", *metrics]]
    paired = base.merge(late, on=["seed", "group"], suffixes=("_global", "_late_boundary"))
    for metric in metrics:
        paired[f"{metric}_delta"] = paired[f"{metric}_late_boundary"] - paired[f"{metric}_global"]
    paired = paired.sort_values(["group", "seed"])
    paired.to_csv(tables_dir / "matched_clean_diagnostics_seed_summary.csv", index=False)

    rows: list[dict[str, float | str | int]] = []
    seeds = sorted(paired["seed"].astype(int).unique().tolist())
    for group, group_df in paired.groupby("group"):
        metric_specs = [(f"{metric}_delta", f"{metric}_delta", 1.0) for metric in metrics]
        rows.extend(
            _paired_stats(
                group_df,
                label=f"{group}_all_matched",
                seed_subset=seeds,
                treatment_label="late_boundary",
                baseline_label="global",
                metrics=metric_specs,
            )
        )
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(tables_dir / "matched_clean_diagnostics_statistics.csv", index=False)
    return stats_df


def _write_vgg_matched(runs: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    global_rows = _condition_rows(
        runs,
        experiment_name="experiment_47_vgg19_bn_global_reference",
        base_lr=0.01024,
        mode="global",
        global_c=0.0625,
    )
    boundary_rows = _condition_rows(
        runs,
        experiment_name="experiment_48_vgg19_bn_boundary_sensitivity_after_late_lr_0p01024",
        base_lr=0.01024,
        mode="location",
        after_late_c=0.0625,
    )
    paired = _paired_run_rows(global_rows, boundary_rows, baseline_label="global", treatment_label="late_boundary")
    paired.to_csv(tables_dir / "vgg19_bn_matched_seed_summary.csv", index=False)
    stats_rows = _paired_stats(
        paired,
        label="all_matched",
        seed_subset=sorted(paired["seed"].astype(int).tolist()),
        treatment_label="late_boundary",
        baseline_label="global",
        metrics=[
            ("best_accuracy_gain_pp", "best_accuracy_gain_pp", 1.0),
            ("final_accuracy_gain_pp", "final_accuracy_gain_pp", 1.0),
            ("best_loss_delta", "best_loss_delta", 1.0),
            ("final_loss_delta", "final_loss_delta", 1.0),
        ],
    )
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(tables_dir / "vgg19_bn_matched_statistics.csv", index=False)
    return stats_df


def main() -> None:
    args = parse_args()
    tables_dir = args.tables
    runs = pd.read_csv(tables_dir / "all_runs_summary.csv")
    diagnostics = pd.read_csv(tables_dir / "diagnostics_summary.csv")
    clean_stats = _write_clean_matched(runs, tables_dir)
    diag_stats = _write_diagnostic_matched(diagnostics, tables_dir)
    vgg_stats = _write_vgg_matched(runs, tables_dir)
    print(f"wrote clean matched statistics: {len(clean_stats)} rows")
    print(f"wrote diagnostic matched statistics: {len(diag_stats)} rows")
    print(f"wrote VGG19-BN matched statistics: {len(vgg_stats)} rows")


if __name__ == "__main__":
    main()
