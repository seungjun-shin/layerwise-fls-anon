#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import pandas as pd
import yaml


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment outputs for paper tables and figures.")
    parser.add_argument("--outputs", default="outputs", help="Root directory containing experiment runs.")
    parser.add_argument("--paper", default="paper", help="Paper directory for figures and tables.")
    parser.add_argument(
        "--from-existing-tables",
        action="store_true",
        help="Regenerate figures from existing paper/tables CSV files without scanning outputs.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _flatten_groups(groups: Any) -> dict[str, float | None]:
    result = {f"{name}_multiplier": None for name in ["early", "middle", "late", "head"]}
    if not isinstance(groups, dict):
        return result
    for name in ["early", "middle", "late", "head"]:
        value = groups.get(name)
        if isinstance(value, dict):
            result[f"{name}_multiplier"] = value.get("output_multiplier")
    return result


def _flatten_boundaries(boundaries: Any) -> dict[str, float | None]:
    result = {f"{name}_multiplier": None for name in ["after_early", "after_middle", "after_late"]}
    if not isinstance(boundaries, dict):
        return result
    for name in ["after_early", "after_middle", "after_late"]:
        value = boundaries.get(name)
        if isinstance(value, dict):
            result[f"{name}_multiplier"] = value.get("output_multiplier")
    return result


def collect_runs(outputs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(outputs_root.glob("**/metrics.csv")):
        try:
            metrics = pd.read_csv(metrics_path)
        except pd.errors.EmptyDataError:
            continue
        if metrics.empty:
            continue
        run_dir = metrics_path.parent
        config = _load_yaml(run_dir / "resolved_config.yaml")
        final = metrics.iloc[-1].to_dict()
        best_idx = metrics["test_accuracy"].idxmax() if "test_accuracy" in metrics else metrics.index[-1]
        best = metrics.loc[best_idx].to_dict()
        fls_cfg = config.get("fls", {})
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        training_cfg = config.get("training", {})
        row = {
            "run_dir": str(run_dir),
            "metrics_csv": str(metrics_path),
            "experiment_name": final.get("experiment_name") or config.get("experiment", {}).get("name"),
            "dataset": data_cfg.get("name"),
            "model": model_cfg.get("name"),
            "width": model_cfg.get("width"),
            "use_bn": model_cfg.get("use_bn"),
            "seed": final.get("seed", training_cfg.get("seed")),
            "epoch": final.get("epoch"),
            "num_epochs_recorded": len(metrics),
            "optimizer": training_cfg.get("optimizer"),
            "base_lr": training_cfg.get("lr"),
            "learning_rate": final.get("learning_rate"),
            "label_noise_rate": config.get("label_noise", {}).get("rate"),
            "fls_mode": fls_cfg.get("mode"),
            "global_output_multiplier": final.get(
                "global_output_multiplier", fls_cfg.get("global", {}).get("output_multiplier")
            ),
            "profile_name": final.get("blockwise_profile_name", fls_cfg.get("profile_name")),
            "final_train_loss": final.get("train_loss"),
            "final_train_accuracy": final.get("train_accuracy"),
            "final_test_loss": final.get("test_loss"),
            "final_test_accuracy": final.get("test_accuracy"),
            "best_epoch": best.get("epoch"),
            "best_test_accuracy": best.get("test_accuracy"),
            "best_test_loss": best.get("test_loss"),
            "wall_clock_time": final.get("wall_clock_time"),
        }
        row.update(_flatten_groups(fls_cfg.get("groups")))
        row.update(_flatten_boundaries(fls_cfg.get("boundaries")))
        rows.append(row)
    return pd.DataFrame(rows)


def collect_diagnostics(outputs_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for diagnostics_path in sorted(outputs_root.glob("**/diagnostics.csv")):
        try:
            df = pd.read_csv(diagnostics_path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df["diagnostics_csv"] = str(diagnostics_path)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_matched_train_accuracy(outputs_root: Path, thresholds: tuple[float, ...] = (0.2, 0.5, 0.8, 0.99)) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(outputs_root.glob("**/metrics.csv")):
        try:
            metrics = pd.read_csv(metrics_path)
        except pd.errors.EmptyDataError:
            continue
        if metrics.empty or "train_accuracy" not in metrics:
            continue
        run_dir = metrics_path.parent
        config = _load_yaml(run_dir / "resolved_config.yaml")
        for threshold in thresholds:
            reached = metrics[pd.to_numeric(metrics["train_accuracy"], errors="coerce") >= threshold]
            if reached.empty:
                continue
            row = reached.iloc[0].to_dict()
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "threshold": threshold,
                    "experiment_name": row.get("experiment_name") or config.get("experiment", {}).get("name"),
                    "dataset": config.get("data", {}).get("name"),
                    "model": config.get("model", {}).get("name"),
                    "seed": row.get("seed", config.get("training", {}).get("seed")),
                    "epoch_at_threshold": row.get("epoch"),
                    "step_at_threshold": row.get("step"),
                    "train_accuracy": row.get("train_accuracy"),
                    "test_accuracy": row.get("test_accuracy"),
                    "test_loss": row.get("test_loss"),
                    "fls_mode": config.get("fls", {}).get("mode"),
                    "profile_name": row.get("blockwise_profile_name", config.get("fls", {}).get("profile_name")),
                    "global_output_multiplier": row.get(
                        "global_output_multiplier", config.get("fls", {}).get("global", {}).get("output_multiplier")
                    ),
                    "label_noise_rate": config.get("label_noise", {}).get("rate"),
                    **_flatten_groups(config.get("fls", {}).get("groups")),
                    **_flatten_boundaries(config.get("fls", {}).get("boundaries")),
                }
            )
    return pd.DataFrame(rows)


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _format_power_of_two(value: float) -> str:
    if value <= 0 or not math.isfinite(value):
        return f"{value:g}"
    exponent = round(math.log2(value))
    if math.isclose(value, 2**exponent, rel_tol=1e-9, abs_tol=1e-12):
        return rf"$2^{{{exponent}}}$"
    return f"{value:g}"


def _format_lr_tick(value: float) -> str:
    base = 6.4e-4
    if value > 0 and math.isfinite(value):
        ratio = value / base
        exponent = round(math.log2(ratio))
        if math.isclose(ratio, 2**exponent, rel_tol=1e-7, abs_tol=1e-9):
            return rf"$2^{{{exponent}}}k$"
    return f"{value:g}"


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}"


def _format_dataset_name(value: Any) -> str:
    text = str(value)
    replacements = {"cifar100": "CIFAR-100", "cifar10": "CIFAR-10"}
    return replacements.get(text.lower(), text)


def write_summary_tables(runs: pd.DataFrame, tables_dir: Path) -> dict[str, Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    numeric_cols = [
        "global_output_multiplier",
        "learning_rate",
        "base_lr",
        "label_noise_rate",
        "width",
        "seed",
        "final_train_accuracy",
        "final_test_accuracy",
        "final_test_loss",
        "best_test_accuracy",
        "best_test_loss",
    ]
    runs = _to_numeric(runs.copy(), numeric_cols)
    outputs: dict[str, Path] = {}

    path = tables_dir / "all_runs_summary.csv"
    runs.to_csv(path, index=False)
    outputs["all_runs"] = path

    global_df = runs[runs["fls_mode"].eq("global")].copy()
    if not global_df.empty:
        grouped = (
            global_df.groupby(
                ["experiment_name", "dataset", "model", "global_output_multiplier", "learning_rate"],
                dropna=False,
            )
            .agg(
                n=("run_dir", "count"),
                mean_final_train_accuracy=("final_train_accuracy", "mean"),
                mean_final_test_accuracy=("final_test_accuracy", "mean"),
                mean_best_test_accuracy=("best_test_accuracy", "mean"),
                std_best_test_accuracy=("best_test_accuracy", "std"),
            )
            .reset_index()
        )
        path = tables_dir / "global_fls_summary.csv"
        grouped.to_csv(path, index=False)
        outputs["global_fls"] = path

    local_df = runs[runs["fls_mode"].isin(["blockwise", "location"])].copy()
    if not local_df.empty:
        grouped = (
            local_df.groupby(
                [
                    "experiment_name",
                    "dataset",
                    "model",
                    "profile_name",
                    "label_noise_rate",
                    "width",
                    "early_multiplier",
                    "middle_multiplier",
                    "late_multiplier",
                    "head_multiplier",
                    "after_early_multiplier",
                    "after_middle_multiplier",
                    "after_late_multiplier",
                ],
                dropna=False,
            )
            .agg(
                n=("run_dir", "count"),
                mean_final_train_accuracy=("final_train_accuracy", "mean"),
                mean_final_test_accuracy=("final_test_accuracy", "mean"),
                mean_best_test_accuracy=("best_test_accuracy", "mean"),
                std_best_test_accuracy=("best_test_accuracy", "std"),
            )
            .reset_index()
        )
        path = tables_dir / "profile_comparison.csv"
        grouped.to_csv(path, index=False)
        outputs["profiles"] = path

    return outputs


def write_diagnostic_tables(diagnostics: pd.DataFrame, tables_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    if diagnostics.empty:
        return outputs
    path = tables_dir / "diagnostics_summary.csv"
    diagnostics.to_csv(path, index=False)
    outputs["diagnostics"] = path
    grouped = (
        diagnostics.groupby(["experiment_name", "dataset", "model", "profile_name", "group"], dropna=False)
        .agg(
            n=("run_dir", "count"),
            mean_cka_drift=("cka_drift", "mean"),
            mean_effective_rank=("effective_rank", "mean"),
            mean_feature_norm=("feature_norm", "mean"),
            mean_clean_alignment=("clean_label_alignment", "mean"),
            mean_train_alignment=("train_label_alignment", "mean"),
            mean_alignment_gap=("alignment_gap_train_minus_clean", "mean"),
        )
        .reset_index()
    )
    path = tables_dir / "diagnostics_by_group.csv"
    grouped.to_csv(path, index=False)
    outputs["diagnostics_by_group"] = path
    return outputs


def write_matched_tables(matched: pd.DataFrame, tables_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    if matched.empty:
        return outputs
    path = tables_dir / "matched_train_accuracy.csv"
    matched.to_csv(path, index=False)
    outputs["matched_train_accuracy"] = path
    grouped = (
        matched.groupby(["threshold", "experiment_name", "dataset", "model", "profile_name", "global_output_multiplier"], dropna=False)
        .agg(
            n=("run_dir", "count"),
            mean_epoch_at_threshold=("epoch_at_threshold", "mean"),
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
        )
        .reset_index()
    )
    path = tables_dir / "matched_train_accuracy_summary.csv"
    grouped.to_csv(path, index=False)
    outputs["matched_train_accuracy_summary"] = path
    return outputs


def write_location_tables(runs: pd.DataFrame, tables_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    required = {
        "experiment_name",
        "dataset",
        "model",
        "seed",
        "base_lr",
        "best_test_accuracy",
        "best_test_loss",
        "after_early_multiplier",
        "after_middle_multiplier",
        "after_late_multiplier",
    }
    if runs.empty or not required.issubset(runs.columns):
        return outputs
    df = _to_numeric(
        runs.copy(),
        [
            "seed",
            "base_lr",
            "best_test_accuracy",
            "best_test_loss",
            "after_early_multiplier",
            "after_middle_multiplier",
            "after_late_multiplier",
        ],
    )
    df = df[
        df["experiment_name"].astype(str).str.startswith("experiment_39_boundary_sensitivity_lrcomp", na=False)
        & df["dataset"].astype(str).str.lower().eq("cifar100")
        & df["model"].astype(str).eq("resnet18_cifar")
    ].copy()
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        candidates = [
            ("early", row.get("after_early_multiplier")),
            ("middle", row.get("after_middle_multiplier")),
            ("late", row.get("after_late_multiplier")),
        ]
        active = [(name, value) for name, value in candidates if pd.notna(value) and not math.isclose(float(value), 1.0)]
        if len(active) != 1:
            continue
        location, c_value = active[0]
        records.append(
            {
                "run_dir": row.get("run_dir"),
                "experiment_name": row.get("experiment_name"),
                "dataset": row.get("dataset"),
                "model": row.get("model"),
                "seed": row.get("seed"),
                "lr": row.get("base_lr"),
                "location": location,
                "c": float(c_value),
                "best_test_accuracy": row.get("best_test_accuracy"),
                "best_test_loss": row.get("best_test_loss"),
            }
        )
    if records:
        path = tables_dir / "location_c_curve_rows.csv"
        pd.DataFrame(records).to_csv(path, index=False)
        outputs["location_c_curve_rows"] = path
    return outputs


def _save_global_heatmap(summary: pd.DataFrame, figures_dir: Path) -> Path | None:
    if summary.empty:
        return None
    required = {"global_output_multiplier", "learning_rate", "mean_best_test_accuracy"}
    if not required.issubset(summary.columns):
        return None
    summary = summary[~summary["experiment_name"].astype(str).str.contains("probe|smoke|pytest|manual", case=False, na=False)]
    preferred = summary[
        summary["experiment_name"].astype(str).str.contains(
            "experiment_42_global_reference_rerun", case=False, na=False
        )
    ]
    if preferred.empty:
        preferred = summary[summary["experiment_name"].astype(str).str.contains("experiment_00|paper_reproduction", case=False, na=False)]
    if not preferred.empty:
        summary = preferred
    for (experiment, dataset), group in summary.groupby(["experiment_name", "dataset"], dropna=False):
        pivot = group.pivot_table(
            index="learning_rate",
            columns="global_output_multiplier",
            values="mean_best_test_accuracy",
            aggfunc="mean",
        )
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        image = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([_format_power_of_two(float(value)) for value in pivot.columns], rotation=0)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([_format_lr_tick(float(value)) for value in pivot.index])
        ax.set_xlabel("Global output multiplier $c$")
        ax.set_ylabel(r"Effective learning rate ($k=6.4{\times}10^{-4}$)")
        ax.set_title(f"{_format_dataset_name(dataset)} global FLS sweep")
        for i, lr_value in enumerate(pivot.index):
            for j, c_value in enumerate(pivot.columns):
                value = pivot.loc[lr_value, c_value]
                if pd.notna(value):
                    text_color = "white" if float(value) < 0.36 else "black"
                    ax.text(j, i, _format_percent(float(value)), ha="center", va="center", fontsize=5.8, color=text_color)
        colorbar = fig.colorbar(image, ax=ax, label="Best test accuracy (%)")
        colorbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{100.0 * value:.0f}"))
        fig.tight_layout()
        path = figures_dir / f"global_fls_heatmap_{experiment}_{dataset}.pdf"
        fig.savefig(path)
        plt.close(fig)
        return path
    return None


def _save_profile_bar(summary: pd.DataFrame, figures_dir: Path) -> Path | None:
    if summary.empty or "profile_name" not in summary:
        return None
    profile_df = summary[summary["profile_name"].notna()].copy()
    profile_df = profile_df[
        ~profile_df["experiment_name"].astype(str).str.contains("probe|smoke|pytest|manual", case=False, na=False)
    ]
    if {"dataset", "model", "experiment_name"}.issubset(profile_df.columns):
        profile_df = profile_df[
            profile_df["dataset"].astype(str).str.lower().eq("cifar100")
            & profile_df["model"].astype(str).eq("resnet18_cifar")
            & profile_df["experiment_name"].astype(str).str.contains("experiment_09_profile_hparam_search", case=False, na=False)
        ]
    canonical = [
        "shuffled_progressive",
        "random",
        "uniform_optimal",
        "progressive_increasing",
        "uniform_same_average",
        "progressive_decreasing",
        "uniform_weak",
        "uniform_strong",
    ]
    profile_df = profile_df[profile_df["profile_name"].isin(canonical)]
    if profile_df.empty:
        return None
    profile_df = (
        profile_df.groupby("profile_name", as_index=False)
        .agg(mean_best_test_accuracy=("mean_best_test_accuracy", "max"))
        .sort_values("mean_best_test_accuracy", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    names = profile_df["profile_name"].astype(str).str.replace("_", " ", regex=False).str.title()
    values = 100.0 * profile_df["mean_best_test_accuracy"]
    y_positions = list(range(len(names)))
    ax.hlines(y_positions, values.min() - 0.15, values, color="#9ECAE1", linewidth=2)
    ax.scatter(values, y_positions, color="#4C78A8", s=44, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names)
    ax.set_ylabel("Profile")
    ax.set_xlabel("Mean best test accuracy (%)")
    ax.set_title("Profile comparison on CIFAR-100")
    ax.grid(axis="x", alpha=0.22)
    ax.invert_yaxis()
    for y_pos, value in enumerate(values):
        ax.text(value + 0.25, y_pos, f"{value:.1f}", va="center", fontsize=8)
    ax.set_xlim(float(values.min()) - 0.35, float(values.max()) + 0.7)
    fig.tight_layout()
    path = figures_dir / "profile_comparison_bar.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_location_sensitivity(rows: pd.DataFrame, figures_dir: Path) -> Path | None:
    required = {"location", "c", "best_test_accuracy", "best_test_loss"}
    if rows.empty or not required.issubset(rows.columns):
        return None
    df = rows.copy()
    df = _to_numeric(df, ["c", "best_test_accuracy", "best_test_loss"])
    df = df[df["location"].isin(["early", "middle", "late"])]
    df = df.dropna(subset=["c", "best_test_accuracy", "best_test_loss"])
    if df.empty:
        return None

    accuracy = (
        df.groupby(["location", "c"], dropna=False)["best_test_accuracy"]
        .max()
        .reset_index()
        .sort_values(["location", "c"])
    )
    loss = (
        df.groupby(["location", "c"], dropna=False)["best_test_loss"]
        .min()
        .reset_index()
        .sort_values(["location", "c"])
    )
    order = ["early", "middle", "late"]
    labels = {
        "early": "after-early",
        "middle": "after-middle",
        "late": "after-late",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharex=False)
    for location in order:
        acc_group = accuracy[accuracy["location"].eq(location)]
        if not acc_group.empty:
            axes[0].plot(acc_group["c"], 100.0 * acc_group["best_test_accuracy"], marker="o", linewidth=2.0, label=labels[location])
        loss_group = loss[loss["location"].eq(location)]
        if not loss_group.empty:
            axes[1].plot(loss_group["c"], loss_group["best_test_loss"], marker="o", linewidth=2.0, label=labels[location])
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Boundary multiplier $c$", fontsize=13)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=12)
    axes[0].set_ylabel("Best test accuracy (%)", fontsize=13)
    axes[1].set_ylabel("Best test loss", fontsize=13)
    axes[0].set_title("Accuracy", fontsize=14)
    axes[1].set_title("Loss", fontsize=14)
    axes[0].legend(loc="best", fontsize=11)
    axes[1].legend(loc="best", fontsize=11)
    fig.tight_layout()
    path = figures_dir / "location_sensitivity_curves.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_matched_seed_comparison(runs: pd.DataFrame, figures_dir: Path) -> Path | None:
    required = {"experiment_name", "seed", "best_test_accuracy"}
    if runs.empty or not required.issubset(runs.columns):
        return None
    df = _to_numeric(runs.copy(), ["seed", "best_test_accuracy"])
    global_df = df[
        df["experiment_name"].astype(str).str.startswith("experiment_49_global_matched_seed_expansion", na=False)
    ].copy()
    boundary_df = df[
        df["experiment_name"].astype(str).str.startswith("experiment_40_boundary_late_seed_expansion", na=False)
    ].copy()
    if global_df.empty or boundary_df.empty:
        return None
    global_df = global_df[["seed", "best_test_accuracy"]].dropna()
    boundary_df = boundary_df[["seed", "best_test_accuracy"]].dropna()
    if global_df.empty or boundary_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    conditions = [
        ("Matched global", global_df, "#4C78A8"),
        ("After-late boundary", boundary_df, "#F58518"),
    ]
    for x_pos, (label, group, color) in enumerate(conditions):
        values = 100.0 * group["best_test_accuracy"]
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ax.bar(x_pos, mean, yerr=std, color=color, alpha=0.78, capsize=4, width=0.58, label=label)
        offsets = [(i - (len(values) - 1) / 2.0) * 0.045 for i in range(len(values))]
        for offset, value in zip(offsets, values):
            ax.scatter(x_pos + offset, value, color="black", s=18, zorder=3, alpha=0.82)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([item[0] for item in conditions])
    ax.set_ylabel("Best test accuracy (%)")
    ax.set_title("Matched clean CIFAR-100 comparison")
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylim(58.0, 63.0)
    fig.tight_layout()
    path = figures_dir / "matched_seed_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_interaction_heatmap(summary: pd.DataFrame, figures_dir: Path) -> Path | None:
    required = {"experiment_name", "early_multiplier", "late_multiplier", "mean_best_test_accuracy"}
    if summary.empty or not required.issubset(summary.columns):
        return None
    interaction = summary[
        summary["experiment_name"].astype(str).str.contains("early_x_late|boundary_early_late", na=False)
    ].copy()
    if interaction.empty:
        return None
    pivot = interaction.pivot_table(
        index="early_multiplier",
        columns="late_multiplier",
        values="mean_best_test_accuracy",
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    image = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([_format_power_of_two(float(value)) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_format_power_of_two(float(value)) for value in pivot.index])
    ax.set_xlabel("After-late boundary multiplier $c$")
    ax.set_ylabel("After-early boundary multiplier $c$")
    ax.set_title("Early-late interaction", fontsize=11)
    for i, early_value in enumerate(pivot.index):
        for j, late_value in enumerate(pivot.columns):
            value = pivot.loc[early_value, late_value]
            if pd.notna(value):
                text_color = "white" if float(value) < 0.45 else "black"
                ax.text(j, i, _format_percent(float(value)), ha="center", va="center", fontsize=6, color=text_color)
    colorbar = fig.colorbar(image, ax=ax, label="Mean best test accuracy (%)")
    colorbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{100.0 * value:.1f}"))
    fig.tight_layout()
    path = figures_dir / "early_late_interaction_heatmap.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_diagnostics_bar(summary: pd.DataFrame, figures_dir: Path) -> Path | None:
    required = {"experiment_name", "group", "mean_cka_drift", "mean_effective_rank", "mean_clean_alignment"}
    if summary.empty or not required.issubset(summary.columns):
        return None
    selected = summary[
        summary["group"].astype(str).isin(["early", "middle", "late", "head"])
        & summary["experiment_name"].astype(str).isin(
            [
                "experiment_49_global_matched_seed_expansion",
                "experiment_40_boundary_late_seed_expansion_after_late",
            ]
        )
    ].copy()
    if selected.empty:
        return None
    label_map = {
        "experiment_49_global_matched_seed_expansion": "Global FLS reference",
        "experiment_40_boundary_late_seed_expansion_after_late": "After-late boundary",
    }
    selected["condition"] = selected["experiment_name"].map(label_map)
    group_order = ["early", "middle", "late", "head"]
    metrics = [
        ("mean_cka_drift", r"$\Delta$ CKA drift from initialization"),
        ("mean_effective_rank", r"$\Delta$ effective rank"),
        ("mean_clean_alignment", r"$\Delta$ class-conditional alignment"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    color = "#4C78A8"
    for ax, (column, title) in zip(axes, metrics):
        pivot = selected.pivot_table(index="group", columns="condition", values=column, aggfunc="mean")
        pivot = pivot.reindex(group_order)
        values = pivot["After-late boundary"].astype(float) - pivot["Global FLS reference"].astype(float)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax.bar(range(len(values)), values, color=color, alpha=0.86, width=0.58)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(value).capitalize() for value in values.index], rotation=20, ha="right")
        ax.set_title(title)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
        ax.grid(axis="y", alpha=0.22)
        ymax = float(values.max())
        ymin = float(values.min())
        margin = max((ymax - ymin) * 0.18, 0.01 if ymax < 10 else 2.0)
        ax.set_ylim(ymin - margin, ymax + margin)
        for x_pos, value in enumerate(values):
            ax.text(x_pos, value, f"{value:.3f}" if value < 10 else f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    path = figures_dir / "diagnostics_by_group.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_figures(tables: dict[str, Path], figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if "global_fls" in tables:
        path = _save_global_heatmap(pd.read_csv(tables["global_fls"]), figures_dir)
        if path is not None:
            written.append(path)
    if "profiles" in tables:
        profiles = pd.read_csv(tables["profiles"])
        path = _save_profile_bar(profiles, figures_dir)
        if path is not None:
            written.append(path)
        path = _save_interaction_heatmap(profiles, figures_dir)
        if path is not None:
            written.append(path)
    location_path = tables.get("location_c_curve_rows")
    if location_path is not None:
        path = _save_location_sensitivity(pd.read_csv(location_path), figures_dir)
        if path is not None:
            written.append(path)
    if "all_runs" in tables:
        path = _save_matched_seed_comparison(pd.read_csv(tables["all_runs"]), figures_dir)
        if path is not None:
            written.append(path)
    if "diagnostics_by_group" in tables:
        path = _save_diagnostics_bar(pd.read_csv(tables["diagnostics_by_group"]), figures_dir)
        if path is not None:
            written.append(path)
    return written


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs)
    paper_dir = Path(args.paper)
    tables_dir = paper_dir / "tables"
    figures_dir = paper_dir / "figures"
    if args.from_existing_tables:
        tables = {
            "global_fls": tables_dir / "global_fls_summary.csv",
            "all_runs": tables_dir / "all_runs_summary.csv",
            "profiles": tables_dir / "profile_comparison.csv",
            "diagnostics_by_group": tables_dir / "diagnostics_by_group.csv",
            "location_c_curve_rows": tables_dir / "location_c_curve_rows.csv",
        }
        tables = {name: path for name, path in tables.items() if path.exists()}
        figures = write_figures(tables, figures_dir)
        for path in figures:
            print(f"figure: {path}")
        return
    runs = collect_runs(outputs_root)
    if runs.empty:
        raise SystemExit(f"No metrics.csv files found under {outputs_root}")
    tables = write_summary_tables(runs, tables_dir)
    tables.update(write_location_tables(runs, tables_dir))
    tables.update(write_diagnostic_tables(collect_diagnostics(outputs_root), tables_dir))
    tables.update(write_matched_tables(collect_matched_train_accuracy(outputs_root), tables_dir))
    figures = write_figures(tables, figures_dir)
    print(f"Collected {len(runs)} runs")
    for name, path in tables.items():
        print(f"table:{name}: {path}")
    for path in figures:
        print(f"figure: {path}")


if __name__ == "__main__":
    main()
