#!/usr/bin/env python
"""Select and summarize the two-stage Tiny-ImageNet validation-controlled study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy import stats


CUTS = tuple(range(2, 9))
DEVELOPMENT_SEEDS = (0, 1, 2)
REPLICATION_SEEDS = (3, 4, 5, 6, 7)
DOSE_MULTIPLIERS = (0.03125, 0.0625, 0.125, 0.25, 0.5)
EXPECTED_EPOCHS = 80


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("phase1", "final"))
    parser.add_argument("--outputs-root", default=Path("outputs"), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase2-jobs", type=Path)
    return parser.parse_args()


def suffix(value: float) -> str:
    """Return the experiment-name suffix for a decimal value."""
    return f"{value:g}".replace(".", "p")


def read_complete_metrics(path: Path) -> list[dict[str, str]]:
    """Read one complete, finite metrics file."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_EPOCHS:
        raise ValueError(f"Expected {EXPECTED_EPOCHS} rows, found {len(rows)}: {path}")
    if int(float(rows[-1]["epoch"])) != EXPECTED_EPOCHS - 1:
        raise ValueError(f"Incomplete run: {path}")
    for row in rows:
        for field in (
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "test_loss",
            "test_accuracy",
        ):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite {field} at epoch {row['epoch']}: {path}")
    return rows


def find_complete_run(outputs_root: Path, experiment: str, seed: int) -> Path:
    """Return the newest complete run for one experiment and seed."""
    candidates = sorted(
        outputs_root.glob(f"{experiment}/*/seed_{seed}/metrics.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            read_complete_metrics(path)
        except ValueError:
            continue
        return path
    raise FileNotFoundError(f"No complete run for {experiment}, seed {seed}")


def validation_checkpoint(rows: list[dict[str, str]]) -> dict[str, str]:
    """Select the earliest epoch attaining maximum validation accuracy."""
    return max(rows, key=lambda row: (float(row["val_accuracy"]), -int(row["epoch"])))


def collect(
    outputs_root: Path, experiment: str, seeds: tuple[int, ...]
) -> list[dict[str, object]]:
    """Collect validation-selected test outcomes for an experiment."""
    records: list[dict[str, object]] = []
    for seed in seeds:
        metrics_path = find_complete_run(outputs_root, experiment, seed)
        rows = read_complete_metrics(metrics_path)
        selected = validation_checkpoint(rows)
        final = rows[-1]
        records.append(
            {
                "experiment": experiment,
                "seed": seed,
                "metrics_csv": str(metrics_path),
                "best_val_epoch": int(float(selected["epoch"])),
                "best_val_accuracy_pct": 100.0 * float(selected["val_accuracy"]),
                "val_loss_at_best_val": float(selected["val_loss"]),
                "test_accuracy_at_best_val_pct": 100.0
                * float(selected["test_accuracy"]),
                "test_loss_at_best_val": float(selected["test_loss"]),
                "final_test_accuracy_pct": 100.0 * float(final["test_accuracy"]),
            }
        )
    return records


def sample_summary(values: list[float]) -> dict[str, float]:
    """Return mean, sample SD, and a two-sided 95% t interval."""
    avg = mean(values)
    sd = stdev(values)
    half_width = float(stats.t.ppf(0.975, len(values) - 1)) * sd / math.sqrt(
        len(values)
    )
    return {
        "mean": avg,
        "sd": sd,
        "ci95_low": avg - half_width,
        "ci95_high": avg + half_width,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a non-empty dictionary table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def phase1_experiment(cut: int, condition: str) -> str:
    """Return a phase-1 experiment name."""
    if condition == "boundary":
        return f"val_tiny72_bnd_cut{cut}"
    if condition == "lr_only":
        return f"val_tiny72_lr_cut{cut}_16x"
    raise ValueError(condition)


def dose_experiment(multiplier: float) -> str:
    """Return a phase-1 dose experiment name, reusing the cut-8 boundary arm."""
    if multiplier == 0.0625:
        return "val_tiny72_bnd_cut8"
    return f"val_tiny72_dose_cut8_c{suffix(multiplier)}"


def mean_field(rows: list[dict[str, object]], field: str) -> float:
    """Return the arithmetic mean of one numeric field."""
    return mean(float(row[field]) for row in rows)


def build_phase2_jobs(path: Path, selected_cut: int) -> None:
    """Write the 15 locked replication jobs for the selected depth cut."""
    jobs: list[str] = []
    for seed in REPLICATION_SEEDS:
        common = (
            "configs/imp_pos_base.yaml "
            f"training.seed={seed} training.max_epochs=80 "
            "data.name=tiny_imagenet data.num_classes=200 model.num_classes=200 "
            "data.val_fraction=0.1 data.val_seed=0 "
            "training.save_checkpoints=false training.checkpoint_at_accs=[] "
            "diagnostics.enabled=false"
        )
        jobs.append(
            f"{common} fls.position=8 fls.output_multiplier=1.0 "
            "experiment.name=val_tiny72_rep_baseline"
        )
        jobs.append(
            f"{common} fls.position={selected_cut} fls.output_multiplier=0.0625 "
            f"experiment.name=val_tiny72_rep_bnd_cut{selected_cut}"
        )
        jobs.append(
            f"{common} fls.position={selected_cut} fls.output_multiplier=1.0 "
            "fls.lr_compensation_multiplier=0.0625 "
            f"experiment.name=val_tiny72_rep_lr_cut{selected_cut}_16x"
        )
    if len(jobs) != 15:
        raise RuntimeError(f"Expected 15 phase-2 jobs, built {len(jobs)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Tiny-ImageNet validation-controlled 72-run package: phase 2/2.\n"
        "# Locked baseline, selected boundary cut, and exact same-cut LR-only control.\n"
        + "\n".join(jobs)
        + "\n",
        encoding="utf-8",
    )


def summarize_phase1(args: argparse.Namespace) -> None:
    """Summarize 57 development runs and freeze the phase-2 cut."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, object]] = []
    depth_rows: list[dict[str, object]] = []
    boundary_by_cut: dict[int, list[dict[str, object]]] = {}

    baseline = collect(args.outputs_root, "val_tiny72_baseline", DEVELOPMENT_SEEDS)
    for row in baseline:
        seed_rows.append({"family": "baseline", "cut": 8, "multiplier": 1.0, **row})

    for cut in CUTS:
        boundary = collect(
            args.outputs_root, phase1_experiment(cut, "boundary"), DEVELOPMENT_SEEDS
        )
        lr_only = collect(
            args.outputs_root, phase1_experiment(cut, "lr_only"), DEVELOPMENT_SEEDS
        )
        boundary_by_cut[cut] = boundary
        for row in boundary:
            seed_rows.append(
                {"family": "depth_boundary", "cut": cut, "multiplier": 0.0625, **row}
            )
        for row in lr_only:
            seed_rows.append(
                {"family": "depth_lr_only", "cut": cut, "multiplier": 1.0, **row}
            )
        boundary_test = [
            float(row["test_accuracy_at_best_val_pct"]) for row in boundary
        ]
        lr_test = [float(row["test_accuracy_at_best_val_pct"]) for row in lr_only]
        margins = [boundary_test[i] - lr_test[i] for i in range(len(boundary_test))]
        margin_stats = sample_summary(margins)
        paired = stats.ttest_rel(boundary_test, lr_test)
        depth_rows.append(
            {
                "cut": cut,
                "n": len(DEVELOPMENT_SEEDS),
                "mean_best_val_accuracy_pct": mean_field(
                    boundary, "best_val_accuracy_pct"
                ),
                "mean_boundary_test_at_best_val_pct": mean(boundary_test),
                "mean_lr_only_test_at_best_val_pct": mean(lr_test),
                "boundary_minus_lr_only_pp": margin_stats["mean"],
                "margin_sd_pp": margin_stats["sd"],
                "margin_ci95_low_pp": margin_stats["ci95_low"],
                "margin_ci95_high_pp": margin_stats["ci95_high"],
                "paired_p_nominal_uncorrected": float(paired.pvalue),
            }
        )

    dose_rows: list[dict[str, object]] = []
    dose_by_multiplier: dict[float, list[dict[str, object]]] = {}
    for multiplier in DOSE_MULTIPLIERS:
        records = collect(
            args.outputs_root, dose_experiment(multiplier), DEVELOPMENT_SEEDS
        )
        dose_by_multiplier[multiplier] = records
        if multiplier != 0.0625:
            for row in records:
                seed_rows.append(
                    {
                        "family": "dose_boundary",
                        "cut": 8,
                        "multiplier": multiplier,
                        **row,
                    }
                )
        dose_rows.append(
            {
                "multiplier": multiplier,
                "n": len(DEVELOPMENT_SEEDS),
                "mean_best_val_accuracy_pct": mean_field(
                    records, "best_val_accuracy_pct"
                ),
                "mean_val_loss_at_best_val": mean_field(
                    records, "val_loss_at_best_val"
                ),
                "mean_test_at_best_val_pct": mean_field(
                    records, "test_accuracy_at_best_val_pct"
                ),
            }
        )

    # The cut is selected without consulting test metrics: highest mean validation
    # accuracy, then lowest mean validation loss, then the shallower cut.
    selected_cut = max(
        CUTS,
        key=lambda cut: (
            mean_field(boundary_by_cut[cut], "best_val_accuracy_pct"),
            -mean_field(boundary_by_cut[cut], "val_loss_at_best_val"),
            -cut,
        ),
    )
    # The cut-8 dose is selected by the same validation-only rule. If validation
    # accuracy and loss tie, prefer the multiplier closer to identity.
    selected_dose = max(
        DOSE_MULTIPLIERS,
        key=lambda multiplier: (
            mean_field(dose_by_multiplier[multiplier], "best_val_accuracy_pct"),
            -mean_field(dose_by_multiplier[multiplier], "val_loss_at_best_val"),
            multiplier,
        ),
    )
    selection = {
        "protocol": "fixed 10% train/validation split; untouched official validation split used as test",
        "checkpoint_selection": "earliest epoch attaining maximum validation accuracy",
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "selected_depth_cut": selected_cut,
        "selected_depth_rule": (
            "highest mean validation accuracy; tie by lower mean validation loss; "
            "then shallower cut"
        ),
        "selected_cut8_dose": selected_dose,
        "selected_dose_rule": (
            "highest mean validation accuracy; tie by lower mean validation loss; "
            "then multiplier closer to identity"
        ),
        "test_metrics_used_for_selection": False,
        "planned_phase1_runs": 57,
        "planned_phase2_runs": 15,
        "planned_total_runs": 72,
    }
    write_csv(args.output_dir / "phase1_seed_level.csv", seed_rows)
    write_csv(args.output_dir / "depth_summary.csv", depth_rows)
    write_csv(args.output_dir / "dose_summary.csv", dose_rows)
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    if args.phase2_jobs is None:
        raise ValueError("--phase2-jobs is required for stage phase1")
    build_phase2_jobs(args.phase2_jobs, selected_cut)
    print(json.dumps(selection, indent=2))
    print(f"Wrote 15 locked replication jobs to {args.phase2_jobs}")


def paired_summary(
    left: list[dict[str, object]], right: list[dict[str, object]]
) -> dict[str, float]:
    """Summarize paired test-accuracy differences left minus right."""
    left_by_seed = {
        int(row["seed"]): float(row["test_accuracy_at_best_val_pct"]) for row in left
    }
    right_by_seed = {
        int(row["seed"]): float(row["test_accuracy_at_best_val_pct"]) for row in right
    }
    differences = [
        left_by_seed[seed] - right_by_seed[seed] for seed in REPLICATION_SEEDS
    ]
    summary = sample_summary(differences)
    paired = stats.ttest_rel(
        [left_by_seed[seed] for seed in REPLICATION_SEEDS],
        [right_by_seed[seed] for seed in REPLICATION_SEEDS],
    )
    return {**summary, "paired_t": float(paired.statistic), "paired_p": float(paired.pvalue)}


def summarize_final(args: argparse.Namespace) -> None:
    """Summarize the locked five-seed replication."""
    selection = json.loads((args.output_dir / "selection.json").read_text())
    selected_cut = int(selection["selected_depth_cut"])
    experiments = {
        "baseline": "val_tiny72_rep_baseline",
        "boundary": f"val_tiny72_rep_bnd_cut{selected_cut}",
        "lr_only_exact_16x": f"val_tiny72_rep_lr_cut{selected_cut}_16x",
    }
    records = {
        condition: collect(args.outputs_root, experiment, REPLICATION_SEEDS)
        for condition, experiment in experiments.items()
    }
    seed_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    for condition, rows in records.items():
        test_values = [float(row["test_accuracy_at_best_val_pct"]) for row in rows]
        final_values = [float(row["final_test_accuracy_pct"]) for row in rows]
        test_stats = sample_summary(test_values)
        condition_rows.append(
            {
                "condition": condition,
                "experiment": experiments[condition],
                "n": len(rows),
                "mean_test_at_best_val_pct": test_stats["mean"],
                "sd_test_at_best_val_pct": test_stats["sd"],
                "ci95_low_pct": test_stats["ci95_low"],
                "ci95_high_pct": test_stats["ci95_high"],
                "mean_final_test_accuracy_pct": mean(final_values),
            }
        )
        for row in rows:
            seed_rows.append({"condition": condition, **row})

    final_summary = {
        **selection,
        "replication_seeds": list(REPLICATION_SEEDS),
        "selected_depth_cut": selected_cut,
        "primary_contrast_boundary_minus_exact_lr_only_pp": paired_summary(
            records["boundary"], records["lr_only_exact_16x"]
        ),
        "secondary_contrast_boundary_minus_baseline_pp": paired_summary(
            records["boundary"], records["baseline"]
        ),
        "diagnostic_contrast_lr_only_minus_baseline_pp": paired_summary(
            records["lr_only_exact_16x"], records["baseline"]
        ),
        "completed_total_runs": 72,
        "interpretation_rule": (
            "The locked replication tests the validation-selected depth cut. "
            "The full per-cut and cut-8 dose profiles remain multiplicity-uncorrected."
        ),
    }
    write_csv(args.output_dir / "phase2_seed_level.csv", seed_rows)
    write_csv(args.output_dir / "phase2_condition_summary.csv", condition_rows)
    (args.output_dir / "final_summary.json").write_text(
        json.dumps(final_summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "COMPLETED").touch()
    print(json.dumps(final_summary, indent=2))


def main() -> None:
    """Run the requested summarization stage."""
    args = parse_args()
    if args.stage == "phase1":
        summarize_phase1(args)
    else:
        summarize_final(args)


if __name__ == "__main__":
    main()
