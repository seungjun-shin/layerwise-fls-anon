#!/usr/bin/env python
"""Build the staged validation-controlled queue that replaces legacy test selection.

Stage 1 runs validation-only selection grids plus every fixed supplementary arm.
Stage 2 reads validation accuracy only, locks one configuration per searched
condition, and emits fresh-seed replication jobs. Test accuracy is never used
to choose a configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ROOT = Path("outputs/_val_preserve_all_20260801")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["initial", "replication"], required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def make_job(
    config: str,
    name: str,
    seed: int,
    overrides: list[str],
    *,
    checkpoints: bool = False,
) -> str:
    common = [
        f"training.seed={seed}",
        "training.max_epochs=80",
        "data.val_fraction=0.1",
        "data.val_seed=0",
        f"training.save_checkpoints={'true' if checkpoints else 'false'}",
        "training.checkpoint_at_accs=[]",
        f"experiment.name={name}",
    ]
    return " ".join([config, *common, *overrides])


def initial_jobs() -> list[str]:
    jobs: list[str] = []

    # CIFAR-10 target-setting search: preserve the original 7x7 global and 9x5 boundary grids.
    for c in [0.0009765625, 0.00390625, 0.015625, 0.0625, 0.25, 1.0, 4.0]:
        for lr in [0.00064, 0.00256, 0.01024, 0.04096, 0.16384, 0.65536, 2.62144]:
            name = f"valp_c10_global_sel_c{slug(c)}_lr{slug(lr)}"
            jobs.append(make_job(
                "configs/experiment_43_cifar10_global_reference.yaml", name, 0,
                [f"fls.global.output_multiplier={c}", f"training.lr={lr}"],
            ))
    for c in [0.0009765625, 0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25]:
        for lr in [0.00256, 0.01024, 0.04096, 0.08192, 0.16384]:
            name = f"valp_c10_boundary_sel_c{slug(c)}_lr{slug(lr)}"
            jobs.append(make_job(
                "configs/experiment_44_cifar10_boundary_sensitivity.yaml", name, 0,
                [f"fls.boundaries.after_late.output_multiplier={c}", f"training.lr={lr}"],
            ))

    # No-BN target-setting search: preserve the original grids.
    for c in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]:
        for lr in [0.04096, 0.08192, 0.16384, 0.32768]:
            name = f"valp_nobn_global_sel_c{slug(c)}_lr{slug(lr)}"
            jobs.append(make_job(
                "configs/experiment_45_nobn_global_reference.yaml", name, 0,
                [f"fls.global.output_multiplier={c}", f"training.lr={lr}"],
            ))
    for c in [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]:
        for lr in [0.04096, 0.08192, 0.16384, 0.32768]:
            name = f"valp_nobn_boundary_sel_c{slug(c)}_lr{slug(lr)}"
            jobs.append(make_job(
                "configs/experiment_46_nobn_boundary_sensitivity.yaml", name, 0,
                [f"fls.boundaries.after_late.output_multiplier={c}", f"training.lr={lr}"],
            ))

    # Practical-regime retuning grids. Checkpoints are retained because the selected
    # seed-0 runs feed the validation-controlled representation-persistence analysis.
    for c in [0.5, 0.75, 1.0, 1.25]:
        for lr in [0.00256, 0.00384, 0.00512]:
            name = f"valp_practical_global_sel_c{slug(c)}_lr{slug(lr)}"
            jobs.append(make_job(
                "configs/experiment_83_practical_global_narrow_c_lr.yaml", name, 0,
                [f"fls.global.output_multiplier={c}", f"training.lr={lr}"], checkpoints=True,
            ))
    for c in [0.75, 1.0, 1.25, 1.5]:
        for lr in [0.00256, 0.00512]:
            name = f"valp_practical_boundary_sel_c{slug(c)}_lr{slug(lr)}"
            jobs.append(make_job(
                "configs/experiment_84_practical_middle_late_identity_neighborhood.yaml", name, 0,
                [f"fls.boundaries.after_late.output_multiplier={c}", f"training.lr={lr}"], checkpoints=True,
            ))

    # Fixed-multiplier augmentation+cosine table (five arms x three seeds).
    for seed in range(3):
        jobs.append(make_job(
            "configs/experiment_93b_persist_global.yaml", "valp_practical_fixed_global_c0p25", seed, [],
        ))
        for c in [0.0625, 0.25, 0.5, 1.0]:
            jobs.append(make_job(
                "configs/experiment_93b_persist_late.yaml",
                f"valp_practical_fixed_boundary_c{slug(c)}", seed,
                [f"fls.boundaries.after_late.output_multiplier={c}"],
            ))

    # Five practical factor pairs (ten fixed arms x three seeds).
    factor_configs = [
        "configs/experiment_65_practical_factor_controlled_global.yaml",
        "configs/experiment_66_practical_factor_controlled_boundary.yaml",
        "configs/experiment_67_practical_factor_aug_only_global.yaml",
        "configs/experiment_68_practical_factor_aug_only_boundary.yaml",
        "configs/experiment_69_practical_factor_momentum_only_global.yaml",
        "configs/experiment_70_practical_factor_momentum_only_boundary.yaml",
        "configs/experiment_71_practical_factor_wd_only_global.yaml",
        "configs/experiment_72_practical_factor_wd_only_boundary.yaml",
        "configs/experiment_73_practical_factor_aug_wd_global.yaml",
        "configs/experiment_74_practical_factor_aug_wd_boundary.yaml",
    ]
    for config in factor_configs:
        base_name = Path(config).stem.replace("experiment_", "valp_factor_")
        for seed in range(3):
            jobs.append(make_job(config, base_name, seed, []))

    # Preserve the early-late 3x4 heatmap with three validation-controlled seeds.
    for seed in range(3):
        for early in [1.0, 2.0, 4.0]:
            for late in [0.03125, 0.0625, 0.125, 0.25]:
                name = f"valp_interaction_e{slug(early)}_l{slug(late)}"
                jobs.append(make_job(
                    "configs/experiment_41_boundary_early_late_interaction.yaml", name, seed,
                    [
                        f"fls.boundaries.after_early.output_multiplier={early}",
                        f"fls.boundaries.after_late.output_multiplier={late}",
                    ],
                ))

    assert len(jobs) == 251, len(jobs)
    assert len(jobs) == len(set(jobs))
    return jobs


def load_best(name: str) -> dict[str, Any]:
    candidates = sorted(Path("outputs").glob(f"{name}/**/seed_0/metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No completed seed-0 metrics found for {name}")
    metrics_path = candidates[-1]
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or max(int(float(row["epoch"])) for row in rows) < 79:
        raise RuntimeError(f"Incomplete metrics: {metrics_path}")
    best_row = max(rows, key=lambda row: float(row["val_accuracy"]))
    with (metrics_path.parent / "resolved_config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return {
        "name": name,
        "metrics_path": str(metrics_path),
        "validation_accuracy": float(best_row["val_accuracy"]),
        "validation_epoch": int(float(best_row["epoch"])),
        "report_only_test_accuracy": float(best_row["test_accuracy"]),
        "config": config,
    }


def select(prefix: str) -> dict[str, Any]:
    names = sorted(path.name for path in Path("outputs").glob(f"{prefix}*") if path.is_dir())
    records = [load_best(name) for name in names]
    if not records:
        raise FileNotFoundError(f"No candidates found for prefix {prefix}")
    # Candidate ordering is the deterministic tie breaker; test accuracy is report-only.
    return max(records, key=lambda record: record["validation_accuracy"])


def selected_values(record: dict[str, Any], kind: str) -> tuple[float, float]:
    config = record["config"]
    lr = float(config["training"]["lr"])
    if kind == "global":
        c = float(config["fls"]["global"]["output_multiplier"])
    else:
        c = float(config["fls"]["boundaries"]["after_late"]["output_multiplier"])
    return c, lr


def replication_jobs(root: Path) -> list[str]:
    selected = {
        "c10_global": select("valp_c10_global_sel_"),
        "c10_boundary": select("valp_c10_boundary_sel_"),
        "nobn_global": select("valp_nobn_global_sel_"),
        "nobn_boundary": select("valp_nobn_boundary_sel_"),
        "practical_global": select("valp_practical_global_sel_"),
        "practical_boundary": select("valp_practical_boundary_sel_"),
    }
    serializable = {
        key: {k: v for k, v in record.items() if k != "config"}
        | {"output_multiplier": selected_values(record, "global" if "global" in key else "boundary")[0],
           "learning_rate": selected_values(record, "global" if "global" in key else "boundary")[1]}
        for key, record in selected.items()
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "validation_selection.json").write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")

    jobs: list[str] = []
    for family, global_config, boundary_config in [
        ("c10", "configs/experiment_43_cifar10_global_reference.yaml", "configs/experiment_44_cifar10_boundary_sensitivity.yaml"),
        ("nobn", "configs/experiment_45_nobn_global_reference.yaml", "configs/experiment_46_nobn_boundary_sensitivity.yaml"),
    ]:
        gc, glr = selected_values(selected[f"{family}_global"], "global")
        bc, blr = selected_values(selected[f"{family}_boundary"], "boundary")
        for seed in [1, 2, 3]:
            jobs.append(make_job(
                global_config, f"valp_{family}_global_locked", seed,
                [f"fls.global.output_multiplier={gc}", f"training.lr={glr}"],
            ))
            jobs.append(make_job(
                boundary_config, f"valp_{family}_boundary_locked", seed,
                [f"fls.boundaries.after_late.output_multiplier={bc}", f"training.lr={blr}"],
            ))

    gc, glr = selected_values(selected["practical_global"], "global")
    bc, blr = selected_values(selected["practical_boundary"], "boundary")
    for seed in [1, 2, 3, 4]:
        jobs.append(make_job(
            "configs/experiment_83_practical_global_narrow_c_lr.yaml", "valp_practical_global_locked", seed,
            [f"fls.global.output_multiplier={gc}", f"training.lr={glr}"], checkpoints=True,
        ))
        jobs.append(make_job(
            "configs/experiment_84_practical_middle_late_identity_neighborhood.yaml",
            "valp_practical_boundary_locked", seed,
            [f"fls.boundaries.after_late.output_multiplier={bc}", f"training.lr={blr}"], checkpoints=True,
        ))

    assert len(jobs) == 20, len(jobs)
    assert len(jobs) == len(set(jobs))
    return jobs


def write_jobs(path: Path, jobs: list[str], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n" + "\n".join(jobs) + "\n", encoding="utf-8")
    print(f"Wrote {len(jobs)} jobs to {path}")


def main() -> None:
    args = parse_args()
    if args.stage == "initial":
        write_jobs(args.root / "jobs_initial.txt", initial_jobs(), "251 validation-controlled preservation jobs")
    else:
        write_jobs(args.root / "jobs_replication.txt", replication_jobs(args.root), "20 validation-locked replication jobs")


if __name__ == "__main__":
    main()
