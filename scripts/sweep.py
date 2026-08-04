#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from fls.scaling.profiles import build_profile
from fls.utils.config import load_config

BOUNDARIES = ["after_early", "after_middle", "after_late"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--multipliers", nargs="*", default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def run_train(command: list[str], continue_on_error: bool) -> None:
    """Run one training command and optionally continue after failed cells."""
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        if not continue_on_error:
            raise
        print(
            f"[sweep] command failed with exit code {exc.returncode}; continuing: {' '.join(command)}",
            file=sys.stderr,
            flush=True,
        )


def last_recorded_epoch(metrics_path: Path) -> int | None:
    """Return the last epoch recorded in a metrics CSV."""
    if not metrics_path.exists():
        return None
    with metrics_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return max(int(float(row.get("epoch", -1) or -1)) for row in rows)


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return fields that define whether a sweep cell is already complete."""
    fls_config = dict(config.get("fls") or {})
    if fls_config.get("mode") == "blockwise" and fls_config.get("groups") is None:
        groups = ["early", "middle", "late", "head"]
        fls_config["groups"] = build_profile(
            fls_config.get("profile_name", "progressive_increasing"),
            groups,
            fls_config.get("base_values"),
            seed=int(config["training"]["seed"]),
        )
    return {
        "experiment": config.get("experiment", {}).get("name"),
        "data": {
            key: config.get("data", {}).get(key)
            for key in ["name", "train_subset_fraction", "num_classes", "augment", "normalize"]
        },
        "model": {
            key: config.get("model", {}).get(key)
            for key in ["name", "width", "use_bn", "freeze_bn_affine", "freeze_bn_stats"]
        },
        "fls": fls_config,
        "label_noise": config.get("label_noise"),
        "training": {
            key: config.get("training", {}).get(key)
            for key in [
                "seed",
                "max_epochs",
                "optimizer",
                "lr",
                "weight_decay",
                "momentum",
                "location_lr_compensation",
                "location_lr_compensation_multipliers",
                "blockwise_lr_compensation",
            ]
        },
    }


def completed_run_exists(config_path: str, overrides: list[str]) -> bool:
    """Check whether a resolved sweep cell already has a complete metrics file."""
    expected = load_config(config_path, overrides)
    output_root = Path(expected.get("output", {}).get("root", "outputs"))
    experiment_name = expected["experiment"]["name"]
    seed = expected["training"]["seed"]
    max_epochs = int(expected["training"]["max_epochs"])
    expected_key = comparable_config(expected)
    for resolved_path in output_root.glob(f"{experiment_name}/**/seed_{seed}/resolved_config.yaml"):
        with resolved_path.open(encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
        if comparable_config(existing) != expected_key:
            continue
        last_epoch = last_recorded_epoch(resolved_path.parent / "metrics.csv")
        if last_epoch is not None and last_epoch >= max_epochs - 1:
            return True
    return False


def run_sweep_cell(config_path: str, overrides: list[str], continue_on_error: bool, skip_completed: bool) -> None:
    """Run one config cell, optionally skipping already completed runs."""
    if skip_completed and completed_run_exists(config_path, overrides):
        print(f"[sweep] skipping completed cell: {' '.join(overrides)}", flush=True)
        return
    run_train([sys.executable, "scripts/train.py", "--config", config_path, *overrides], continue_on_error)


def profile_base_value_overrides(base_values: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    """Return CLI overrides and name suffixes for a profile base-value setting."""
    if base_values is None:
        return [], []
    overrides = []
    suffix = []
    for key in ["weak", "optimal", "strong"]:
        if key in base_values:
            value = base_values[key]
            overrides.append(f"fls.base_values.{key}={value}")
            suffix.append(f"{key}_{value}")
    return overrides, suffix


def sanitized(value: Any) -> str:
    """Return a compact string safe for experiment-name suffixes."""
    return str(value).replace(".", "p").replace("-", "m")


def bn_mode_overrides(mode: str) -> tuple[list[str], list[str]]:
    """Return CLI overrides and name suffixes for a BN-control mode."""
    if mode == "default":
        return [], []
    if mode == "freeze_affine":
        return ["model.freeze_bn_affine=true", "model.freeze_bn_stats=false"], ["freeze_affine"]
    if mode == "freeze_stats":
        return ["model.freeze_bn_affine=false", "model.freeze_bn_stats=true"], ["freeze_stats"]
    if mode == "freeze_all":
        return ["model.freeze_bn_affine=true", "model.freeze_bn_stats=true"], ["freeze_all"]
    raise ValueError(f"Unknown bn_mode: {mode}")


def group_overrides(groups: dict[str, float]) -> list[str]:
    """Return block-wise FLS group overrides for explicit output multipliers."""
    overrides = ["fls.profile_name=null"]
    for block_group in ["early", "middle", "late", "head"]:
        multiplier = groups[block_group]
        overrides.append(f"fls.groups.{block_group}.output_multiplier={multiplier}")
        overrides.append(f"fls.groups.{block_group}.init_scale=1.0")
    return overrides


def boundary_overrides(boundaries: dict[str, float]) -> list[str]:
    """Return location-boundary FLS overrides for explicit output multipliers."""
    overrides: list[str] = []
    for boundary in BOUNDARIES:
        multiplier = boundaries[boundary]
        overrides.append(f"fls.boundaries.{boundary}.output_multiplier={multiplier}")
        overrides.append(f"fls.boundaries.{boundary}.init_scale=1.0")
    return overrides


def extra_axis_values(sweep: dict[str, Any]) -> tuple[list[Any], list[Any], list[str]]:
    """Return optional stress axes shared by sweep branches."""
    return (
        sweep.get("label_noise_rates", [None]),
        sweep.get("train_subset_fractions", [None]),
        sweep.get("bn_modes", ["default"]),
    )


def extra_axis_overrides(
    noise_rate: Any,
    subset_fraction: Any,
    bn_mode: str,
) -> tuple[list[str], list[str]]:
    """Return CLI overrides and name suffixes for optional stress axes."""
    overrides: list[str] = []
    suffix: list[str] = []
    if noise_rate is not None:
        overrides.append(f"label_noise.rate={noise_rate}")
        suffix.append(f"noise_{sanitized(noise_rate)}")
    if subset_fraction is not None:
        overrides.append(f"data.train_subset_fraction={subset_fraction}")
        suffix.append(f"subset_{sanitized(subset_fraction)}")
    bn_overrides, bn_suffix = bn_mode_overrides(str(bn_mode))
    overrides.extend(bn_overrides)
    suffix.extend(bn_suffix)
    return overrides, suffix


def maybe_name_override(config: dict[str, Any], suffix: list[str]) -> list[str]:
    """Return an experiment.name override when a sweep cell needs a unique namespace."""
    if not suffix:
        return []
    return [f"experiment.name={config['experiment']['name']}_{'_'.join(suffix)}"]


def anchored_profile_groups(anchor: float, ratios: list[float]) -> dict[str, float]:
    """Build explicit group multipliers from an anchor and early/middle/late/head ratios."""
    if len(ratios) != 4:
        raise ValueError(f"anchor profile ratios must contain four values, got {ratios}")
    return {
        group: float(anchor) * float(ratio)
        for group, ratio in zip(["early", "middle", "late", "head"], ratios, strict=True)
    }


def interaction_values_for_pair(sweep: dict[str, Any], first: str, second: str) -> tuple[list[str], list[str]]:
    """Return possibly group-specific values for a two-group interaction sweep."""
    default_values = [str(value) for value in sweep.get("output_multipliers", [1.0])]
    values_by_group = sweep.get("interaction_values")
    if not isinstance(values_by_group, dict):
        return default_values, default_values
    first_values = [str(value) for value in values_by_group.get(first, default_values)]
    second_values = [str(value) for value in values_by_group.get(second, default_values)]
    return first_values, second_values


def interaction_values_for_boundary_pair(sweep: dict[str, Any], first: str, second: str) -> tuple[list[str], list[str]]:
    """Return possibly boundary-specific values for a two-boundary interaction sweep."""
    default_values = [str(value) for value in sweep.get("output_multipliers", [1.0])]
    values_by_boundary = sweep.get("interaction_values")
    if not isinstance(values_by_boundary, dict):
        return default_values, default_values
    first_values = [str(value) for value in values_by_boundary.get(first, default_values)]
    second_values = [str(value) for value in values_by_boundary.get(second, default_values)]
    return first_values, second_values


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    sweep = config.get("sweep", {})
    continue_on_error = bool(sweep.get("continue_on_error", True))
    skip_completed = bool(sweep.get("skip_completed", True))
    base_overrides = args.overrides
    seeds = [str(seed) for seed in sweep.get("seeds", [config["training"].get("seed", 0)])]
    learning_rates = [str(lr) for lr in sweep.get("learning_rates", [config["training"].get("lr")])]
    label_noise_rates, subset_fractions, bn_modes = extra_axis_values(sweep)
    if config["fls"]["mode"] == "global":
        multipliers = args.multipliers or [str(value) for value in sweep.get("global_output_multipliers", ["0.5", "1.0"])]
        for seed, lr, multiplier, noise_rate, subset_fraction, bn_mode in product(
            seeds, learning_rates, multipliers, label_noise_rates, subset_fractions, bn_modes
        ):
            axis_overrides, axis_suffix = extra_axis_overrides(noise_rate, subset_fraction, str(bn_mode))
            suffix = [f"c_{sanitized(multiplier)}"]
            if len(learning_rates) > 1:
                suffix.append(f"lr_{sanitized(lr)}")
            suffix.extend(axis_suffix)
            run_sweep_cell(
                args.config,
                [
                    *base_overrides,
                    f"training.seed={seed}",
                    f"training.lr={lr}",
                    f"fls.global.output_multiplier={multiplier}",
                    *axis_overrides,
                    *maybe_name_override(config, suffix if axis_suffix else []),
                ],
                continue_on_error,
                skip_completed,
            )
    elif config["fls"]["mode"] == "location" and "interaction_boundaries" in sweep:
        fixed = sweep.get("fixed_output_multiplier", 1.0)
        for seed, lr, pair, noise_rate, subset_fraction, bn_mode in product(
            seeds, learning_rates, sweep["interaction_boundaries"], label_noise_rates, subset_fractions, bn_modes
        ):
            if len(pair) != 2:
                raise ValueError(f"interaction pair must contain exactly two boundaries: {pair}")
            first, second = pair
            first_values, second_values = interaction_values_for_boundary_pair(sweep, first, second)
            for first_value, second_value in product(first_values, second_values):
                axis_overrides, axis_suffix = extra_axis_overrides(noise_rate, subset_fraction, str(bn_mode))
                values = {boundary: fixed for boundary in BOUNDARIES}
                values[first] = first_value
                values[second] = second_value
                suffix_parts = [f"{first}_x_{second}"]
                if len(learning_rates) > 1:
                    suffix_parts.append(f"lr_{sanitized(lr)}")
                suffix_parts.extend(axis_suffix)
                run_sweep_cell(
                    args.config,
                    [
                        *base_overrides,
                        f"training.seed={seed}",
                        f"training.lr={lr}",
                        *boundary_overrides(values),
                        *axis_overrides,
                        f"experiment.name={config['experiment']['name']}_{'_'.join(suffix_parts)}",
                    ],
                    continue_on_error,
                    skip_completed,
                )
    elif config["fls"]["mode"] == "location" and "varied_boundaries" in sweep:
        values = [str(value) for value in sweep.get("output_multipliers", [1.0])]
        fixed = sweep.get("fixed_output_multiplier", 1.0)
        for seed, lr, boundary, value, noise_rate, subset_fraction, bn_mode in product(
            seeds, learning_rates, sweep["varied_boundaries"], values, label_noise_rates, subset_fractions, bn_modes
        ):
            axis_overrides, axis_suffix = extra_axis_overrides(noise_rate, subset_fraction, str(bn_mode))
            multipliers = {name: fixed for name in BOUNDARIES}
            multipliers[boundary] = value
            suffix_parts = [str(boundary), f"c_{sanitized(value)}"]
            if len(learning_rates) > 1:
                suffix_parts.append(f"lr_{sanitized(lr)}")
            suffix_parts.extend(axis_suffix)
            run_sweep_cell(
                args.config,
                [
                    *base_overrides,
                    f"training.seed={seed}",
                    f"training.lr={lr}",
                    *boundary_overrides(multipliers),
                    *axis_overrides,
                    f"experiment.name={config['experiment']['name']}_{'_'.join(suffix_parts)}",
                ],
                continue_on_error,
                skip_completed,
            )
    elif config["fls"]["mode"] == "blockwise" and "anchor_profiles" in sweep:
        anchors = [float(value) for value in sweep.get("anchor_values", [1.0])]
        for seed, lr, anchor, profile, noise_rate, subset_fraction, bn_mode in product(
            seeds, learning_rates, anchors, sweep["anchor_profiles"], label_noise_rates, subset_fractions, bn_modes
        ):
            profile_name = profile["name"]
            groups = anchored_profile_groups(anchor, profile["ratios"])
            axis_overrides, axis_suffix = extra_axis_overrides(noise_rate, subset_fraction, str(bn_mode))
            suffix = [str(profile_name), f"anchor_{sanitized(anchor)}"]
            if len(learning_rates) > 1:
                suffix.append(f"lr_{sanitized(lr)}")
            suffix.extend(axis_suffix)
            run_sweep_cell(
                args.config,
                [
                    *base_overrides,
                    f"training.seed={seed}",
                    f"training.lr={lr}",
                    *group_overrides(groups),
                    *axis_overrides,
                    *maybe_name_override(config, suffix),
                ],
                continue_on_error,
                skip_completed,
            )
    elif config["fls"]["mode"] == "blockwise" and "anchor_interaction_pairs" in sweep:
        anchors = [float(value) for value in sweep.get("anchor_values", [1.0])]
        ratios = [float(value) for value in sweep.get("interaction_ratios", [1.0])]
        for seed, lr, anchor, pair, noise_rate, subset_fraction, bn_mode in product(
            seeds, learning_rates, anchors, sweep["anchor_interaction_pairs"], label_noise_rates, subset_fractions, bn_modes
        ):
            if len(pair) != 2:
                raise ValueError(f"interaction pair must contain exactly two groups: {pair}")
            first, second = pair
            for first_ratio, second_ratio in product(ratios, ratios):
                groups = {block_group: anchor for block_group in ["early", "middle", "late", "head"]}
                groups[first] = anchor * first_ratio
                groups[second] = anchor * second_ratio
                axis_overrides, axis_suffix = extra_axis_overrides(noise_rate, subset_fraction, str(bn_mode))
                suffix = [
                    f"{first}_x_{second}",
                    f"anchor_{sanitized(anchor)}",
                    f"{first}_{sanitized(first_ratio)}",
                    f"{second}_{sanitized(second_ratio)}",
                ]
                if len(learning_rates) > 1:
                    suffix.append(f"lr_{sanitized(lr)}")
                suffix.extend(axis_suffix)
                run_sweep_cell(
                    args.config,
                    [
                        *base_overrides,
                        f"training.seed={seed}",
                        f"training.lr={lr}",
                        *group_overrides(groups),
                        *axis_overrides,
                        *maybe_name_override(config, suffix),
                    ],
                    continue_on_error,
                    skip_completed,
                )
    elif config["fls"]["mode"] == "blockwise" and "profiles" in sweep:
        profile_base_values = sweep.get("profile_base_values", [None])
        for seed, lr, profile_name, base_values, noise_rate, subset_fraction, bn_mode in product(
            seeds, learning_rates, sweep["profiles"], profile_base_values, label_noise_rates, subset_fractions, bn_modes
        ):
            base_value_overrides, base_value_suffix = profile_base_value_overrides(base_values)
            axis_overrides, axis_suffix = extra_axis_overrides(noise_rate, subset_fraction, str(bn_mode))
            overrides = [
                f"training.seed={seed}",
                f"training.lr={lr}",
                f"fls.profile_name={profile_name}",
                "fls.groups=null",
                *base_value_overrides,
                *axis_overrides,
            ]
            name_suffix = [str(profile_name)]
            name_suffix.extend(base_value_suffix)
            name_suffix.extend(axis_suffix)
            if len(learning_rates) > 1:
                name_suffix.append(f"lr_{sanitized(lr)}")
            if base_value_suffix or axis_suffix or len(learning_rates) > 1:
                overrides.extend(maybe_name_override(config, name_suffix))
            run_sweep_cell(args.config, [*base_overrides, *overrides], continue_on_error, skip_completed)
    elif config["fls"]["mode"] == "blockwise" and "interaction_pairs" in sweep:
        fixed = sweep.get("fixed_output_multiplier", 1.0)
        for seed, lr, pair in product(seeds, learning_rates, sweep["interaction_pairs"]):
            if len(pair) != 2:
                raise ValueError(f"interaction pair must contain exactly two groups: {pair}")
            first, second = pair
            first_values, second_values = interaction_values_for_pair(sweep, first, second)
            for first_value, second_value in product(first_values, second_values):
                overrides = [f"training.seed={seed}", f"training.lr={lr}", "fls.profile_name=null"]
                for block_group in ["early", "middle", "late", "head"]:
                    if block_group == first:
                        multiplier = first_value
                    elif block_group == second:
                        multiplier = second_value
                    else:
                        multiplier = fixed
                    overrides.append(f"fls.groups.{block_group}.output_multiplier={multiplier}")
                    overrides.append(f"fls.groups.{block_group}.init_scale=1.0")
                suffix = f"{first}_x_{second}"
                if len(learning_rates) > 1:
                    suffix = f"{suffix}_lr_{lr}"
                overrides.append(f"experiment.name={config['experiment']['name']}_{suffix}")
                run_sweep_cell(args.config, [*base_overrides, *overrides], continue_on_error, skip_completed)
    elif config["fls"]["mode"] == "blockwise" and "varied_groups" in sweep:
        values = [str(value) for value in sweep.get("output_multipliers", [1.0])]
        fixed = sweep.get("fixed_output_multiplier", 1.0)
        for seed, lr, group, value in product(seeds, learning_rates, sweep["varied_groups"], values):
            overrides = [f"training.seed={seed}", f"training.lr={lr}", "fls.profile_name=null"]
            for block_group in ["early", "middle", "late", "head"]:
                multiplier = value if block_group == group else fixed
                overrides.append(f"fls.groups.{block_group}.output_multiplier={multiplier}")
                overrides.append(f"fls.groups.{block_group}.init_scale=1.0")
            suffix = str(group)
            if len(learning_rates) > 1:
                suffix = f"{suffix}_lr_{lr}"
            overrides.append(f"experiment.name={config['experiment']['name']}_{suffix}")
            run_sweep_cell(args.config, [*base_overrides, *overrides], continue_on_error, skip_completed)
    else:
        run_sweep_cell(args.config, base_overrides, continue_on_error, skip_completed)


if __name__ == "__main__":
    main()
