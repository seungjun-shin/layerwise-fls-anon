#!/usr/bin/env python
"""Audit experiment runs for epoch completeness.

Scans outputs/<prefix>*/**/seed_*/metrics.csv and reports, per run, the last
recorded epoch vs the configured max_epochs (read from resolved_config.yaml).
Flags any run that did not reach max_epochs-1 (incomplete / early-stopped /
crashed), and any experiment-name prefix with missing seeds.

Usage:
    python scripts/check_epochs.py experiment_88 experiment_90 ...
    python scripts/check_epochs.py --expect-seeds 5 experiment_88
    python scripts/check_epochs.py --all          # audit everything under outputs/
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import yaml


def last_epoch(metrics: Path) -> int | None:
    if not metrics.exists():
        return None
    with metrics.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return max(int(float(r.get("epoch", -1) or -1)) for r in rows)


def expected_epochs(resolved: Path) -> int | None:
    if not resolved.exists():
        return None
    cfg = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    try:
        return int(cfg["training"]["max_epochs"])
    except (KeyError, TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="*", help="Experiment-name prefixes to audit.")
    ap.add_argument("--all", action="store_true", help="Audit every run under outputs/.")
    ap.add_argument("--expect-seeds", type=int, default=None, help="Flag experiments with fewer seeds than this.")
    ap.add_argument("--outputs", default="outputs")
    args = ap.parse_args()

    out = Path(args.outputs)
    run_dirs = sorted(out.glob("**/seed_*/metrics.csv"))
    incomplete: list[str] = []
    ok = 0
    by_experiment: dict[str, set[int]] = defaultdict(set)

    for metrics in run_dirs:
        seed_dir = metrics.parent
        exp_name = None
        # experiment name = the directory two levels up from seed dir is timestamp; name is above that
        # outputs/<name>/<timestamp>/seed_x/metrics.csv
        try:
            exp_name = seed_dir.parent.parent.name
        except Exception:
            exp_name = str(seed_dir)
        if args.prefixes and not any(exp_name.startswith(p) for p in args.prefixes):
            continue
        seed = seed_dir.name.replace("seed_", "")
        try:
            by_experiment[exp_name].add(int(seed))
        except ValueError:
            pass
        le = last_epoch(metrics)
        exp = expected_epochs(seed_dir / "resolved_config.yaml")
        tag = f"{exp_name}/seed_{seed}"
        if le is None:
            incomplete.append(f"{tag}: NO metrics rows")
        elif exp is None:
            incomplete.append(f"{tag}: last_epoch={le}, max_epochs UNKNOWN")
        elif le < exp - 1:
            incomplete.append(f"{tag}: last_epoch={le} < expected {exp - 1}  (INCOMPLETE)")
        else:
            ok += 1

    print(f"[check_epochs] audited {ok + len(incomplete)} runs: {ok} complete, {len(incomplete)} flagged")
    for line in incomplete:
        print(f"  [!] {line}")
    if args.expect_seeds is not None:
        for exp_name, seeds in sorted(by_experiment.items()):
            if len(seeds) < args.expect_seeds:
                missing = set(range(args.expect_seeds)) - seeds
                print(f"  [!] {exp_name}: only {len(seeds)}/{args.expect_seeds} seeds present (missing {sorted(missing)})")
    if incomplete:
        raise SystemExit(1)
    print("[check_epochs] all audited runs reached full epoch budget.")


if __name__ == "__main__":
    main()
