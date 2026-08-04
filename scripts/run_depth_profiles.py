#!/usr/bin/env python
"""Batch-compute per-block depth profiles for the existing ResNet position runs.

For each (experiment, seed) it picks the latest timestamped run that has a
checkpoint, computes the depth profile via compute_depth_profile.main(), and
writes outputs/depth_profiles/<experiment>__seed<seed>.csv. Re-analysis only --
no training, no checkpoint writes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PRESETS = {
    "legacy": [
        ("imp_pos_baseline", "imp_pos_baseline"),
        ("imp_pos_cut2", "imp_pos_cut2"),
        ("imp_pos_cut3", "imp_pos_cut3"),
        ("imp_pos_cut4", "imp_pos_cut4"),
        ("imp_pos_cut5", "imp_pos_cut5"),
        ("imp_pos_cut6", "imp_pos_cut6"),
        ("imp_pos_cut7", "imp_pos_cut7"),
        ("imp_pos_cut8", "imp_pos_cut8"),
    ],
    "validation_core": [
        ("val60_core_global_c0p25", "global"),
        ("val60_core_boundary_cut2", "cut2"),
        ("val60_core_boundary_cut8", "cut8"),
    ],
}
CKPT = "checkpoint_best.pt"


def parse_args() -> argparse.Namespace:
    """Parse the experiment preset and diagnostic device."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="legacy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=256)
    return parser.parse_args()


def latest_seed_dirs(experiment: str) -> dict[int, Path]:
    """Map seed -> newest run dir (by timestamp name) that has the checkpoint."""
    best: dict[int, Path] = {}
    best_stamp: dict[int, str] = {}
    for ckpt in Path("outputs", experiment).glob(f"*/seed_*/{CKPT}"):
        seed_dir = ckpt.parent
        seed = int(seed_dir.name.split("_")[1])
        stamp = seed_dir.parent.name  # timestamp dir
        if seed not in best_stamp or stamp > best_stamp[seed]:
            best_stamp[seed] = stamp
            best[seed] = seed_dir
    return best


def main() -> None:
    args = parse_args()
    out_dir = (
        Path("outputs/_summaries/validation_depth_repr_20260803")
        if args.preset == "validation_core"
        else Path("outputs/depth_profiles")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    total = ok = 0
    for exp, output_name in PRESETS[args.preset]:
        seed_dirs = latest_seed_dirs(exp)
        if not seed_dirs:
            print(f"[skip] {exp}: no checkpoints found", flush=True)
            continue
        for seed, run_dir in sorted(seed_dirs.items()):
            separator = "_" if args.preset == "validation_core" else "__"
            out = out_dir / f"{output_name}{separator}seed{seed}.csv"
            total += 1
            if out.exists():
                print(f"[done] {out} (exists)", flush=True)
                ok += 1
                continue
            cmd = [
                py, "scripts/compute_depth_profile.py",
                "--run-dir", str(run_dir),
                "--checkpoint", CKPT,
                "--split", "train",
                "--max-samples", str(args.max_samples),
                "--device", args.device,
                "--output", str(out),
            ]
            print(f"[run ] {exp} seed{seed} <- {run_dir}", flush=True)
            r = subprocess.run(cmd)
            ok += r.returncode == 0
    print(f"[depth-profiles] {ok}/{total} written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
