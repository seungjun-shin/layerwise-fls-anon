#!/usr/bin/env python
"""Test whether a held-out separability metric orders a confound-control family.

For each condition (global / LR-only / no-comp / late-boundary), load each seed's
checkpoint_best, extract late-stage TEST features, and compute a scale-invariant
Fisher discriminant ratio (between-class / within-class scatter) plus test-set
clean-label alignment. Aggregate mean+-sd across seeds.

The script derives both the accuracy ordering and the Fisher-ratio ordering
directly from the selected run outputs.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from fls.data.datasets import build_dataset
from fls.diagnostics.alignment import class_means, mean_class_alignment
from fls.training.seed import set_seed

# reuse loading/feature extraction from the existing diagnostics script
import importlib.util
_spec = importlib.util.spec_from_file_location("compute_diagnostics", Path(__file__).parent / "compute_diagnostics.py")
_cd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cd)
load_yaml = _cd.load_yaml
build_scaled_model = _cd.build_scaled_model
collect_features = _cd.collect_features

LEGACY_CONDITIONS = {
    "global":     "experiment_49_global_matched_seed_expansion",
    "lr_only":    "experiment_75_boundary_late_lr_only_ablation",
    "no_comp":    "experiment_63_boundary_late_no_comp_ablation",
    "late_bound": "experiment_40_boundary_late_seed_expansion",
}
VAL60_CONDITIONS = {
    "global": "val60_core_global_c0p25",
    "lr_only": "val60_core_lronly_cut8",
    "no_comp": "val60_core_nocomp_cut8",
    "late_bound": "val60_core_boundary_cut8",
}


def reported_accuracy(run_dir: Path) -> float:
    """Return test accuracy at the validation-selected checkpoint when available."""
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and rows[0].get("val_accuracy") not in {None, ""}:
        selected = max(rows, key=lambda row: float(row["val_accuracy"]))
    else:
        selected = max(rows, key=lambda row: float(row["test_accuracy"]))
    return 100.0 * float(selected["test_accuracy"])


def fisher_ratio(features: torch.Tensor, labels: torch.Tensor) -> float:
    """Scale-invariant between/within class scatter ratio."""
    x = features.detach().float().reshape(features.shape[0], -1)
    means = class_means(x, labels)
    if len(means) < 2:
        return 0.0
    within = torch.stack([(x[labels == c] - m).pow(2).sum(1).mean() for c, m in means.items()]).mean()
    mvals = torch.stack(list(means.values()))
    gmean = mvals.mean(0)
    between = (mvals - gmean).pow(2).sum(1).mean()
    return float((between / (within + 1e-12)).item())


def run_one(run_dir: Path, split: str, max_samples: int, device: torch.device) -> dict | None:
    cfg_path = run_dir / "resolved_config.yaml"
    ckpt_path = run_dir / "checkpoint_best.pt"
    if not cfg_path.exists() or not ckpt_path.exists():
        return None
    config = load_yaml(cfg_path)
    set_seed(int(config["training"]["seed"]))
    dataset = build_dataset(config, train=(split == "train"))
    n = min(max_samples, len(dataset))
    gen = torch.Generator().manual_seed(12345)
    idx = torch.randperm(len(dataset), generator=gen)[:n].tolist()
    loader = DataLoader(Subset(dataset, idx), batch_size=256, shuffle=False, num_workers=0)
    set_seed(int(config["training"]["seed"]))
    model = build_scaled_model(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    feats, y_clean, _, _ = collect_features(model, loader, device, n)
    late = feats.get("late")
    if late is None:
        return None
    return {
        "fisher_ratio": fisher_ratio(late, y_clean),
        "clean_alignment": mean_class_alignment(late, y_clean),
        "accuracy": reported_accuracy(run_dir),
        "seed": int(config["training"]["seed"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--max-samples", type=int, default=5000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/discriminating_diagnostic.md")
    ap.add_argument("--protocol", choices=["legacy", "val60"], default="legacy")
    args = ap.parse_args()
    device = torch.device(args.device)

    lines = [f"# M2 discriminating diagnostic ({args.split} split, {args.max_samples} samples, late stage)", "",
             "| condition | accuracy | Fisher ratio (between/within) | clean alignment | n seeds |",
             "|---|---|---|---|---|"]
    table = []
    conditions = VAL60_CONDITIONS if args.protocol == "val60" else LEGACY_CONDITIONS
    for cond, prefix in conditions.items():
        recs = []
        for run_dir in sorted(Path("outputs").glob(f"{prefix}*/**/seed_*")):
            if not run_dir.is_dir():
                continue
            r = run_one(run_dir, args.split, args.max_samples, device)
            if r:
                recs.append(r)
        if not recs:
            lines.append(f"| {cond} | — | — | — | 0 |")
            continue
        fr = [r["fisher_ratio"] for r in recs]
        al = [r["clean_alignment"] for r in recs]
        acc = [r["accuracy"] for r in recs]
        frm = statistics.mean(fr)
        accm = statistics.mean(acc)
        table.append((cond, accm, frm))
        sd = statistics.pstdev(fr) if len(fr) > 1 else 0.0
        asd = statistics.pstdev(al) if len(al) > 1 else 0.0
        lines.append(f"| {cond} | {accm:.2f} | {frm:.3f}±{sd:.3f} | {statistics.mean(al):.3f}±{asd:.3f} | {len(recs)} |")
        print(f"{cond:11s} acc={accm:.2f}  fisher={frm:.3f}  align={statistics.mean(al):.3f}  (n={len(recs)})", flush=True)

    # does Fisher ratio order match accuracy order?
    if len(table) >= 2:
        by_acc = [c for c, _, _ in sorted(table, key=lambda t: -t[1])]
        by_fisher = [c for c, _, _ in sorted(table, key=lambda t: -t[2])]
        lines += ["", f"Accuracy order:  {' > '.join(by_acc)}", f"Fisher order:    {' > '.join(by_fisher)}",
                  f"**Orders match: {by_acc == by_fisher}** — if true, test-set Fisher ratio discriminates good/bad representations (resolves M2)."]
        print("accuracy order:", by_acc)
        print("fisher order:  ", by_fisher, "MATCH" if by_acc == by_fisher else "NO MATCH")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[written {args.out}]")


if __name__ == "__main__":
    main()
