#!/usr/bin/env python
"""Validation-selected multi-architecture / multi-dataset depth profiles.

Overlays the boundary-scaling depth response across:
  * CIFAR-100 ResNet18-CIFAR  (9-block axis, cuts 2..8)   -- the original result
  * CIFAR-100 VGG19-BN-CIFAR  (16-block axis, cuts 1..15) -- architecture generality
  * Tiny-ImageNet ResNet18    (9-block axis, cuts 2..8)   -- dataset generality

The x-axis is normalized cut depth (0=shallow, 1=deep) so the different block
counts overlay. The y-axis is the change in validation-selected test accuracy
from the shallowest tested cut of the same architecture/dataset. A right-hand
panel shows the validation-selected Tiny-ImageNet late-boundary dose response.
Error bars are sample standard deviations across seeds.

Robust to partial data: only conditions with >=1 completed seed (epoch>=max-1)
are plotted, and missing series are skipped with a logged note.
Output: paper/figures/depth_profile_multiarch.pdf
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.03, "savefig.dpi": 300})

RESNET_DATA = Path("paper/tables/depth_profile_val_plot.csv")
VGG_DATA = Path("outputs/_summaries/vgg_depth_paired_20260731/cut_summary.csv")
TINY_DATA = Path("outputs/_summaries/tiny72_20260731/depth_summary.csv")
TINY_SEED_DATA = Path("outputs/_summaries/tiny72_20260731/phase1_seed_level.csv")
DOSE_DATA = Path("outputs/_summaries/tiny72_20260731/dose_summary.csv")


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read one CSV table."""

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tiny_seed_sd(cut: int) -> float:
    """Return the sample SD of validation-selected Tiny-ImageNet test accuracy."""

    values = [
        float(row["test_accuracy_at_best_val_pct"])
        for row in read_rows(TINY_SEED_DATA)
        if row["family"] == "depth_boundary" and int(row["cut"]) == cut
    ]
    return stdev(values)


def tiny_dose_sd(multiplier: float) -> float:
    """Return the sample SD for one validation-controlled dose."""

    family = "depth_boundary" if multiplier == 0.0625 else "dose_boundary"
    values = [
        float(row["test_accuracy_at_best_val_pct"])
        for row in read_rows(TINY_SEED_DATA)
        if row["family"] == family
        and int(row["cut"]) == 8
        and float(row["multiplier"]) == multiplier
    ]
    return stdev(values)


def main() -> None:
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.15, 2.8), gridspec_kw={"width_ratios": [2.0, 1.0]})

    style_by_label = {
        "CIFAR-100 ResNet18": ("o", "-"),
        "CIFAR-100 VGG19-BN": ("s", "--"),
        "Tiny-ImageNet ResNet18": ("^", "-."),
    }

    resnet_rows = read_rows(RESNET_DATA)
    vgg_rows = read_rows(VGG_DATA)
    tiny_rows = read_rows(TINY_DATA)
    profiles = {
        "CIFAR-100 ResNet18": (
            [int(row["cut"]) for row in resnet_rows],
            [int(row["cut"]) / 8.0 for row in resnet_rows],
            [float(row["boundary_mean"]) for row in resnet_rows],
            [float(row["boundary_sd"]) for row in resnet_rows],
        ),
        "CIFAR-100 VGG19-BN": (
            [int(row["cut"]) for row in vgg_rows],
            [float(row["normalized_depth"]) for row in vgg_rows],
            [float(row["boundary_mean_test_at_best_val_pct"]) for row in vgg_rows],
            [float(row["boundary_sd_pct"]) for row in vgg_rows],
        ),
        "Tiny-ImageNet ResNet18": (
            [int(row["cut"]) for row in tiny_rows],
            [int(row["cut"]) / 8.0 for row in tiny_rows],
            [float(row["mean_boundary_test_at_best_val_pct"]) for row in tiny_rows],
            [tiny_seed_sd(int(row["cut"])) for row in tiny_rows],
        ),
    }
    colors = {"CIFAR-100 ResNet18": "#2C3E50", "CIFAR-100 VGG19-BN": "#C0392B",
              "Tiny-ImageNet ResNet18": "#1f77b4"}
    for label in colors:
        raw_cuts, fracs, accuracies, sds = profiles[label]
        deltas = [accuracy - accuracies[0] for accuracy in accuracies]
        rho, p = spearmanr(raw_cuts, accuracies)
        color = colors[label]
        marker, linestyle = style_by_label[label]
        axL.errorbar(fracs, deltas, yerr=sds, marker=marker, linestyle=linestyle,
                     color=color, ms=4, lw=1.5, capsize=2.5,
                     label=f"{label}  ($\\rho={rho:.2f}$)" if rho == rho else label)
        print(f"[{label}] Spearman(cut, gain) rho={rho:.3f} p={p:.3g}")

    axL.axhline(0.0, color="#999999", ls="--", lw=1.2)
    axL.set_xlabel("Normalized cut depth (shallow $\\to$ deep)")
    axL.set_ylabel(r"$\Delta$ validation-selected accuracy (pp)")
    axL.set_title("(a) Cut-depth response")
    axL.legend(loc="best", frameon=True, framealpha=0.9)
    axL.grid(True, color="#DDDDDD", lw=0.6)
    axL.set_axisbelow(True)

    # Tiny-ImageNet dose-response panel
    dose = read_rows(DOSE_DATA)
    cs = [float(r["multiplier"]) for r in dose]
    dms = [float(r["mean_test_at_best_val_pct"]) for r in dose]
    dsd = [tiny_dose_sd(float(r["multiplier"])) for r in dose]
    axR.errorbar(cs, dms, yerr=dsd, fmt="-o", color="#1f77b4", ms=4, lw=1.5, capsize=2.5)
    axR.set_xscale("log", base=2)
    axR.set_xlabel("After-late multiplier $c$ (log scale)")
    axR.set_ylabel("Validation-selected test accuracy (%)")
    axR.set_title("(b) Tiny-ImageNet dose response")
    axR.grid(True, color="#DDDDDD", lw=0.6)
    axR.set_axisbelow(True)

    fig.tight_layout(pad=0.4, w_pad=1.0)
    out = "paper/figures/depth_profile_multiarch.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
