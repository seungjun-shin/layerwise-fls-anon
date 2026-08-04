"""Plot the validation-controlled ACT-minus-LR scope profile.

The source table is generated from the frozen scope experiment family. Each
row uses paired seeds 0--4 and validation-selected checkpoints.  Error bars are
95% paired t intervals; Holm-adjusted p-values cover the seven ACT--LR scope
contrasts.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOURCE = Path(
    "paper_reproduction/analysis/statistics/accuracy_paired_summary.csv"
)
OUTPUT = Path("paper/figures/scope_stress_forest.pdf")
GROUPS = [
    ("Dataset transfer", ["cifar10", "stl10", "svhn"]),
    ("Architecture transfer", ["nobn"]),
    ("Training stress", ["noise20", "noise50", "persistence"]),
]
ORDER = [key for _, keys in GROUPS for key in keys]
LABELS = {
    "cifar10": "CIFAR-10",
    "nobn": "CIFAR-100, no BN",
    "noise20": "CIFAR-100, 20% noise",
    "noise50": "CIFAR-100, 50% noise",
    "persistence": "Augmentation + cosine",
    "stl10": "STL-10",
    "svhn": "SVHN",
}


def main() -> None:
    """Read the frozen paired summary and write the publication figure."""

    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = {
            row["evidence_family"]: row
            for row in csv.DictReader(handle)
            if row["comparison"] == "act - lr"
            and row["multiplicity_family"] == "scope_act_lr"
        }

    missing = [key for key in ORDER if key not in rows]
    if missing:
        raise ValueError(f"missing scope rows: {missing}")

    values = [float(rows[key]["mean_difference_pp"]) for key in ORDER]
    low = [float(rows[key]["ci95_low_pp"]) for key in ORDER]
    high = [float(rows[key]["ci95_high_pp"]) for key in ORDER]
    # Leave a compact header row above each family.  The grouped layout restores
    # the visual hierarchy of the earlier scope figure without changing the
    # comparisons or their inferential quantities.
    header_y: dict[str, float] = {}
    row_y: dict[str, float] = {}
    cursor = 0.40
    for group_name, keys in GROUPS:
        header_y[group_name] = cursor
        cursor += 0.62
        for key in keys:
            row_y[key] = cursor
            cursor += 0.58
        cursor += 0.18

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.2,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 8.2,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "savefig.dpi": 300,
        }
    )
    fig, ax = plt.subplots(figsize=(7.15, 3.62))
    x_min, x_max = -5.25, 8.15
    ax.axvspan(x_min, 0.0, color="#B85C4A", alpha=0.055, linewidth=0)
    ax.axvspan(0.0, x_max, color="#4F7F65", alpha=0.055, linewidth=0)
    ax.axvline(0.0, color="#3E4349", linewidth=1.05, zorder=2)

    estimate_color = "#244F73"
    for key, mean, lo, hi in zip(ORDER, values, low, high):
        y = row_y[key]
        ax.errorbar(
            mean,
            y,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o",
            markersize=4.7,
            color=estimate_color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            ecolor=estimate_color,
            elinewidth=1.05,
            capsize=2.4,
            capthick=0.9,
            zorder=4,
        )
        if mean >= 0:
            text_x, text_y, horizontal_alignment = hi + 0.12, y, "left"
            text_box = None
        else:
            text_x, text_y, horizontal_alignment = mean, y - 0.17, "center"
            text_box = {
                "facecolor": "#FAF6F4",
                "edgecolor": "none",
                "boxstyle": "square,pad=0.08",
            }
        ax.text(
            text_x,
            text_y,
            f"{mean:+.3f}".replace("-", "−"),
            ha=horizontal_alignment,
            va="center",
            color=estimate_color,
            fontsize=7.9,
            fontweight="semibold",
            bbox=text_box,
            zorder=5,
        )

    for group_name, _ in GROUPS:
        ax.text(
            x_min + 0.16,
            header_y[group_name],
            group_name,
            ha="left",
            va="center",
            color="#2F3337",
            fontsize=8.4,
            fontweight="bold",
            fontstyle="italic",
        )

    ax.set_yticks([row_y[key] for key in ORDER])
    ax.set_yticklabels([LABELS[key] for key in ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Test-accuracy difference, ACT − exact LR-only (pp)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(cursor - 0.08, -0.30)
    ax.grid(axis="x", color="#B7BCC2", alpha=0.42, linewidth=0.55)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.tick_params(axis="x", length=3.0, width=0.7, color="#4D5155")
    ax.text(
        0.014,
        0.99,
        "LR-only higher",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#A65345",
        fontsize=7.7,
        fontstyle="italic",
    )
    ax.text(
        0.986,
        0.99,
        "ACT higher",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#3E7355",
        fontsize=7.7,
        fontstyle="italic",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#4D5155")
    ax.spines["bottom"].set_linewidth(0.75)
    fig.subplots_adjust(left=0.255, right=0.975, bottom=0.17, top=0.97)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
