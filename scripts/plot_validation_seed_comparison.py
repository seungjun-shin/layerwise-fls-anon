#!/usr/bin/env python
"""Plot the validation-selected seed-level final-boundary comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE = Path("paper/tables/matched_final_boundary_seed_plot.csv")
OUTPUT = Path("paper/figures/matched_seed_comparison.pdf")
CONDITIONS = (
    ("G", "Global FLS", "#7f7f7f"),
    ("Rbest", "Best tested LR-only (cut 6)", "#9467bd"),
    ("Rsame", "Exact same-cut LR-only", "#ff7f0e"),
    ("L", "After-late ACT", "#2ca02c"),
)


def main() -> None:
    """Read the audit table and write the manuscript figure."""

    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seeds = sorted({int(row["seed"]) for row in rows})
    if seeds != list(range(10)):
        raise RuntimeError(f"Expected seeds 0--9, found {seeds}")

    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    x = np.arange(len(CONDITIONS), dtype=float)
    offsets = np.linspace(-0.16, 0.16, len(seeds))
    for seed, offset in zip(seeds, offsets, strict=True):
        seed_rows = {row["condition"]: row for row in rows if int(row["seed"]) == seed}
        values = [float(seed_rows[name]["test_accuracy_percent"]) for name, _, _ in CONDITIONS]
        ax.plot(x + offset, values, color="#BDBDBD", linewidth=0.55, alpha=0.55, zorder=1)
        ax.scatter(x + offset, values, s=9, color="#555555", alpha=0.7, zorder=2)

    for index, (name, _, color) in enumerate(CONDITIONS):
        values = [
            float(row["test_accuracy_percent"])
            for row in rows
            if row["condition"] == name
        ]
        ax.scatter(index, np.mean(values), s=55, color=color, edgecolor="white", linewidth=0.8, zorder=4)

    ax.set_xticks(x, [label for _, label, _ in CONDITIONS], rotation=17, ha="right")
    ax.set_ylabel("Validation-selected test accuracy (%)")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.35)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", dpi=300)
    print(OUTPUT)


if __name__ == "__main__":
    main()
