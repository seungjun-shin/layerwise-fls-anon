from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_curve(metrics_csv: str | Path, x: str, y: str, output_path: str | Path) -> None:
    """Plot a metric curve from a metrics CSV file."""
    df = pd.read_csv(metrics_csv)
    ax = df.plot(x=x, y=y)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path)
    plt.close(ax.figure)

