from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(values: np.ndarray, output_path: str | Path, title: str = "") -> None:
    """Save a simple matplotlib heatmap."""
    fig, ax = plt.subplots()
    image = ax.imshow(values)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

