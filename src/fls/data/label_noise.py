from __future__ import annotations

import numpy as np


def symmetric_label_noise(labels: np.ndarray, num_classes: int, rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return noisy labels and a boolean corruption mask."""
    rng = np.random.default_rng(seed)
    noisy = labels.copy()
    mask = rng.random(len(labels)) < rate
    for idx in np.where(mask)[0]:
        choices = [c for c in range(num_classes) if c != labels[idx]]
        noisy[idx] = rng.choice(choices)
    return noisy, mask

