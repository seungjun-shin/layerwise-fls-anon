from __future__ import annotations

import random


def build_profile(
    name: str,
    groups: list[str],
    base_values: dict | None = None,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Build a block-wise output multiplier profile.

    Progressive increasing means stronger feature learning in deeper blocks, so output
    multipliers decrease with depth under the global FLS convention.
    """
    base_values = base_values or {"weak": 4.0, "optimal": 1.0, "strong": 0.25}
    weak = float(base_values.get("weak", 4.0))
    optimal = float(base_values.get("optimal", 1.0))
    strong = float(base_values.get("strong", 0.25))
    if name == "uniform_weak":
        values = [weak] * len(groups)
    elif name == "uniform_optimal":
        values = [optimal] * len(groups)
    elif name == "uniform_strong":
        values = [strong] * len(groups)
    elif name == "uniform_same_average":
        progressive_values = [weak, optimal, strong, optimal][: len(groups)]
        values = [sum(progressive_values) / len(progressive_values)] * len(groups)
    elif name == "progressive_increasing":
        values = [weak, optimal, strong, optimal][: len(groups)]
    elif name == "progressive_decreasing":
        values = [strong, optimal, weak, optimal][: len(groups)]
    elif name in {"random", "shuffled", "shuffled_progressive"}:
        values = [weak, optimal, strong, optimal][: len(groups)]
        rng = random.Random(seed)
        rng.shuffle(values)
    else:
        raise ValueError(f"Unknown profile: {name}")
    return {group: {"output_multiplier": value, "init_scale": 1.0} for group, value in zip(groups, values, strict=False)}
