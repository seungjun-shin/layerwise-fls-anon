from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_run_dir(config: dict) -> Path:
    """Create an output directory of form outputs/name/timestamp/seed_seed."""
    root = Path(config["output"]["root"])
    name = config["experiment"]["name"]
    seed = config["training"]["seed"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / name / timestamp / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "diagnostics").mkdir()
    (run_dir / "plots").mkdir()
    return run_dir

