import subprocess
import sys
from pathlib import Path


def test_smoke_train_creates_outputs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", "configs/smoke_test.yaml", "experiment.name=pytest_smoke"],
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = result.stdout.strip().splitlines()[-1]
    assert run_dir
    assert (Path(run_dir) / "metrics.csv").exists()
    assert (Path(run_dir) / "resolved_config.yaml").exists()
    assert (Path(run_dir) / "checkpoint_last.pt").exists()
