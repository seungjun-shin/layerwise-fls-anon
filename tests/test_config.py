from pathlib import Path

from fls.utils.config import load_config, save_config


def test_config_load_override_and_save(tmp_path: Path) -> None:
    config = load_config("configs/smoke_test.yaml", ["training.seed=3", "fls.global.output_multiplier=0.5"])
    assert config["training"]["seed"] == 3
    assert config["fls"]["global"]["output_multiplier"] == 0.5
    path = tmp_path / "resolved_config.yaml"
    save_config(config, path)
    assert path.exists()

