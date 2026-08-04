from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def parse_value(raw: str) -> Any:
    """Parse a CLI override value using YAML scalar rules."""
    return yaml.safe_load(raw)


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested config key using dot notation."""
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Return a config with `key=value` CLI overrides applied."""
    resolved = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must have form key=value: {override}")
        key, raw_value = override.split("=", 1)
        set_nested(resolved, key, parse_value(raw_value))
    return resolved


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML and apply optional CLI overrides."""
    config = load_yaml(path)
    resolved = apply_overrides(config, overrides or [])
    validate_config(resolved)
    return resolved


def validate_config(config: dict[str, Any]) -> None:
    """Validate fields required by the first implementation."""
    required = ["experiment", "output", "data", "model", "fls", "training"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a resolved config to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

