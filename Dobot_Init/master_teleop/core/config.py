from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


MASTER_TELEOP_ROOT = Path(__file__).resolve().parents[1]
MASTER_CONFIG_DIR = MASTER_TELEOP_ROOT / "config"
MASTER_RUNTIME_CONFIG = MASTER_CONFIG_DIR / "master_runtime.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Master-hand config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping: {config_path}")
    return loaded


def load_master_project_config(config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load baseline calibration and the generated runtime override."""
    config_root = Path(config_dir).expanduser().resolve() if config_dir else MASTER_CONFIG_DIR
    config = load_yaml(config_root / "master_hand.yaml")
    runtime_path = config_root / "master_runtime.yaml"
    if runtime_path.exists():
        config = _deep_merge(config, load_yaml(runtime_path))
    return config


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
