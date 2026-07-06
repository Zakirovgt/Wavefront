from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml


def _flatten_sections(config: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten nested YAML sections into a dictionary compatible with trainers.

    Example:
        {"training": {"lr": 1e-3}, "model": {"width": 32}}
    becomes:
        {"lr": 1e-3, "width": 32}
    """
    flat: dict[str, Any] = {}

    for key, value in config.items():
        if key in {"training", "model", "loss"}:
            if value is None:
                continue
            if not isinstance(value, dict):
                raise TypeError(
                    f"Configuration section '{key}' must be a mapping."
                )
            flat.update(value)
        else:
            flat[key] = value

    return flat


def _read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML configuration file and return an empty dict for empty files."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return {} if data is None else data


def load_config(
    operator_type: str,
    config_dir: str | Path = "configs",
    **overrides: Any,
) -> Namespace:
    """
    Load common and operator-specific configuration as argparse.Namespace.

    Args:
        operator_type: Either "deeponet" or "fno".
        config_dir: Directory containing common.yaml and model YAML files.
        **overrides: Values that replace YAML values at runtime.

    Returns:
        Namespace compatible with the current training functions.
    """
    if operator_type not in {"deeponet", "fno"}:
        raise ValueError(
            f"Unknown operator_type={operator_type!r}. "
            "Expected 'deeponet' or 'fno'."
        )

    config_dir = Path(config_dir)

    common = _flatten_sections(
        _read_yaml(config_dir / "common.yaml")
    )
    operator = _flatten_sections(
        _read_yaml(config_dir / f"{operator_type}.yaml")
    )

    config = {**common, **operator, **overrides}
    config["operator_type"] = operator_type

    grid_size = int(config["grid_size"])

    if operator_type == "deeponet":
        config["p_test"] = grid_size * grid_size
        config["p_ics_train"] = grid_size * grid_size

        if config.get("n_sensors") is None:
            config["n_sensors"] = config["p_sensors"]

    elif operator_type == "fno":
        config["nx"] = grid_size
        config["ny"] = grid_size

    # Preserves compatibility with the existing notebook convention.
    config["epochs"] = config["steps"]

    return Namespace(**config)