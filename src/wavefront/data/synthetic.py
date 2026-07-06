from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wavefront.data.generators import generate_mixed_span_dataset


@dataclass
class SyntheticDatasetConfig:
    """Configuration for one synthetic Wavefront train/test dataset."""

    grid_size: int = 24

    n_train: int = 10_000
    n_test: int = 1_000

    branch_grid_path: str = "1.csv"

    frac_zernike: float = 0.0
    frac_spiral: float = 0.0
    frac_distortion: float = 1.0

    with_noise: bool = True
    noise_percentage: float = 1.0
    noise_lambda: float = 0.03

    apply_blur: bool = False
    sigma_pix: float = 1.0

    seed: int = 42

    def __post_init__(self) -> None:
        """Validate dataset dimensions and mixture settings."""
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2.")

        if self.n_train < 1:
            raise ValueError("n_train must be at least 1.")

        if self.n_test < 1:
            raise ValueError("n_test must be at least 1.")

        fractions = np.asarray(
            [
                self.frac_zernike,
                self.frac_spiral,
                self.frac_distortion,
            ],
            dtype=np.float64,
        )

        if np.any(fractions < 0.0):
            raise ValueError(
                "Dataset mixture fractions must be non-negative."
            )

        if not np.isclose(
            np.sum(fractions),
            1.0,
            atol=1e-8,
        ):
            raise ValueError(
                "frac_zernike + frac_spiral + frac_distortion "
                "must equal 1.0."
            )

        if self.noise_percentage < 0.0:
            raise ValueError(
                "noise_percentage must be non-negative."
            )

        if self.noise_lambda < 0.0:
            raise ValueError(
                "noise_lambda must be non-negative."
            )

        if self.sigma_pix < 0.0:
            raise ValueError(
                "sigma_pix must be non-negative."
            )


@dataclass
class SyntheticDataset:
    """Generated Wavefront data used by standalone training."""

    wavefronts: np.ndarray
    sensor_gradients: np.ndarray
    grid_gradients: np.ndarray


def _read_yaml_mapping(
    path: str | Path,
) -> dict[str, Any]:
    """Read one YAML configuration file as a mapping."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    if payload is None:
        return {}

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a YAML mapping in {path}, "
            f"got {type(payload).__name__}."
        )

    return payload


def load_synthetic_dataset_config(
    path: str | Path,
) -> SyntheticDatasetConfig:
    """Load and validate a synthetic dataset YAML configuration."""
    payload = _read_yaml_mapping(path)

    allowed_keys = {
        "grid_size",
        "n_train",
        "n_test",
        "branch_grid_path",
        "frac_zernike",
        "frac_spiral",
        "frac_distortion",
        "with_noise",
        "noise_percentage",
        "noise_lambda",
        "apply_blur",
        "sigma_pix",
        "seed",
    }

    unknown_keys = set(payload) - allowed_keys

    if unknown_keys:
        unknown_text = ", ".join(sorted(unknown_keys))
        raise KeyError(
            "Unknown key(s) in standalone dataset configuration: "
            f"{unknown_text}."
        )

    return SyntheticDatasetConfig(**payload)


def generate_synthetic_dataset(
    cfg: SyntheticDatasetConfig,
) -> SyntheticDataset:
    """
    Generate one synthetic Wavefront train/test dataset.

    Returns:
        SyntheticDataset containing:

            wavefronts:
                Shape ``(N, grid_size * grid_size)`` or equivalent.

            sensor_gradients:
                Shape ``(N, P_sensor, 2)``.

            grid_gradients:
                Shape ``(N, grid_size * grid_size, 2)`` or equivalent.
    """
    (
        wavefronts,
        sensor_gradients,
        grid_gradients,
        *_,
    ) = generate_mixed_span_dataset(
        coarse_size=int(cfg.grid_size),
        num_train=int(cfg.n_train),
        num_test=int(cfg.n_test),
        frac_zernike=float(cfg.frac_zernike),
        frac_spiral=float(cfg.frac_spiral),
        frac_distortion=float(cfg.frac_distortion),
        with_noise=bool(cfg.with_noise),
        noise_percentage=float(cfg.noise_percentage),
        noise_lambda=float(cfg.noise_lambda),
        apply_blur=bool(cfg.apply_blur),
        sigma_pix=float(cfg.sigma_pix),
        seed=int(cfg.seed),
        save_dir=None,
        sensor_csv_path=cfg.branch_grid_path,
    )

    return SyntheticDataset(
        wavefronts=np.asarray(
            wavefronts,
            dtype=np.float32,
        ),
        sensor_gradients=np.asarray(
            sensor_gradients,
            dtype=np.float32,
        ),
        grid_gradients=np.asarray(
            grid_gradients,
            dtype=np.float32,
        ),
    )