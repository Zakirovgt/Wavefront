from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wavefront.data.dataset_artifacts import (
    load_synthetic_dataset_artifact,
)


@dataclass(frozen=True)
class PipelineDataset:
    """One validated dataset artifact prepared for multi-stage training."""

    root_dir: Path
    wavefronts: np.ndarray
    sensor_gradients: np.ndarray
    grid_gradients: np.ndarray

    grid_size: int
    n_train: int
    n_test: int

    branch_grid_path: str
    provenance: dict[str, Any]

    @property
    def n_samples(self) -> int:
        """Return the total number of stored samples."""
        return int(self.wavefronts.shape[0])

    @property
    def train_slice(self) -> slice:
        """Return the training subset slice."""
        return slice(0, self.n_train)

    @property
    def test_slice(self) -> slice:
        """Return the held-out validation subset slice."""
        return slice(
            self.n_train,
            self.n_train + self.n_test,
        )

    @property
    def train_wavefronts(self) -> np.ndarray:
        """Return training wavefront targets."""
        return self.wavefronts[self.train_slice]

    @property
    def test_wavefronts(self) -> np.ndarray:
        """Return held-out wavefront targets."""
        return self.wavefronts[self.test_slice]

    @property
    def train_sensor_gradients(self) -> np.ndarray:
        """Return training sparse sensor gradients."""
        return self.sensor_gradients[self.train_slice]

    @property
    def test_sensor_gradients(self) -> np.ndarray:
        """Return held-out sparse sensor gradients."""
        return self.sensor_gradients[self.test_slice]

    @property
    def train_grid_gradients(self) -> np.ndarray:
        """Return training clean regular-grid gradients."""
        return self.grid_gradients[self.train_slice]

    @property
    def test_grid_gradients(self) -> np.ndarray:
        """Return held-out clean regular-grid gradients."""
        return self.grid_gradients[self.test_slice]


def _flatten_grid_gradients(
        grid_gradients,
        *,
        grid_size: int,
) -> np.ndarray:
    """
    Convert grid gradients to shape ``(N, grid_size**2, 2)``.

    The artifact may store either flattened gradients or channel-last grids.
    Stage-1 DeepONet uses the flattened representation.
    """
    grid_gradients = np.asarray(
        grid_gradients,
        dtype=np.float32,
    )

    n_points = int(grid_size) ** 2

    if grid_gradients.ndim == 3:
        if grid_gradients.shape[1:] != (n_points, 2):
            raise ValueError(
                "Flattened grid gradients must have shape "
                f"(N, {n_points}, 2), got {grid_gradients.shape}."
            )

        return grid_gradients

    if grid_gradients.ndim == 4:
        expected_shape = (
            grid_size,
            grid_size,
            2,
        )

        if grid_gradients.shape[1:] != expected_shape:
            raise ValueError(
                "Grid-shaped gradients must have shape "
                f"(N, {grid_size}, {grid_size}, 2), "
                f"got {grid_gradients.shape}."
            )

        return grid_gradients.reshape(
            grid_gradients.shape[0],
            n_points,
            2,
        )

    raise ValueError(
        "grid_gradients must have shape (N, P, 2) or "
        f"(N, {grid_size}, {grid_size}, 2), "
        f"got {grid_gradients.shape}."
    )


def _validate_expected_dimensions(
        *,
        artifact_config,
        grid_size: int,
        n_train: int,
        n_test: int,
) -> None:
    """Check that an artifact matches the pipeline configuration."""
    expected = {
        "grid_size": int(grid_size),
        "n_train": int(n_train),
        "n_test": int(n_test),
    }

    mismatches = []

    for name, expected_value in expected.items():
        artifact_value = int(
            getattr(artifact_config, name)
        )

        if artifact_value != expected_value:
            mismatches.append(
                f"{name}: artifact={artifact_value}, "
                f"pipeline={expected_value}"
            )

    if mismatches:
        raise ValueError(
            "Dataset artifact is incompatible with the pipeline "
            "configuration: "
            + "; ".join(mismatches)
        )


def validate_stage1_dimensions(
        *,
        data_cfg,
        stage1_cfg,
) -> None:
    """
    Ensure Stage-1 and pipeline data use exactly the same split and grid.
    """
    mismatches = []

    for name in ("grid_size", "n_train", "n_test"):
        data_value = int(getattr(data_cfg, name))
        stage1_value = int(getattr(stage1_cfg, name))

        if data_value != stage1_value:
            mismatches.append(
                f"{name}: data_cfg={data_value}, "
                f"stage1_cfg={stage1_value}"
            )

    if mismatches:
        raise ValueError(
            "Stage-1 configuration is incompatible with pipeline data: "
            + "; ".join(mismatches)
        )


def load_pipeline_dataset_artifact(
        dataset_dir: str | Path,
        *,
        grid_size: int,
        n_train: int,
        n_test: int,
) -> PipelineDataset:
    """
    Load a portable dataset artifact for a two- or three-stage pipeline.

    The artifact's local ``sensor_layout.csv`` is preferred over the original
    generation-machine path, making the pipeline portable between machines.
    """
    artifact = load_synthetic_dataset_artifact(dataset_dir)

    _validate_expected_dimensions(
        artifact_config=artifact.config,
        grid_size=grid_size,
        n_train=n_train,
        n_test=n_test,
    )

    sensor_layout_path = artifact.sensor_layout_path

    if sensor_layout_path is None:
        fallback_path = Path(
            artifact.config.branch_grid_path
        ).expanduser()

        if not fallback_path.is_file():
            raise FileNotFoundError(
                "Dataset artifact does not contain sensor_layout.csv and "
                "the original branch_grid_path is unavailable."
            )

        sensor_layout_path = fallback_path

    wavefronts = np.asarray(
        artifact.wavefronts,
        dtype=np.float32,
    )
    sensor_gradients = np.asarray(
        artifact.sensor_gradients,
        dtype=np.float32,
    )
    grid_gradients = _flatten_grid_gradients(
        artifact.grid_gradients,
        grid_size=int(grid_size),
    )

    expected_samples = int(n_train) + int(n_test)

    if wavefronts.shape[0] != expected_samples:
        raise ValueError(
            "Artifact wavefront count does not match the requested split: "
            f"expected {expected_samples}, got {wavefronts.shape[0]}."
        )

    if sensor_gradients.shape[0] != expected_samples:
        raise ValueError(
            "Artifact sensor-gradient count does not match the requested "
            f"split: expected {expected_samples}, "
            f"got {sensor_gradients.shape[0]}."
        )

    if grid_gradients.shape[0] != expected_samples:
        raise ValueError(
            "Artifact grid-gradient count does not match the requested "
            f"split: expected {expected_samples}, "
            f"got {grid_gradients.shape[0]}."
        )

    return PipelineDataset(
        root_dir=artifact.root_dir,
        wavefronts=wavefronts,
        sensor_gradients=sensor_gradients,
        grid_gradients=grid_gradients,
        grid_size=int(grid_size),
        n_train=int(n_train),
        n_test=int(n_test),
        branch_grid_path=str(sensor_layout_path),
        provenance={
            "source": "dataset_artifact",
            "dataset_dir": str(artifact.root_dir),
            "schema_version": artifact.metadata.get(
                "schema_version"
            ),
            "checksums": artifact.metadata.get(
                "checksums",
                {},
            ),
            "generation_config": asdict(artifact.config),
            "sensor_layout_path": str(sensor_layout_path),
            "dataset_reuse_policy": (
                "The same artifact is used by every pipeline stage. "
                "Only its first n_train samples are used for optimization; "
                "the following n_test samples are validation-only."
            ),
        },
    )
