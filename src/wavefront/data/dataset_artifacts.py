from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wavefront.data.synthetic import (
    SyntheticDataset,
    SyntheticDatasetConfig,
    generate_synthetic_dataset,
    load_synthetic_dataset_config,
)

DATASET_SCHEMA_VERSION = 1


@dataclass
class DatasetArtifact:
    """Loaded synthetic dataset and its reproducibility metadata."""

    root_dir: Path
    wavefronts: np.ndarray
    sensor_gradients: np.ndarray
    grid_gradients: np.ndarray
    config: SyntheticDatasetConfig
    metadata: dict[str, Any]
    sensor_layout_path: Path | None

    @property
    def n_samples(self) -> int:
        """Return the total number of stored samples."""
        return int(self.wavefronts.shape[0])

    @property
    def n_train(self) -> int:
        """Return the number of training samples."""
        return int(self.config.n_train)

    @property
    def n_test(self) -> int:
        """Return the number of held-out test samples."""
        return int(self.config.n_test)

    @property
    def train_slice(self) -> slice:
        """Return the slice containing training samples."""
        return slice(0, self.n_train)

    @property
    def test_slice(self) -> slice:
        """Return the slice containing held-out test samples."""
        return slice(self.n_train, self.n_train + self.n_test)


def _sha256(path: Path) -> str:
    """Compute a SHA-256 checksum for one file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _read_json_mapping(path: Path) -> dict[str, Any]:
    """Read a JSON file and require a mapping at the top level."""
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file was not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a JSON object in {path}, "
            f"got {type(payload).__name__}."
        )

    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one UTF-8 JSON file."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write one UTF-8 YAML file."""
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            payload,
            file,
            allow_unicode=True,
            sort_keys=False,
        )


def _validate_dataset_arrays(
        dataset: SyntheticDataset,
        cfg: SyntheticDatasetConfig,
) -> None:
    """Validate array shapes before storing or returning a dataset."""
    wavefronts = np.asarray(dataset.wavefronts)
    sensor_gradients = np.asarray(dataset.sensor_gradients)
    grid_gradients = np.asarray(dataset.grid_gradients)

    n_expected = int(cfg.n_train) + int(cfg.n_test)
    grid_size = int(cfg.grid_size)
    n_grid_points = grid_size * grid_size

    if wavefronts.shape[0] != n_expected:
        raise ValueError(
            "wavefronts has an unexpected sample count: "
            f"expected {n_expected}, got {wavefronts.shape[0]}."
        )

    if sensor_gradients.shape[0] != n_expected:
        raise ValueError(
            "sensor_gradients has an unexpected sample count: "
            f"expected {n_expected}, got {sensor_gradients.shape[0]}."
        )

    if grid_gradients.shape[0] != n_expected:
        raise ValueError(
            "grid_gradients has an unexpected sample count: "
            f"expected {n_expected}, got {grid_gradients.shape[0]}."
        )

    if sensor_gradients.ndim != 3 or sensor_gradients.shape[-1] != 2:
        raise ValueError(
            "sensor_gradients must have shape (N, P_sensor, 2), "
            f"got {sensor_gradients.shape}."
        )

    if wavefronts.ndim == 2:
        if wavefronts.shape[1] != n_grid_points:
            raise ValueError(
                "Flattened wavefronts must have shape "
                f"(N, {n_grid_points}), got {wavefronts.shape}."
            )

    elif wavefronts.ndim == 3:
        if wavefronts.shape[1:] != (grid_size, grid_size):
            raise ValueError(
                "Grid-shaped wavefronts must have shape "
                f"(N, {grid_size}, {grid_size}), got {wavefronts.shape}."
            )

    else:
        raise ValueError(
            "wavefronts must have shape (N, P) or (N, H, W), "
            f"got {wavefronts.shape}."
        )

    if grid_gradients.ndim == 3:
        if grid_gradients.shape[1:] != (n_grid_points, 2):
            raise ValueError(
                "Flattened grid_gradients must have shape "
                f"(N, {n_grid_points}, 2), got {grid_gradients.shape}."
            )

    elif grid_gradients.ndim == 4:
        if grid_gradients.shape[1:] != (grid_size, grid_size, 2):
            raise ValueError(
                "Grid-shaped grid_gradients must have shape "
                f"(N, {grid_size}, {grid_size}, 2), got "
                f"{grid_gradients.shape}."
            )

    else:
        raise ValueError(
            "grid_gradients must have shape (N, P, 2) or (N, H, W, 2), "
            f"got {grid_gradients.shape}."
        )


def _prepare_output_directory(
        output_dir: Path,
        *,
        overwrite: bool,
) -> None:
    """Create an artifact directory or reject unsafe overwrite attempts."""
    if output_dir.exists():
        has_contents = any(output_dir.iterdir())

        if has_contents and not overwrite:
            raise FileExistsError(
                f"Dataset directory already exists and is not empty: "
                f"{output_dir}. Use overwrite=True only when replacement "
                "is intentional."
            )

    output_dir.mkdir(parents=True, exist_ok=True)


def save_synthetic_dataset_artifact(
        output_dir: str | Path,
        *,
        dataset: SyntheticDataset,
        config: SyntheticDatasetConfig,
        overwrite: bool = False,
        copy_sensor_layout: bool = True,
) -> DatasetArtifact:
    """
    Save a generated synthetic dataset in a portable artifact directory.

    The artifact contains arrays, generator settings, metadata, and optionally
    a copy of the sensor-layout CSV used to generate sparse sensor gradients.
    """
    output_dir = Path(output_dir).expanduser().resolve()

    _validate_dataset_arrays(
        dataset=dataset,
        cfg=config,
    )
    _prepare_output_directory(
        output_dir,
        overwrite=overwrite,
    )

    arrays_path = output_dir / "arrays.npz"
    config_path = output_dir / "generation_config.yaml"
    metadata_path = output_dir / "metadata.json"
    sensor_layout_path = output_dir / "sensor_layout.csv"

    np.savez_compressed(
        arrays_path,
        wavefronts=np.asarray(
            dataset.wavefronts,
            dtype=np.float32,
        ),
        sensor_gradients=np.asarray(
            dataset.sensor_gradients,
            dtype=np.float32,
        ),
        grid_gradients=np.asarray(
            dataset.grid_gradients,
            dtype=np.float32,
        ),
    )

    _write_yaml(
        config_path,
        asdict(config),
    )

    copied_layout_path: Path | None = None
    original_layout_path = Path(config.branch_grid_path).expanduser()

    if copy_sensor_layout:
        if not original_layout_path.is_file():
            raise FileNotFoundError(
                "The configured sensor layout could not be copied: "
                f"{original_layout_path}."
            )

        shutil.copy2(
            original_layout_path,
            sensor_layout_path,
        )
        copied_layout_path = sensor_layout_path

    metadata = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_type": "synthetic_wavefront",
        "normalization_space": "per_sample_normalized_model_space",
        "split": {
            "n_train": int(config.n_train),
            "n_test": int(config.n_test),
            "test_start_index": int(config.n_train),
        },
        "arrays": {
            "wavefronts": {
                "shape": list(dataset.wavefronts.shape),
                "dtype": str(dataset.wavefronts.dtype),
            },
            "sensor_gradients": {
                "shape": list(dataset.sensor_gradients.shape),
                "dtype": str(dataset.sensor_gradients.dtype),
            },
            "grid_gradients": {
                "shape": list(dataset.grid_gradients.shape),
                "dtype": str(dataset.grid_gradients.dtype),
            },
        },
        "sensor_layout": {
            "copied_into_artifact": copied_layout_path is not None,
            "original_path": str(original_layout_path),
            "artifact_path": (
                copied_layout_path.name
                if copied_layout_path is not None
                else None
            ),
        },
        "checksums": {
            "arrays.npz": _sha256(arrays_path),
            "generation_config.yaml": _sha256(config_path),
        },
    }

    if copied_layout_path is not None:
        metadata["checksums"]["sensor_layout.csv"] = _sha256(
            copied_layout_path
        )

    _write_json(
        metadata_path,
        metadata,
    )

    loaded_config = SyntheticDatasetConfig(
        **asdict(config),
    )

    if copied_layout_path is not None:
        loaded_config.branch_grid_path = str(copied_layout_path)

    return DatasetArtifact(
        root_dir=output_dir,
        wavefronts=np.asarray(
            dataset.wavefronts,
            dtype=np.float32,
        ),
        sensor_gradients=np.asarray(
            dataset.sensor_gradients,
            dtype=np.float32,
        ),
        grid_gradients=np.asarray(
            dataset.grid_gradients,
            dtype=np.float32,
        ),
        config=loaded_config,
        metadata=metadata,
        sensor_layout_path=copied_layout_path,
    )


def load_synthetic_dataset_artifact(
        dataset_dir: str | Path,
        *,
        verify_checksums: bool = True,
) -> DatasetArtifact:
    """
    Load and validate a portable synthetic dataset artifact.

    When ``sensor_layout.csv`` exists inside the artifact, the returned config
    uses its local path instead of the original generation-machine path.
    """
    dataset_dir = Path(dataset_dir).expanduser().resolve()

    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset artifact directory was not found: {dataset_dir}"
        )

    arrays_path = dataset_dir / "arrays.npz"
    config_path = dataset_dir / "generation_config.yaml"
    metadata_path = dataset_dir / "metadata.json"
    sensor_layout_path = dataset_dir / "sensor_layout.csv"

    metadata = _read_json_mapping(metadata_path)

    schema_version = metadata.get("schema_version")

    if schema_version != DATASET_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported dataset schema version: "
            f"{schema_version!r}. Expected {DATASET_SCHEMA_VERSION}."
        )

    if verify_checksums:
        checksums = metadata.get("checksums", {})

        if not isinstance(checksums, dict):
            raise TypeError(
                "metadata.checksums must be a mapping."
            )

        for filename, expected_digest in checksums.items():
            path = dataset_dir / filename

            if not path.is_file():
                raise FileNotFoundError(
                    f"Checksum references a missing file: {path}"
                )

            actual_digest = _sha256(path)

            if actual_digest != expected_digest:
                raise ValueError(
                    f"Checksum mismatch for {path}. The dataset artifact "
                    "may be corrupted or modified."
                )

    config = load_synthetic_dataset_config(config_path)

    with np.load(arrays_path, allow_pickle=False) as arrays:
        required_names = {
            "wavefronts",
            "sensor_gradients",
            "grid_gradients",
        }

        missing_names = required_names - set(arrays.files)

        if missing_names:
            missing_text = ", ".join(sorted(missing_names))
            raise KeyError(
                f"arrays.npz is missing required array(s): {missing_text}."
            )

        dataset = SyntheticDataset(
            wavefronts=np.asarray(
                arrays["wavefronts"],
                dtype=np.float32,
            ),
            sensor_gradients=np.asarray(
                arrays["sensor_gradients"],
                dtype=np.float32,
            ),
            grid_gradients=np.asarray(
                arrays["grid_gradients"],
                dtype=np.float32,
            ),
        )

    _validate_dataset_arrays(
        dataset=dataset,
        cfg=config,
    )

    if sensor_layout_path.is_file():
        config.branch_grid_path = str(sensor_layout_path)
        local_sensor_layout_path: Path | None = sensor_layout_path
    else:
        local_sensor_layout_path = None

    return DatasetArtifact(
        root_dir=dataset_dir,
        wavefronts=dataset.wavefronts,
        sensor_gradients=dataset.sensor_gradients,
        grid_gradients=dataset.grid_gradients,
        config=config,
        metadata=metadata,
        sensor_layout_path=local_sensor_layout_path,
    )


def generate_and_save_synthetic_dataset(
        output_dir: str | Path,
        *,
        config: SyntheticDatasetConfig,
        overwrite: bool = False,
        copy_sensor_layout: bool = True,
) -> DatasetArtifact:
    """Generate a synthetic dataset and immediately save it as an artifact."""
    dataset = generate_synthetic_dataset(config)

    return save_synthetic_dataset_artifact(
        output_dir,
        dataset=dataset,
        config=config,
        overwrite=overwrite,
        copy_sensor_layout=copy_sensor_layout,
    )
