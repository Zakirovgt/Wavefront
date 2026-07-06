from __future__ import annotations

import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from wavefront.config import load_config
from wavefront.data.dataset_artifacts import (
    DatasetArtifact,
    load_synthetic_dataset_artifact,
)
from wavefront.data.synthetic import (
    SyntheticDataset,
    SyntheticDatasetConfig,
    generate_synthetic_dataset,
)
from wavefront.training.runner import main_routine


def _to_jsonable(value: Any) -> Any:
    """Convert common Python and NumPy values into JSON-safe objects."""
    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    if isinstance(value, np.generic):
        return _to_jsonable(value.item())

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]

    return str(value)


def _write_json(
        path: Path,
        payload: dict[str, Any],
) -> None:
    """Write one UTF-8 JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            _to_jsonable(payload),
            file,
            ensure_ascii=False,
            indent=2,
        )


def _resolve_dataset_source(
        *,
        data_cfg: SyntheticDatasetConfig | None,
        dataset_dir: str | Path | None,
        dataset_artifact: DatasetArtifact | None,
) -> tuple[
    SyntheticDataset,
    SyntheticDatasetConfig,
    dict[str, Any],
]:
    """
    Resolve exactly one training dataset source.

    A caller may provide one of:

        1. ``data_cfg``:
           Generate a fresh synthetic dataset.

        2. ``dataset_dir``:
           Load a saved portable dataset artifact.

        3. ``dataset_artifact``:
           Reuse an already loaded artifact without reading arrays twice.
    """
    source_count = sum(
        value is not None
        for value in (
            data_cfg,
            dataset_dir,
            dataset_artifact,
        )
    )

    if source_count != 1:
        raise ValueError(
            "Provide exactly one dataset source: data_cfg, dataset_dir, "
            "or dataset_artifact."
        )

    if dataset_artifact is not None:
        dataset = SyntheticDataset(
            wavefronts=dataset_artifact.wavefronts,
            sensor_gradients=dataset_artifact.sensor_gradients,
            grid_gradients=dataset_artifact.grid_gradients,
        )

        provenance = {
            "source": "dataset_artifact",
            "dataset_dir": str(dataset_artifact.root_dir),
            "schema_version": dataset_artifact.metadata.get(
                "schema_version"
            ),
            "checksums": dataset_artifact.metadata.get(
                "checksums",
                {},
            ),
            "sensor_layout_path": (
                str(dataset_artifact.sensor_layout_path)
                if dataset_artifact.sensor_layout_path is not None
                else None
            ),
        }

        return dataset, dataset_artifact.config, provenance

    if dataset_dir is not None:
        artifact = load_synthetic_dataset_artifact(dataset_dir)

        dataset = SyntheticDataset(
            wavefronts=artifact.wavefronts,
            sensor_gradients=artifact.sensor_gradients,
            grid_gradients=artifact.grid_gradients,
        )

        provenance = {
            "source": "dataset_artifact",
            "dataset_dir": str(artifact.root_dir),
            "schema_version": artifact.metadata.get(
                "schema_version"
            ),
            "checksums": artifact.metadata.get(
                "checksums",
                {},
            ),
            "sensor_layout_path": (
                str(artifact.sensor_layout_path)
                if artifact.sensor_layout_path is not None
                else None
            ),
        }

        return dataset, artifact.config, provenance

    if data_cfg is None:
        raise RuntimeError("data_cfg unexpectedly resolved to None.")

    dataset = generate_synthetic_dataset(data_cfg)

    provenance = {
        "source": "generated_in_process",
        "data_config": asdict(data_cfg),
    }

    return dataset, data_cfg, provenance


def build_standalone_args(
        operator_type: str,
        *,
        config_dir: str | Path,
        data_cfg: SyntheticDatasetConfig,
        result_parent: str | Path,
        model_overrides: dict[str, Any] | None = None,
):
    """
    Build trainer arguments from model YAML and resolved dataset settings.

    The dataset source remains authoritative for grid dimensions, train/test
    split, and the sensor-layout path.
    """
    if operator_type not in {"deeponet", "fno"}:
        raise ValueError(
            f"Unknown operator_type={operator_type!r}. "
            "Expected 'deeponet' or 'fno'."
        )

    overrides = {
        "grid_size": int(data_cfg.grid_size),
        "n_train": int(data_cfg.n_train),
        "n_test": int(data_cfg.n_test),
    }

    if model_overrides is not None:
        overrides.update(model_overrides)

    args = load_config(
        operator_type=operator_type,
        config_dir=config_dir,
        **overrides,
    )

    args.result_dir = str(result_parent)
    args.branch_grid_path = str(data_cfg.branch_grid_path)

    if operator_type == "fno":
        args.nx = int(data_cfg.grid_size)
        args.ny = int(data_cfg.grid_size)
        args.in_channels = 2
        args.num_outputs = 1

    return args


def run_standalone_training(
        operator_type: str,
        *,
        config_dir: str | Path,
        result_parent: str | Path,
        data_cfg: SyntheticDatasetConfig | None = None,
        dataset_dir: str | Path | None = None,
        dataset_artifact: DatasetArtifact | None = None,
        model_overrides: dict[str, Any] | None = None,
) -> Path:
    """
    Train one standalone operator from generated or saved dataset arrays.

    Returns:
        Final timestamped result directory created by the trainer.
    """
    (
        dataset,
        resolved_data_cfg,
        dataset_provenance,
    ) = _resolve_dataset_source(
        data_cfg=data_cfg,
        dataset_dir=dataset_dir,
        dataset_artifact=dataset_artifact,
    )

    args = build_standalone_args(
        operator_type=operator_type,
        config_dir=config_dir,
        data_cfg=resolved_data_cfg,
        result_parent=result_parent,
        model_overrides=model_overrides,
    )

    if operator_type == "deeponet":
        data_mode = str(
            getattr(args, "data_mode", "sensor")
        )

        if data_mode in {"regular_grid", "grid"}:
            branch_inputs = dataset.grid_gradients
        else:
            branch_inputs = dataset.sensor_gradients

        trained_state = main_routine(
            args=args,
            grad_sensor_data=branch_inputs,
            wavefront_true_data=dataset.wavefronts,
        )

    elif operator_type == "fno":
        trained_state = main_routine(
            args=args,
            grad_sensor_data=dataset.sensor_gradients,
            wavefront_true_data=dataset.wavefronts,
            grad_grid_data=dataset.grid_gradients,
        )

    else:
        raise RuntimeError(
            f"Unsupported operator type: {operator_type!r}."
        )

    if not isinstance(trained_state, dict):
        raise TypeError(
            "The trainer must return a state dictionary."
        )

    if "result_dir" not in trained_state:
        raise KeyError(
            "The trainer state does not contain 'result_dir'."
        )

    run_dir = Path(trained_state["result_dir"])
    trained_args = trained_state.get("args", args)

    _write_json(
        run_dir / "run_manifest.json",
        {
            "operator_type": operator_type,
            "dataset": dataset_provenance,
            "data_config": asdict(resolved_data_cfg),
            "training_args": vars(trained_args),
            "dataset_shapes": {
                "wavefronts": list(dataset.wavefronts.shape),
                "sensor_gradients": list(
                    dataset.sensor_gradients.shape
                ),
                "grid_gradients": list(
                    dataset.grid_gradients.shape
                ),
            },
            "data_mode": str(
                getattr(trained_args, "data_mode", "unknown")
            ),
        },
    )

    del trained_state
    del dataset
    gc.collect()

    return run_dir


def run_standalone_synthetic_training(
        operator_type: str,
        *,
        config_dir: str | Path,
        data_cfg: SyntheticDatasetConfig,
        result_parent: str | Path,
        model_overrides: dict[str, Any] | None = None,
) -> Path:
    """
    Backward-compatible wrapper for generated synthetic-data training.
    """
    return run_standalone_training(
        operator_type=operator_type,
        config_dir=config_dir,
        result_parent=result_parent,
        data_cfg=data_cfg,
        model_overrides=model_overrides,
    )
