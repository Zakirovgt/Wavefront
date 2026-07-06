from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _to_jsonable(value: Any) -> Any:
    """Convert common NumPy and Path values into JSON-safe objects."""
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
        return [
            _to_jsonable(item)
            for item in value
        ]

    return str(value)


def _as_wavefront_batch(
        wavefronts,
) -> np.ndarray:
    """Validate a batch of predicted wavefronts."""
    wavefronts = np.asarray(
        wavefronts,
        dtype=np.float32,
    )

    if wavefronts.ndim == 2:
        wavefronts = wavefronts[None, ...]

    if wavefronts.ndim != 3:
        raise ValueError(
            "wavefronts must have shape (H, W) or (B, H, W), "
            f"got {wavefronts.shape}."
        )

    return wavefronts


def _save_prediction_csv(
        path: Path,
        wavefronts: np.ndarray,
        *,
        value_name: str = "wavefront_prediction",
) -> None:
    """Save all predicted wavefront samples in one long-form CSV file."""
    batch_size, height, width = wavefronts.shape

    x_values = np.linspace(
        -1.0,
        1.0,
        width,
        dtype=np.float32,
    )
    y_values = np.linspace(
        -1.0,
        1.0,
        height,
        dtype=np.float32,
    )

    with path.open(
            "w",
            encoding="utf-8",
            newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "sample_index",
                "x",
                "y",
                value_name,
            ]
        )

        for sample_index in range(batch_size):
            for row, y in enumerate(y_values):
                for column, x in enumerate(x_values):
                    writer.writerow(
                        [
                            sample_index,
                            float(x),
                            float(y),
                            float(
                                wavefronts[
                                    sample_index,
                                    row,
                                    column,
                                ]
                            ),
                        ]
                    )


def _save_prediction_figures(
        figures_dir: Path,
        wavefronts: np.ndarray,
        *,
        operator_name: str,
        dpi: int,
) -> list[str]:
    """Save one normalized-wavefront image for every predicted sample."""
    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = []

    for sample_index, wavefront in enumerate(wavefronts):
        finite = wavefront[np.isfinite(wavefront)]

        if finite.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            limit = float(np.max(np.abs(finite)))

            if limit == 0.0:
                limit = 1.0

            vmin, vmax = -limit, limit

        figure, axis = plt.subplots(
            figsize=(5.5, 4.5),
        )

        image = axis.imshow(
            wavefront,
            cmap="seismic",
            vmin=vmin,
            vmax=vmax,
            extent=[-1, 1, 1, -1],
            origin="upper",
        )

        axis.set_title(
            f"{operator_name}: normalized wavefront "
            f"#{sample_index}"
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")

        figure.colorbar(
            image,
            ax=axis,
            label="Normalized wavefront",
        )

        figure.tight_layout()

        path = (
                figures_dir
                / f"prediction_{sample_index:04d}.png"
        )

        figure.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(figure)

        paths.append(str(path))

    return paths


def save_inference_outputs(
        output_dir: str | Path,
        *,
        wavefronts,
        operator_name: str,
        metadata: dict[str, Any] | None = None,
        input_gradient_grids=None,
        predicted_gradient_grids=None,
        dpi: int = 180,
) -> dict[str, Any]:
    """
    Save model-space inference outputs in NPY, CSV, PNG, and JSON formats.

    All saved wavefront predictions are normalized model outputs. Physical-unit
    conversion must be applied explicitly by a calibration layer.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    wavefronts = _as_wavefront_batch(wavefronts)

    wavefront_path = (
            output_dir
            / "wavefront_predictions_normalized.npy"
    )
    np.save(
        wavefront_path,
        wavefronts,
    )

    csv_path = (
            output_dir
            / "wavefront_predictions_normalized.csv"
    )
    _save_prediction_csv(
        csv_path,
        wavefronts,
    )

    saved_paths = {
        "wavefronts_npy": str(wavefront_path),
        "wavefronts_csv": str(csv_path),
    }

    if input_gradient_grids is not None:
        input_gradient_path = (
                output_dir
                / "input_gradient_grids_normalized.npy"
        )
        np.save(
            input_gradient_path,
            np.asarray(
                input_gradient_grids,
                dtype=np.float32,
            ),
        )
        saved_paths["input_gradient_grids_npy"] = str(
            input_gradient_path
        )

    if predicted_gradient_grids is not None:
        predicted_gradient_path = (
                output_dir
                / "predicted_gradient_grids_normalized.npy"
        )
        np.save(
            predicted_gradient_path,
            np.asarray(
                predicted_gradient_grids,
                dtype=np.float32,
            ),
        )
        saved_paths["predicted_gradient_grids_npy"] = str(
            predicted_gradient_path
        )

    figure_paths = _save_prediction_figures(
        output_dir / "figures",
        wavefronts,
        operator_name=operator_name,
        dpi=int(dpi),
    )

    saved_paths["figures"] = figure_paths

    metadata_payload = {
        "operator_name": operator_name,
        "wavefront_shape": list(wavefronts.shape),
        "output_space": "normalized_model_space",
        "saved_files": saved_paths,
        "metadata": metadata or {},
    }

    metadata_path = output_dir / "metadata.json"

    with metadata_path.open(
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            _to_jsonable(metadata_payload),
            file,
            ensure_ascii=False,
            indent=2,
        )

    saved_paths["metadata_json"] = str(metadata_path)

    return saved_paths


def save_physical_wavefront_outputs(
        output_dir: str | Path,
        *,
        wavefronts,
        units: str | None = None,
) -> dict[str, str]:
    """
    Save calibrated wavefront predictions in NPY and CSV formats.

    These outputs use the scale and piston offset explicitly selected by the
    caller. They are not automatically absolute measurements.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    wavefronts = _as_wavefront_batch(wavefronts)

    npy_path = output_dir / "wavefront_predictions_physical.npy"
    csv_path = output_dir / "wavefront_predictions_physical.csv"

    np.save(npy_path, wavefronts)

    value_name = "wavefront_prediction_physical"

    if units:
        value_name = f"{value_name}_{units}"

    _save_prediction_csv(
        csv_path,
        wavefronts,
        value_name=value_name,
    )

    return {
        "wavefronts_physical_npy": str(npy_path),
        "wavefronts_physical_csv": str(csv_path),
    }
