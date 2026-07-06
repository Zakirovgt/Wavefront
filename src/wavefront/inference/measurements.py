from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MeasurementColumns:
    """Column names used by one sensor-gradient measurement CSV."""

    gx: str = "gx"
    gy: str = "gy"

    sample_id: str | None = None
    sensor_id: str | None = None
    valid: str | None = None


@dataclass(frozen=True)
class SensorLayout:
    """Expected sensor order and normalized coordinates for one model."""

    coordinates: np.ndarray
    sensor_ids: tuple[str, ...] | None
    source_path: str

    @property
    def n_sensors(self) -> int:
        """Return the number of expected sensor positions."""
        return int(self.coordinates.shape[0])


@dataclass(frozen=True)
class SensorGradientBatch:
    """Validated sparse sensor gradients loaded from a measurement CSV."""

    gradients: np.ndarray
    sample_ids: tuple[str, ...]
    invalid_sensor_count: int
    missing_sensor_count: int

    @property
    def n_samples(self) -> int:
        """Return the number of measurement samples."""
        return int(self.gradients.shape[0])


_TRUE_VALUES = {
    "1",
    "true",
    "t",
    "yes",
    "y",
    "on",
}

_FALSE_VALUES = {
    "0",
    "false",
    "f",
    "no",
    "n",
    "off",
    "",
}


def _read_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read a UTF-8 CSV file and return its rows and header names."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"CSV file was not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {path}")

        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if not rows:
        raise ValueError(f"CSV file contains no data rows: {path}")

    return rows, fieldnames


def _require_columns(
        available_columns: list[str],
        required_columns: list[str | None],
        *,
        source_name: str,
) -> None:
    """Raise an informative error when required CSV columns are missing."""
    missing_columns = [
        column
        for column in required_columns
        if column is not None and column not in available_columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        available_text = ", ".join(available_columns)

        raise KeyError(
            f"{source_name} is missing required column(s): {missing_text}. "
            f"Available columns: {available_text}."
        )


def _parse_float(
        value: str | None,
        *,
        column_name: str,
        row_index: int,
) -> float:
    """Parse one finite floating-point CSV value."""
    if value is None:
        raise ValueError(
            f"Missing value in column '{column_name}' at CSV row {row_index}."
        )

    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(
            f"Invalid float in column '{column_name}' at CSV row "
            f"{row_index}: {value!r}."
        ) from error

    if not np.isfinite(parsed):
        raise ValueError(
            f"Non-finite value in column '{column_name}' at CSV row "
            f"{row_index}: {value!r}."
        )

    return parsed


def _parse_valid_flag(
        value: str | None,
        *,
        column_name: str,
        row_index: int,
) -> bool:
    """Parse a common boolean CSV representation."""
    normalized = "" if value is None else str(value).strip().lower()

    if normalized in _TRUE_VALUES:
        return True

    if normalized in _FALSE_VALUES:
        return False

    raise ValueError(
        f"Invalid boolean value in column '{column_name}' at CSV row "
        f"{row_index}: {value!r}. Use values such as 1/0, true/false, "
        "or yes/no."
    )


def load_sensor_layout(
        path: str | Path,
        *,
        x_column: str = "X",
        y_column: str = "Y",
        sensor_id_column: str | None = None,
) -> SensorLayout:
    """
    Load expected sensor positions and optional IDs from a layout CSV.

    The layout must preserve the exact sensor ordering used during training.
    Coordinates must be in the same normalized coordinate system as training,
    typically within [-1, 1].
    """
    rows, fieldnames = _read_csv_rows(path)

    _require_columns(
        fieldnames,
        [x_column, y_column, sensor_id_column],
        source_name="Sensor layout CSV",
    )

    coordinates = np.empty((len(rows), 2), dtype=np.float32)
    sensor_ids: list[str] = []

    for row_index, row in enumerate(rows, start=2):
        coordinates[row_index - 2, 0] = _parse_float(
            row.get(x_column),
            column_name=x_column,
            row_index=row_index,
        )
        coordinates[row_index - 2, 1] = _parse_float(
            row.get(y_column),
            column_name=y_column,
            row_index=row_index,
        )

        if sensor_id_column is not None:
            sensor_id = str(
                row.get(sensor_id_column, "")
            ).strip()

            if not sensor_id:
                raise ValueError(
                    f"Empty sensor ID in column '{sensor_id_column}' "
                    f"at CSV row {row_index}."
                )

            sensor_ids.append(sensor_id)

    if sensor_id_column is not None:
        if len(set(sensor_ids)) != len(sensor_ids):
            raise ValueError(
                "Sensor layout IDs must be unique."
            )

        layout_ids: tuple[str, ...] | None = tuple(sensor_ids)
    else:
        layout_ids = None

    return SensorLayout(
        coordinates=coordinates,
        sensor_ids=layout_ids,
        source_path=str(Path(path)),
    )


def load_sensor_gradient_csv(
        path: str | Path,
        *,
        layout: SensorLayout,
        columns: MeasurementColumns,
        invalid_sensor_policy: str = "error",
) -> SensorGradientBatch:
    """
    Load measurement gradients and align them to the training sensor order.

    Args:
        path:
            Measurement CSV path.

        layout:
            Expected sensor order and coordinates.

        columns:
            CSV-column mapping.

        invalid_sensor_policy:
            ``"error"`` rejects missing or invalid sensors.
            ``"zero"`` replaces missing or invalid sensor gradients with zero.

    Returns:
        Sparse gradients with shape ``(B, P_sensor, 2)``.
    """
    if invalid_sensor_policy not in {"error", "zero"}:
        raise ValueError(
            "invalid_sensor_policy must be either 'error' or 'zero'."
        )

    rows, fieldnames = _read_csv_rows(path)

    _require_columns(
        fieldnames,
        [
            columns.gx,
            columns.gy,
            columns.sample_id,
            columns.sensor_id,
            columns.valid,
        ],
        source_name="Measurement CSV",
    )

    if columns.sensor_id is not None and layout.sensor_ids is None:
        raise ValueError(
            "sensor_id_column was provided, but layout_sensor_id_column "
            "was not provided when loading the sensor layout."
        )

    grouped_rows: dict[str, list[tuple[int, dict[str, str]]]] = {}

    for row_index, row in enumerate(rows, start=2):
        if columns.sample_id is None:
            sample_key = "0"
        else:
            sample_key = str(
                row.get(columns.sample_id, "")
            ).strip()

            if not sample_key:
                raise ValueError(
                    f"Empty sample ID in column '{columns.sample_id}' "
                    f"at CSV row {row_index}."
                )

        grouped_rows.setdefault(sample_key, []).append(
            (row_index, row)
        )

    sensor_index = None

    if layout.sensor_ids is not None:
        sensor_index = {
            sensor_id: index
            for index, sensor_id in enumerate(layout.sensor_ids)
        }

    gradients = np.zeros(
        (
            len(grouped_rows),
            layout.n_sensors,
            2,
        ),
        dtype=np.float32,
    )

    invalid_sensor_count = 0
    missing_sensor_count = 0

    for sample_index, (sample_id, sample_rows) in enumerate(
            grouped_rows.items()
    ):
        filled = np.zeros(layout.n_sensors, dtype=bool)

        if (
                columns.sensor_id is None
                and len(sample_rows) > layout.n_sensors
        ):
            raise ValueError(
                f"Sample {sample_id!r} contains {len(sample_rows)} rows, "
                f"but the layout contains only {layout.n_sensors} sensors."
            )

        for local_index, (row_index, row) in enumerate(sample_rows):
            if columns.sensor_id is None:
                expected_index = local_index
            else:
                sensor_id = str(
                    row.get(columns.sensor_id, "")
                ).strip()

                if not sensor_id:
                    raise ValueError(
                        f"Empty sensor ID in column '{columns.sensor_id}' "
                        f"at CSV row {row_index}."
                    )

                if sensor_id not in sensor_index:
                    raise ValueError(
                        f"Unknown sensor ID {sensor_id!r} at CSV row "
                        f"{row_index}. It does not exist in the layout CSV."
                    )

                expected_index = sensor_index[sensor_id]

            if filled[expected_index]:
                raise ValueError(
                    f"Duplicate measurement for sensor index "
                    f"{expected_index} in sample {sample_id!r}."
                )

            is_valid = True

            if columns.valid is not None:
                is_valid = _parse_valid_flag(
                    row.get(columns.valid),
                    column_name=columns.valid,
                    row_index=row_index,
                )

            if not is_valid:
                invalid_sensor_count += 1

                if invalid_sensor_policy == "error":
                    raise ValueError(
                        f"Invalid sensor measurement at CSV row {row_index}. "
                        "Use --invalid-sensor-policy zero only when zero "
                        "replacement is physically justified."
                    )

                filled[expected_index] = True
                continue

            gradients[sample_index, expected_index, 0] = _parse_float(
                row.get(columns.gx),
                column_name=columns.gx,
                row_index=row_index,
            )
            gradients[sample_index, expected_index, 1] = _parse_float(
                row.get(columns.gy),
                column_name=columns.gy,
                row_index=row_index,
            )
            filled[expected_index] = True

        missing_mask = ~filled
        missing_count = int(np.sum(missing_mask))

        if missing_count:
            missing_sensor_count += missing_count

            if invalid_sensor_policy == "error":
                raise ValueError(
                    f"Sample {sample_id!r} is missing {missing_count} "
                    f"of {layout.n_sensors} expected sensors."
                )

    return SensorGradientBatch(
        gradients=gradients,
        sample_ids=tuple(grouped_rows.keys()),
        invalid_sensor_count=invalid_sensor_count,
        missing_sensor_count=missing_sensor_count,
    )


def normalize_sensor_gradients(
        sensor_gradients,
        *,
        normalization_scale: float,
) -> np.ndarray:
    """
    Convert physical gradients to the normalized model space.

    ``normalization_scale`` must match the scale used during training:
    gradients were divided by the sample-specific wavefront standard deviation.
    """
    normalization_scale = float(normalization_scale)

    if not np.isfinite(normalization_scale) or normalization_scale <= 0.0:
        raise ValueError(
            "normalization_scale must be a finite positive number."
        )

    sensor_gradients = np.asarray(
        sensor_gradients,
        dtype=np.float32,
    )

    if (
            sensor_gradients.ndim != 3
            or sensor_gradients.shape[-1] != 2
    ):
        raise ValueError(
            "sensor_gradients must have shape (B, P_sensor, 2), "
            f"got {sensor_gradients.shape}."
        )

    return sensor_gradients / normalization_scale


def denormalize_wavefronts(
        normalized_wavefronts,
        *,
        normalization_scale: float,
        wavefront_offset: float = 0.0,
) -> np.ndarray:
    """
    Convert normalized model outputs into a chosen physical wavefront scale.

    The offset represents the unobservable piston term. Gradients alone cannot
    determine this value, so it must be supplied externally when needed.
    """
    normalization_scale = float(normalization_scale)
    wavefront_offset = float(wavefront_offset)

    if not np.isfinite(normalization_scale) or normalization_scale <= 0.0:
        raise ValueError(
            "normalization_scale must be a finite positive number."
        )

    if not np.isfinite(wavefront_offset):
        raise ValueError(
            "wavefront_offset must be finite."
        )

    normalized_wavefronts = np.asarray(
        normalized_wavefronts,
        dtype=np.float32,
    )

    return (
            normalized_wavefronts * normalization_scale
            + wavefront_offset
    ).astype(np.float32)
