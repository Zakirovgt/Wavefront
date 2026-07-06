from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Literal

from wavefront.evaluation.loaders import (
    load_standalone_fno_state,
    load_three_stage_metamodel_state,
    load_two_stage_metamodel_state,
)
from wavefront.inference.measurements import (
    MeasurementColumns,
    denormalize_wavefronts,
    load_sensor_gradient_csv,
    load_sensor_layout,
    normalize_sensor_gradients,
)
from wavefront.inference.operators import (
    predict_fno_from_sensor_gradients,
    predict_metamodel_from_sensor_gradients,
)
from wavefront.inference.outputs import (
    save_inference_outputs,
    save_physical_wavefront_outputs,
)
from wavefront.training.precision import set_mixed_precision

OperatorName = Literal[
    "fno",
    "two_stage",
    "three_stage",
]


def _parse_args(
        operator_name: OperatorName,
) -> argparse.Namespace:
    """Parse command-line options for one saved operator."""
    parser = argparse.ArgumentParser(
        description=(
            f"Run {operator_name} Wavefront inference from sensor gradients."
        )
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Saved standalone or pipeline run directory.",
    )

    parser.add_argument(
        "--measurements",
        type=Path,
        required=True,
        help="Measurement CSV containing sensor gradients.",
    )

    parser.add_argument(
        "--sensor-layout",
        type=Path,
        required=True,
        help=(
            "Sensor-layout CSV used during training, with normalized X/Y "
            "coordinates in the original sensor order."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "results/predictions/<operator>/<timestamp>."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
        help="Saved checkpoint to restore.",
    )

    parser.add_argument(
        "--gx-column",
        type=str,
        default="gx",
        help="Measurement CSV column containing dU/dx.",
    )

    parser.add_argument(
        "--gy-column",
        type=str,
        default="gy",
        help="Measurement CSV column containing dU/dy.",
    )

    parser.add_argument(
        "--sample-id-column",
        type=str,
        default=None,
        help=(
            "Optional measurement CSV column grouping rows into samples. "
            "Without it, the entire CSV is treated as one sample."
        ),
    )

    parser.add_argument(
        "--sensor-id-column",
        type=str,
        default=None,
        help=(
            "Optional measurement CSV sensor-ID column. Use this together "
            "with --layout-sensor-id-column for ID-based alignment."
        ),
    )

    parser.add_argument(
        "--layout-sensor-id-column",
        type=str,
        default=None,
        help=(
            "Optional sensor-layout CSV ID column used for alignment with "
            "--sensor-id-column."
        ),
    )

    parser.add_argument(
        "--layout-x-column",
        type=str,
        default="X",
        help="Sensor-layout CSV x-coordinate column.",
    )

    parser.add_argument(
        "--layout-y-column",
        type=str,
        default="Y",
        help="Sensor-layout CSV y-coordinate column.",
    )

    parser.add_argument(
        "--valid-column",
        type=str,
        default=None,
        help=(
            "Optional measurement CSV boolean validity column. "
            "Accepted values include 1/0, true/false, and yes/no."
        ),
    )

    parser.add_argument(
        "--invalid-sensor-policy",
        choices=["error", "zero"],
        default="error",
        help=(
            "How to handle missing or invalid sensors. 'error' is the safe "
            "default; 'zero' must be physically justified."
        ),
    )

    parser.add_argument(
        "--normalization-scale",
        type=float,
        required=True,
        help=(
            "Positive scale used to divide physical gradients before model "
            "inference and multiply normalized wavefront outputs afterward."
        ),
    )

    parser.add_argument(
        "--wavefront-offset",
        type=float,
        default=0.0,
        help=(
            "Chosen physical piston offset added after denormalization. "
            "Gradients alone cannot determine this value."
        ),
    )

    parser.add_argument(
        "--units",
        type=str,
        default=None,
        help=(
            "Optional physical-unit label written into output metadata, "
            "for example 'um' or 'rad'."
        ),
    )

    parser.add_argument(
        "--interpolation-method",
        choices=["linear", "nearest", "cubic"],
        default=None,
        help=(
            "Optional sparse-sensor interpolation method for standalone FNO. "
            "Ignored by metamodel operators."
        ),
    )

    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable JAX mixed precision for inference.",
    )

    parser.add_argument(
        "--mp-dtype",
        type=str,
        default="bfloat16",
        help="Mixed-precision dtype when --mixed-precision is enabled.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI for normalized prediction figures.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate CSV schemas and print the resolved configuration "
            "without loading model parameters."
        ),
    )

    return parser.parse_args()


def _load_operator_state(
        operator_name: OperatorName,
        run_dir: Path,
        checkpoint: str,
) -> dict:
    """Load one saved operator state."""
    if operator_name == "fno":
        return load_standalone_fno_state(
            run_dir,
            checkpoint=checkpoint,
        )

    if operator_name == "two_stage":
        return load_two_stage_metamodel_state(
            run_dir,
            checkpoint=checkpoint,
        )

    if operator_name == "three_stage":
        return load_three_stage_metamodel_state(
            run_dir,
            checkpoint=checkpoint,
        )

    raise ValueError(
        f"Unsupported operator {operator_name!r}."
    )


def _write_json(
        path: Path,
        payload: dict,
) -> None:
    """Write a JSON artifact with UTF-8 encoding."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def run_prediction_cli(
        operator_name: OperatorName,
) -> None:
    """Run one saved operator on aligned sensor-gradient CSV measurements."""
    args = _parse_args(operator_name)

    if args.normalization_scale <= 0.0:
        raise ValueError(
            "--normalization-scale must be positive."
        )

    if args.sensor_id_column is None:
        if args.layout_sensor_id_column is not None:
            raise ValueError(
                "--layout-sensor-id-column requires --sensor-id-column."
            )

    layout = load_sensor_layout(
        args.sensor_layout,
        x_column=args.layout_x_column,
        y_column=args.layout_y_column,
        sensor_id_column=args.layout_sensor_id_column,
    )

    measurement_batch = load_sensor_gradient_csv(
        args.measurements,
        layout=layout,
        columns=MeasurementColumns(
            gx=args.gx_column,
            gy=args.gy_column,
            sample_id=args.sample_id_column,
            sensor_id=args.sensor_id_column,
            valid=args.valid_column,
        ),
        invalid_sensor_policy=args.invalid_sensor_policy,
    )

    normalized_sensor_gradients = normalize_sensor_gradients(
        measurement_batch.gradients,
        normalization_scale=args.normalization_scale,
    )

    output_dir = args.output_dir

    if output_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = (
                Path("results")
                / "predictions"
                / operator_name
                / timestamp
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        "operator_name": operator_name,
        "run_dir": str(args.run_dir),
        "checkpoint": args.checkpoint,
        "measurements": str(args.measurements),
        "sensor_layout": str(args.sensor_layout),
        "n_samples": measurement_batch.n_samples,
        "n_sensors": layout.n_sensors,
        "sample_ids": list(measurement_batch.sample_ids),
        "invalid_sensor_count": measurement_batch.invalid_sensor_count,
        "missing_sensor_count": measurement_batch.missing_sensor_count,
        "invalid_sensor_policy": args.invalid_sensor_policy,
        "normalization_scale": float(args.normalization_scale),
        "wavefront_offset": float(args.wavefront_offset),
        "units": args.units,
        "mixed_precision": bool(args.mixed_precision),
        "mp_dtype": args.mp_dtype,
        "interpolation_method": args.interpolation_method,
    }

    if args.dry_run:
        print(
            json.dumps(
                resolved,
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    set_mixed_precision(
        enabled=bool(args.mixed_precision),
        dtype=str(args.mp_dtype),
    )

    state = None

    try:
        state = _load_operator_state(
            operator_name,
            args.run_dir,
            args.checkpoint,
        )

        if operator_name == "fno":
            prediction_result = predict_fno_from_sensor_gradients(
                state=state,
                sensor_gradients=normalized_sensor_gradients,
                interpolation_method=args.interpolation_method,
                sensor_coords=layout.coordinates,
            )

            normalized_wavefronts = prediction_result["wavefronts"]

            normalized_files = save_inference_outputs(
                output_dir,
                wavefronts=normalized_wavefronts,
                operator_name="FNO",
                metadata=resolved,
                input_gradient_grids=prediction_result[
                    "input_gradient_grids"
                ],
                dpi=int(args.dpi),
            )

        else:
            prediction_result = predict_metamodel_from_sensor_gradients(
                state=state,
                sensor_gradients=normalized_sensor_gradients,
            )

            normalized_wavefronts = prediction_result["wavefronts"]

            normalized_files = save_inference_outputs(
                output_dir,
                wavefronts=normalized_wavefronts,
                operator_name=operator_name,
                metadata=resolved,
                predicted_gradient_grids=prediction_result[
                    "predicted_gradient_grids"
                ],
                dpi=int(args.dpi),
            )

        physical_wavefronts = denormalize_wavefronts(
            normalized_wavefronts,
            normalization_scale=args.normalization_scale,
            wavefront_offset=args.wavefront_offset,
        )

        physical_files = save_physical_wavefront_outputs(
            output_dir,
            wavefronts=physical_wavefronts,
            units=args.units,
        )

        resolved["saved_normalized_files"] = normalized_files
        resolved["saved_physical_files"] = physical_files

        _write_json(
            output_dir / "inference_manifest.json",
            resolved,
        )

        print(
            "Inference completed successfully.\n"
            f"Output directory: {output_dir}"
        )

    finally:
        if state is not None:
            state.clear()

        gc.collect()
