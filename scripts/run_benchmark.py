from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from wavefront.data.generators import generate_mixed_span_dataset
from wavefront.evaluation import (
    load_benchmark_config,
    print_report,
    run_sequential_benchmark,
    run_sequential_sample_visualization,
    take_common_test,
)
from wavefront.data.dataset_artifacts import (
    load_synthetic_dataset_artifact,
)
from wavefront.training.precision import set_mixed_precision


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for a sequential benchmark run.

    Returns:
        Parsed command-line namespace containing:

            config:
                Path to the benchmark YAML configuration file.

            run_name:
                Optional benchmark output-directory name overriding
                ``runtime.run_name`` from YAML.

            output_root:
                Optional benchmark output root overriding
                ``runtime.output_root`` from YAML.

            fail_fast:
                Whether benchmarking stops at the first failed operator load
                or evaluation.

            dry_run:
                Whether to validate and print the resolved configuration
                without generating data or loading any models.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate wavefront reconstruction operators sequentially on one "
            "shared synthetic holdout dataset."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark.yaml"),
        help="Path to the benchmark YAML configuration file.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional benchmark directory name. Overrides runtime.run_name "
            "from the YAML configuration."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional benchmark output root. Overrides runtime.output_root "
            "from the YAML configuration."
        ),
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop at the first failed model load or evaluation instead of "
            "recording the failure and continuing."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate and print the resolved configuration without generating "
            "data or loading models."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Use a saved portable dataset artifact instead of generating "
            "synthetic arrays from benchmark.yaml."
        ),
    )
    return parser.parse_args()


def _to_jsonable(
        value: Any,
) -> Any:
    """
    Convert Python, NumPy, and filesystem values into JSON-compatible objects.

    Args:
        value: Arbitrary object that may contain NumPy scalars, arrays, paths,
            mappings, sequences, or non-finite floating-point values.

    Returns:
        A value compatible with ``json.dump``.

    Notes:
        NaN and infinite floating-point values are converted to None because
        they are not portable JSON values.
    """
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


def _write_json(
        path: Path,
        payload: dict[str, Any],
) -> None:
    """
    Write one UTF-8 JSON artifact.

    Args:
        path: Destination JSON file path.
        payload: Dictionary to serialize.

    Side Effects:
        Creates parent directories when they do not already exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            _to_jsonable(payload),
            file,
            ensure_ascii=False,
            indent=2,
        )


def _generate_benchmark_dataset(
        config,
        branch_grid_path: str,
):
    """
    Generate one complete synthetic dataset for fair operator comparison.

    The resulting dataset contains both training and held-out partitions. Only
    the held-out partition is evaluated by ``run_sequential_benchmark``, but
    the training portion must be generated because the common test split begins
    at ``data_cfg.n_train``.

    Args:
        config: Synthetic benchmark-data configuration.
        branch_grid_path: CSV path defining the sensor layout used to generate
            sparse sensor-gradient measurements.

    Returns:
        A tuple containing:

            wavefronts_all:
                Wavefront targets with sample dimension first.

            sensor_gradients_all:
                Sparse sensor-gradient measurements with shape
                ``(N, P_sensor, 2)``.

            grid_gradients_all:
                Regular-grid gradient fields with shape
                ``(N, grid_size * grid_size, 2)``, or an equivalent
                grid-shaped representation.

    Notes:
        Passing ``branch_grid_path`` ensures generated sensor gradients follow
        the same coordinate layout and sensor ordering expected by the loaded
        DeepONet and metamodel checkpoints.
    """
    (
        wavefronts_all,
        sensor_gradients_all,
        grid_gradients_all,
        *_,
    ) = generate_mixed_span_dataset(
        coarse_size=int(config.grid_size),
        num_train=int(config.n_train),
        num_test=int(config.n_test),
        frac_zernike=float(config.frac_zernike),
        frac_spiral=float(config.frac_spiral),
        frac_distortion=float(config.frac_distortion),
        with_noise=bool(config.with_noise),
        noise_percentage=float(config.noise_percentage),
        noise_lambda=float(config.noise_lambda),
        apply_blur=bool(config.apply_blur),
        sigma_pix=float(config.sigma_pix),
        seed=int(config.seed),
        save_dir=None,
        sensor_csv_path=branch_grid_path,
    )

    return (
        np.asarray(
            wavefronts_all,
            dtype=np.float32,
        ),
        np.asarray(
            sensor_gradients_all,
            dtype=np.float32,
        ),
        np.asarray(
            grid_gradients_all,
            dtype=np.float32,
        ),
    )


def _validate_artifact_for_benchmark(
        artifact,
        config,
) -> Path:
    """
    Validate that the saved artifact matches benchmark split dimensions.

    The artifact remains authoritative for arrays and sensor layout. The YAML
    configuration remains authoritative for benchmark mode, methods, paths to
    model runs, visualization, and runtime settings.
    """
    artifact_cfg = artifact.config
    benchmark_data_cfg = config.data_cfg

    mismatches = []

    for field_name in (
            "grid_size",
            "n_train",
            "n_test",
    ):
        artifact_value = getattr(artifact_cfg, field_name)
        benchmark_value = getattr(
            benchmark_data_cfg,
            field_name,
        )

        if int(artifact_value) != int(benchmark_value):
            mismatches.append(
                f"{field_name}: artifact={artifact_value}, "
                f"benchmark={benchmark_value}"
            )

    if mismatches:
        mismatch_text = "; ".join(mismatches)

        raise ValueError(
            "Saved dataset artifact is incompatible with benchmark.yaml: "
            f"{mismatch_text}."
        )

    sensor_layout_path = artifact.sensor_layout_path

    if sensor_layout_path is None:
        fallback_path = Path(
            artifact_cfg.branch_grid_path
        ).expanduser()

        if fallback_path.is_file():
            sensor_layout_path = fallback_path
        else:
            raise FileNotFoundError(
                "Dataset artifact does not contain sensor_layout.csv and "
                "its original branch_grid_path is unavailable."
            )

    # The artifact's sensor layout is authoritative for sensor interpolation.
    config.eval_cfg.branch_grid_path = str(sensor_layout_path)

    return sensor_layout_path


def _resolve_benchmark_dataset(
        *,
        config,
        dataset_dir: Path | None,
):
    """
    Load a saved dataset artifact or generate fresh benchmark arrays.

    Returns:
        wavefronts_all, sensor_gradients_all, grid_gradients_all,
        dataset_info
    """
    if dataset_dir is None:
        (
            wavefronts_all,
            sensor_gradients_all,
            grid_gradients_all,
        ) = _generate_benchmark_dataset(
            config.data_cfg,
            branch_grid_path=config.branch_grid_path,
        )

        return (
            wavefronts_all,
            sensor_gradients_all,
            grid_gradients_all,
            {
                "source": "generated_in_process",
                "generation_config": asdict(config.data_cfg),
                "sensor_layout_path": config.branch_grid_path,
            },
        )

    artifact = load_synthetic_dataset_artifact(dataset_dir)

    sensor_layout_path = _validate_artifact_for_benchmark(
        artifact,
        config,
    )

    return (
        artifact.wavefronts,
        artifact.sensor_gradients,
        artifact.grid_gradients,
        {
            "source": "dataset_artifact",
            "dataset_dir": str(artifact.root_dir),
            "schema_version": artifact.metadata.get(
                "schema_version"
            ),
            "checksums": artifact.metadata.get(
                "checksums",
                {},
            ),
            "sensor_layout_path": str(sensor_layout_path),
        },
    )


def main() -> None:
    """
    Generate a shared synthetic holdout dataset and benchmark selected models.

    The procedure:

        1. Loads and validates benchmark YAML settings.
        2. Applies command-line runtime overrides.
        3. Optionally prints the resolved configuration and exits in dry-run
           mode.
        4. Generates one synthetic dataset shared by all operators.
        5. Evaluates the Poisson baseline and requested learned operators
           sequentially.
        6. Saves configuration, dataset metadata, and the final report.

    Side Effects:
        Creates a timestamped benchmark output directory containing:

            benchmark_config.json
            dataset_info.json
            report.json

        Also prints a compact human-readable comparison report.
    """
    cli_args = parse_args()

    # Load YAML configuration and resolve all benchmark defaults.
    config = load_benchmark_config(cli_args.config)

    # Explicit command-line settings take precedence over YAML values.
    if cli_args.run_name is not None:
        config.run_name = cli_args.run_name

    if cli_args.output_root is not None:
        config.output_root = str(cli_args.output_root)

    if cli_args.fail_fast:
        config.fail_fast = True

    # Save a transparent, fully resolved representation of the benchmark
    # settings before any data generation or model loading begins.
    resolved = {
        "runtime": {
            "branch_grid_path": config.branch_grid_path,
            "output_root": config.output_root,
            "run_name": config.run_name,
            "checkpoint": config.checkpoint,
            "fail_fast": config.fail_fast,
            "mixed_precision": config.mixed_precision,
            "mp_dtype": config.mp_dtype,
        },
        "data": asdict(config.data_cfg),
        "evaluation": asdict(config.eval_cfg),
        "visualization": asdict(config.visualization_cfg),
        "runs": {
            "deeponet": config.deeponet_run,
            "fno": config.fno_run,
            "two_stage": config.two_stage_run,
            "three_stage": config.three_stage_run,
        },
    }

    # Validate and display settings without allocating data or accelerator
    # memory when dry-run mode is requested.
    if cli_args.dataset_dir is not None:
        resolved["dataset"] = {
            "source": "dataset_artifact",
            "dataset_dir": str(cli_args.dataset_dir),
        }
    else:
        resolved["dataset"] = {
            "source": "generated_in_process",
        }
    if cli_args.dry_run:
        print(
            json.dumps(
                resolved,
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    # Configure global mixed-precision behavior before loading JAX models.
    set_mixed_precision(
        enabled=config.mixed_precision,
        dtype=config.mp_dtype,
    )

    # Use an explicit name when supplied; otherwise create a timestamped
    # benchmark artifact directory.
    run_id = config.run_name or time.strftime("%Y%m%d-%H%M%S")

    output_dir = Path(config.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        wavefronts_all,
        sensor_gradients_all,
        grid_gradients_all,
        dataset_source,
    ) = _resolve_benchmark_dataset(
        config=config,
        dataset_dir=cli_args.dataset_dir,
    )

    resolved["dataset"] = dataset_source

    # The artifact may provide a portable sensor_layout.csv, so save the final
    # resolved evaluation settings only after the dataset source is known.
    resolved["evaluation"] = asdict(config.eval_cfg)
    resolved["runtime"]["branch_grid_path"] = str(
        config.eval_cfg.branch_grid_path
    )

    _write_json(
        output_dir / "benchmark_config.json",
        resolved,
    )

    # Record generated-array shapes and the random seed for reproducibility.
    _write_json(
        output_dir / "dataset_info.json",
        {
            "source": dataset_source,
            "wavefronts_shape": list(wavefronts_all.shape),
            "sensor_gradients_shape": list(
                sensor_gradients_all.shape
            ),
            "grid_gradients_shape": list(
                grid_gradients_all.shape
            ),
            "seed": int(config.data_cfg.seed),
        },
    )

    # Load, evaluate, and release requested learned operators sequentially.
    report = run_sequential_benchmark(
        cfg=config.eval_cfg,
        wavefronts_all=wavefronts_all,
        sensor_gradients_all=sensor_gradients_all,
        grid_gradients_all=grid_gradients_all,
        deeponet_run=config.deeponet_run,
        fno_run=config.fno_run,
        two_stage_run=config.two_stage_run,
        three_stage_run=config.three_stage_run,
        checkpoint=config.checkpoint,
        fail_fast=config.fail_fast,
    )

    if config.visualization_cfg.enabled:
        U_test, D_sensor_test, D_grid_test = take_common_test(
            wavefronts_all=wavefronts_all,
            sensor_gradients_all=sensor_gradients_all,
            grid_gradients_all=grid_gradients_all,
            cfg=config.eval_cfg,
        )

        visualization_report = run_sequential_sample_visualization(
            cfg=config.eval_cfg,
            U_test=U_test,
            D_sensor_test=D_sensor_test,
            D_grid_test=D_grid_test,
            output_dir=output_dir,
            sample_indices=config.visualization_cfg.sample_indices,
            deeponet_run=config.deeponet_run,
            fno_run=config.fno_run,
            two_stage_run=config.two_stage_run,
            three_stage_run=config.three_stage_run,
            checkpoint=config.checkpoint,
            error_mode=config.visualization_cfg.error_mode,
            solution_percentiles=tuple(
                config.visualization_cfg.solution_percentiles
            ),
            error_percentile=float(
                config.visualization_cfg.error_percentile
            ),
            separate_poisson_scale=bool(
                config.visualization_cfg.separate_poisson_scale
            ),
            save_stage1_gradient=bool(
                config.visualization_cfg.save_stage1_gradient
            ),
            stage1_source=config.visualization_cfg.stage1_source,
            dpi=int(config.visualization_cfg.dpi),
            fail_fast=config.fail_fast,
        )

        report["visualizations"] = visualization_report

        _write_json(
            output_dir / "sample_visualizations.json",
            visualization_report,
        )
    _write_json(
        output_dir / "report.json",
        report,
    )

    # Display headline reconstruction and inference metrics in the terminal.
    print_report(report)

    print(f"Benchmark artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
