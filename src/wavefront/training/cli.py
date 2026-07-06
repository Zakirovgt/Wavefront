from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from wavefront.data.dataset_artifacts import (
    load_synthetic_dataset_artifact,
)
from wavefront.data.synthetic import (
    load_synthetic_dataset_config,
)
from wavefront.training.standalone_runner import (
    build_standalone_args,
    run_standalone_training,
)

DEFAULT_DATA_CONFIG = Path("configs/standalone_data.yaml")


def _parse_args(
        operator_type: str,
) -> argparse.Namespace:
    """Parse CLI arguments for one standalone model."""
    parser = argparse.ArgumentParser(
        description=(
            f"Train a standalone Wavefront {operator_type} model."
        )
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help=(
            "Directory containing common.yaml and the "
            f"{operator_type}.yaml model configuration."
        ),
    )

    source_group = parser.add_mutually_exclusive_group()

    source_group.add_argument(
        "--data-config",
        type=Path,
        default=None,
        help=(
            "Synthetic dataset YAML configuration. When omitted and "
            "--dataset-dir is absent, uses "
            "configs/standalone_data.yaml."
        ),
    )

    source_group.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Portable dataset artifact directory containing arrays.npz, "
            "metadata.json, generation_config.yaml, and optionally "
            "sensor_layout.csv."
        ),
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/standalone"),
        help="Root directory for standalone training results.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional experiment parent directory. The trainer creates "
            "a timestamped run directory inside it."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional override for the number of training iterations.",
    )

    parser.add_argument(
        "--data-mode",
        choices=["sensor", "regular_grid"],
        default=None,
        help=(
            "Optional input-mode override. Use FNO sensor mode for "
            "fair sensor-based experiments."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate configuration and dataset source without training."
        ),
    )

    return parser.parse_args()


def _resolve_cli_dataset(
        cli_args: argparse.Namespace,
):
    """
    Resolve either a generated-data configuration or a saved artifact.

    Returns:
        data_cfg:
            Dataset configuration used to construct training arguments.

        dataset_artifact:
            Loaded artifact when --dataset-dir was used, otherwise None.

        source:
            JSON-safe dataset source metadata.
    """
    if cli_args.dataset_dir is not None:
        artifact = load_synthetic_dataset_artifact(
            cli_args.dataset_dir
        )

        return (
            artifact.config,
            artifact,
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
                "sensor_layout_path": (
                    str(artifact.sensor_layout_path)
                    if artifact.sensor_layout_path is not None
                    else None
                ),
            },
        )

    data_config_path = (
        cli_args.data_config
        if cli_args.data_config is not None
        else DEFAULT_DATA_CONFIG
    )

    data_cfg = load_synthetic_dataset_config(
        data_config_path
    )

    return (
        data_cfg,
        None,
        {
            "source": "generated_in_process",
            "data_config_path": str(data_config_path),
        },
    )


def run_standalone_cli(
        operator_type: str,
) -> None:
    """Execute standalone DeepONet or FNO training."""
    cli_args = _parse_args(operator_type)

    (
        data_cfg,
        dataset_artifact,
        dataset_source,
    ) = _resolve_cli_dataset(cli_args)

    model_overrides = {}

    if cli_args.steps is not None:
        model_overrides["steps"] = int(cli_args.steps)

    if cli_args.data_mode is not None:
        model_overrides["data_mode"] = cli_args.data_mode

    result_parent = cli_args.results_root / operator_type

    if cli_args.run_name is not None:
        result_parent = result_parent / cli_args.run_name

    resolved_args = build_standalone_args(
        operator_type=operator_type,
        config_dir=cli_args.config_dir,
        data_cfg=data_cfg,
        result_parent=result_parent,
        model_overrides=model_overrides,
    )

    if cli_args.dry_run:
        print(
            json.dumps(
                {
                    "operator_type": operator_type,
                    "dataset": dataset_source,
                    "data": asdict(data_cfg),
                    "training_args": vars(resolved_args),
                    "result_parent": str(result_parent),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return

    run_dir = run_standalone_training(
        operator_type=operator_type,
        config_dir=cli_args.config_dir,
        result_parent=result_parent,
        dataset_artifact=dataset_artifact,
        data_cfg=(
            data_cfg
            if dataset_artifact is None
            else None
        ),
        model_overrides=model_overrides,
    )

    print("\nTraining completed successfully.")
    print(f"Run directory: {run_dir}")
