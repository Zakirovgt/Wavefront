from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from wavefront.data.dataset_artifacts import (
    generate_and_save_synthetic_dataset,
)
from wavefront.data.synthetic import (
    load_synthetic_dataset_config,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for synthetic dataset generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate and save a portable synthetic Wavefront dataset."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/standalone_data.yaml"),
        help="Path to the synthetic dataset YAML configuration.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Dataset artifact directory. Defaults to "
            "datasets/synthetic/<timestamp>."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace files in an existing dataset directory. "
            "Use only when intentional."
        ),
    )

    parser.add_argument(
        "--no-copy-sensor-layout",
        action="store_true",
        help=(
            "Do not copy the sensor-layout CSV into the dataset artifact."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate and print dataset settings without generating arrays."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Generate one reproducible synthetic dataset artifact."""
    args = parse_args()

    config = load_synthetic_dataset_config(args.config)

    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("datasets") / "synthetic" / timestamp
    else:
        output_dir = args.output_dir

    resolved = {
        "output_dir": str(output_dir),
        "overwrite": bool(args.overwrite),
        "copy_sensor_layout": not args.no_copy_sensor_layout,
        "config": asdict(config),
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

    artifact = generate_and_save_synthetic_dataset(
        output_dir=output_dir,
        config=config,
        overwrite=bool(args.overwrite),
        copy_sensor_layout=not args.no_copy_sensor_layout,
    )

    print("Dataset generation completed successfully.")
    print(f"Dataset directory: {artifact.root_dir}")
    print(f"Wavefronts shape: {artifact.wavefronts.shape}")
    print(
        "Sensor gradients shape: "
        f"{artifact.sensor_gradients.shape}"
    )
    print(
        "Grid gradients shape: "
        f"{artifact.grid_gradients.shape}"
    )


if __name__ == "__main__":
    main()
