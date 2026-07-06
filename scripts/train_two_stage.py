from __future__ import annotations

import argparse
import json
from pathlib import Path

from wavefront.pipelines import (
    load_two_stage_run_kwargs,
    run_two_stage_pipeline,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the two-stage baseline training pipeline.

    Returns:
        Parsed command-line argument namespace containing:

            config:
                Path to the two-stage YAML configuration file.

            run_name:
                Optional output-directory name that overrides
                ``runtime.run_name`` from the YAML configuration.

            results_root:
                Optional root directory that overrides
                ``runtime.results_root`` from the YAML configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train the two-stage wavefront reconstruction baseline: "
            "Stage 1 DeepONet gradient mapping followed by end-to-end joint "
            "fine-tuning with a randomly initialized FNO."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/two_stage.yaml"),
        help=(
            "Path to the YAML configuration file for the two-stage "
            "training pipeline."
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional run directory name. Overrides runtime.run_name "
            "from the YAML configuration."
        ),
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory for training results. Overrides "
            "runtime.results_root from the YAML configuration."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Portable dataset artifact directory. When supplied, all pipeline "
            "stages reuse its fixed train/test split instead of generating "
            "synthetic datasets internally."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Load two-stage pipeline settings, launch training, and print the manifest.

    Command-line arguments override the corresponding runtime values from the
    YAML configuration when explicitly supplied.

    Side Effects:
        - Trains the Stage-1 DeepONet gradient-map model.
        - Creates a randomly initialized FNO.
        - Jointly fine-tunes the DeepONet–FNO pipeline against wavefront
          targets.
        - Writes checkpoints, metrics, configuration files, and manifests
          under the configured results directory.
        - Prints the final pipeline manifest as formatted JSON.
    """
    cli_args = parse_args()

    # Load the YAML file and convert its contents into keyword arguments
    # accepted by run_two_stage_pipeline.
    run_kwargs = load_two_stage_run_kwargs(cli_args.config)

    # Explicit command-line options take precedence over YAML runtime settings.
    if cli_args.run_name is not None:
        run_kwargs["run_name"] = cli_args.run_name

    if cli_args.results_root is not None:
        run_kwargs["results_root"] = str(cli_args.results_root)

    if cli_args.dataset_dir is not None:
        run_kwargs["dataset_dir"] = str(cli_args.dataset_dir)
    # Run both training stages and collect artifact locations plus summary
    # validation metrics.
    manifest = run_two_stage_pipeline(**run_kwargs)

    print("\nTwo-stage baseline training completed.")

    # Print the final manifest in a readable UTF-8 JSON representation.
    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
