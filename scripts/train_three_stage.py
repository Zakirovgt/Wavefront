from __future__ import annotations

import argparse
import json
from pathlib import Path

from wavefront.pipelines import (
    load_three_stage_run_kwargs,
    run_three_stage_pipeline,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for a three-stage training pipeline run.

    Returns:
        Parsed command-line argument namespace containing:

            config:
                Path to the three-stage YAML configuration file.

            run_name:
                Optional output-directory name that overrides
                ``runtime.run_name`` from the YAML configuration.

            results_root:
                Optional results root directory that overrides
                ``runtime.results_root`` from the YAML configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train the complete three-stage wavefront reconstruction pipeline: "
            "Stage 1 DeepONet gradient mapping, Stage 2 FNO reconstruction, "
            "and Stage 3 joint end-to-end fine-tuning."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/three_stage.yaml"),
        help=(
            "Path to the YAML configuration file for the three-stage "
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
    Load pipeline configuration, run three-stage training, and print metadata.

    The command-line arguments override selected runtime values from the YAML
    configuration before the pipeline begins.

    Side Effects:
        - Trains the Stage-1 DeepONet gradient-map model.
        - Trains the Stage-2 FNO wavefront-reconstruction model.
        - Performs Stage-3 joint DeepONet–FNO fine-tuning.
        - Writes model checkpoints, metrics, configuration files, and
          manifests under the configured results directory.
        - Prints the final pipeline manifest as formatted JSON.
    """
    cli_args = parse_args()

    # Load YAML configuration and convert it into keyword arguments accepted by
    # run_three_stage_pipeline.
    run_kwargs = load_three_stage_run_kwargs(cli_args.config)

    # Command-line values take precedence over the corresponding YAML runtime
    # settings when explicitly supplied.
    if cli_args.run_name is not None:
        run_kwargs["run_name"] = cli_args.run_name

    if cli_args.results_root is not None:
        run_kwargs["results_root"] = str(cli_args.results_root)
    if cli_args.dataset_dir is not None:
        run_kwargs["dataset_dir"] = str(cli_args.dataset_dir)
    # Execute all three training stages and receive paths plus summary metrics.
    manifest = run_three_stage_pipeline(**run_kwargs)

    print("\nThree-stage training completed.")

    # Print the final manifest in a human-readable UTF-8 JSON representation.
    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
