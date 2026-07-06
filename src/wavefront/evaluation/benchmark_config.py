from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from wavefront.evaluation.config import EvalConfig


@dataclass
class BenchmarkDataConfig:
    """
    Synthetic holdout-dataset settings used for a benchmark run.

    The generated dataset includes a training partition and a held-out test
    partition. The training portion is retained to reproduce the same split
    convention used by the trained operators, while benchmarking is performed
    on the held-out samples.

    Attributes:
        grid_size: Number of points along each spatial axis of the regular
            wavefront and gradient grid.
        n_train: Number of samples assigned to the training partition.
        n_test: Number of held-out samples available for benchmarking.
        seed: Random seed used to generate the synthetic dataset.
        frac_zernike: Fraction of generated samples based on Zernike modes.
        frac_spiral: Fraction of generated samples based on spiral wavefronts.
        frac_distortion: Fraction of generated samples based on distortion
            wavefronts.
        with_noise: Whether noise is added to sensor-gradient measurements.
        noise_percentage: Relative amplitude of added sensor noise.
        noise_lambda: Noise correlation or filtering parameter.
        apply_blur: Whether generated fields receive optional spatial blur.
        sigma_pix: Gaussian blur standard deviation in pixel units.
    """

    grid_size: int = 24
    n_train: int = 10_000
    n_test: int = 1_000
    seed: int = 20260706

    frac_zernike: float = 0.0
    frac_spiral: float = 0.0
    frac_distortion: float = 1.0

    with_noise: bool = True
    noise_percentage: float = 1.0
    noise_lambda: float = 0.03

    apply_blur: bool = False
    sigma_pix: float = 1.0

    def __post_init__(self) -> None:
        """
        Validate synthetic dataset dimensions and mixture weights.

        Raises:
            ValueError: If grid size or split sizes are invalid, if any
                dataset-mixture fraction is negative, or if all fractions
                are zero.
        """
        # A two-dimensional regular grid needs at least two points per axis.
        if self.grid_size < 2:
            raise ValueError(
                "grid_size must be at least 2."
            )

        # The training partition may be empty for special evaluation workflows,
        # but it cannot have a negative size.
        if self.n_train < 0:
            raise ValueError(
                "n_train must be non-negative."
            )

        # Benchmarking requires at least one held-out sample.
        if self.n_test < 1:
            raise ValueError(
                "n_test must be at least 1."
            )

        fractions = (
            float(self.frac_zernike),
            float(self.frac_spiral),
            float(self.frac_distortion),
        )

        # Negative mixture weights are not meaningful.
        if any(value < 0.0 for value in fractions):
            raise ValueError(
                "Dataset mixture fractions must be non-negative."
            )

        # At least one waveform family must contribute samples.
        if sum(fractions) <= 0.0:
            raise ValueError(
                "At least one dataset mixture fraction must be positive."
            )


@dataclass
class BenchmarkVisualizationConfig:
    """Configuration for benchmark sample figures."""

    enabled: bool = True
    sample_indices: list[int] | None = None

    error_mode: str = "global"
    solution_percentiles: tuple[float, float] = (0.0, 100.0)
    error_percentile: float = 100.0
    separate_poisson_scale: bool = False

    save_stage1_gradient: bool = True
    stage1_source: str = "auto"
    dpi: int = 180

    def __post_init__(self) -> None:
        """Validate visualization settings."""
        if self.sample_indices is None:
            self.sample_indices = [0]

        self.sample_indices = sorted(
            {int(index) for index in self.sample_indices}
        )

        if not self.sample_indices:
            raise ValueError(
                "visualization.sample_indices must not be empty."
            )

        if any(index < 0 for index in self.sample_indices):
            raise ValueError(
                "visualization.sample_indices must be non-negative."
            )

        if self.error_mode not in {"global", "l2", "pointwise"}:
            raise ValueError(
                "visualization.error_mode must be one of: "
                "'global', 'l2', 'pointwise'."
            )

        if self.stage1_source not in {
            "auto",
            "two_stage",
            "three_stage",
            "none",
        }:
            raise ValueError(
                "visualization.stage1_source must be one of: "
                "'auto', 'two_stage', 'three_stage', 'none'."
            )

        if self.dpi < 1:
            raise ValueError("visualization.dpi must be at least 1.")

        if len(self.solution_percentiles) != 2:
            raise ValueError(
                "visualization.solution_percentiles must contain "
                "exactly two values."
            )


@dataclass
class BenchmarkRunConfig:
    """
    Fully resolved benchmark settings loaded from a YAML configuration file.

    Attributes:
        branch_grid_path: CSV path defining the sensor layout and ordering.
        output_root: Root directory for benchmark reports and artifacts.
        run_name: Optional benchmark output-directory name.
        checkpoint: Checkpoint selector, either ``"best"`` or ``"last"``.
        fail_fast: Whether evaluation stops at the first operator failure.
        mixed_precision: Whether mixed precision is enabled for evaluation.
        mp_dtype: Mixed-precision dtype, such as ``"bfloat16"``.
        data_cfg: Synthetic holdout-dataset configuration.
        eval_cfg: Shared operator-evaluation configuration.
        deeponet_run: Optional standalone DeepONet run directory.
        fno_run: Optional standalone FNO run directory.
        two_stage_run: Optional two-stage metamodel run directory.
        three_stage_run: Optional three-stage metamodel run directory.
    """

    branch_grid_path: str
    output_root: str
    run_name: str | None

    checkpoint: str
    fail_fast: bool

    mixed_precision: bool
    mp_dtype: str

    data_cfg: BenchmarkDataConfig
    eval_cfg: EvalConfig

    deeponet_run: str | None
    fno_run: str | None
    two_stage_run: str | None
    three_stage_run: str | None
    visualization_cfg: BenchmarkVisualizationConfig


def _read_yaml_mapping(
        path: str | Path,
) -> dict[str, Any]:
    """
    Load a YAML file and require a mapping at the top level.

    Args:
        path: Path to the benchmark YAML configuration file.

    Returns:
        Parsed YAML content as a dictionary. An empty YAML document produces
        an empty dictionary.

    Raises:
        FileNotFoundError: If the requested YAML file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
        TypeError: If the top-level YAML value is not a mapping.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    # Treat an empty YAML document as an empty configuration.
    if payload is None:
        return {}

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a YAML mapping in {path}, "
            f"but got {type(payload).__name__}."
        )

    return payload


def _validate_keys(
        section_name: str,
        values: dict[str, Any],
        allowed_keys: set[str],
) -> None:
    """
    Validate that a YAML section contains only supported configuration keys.

    Args:
        section_name: Human-readable YAML section name.
        values: Parsed key-value mapping for that section.
        allowed_keys: Set of accepted keys for the section.

    Raises:
        KeyError: If one or more unknown keys are present.

    Notes:
        Early key validation helps catch misspellings that would otherwise
        silently fall back to defaults or be ignored.
    """
    unknown_keys = set(values) - allowed_keys

    if unknown_keys:
        unknown_text = ", ".join(sorted(unknown_keys))

        raise KeyError(
            f"Unknown key(s) in '{section_name}': {unknown_text}."
        )


def _optional_path(
        value: Any,
        field_name: str,
) -> str | None:
    """
    Convert an optional YAML run-path value into a string or None.

    Args:
        value: Value read from the YAML ``runs`` section.
        field_name: Name of the run-path field, used in error messages.

    Returns:
        The path string when supplied, otherwise None.

    Raises:
        TypeError: If a non-null value is not a string.
    """
    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError(
            f"runs.{field_name} must be a path string or null, "
            f"but got {type(value).__name__}."
        )

    return value


def load_benchmark_config(
        config_path: str | Path,
) -> BenchmarkRunConfig:
    """
    Load a benchmark YAML file and return fully validated runtime settings.

    The synthetic-data configuration and ``EvalConfig`` use the same grid
    resolution and train/test split. This ensures that all selected operators
    are evaluated on exactly the same held-out examples.

    Args:
        config_path: Path to the benchmark YAML configuration file.

    Returns:
        A fully resolved ``BenchmarkRunConfig`` instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
        TypeError: If a named section is not a mapping or a run path has an
            invalid type.
        KeyError: If a section contains unsupported keys.
        ValueError: If data, evaluation, or visualization settings are invalid.
    """
    config = _read_yaml_mapping(config_path)

    # Missing sections are treated as empty mappings so defaults can be used.
    runtime = config.get("runtime", {})
    data = config.get("data", {})
    evaluation = config.get("evaluation", {})
    visualization = config.get("visualization", {})
    runs = config.get("runs", {})

    # Every named section must be a YAML mapping.
    for section_name, section in {
        "runtime": runtime,
        "data": data,
        "evaluation": evaluation,
        "visualization": visualization,
        "runs": runs,
    }.items():
        if not isinstance(section, dict):
            raise TypeError(
                f"Section '{section_name}' must be a YAML mapping."
            )

    # Reject unsupported settings before constructing configuration objects.
    _validate_keys(
        "runtime",
        runtime,
        {
            "branch_grid_path",
            "output_root",
            "run_name",
            "checkpoint",
            "fail_fast",
            "mixed_precision",
            "mp_dtype",
        },
    )

    _validate_keys(
        "data",
        data,
        {
            "grid_size",
            "n_train",
            "n_test",
            "seed",
            "frac_zernike",
            "frac_spiral",
            "frac_distortion",
            "with_noise",
            "noise_percentage",
            "noise_lambda",
            "apply_blur",
            "sigma_pix",
        },
    )

    _validate_keys(
        "evaluation",
        evaluation,
        {
            "n_eval",
            "mode",
            "sensor_to_grid_interp",
            "bench_batch",
        },
    )

    _validate_keys(
        "visualization",
        visualization,
        {
            "enabled",
            "sample_indices",
            "error_mode",
            "solution_percentiles",
            "error_percentile",
            "separate_poisson_scale",
            "save_stage1_gradient",
            "stage1_source",
            "dpi",
        },
    )

    _validate_keys(
        "runs",
        runs,
        {
            "deeponet",
            "fno",
            "two_stage",
            "three_stage",
        },
    )

    # Build and validate the shared synthetic holdout-dataset configuration.
    data_cfg = BenchmarkDataConfig(**data)

    # The dataset configuration is the single source of truth for the split
    # and regular-grid dimensions used by all compared operators.
    eval_cfg = EvalConfig(
        n_train=int(data_cfg.n_train),
        n_test=int(data_cfg.n_test),
        n_eval=int(
            evaluation.get("n_eval", data_cfg.n_test)
        ),
        seed=int(data_cfg.seed),
        grid_size=int(data_cfg.grid_size),
        branch_grid_path=str(
            runtime.get("branch_grid_path", "1.csv")
        ),
        sensor_to_grid_interp=str(
            evaluation.get("sensor_to_grid_interp", "linear")
        ),
        bench_batch=int(
            evaluation.get("bench_batch", 64)
        ),
        mode=str(
            evaluation.get("mode", "sensor")
        ),
    )

    if eval_cfg.n_eval > data_cfg.n_test:
        raise ValueError(
            "evaluation.n_eval cannot exceed data.n_test: "
            f"{eval_cfg.n_eval} > {data_cfg.n_test}."
        )

    # YAML represents tuples as lists. Convert the plotting percentiles to a
    # tuple before creating the dataclass.
    visualization_values = dict(visualization)

    if "solution_percentiles" in visualization_values:
        visualization_values["solution_percentiles"] = tuple(
            visualization_values["solution_percentiles"]
        )

    visualization_cfg = BenchmarkVisualizationConfig(
        **visualization_values,
    )

    # Visualization indices are relative to the held-out test split used by
    # benchmark evaluation, not to the complete generated dataset.
    if any(
            index >= eval_cfg.n_eval
            for index in visualization_cfg.sample_indices
    ):
        raise ValueError(
            "Every visualization.sample_indices value must be smaller than "
            f"evaluation.n_eval={eval_cfg.n_eval}."
        )

    # The same checkpoint choice is applied to every learned operator.
    checkpoint = str(
        runtime.get("checkpoint", "best")
    )

    if checkpoint not in {"best", "last"}:
        raise ValueError(
            "runtime.checkpoint must be either 'best' or 'last'."
        )

    return BenchmarkRunConfig(
        branch_grid_path=str(
            runtime.get("branch_grid_path", "1.csv")
        ),
        output_root=str(
            runtime.get("output_root", "results/benchmarks")
        ),
        run_name=runtime.get("run_name"),
        checkpoint=checkpoint,
        fail_fast=bool(
            runtime.get("fail_fast", False)
        ),
        mixed_precision=bool(
            runtime.get("mixed_precision", False)
        ),
        mp_dtype=str(
            runtime.get("mp_dtype", "bfloat16")
        ),
        data_cfg=data_cfg,
        eval_cfg=eval_cfg,
        visualization_cfg=visualization_cfg,
        deeponet_run=_optional_path(
            runs.get("deeponet"),
            "deeponet",
        ),
        fno_run=_optional_path(
            runs.get("fno"),
            "fno",
        ),
        two_stage_run=_optional_path(
            runs.get("two_stage"),
            "two_stage",
        ),
        three_stage_run=_optional_path(
            runs.get("three_stage"),
            "three_stage",
        ),
    )
