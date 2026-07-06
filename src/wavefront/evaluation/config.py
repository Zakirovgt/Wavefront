from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalConfig:
    """
    Shared configuration for evaluating wavefront reconstruction operators.

    This configuration supports evaluation workflows for DeepONet, FNO, and
    related baselines that reconstruct wavefronts from either sparse sensor
    gradients or regular-grid gradient fields.

    Attributes:
        n_train: Number of samples reserved for the training partition when
            reproducing the dataset split used by an operator.
        n_test: Number of samples reserved for the test partition.
        n_eval: Maximum number of samples used during evaluation and inference
            benchmarking.
        seed: Random seed used for reproducible dataset generation or sample
            selection.
        grid_size: Number of points along each spatial dimension of the
            regular reconstruction grid.
        deeponet_p_test: Number of DeepONet output query points. When None,
            it is derived as ``grid_size * grid_size``.
        branch_grid_path: Path to the CSV file describing the sensor layout
            used by the DeepONet branch input.
        sensor_to_grid_interp: Interpolation method used to map sparse sensor
            gradients to a regular grid for FNO evaluation.
        bench_batch: Number of complete functions processed per batch during
            inference benchmarking.
        mode: Gradient-input mode:
            - ``"sensor"``: Interpolate sparse sensor gradients onto a
              regular grid.
            - ``"grid"``: Use precomputed regular-grid gradients directly.
    """

    n_train: int = 10_000
    n_test: int = 1_000
    n_eval: int = 1_000
    seed: int = 1234

    grid_size: int = 24
    deeponet_p_test: int | None = None

    branch_grid_path: str = "1.csv"
    sensor_to_grid_interp: str = "linear"
    bench_batch: int = 64

    # "sensor": interpolate sparse sensor gradients onto a regular grid.
    # "grid": use precomputed regular-grid gradients directly.
    mode: str = "sensor"

    def __post_init__(self) -> None:
        """
        Validate configuration values and derive omitted dependent settings.

        Raises:
            ValueError: If grid resolution, dataset sizes, evaluation count,
                benchmark batch size, or evaluation mode is invalid.
        """
        # A two-dimensional grid requires at least two points per axis.
        if self.grid_size < 2:
            raise ValueError(
                "grid_size must be at least 2."
            )

        # Dataset split sizes may be zero individually, but cannot be negative.
        if self.n_train < 0 or self.n_test < 0:
            raise ValueError(
                "n_train and n_test must be non-negative."
            )

        # At least one sample must be available for evaluation.
        if self.n_eval < 1:
            raise ValueError(
                "n_eval must be at least 1."
            )

        # Benchmarking requires a positive inference batch size.
        if self.bench_batch < 1:
            raise ValueError(
                "bench_batch must be at least 1."
            )

        # Restrict gradient-source handling to supported modes.
        if self.mode not in {"sensor", "grid"}:
            raise ValueError(
                f"Unknown evaluation mode {self.mode!r}. "
                "Expected 'sensor' or 'grid'."
            )

        # By default, DeepONet is evaluated at every point of the regular
        # reconstruction grid.
        if self.deeponet_p_test is None:
            self.deeponet_p_test = (
                    self.grid_size * self.grid_size
            )
