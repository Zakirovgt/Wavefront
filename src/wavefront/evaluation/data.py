from __future__ import annotations

import numpy as np

from wavefront.data.interpolation import (
    interpolate_sensor_derivatives_to_grid,
)
from wavefront.data.sensors import load_branch_sensor_grid
from wavefront.evaluation.config import EvalConfig


def take_common_test(
        wavefronts_all,
        sensor_gradients_all,
        grid_gradients_all,
        cfg: EvalConfig,
):
    """
    Select the same held-out test partition for every evaluated operator.

    The split is defined by the shared evaluation configuration:

        test_start = cfg.n_train
        test_stop = cfg.n_train + cfg.n_test

    Args:
        wavefronts_all: Wavefront targets with the sample dimension first.
            Typical shapes are ``(N, p)`` or ``(N, H, W)``.
        sensor_gradients_all: Sparse sensor-gradient data with shape
            ``(N, P_sensor, 2)``.
        grid_gradients_all: Optional regular-grid gradient data with sample
            dimension first. Typical shapes are ``(N, p, 2)`` or
            ``(N, H, W, 2)``.
        cfg: Shared evaluation configuration defining the train/test split.

    Returns:
        A tuple containing:

            U_test:
                Held-out wavefront targets.

            D_sensor_test:
                Held-out sparse sensor gradients.

            D_grid_test:
                Held-out regular-grid gradients, or None when
                ``grid_gradients_all`` is None.

    Raises:
        ValueError: If input arrays do not contain the same number of samples,
            or if there are not enough samples for the configured split.
    """
    wavefronts_all = np.asarray(wavefronts_all)
    sensor_gradients_all = np.asarray(sensor_gradients_all)

    # All operator inputs and targets must share the same sample ordering.
    if wavefronts_all.shape[0] != sensor_gradients_all.shape[0]:
        raise ValueError(
            "wavefronts_all and sensor_gradients_all must contain the same "
            f"number of samples, but got {wavefronts_all.shape[0]} and "
            f"{sensor_gradients_all.shape[0]}."
        )

    # Validate regular-grid gradients when they are available.
    if grid_gradients_all is not None:
        grid_gradients_all = np.asarray(grid_gradients_all)

        if grid_gradients_all.shape[0] != wavefronts_all.shape[0]:
            raise ValueError(
                "grid_gradients_all and wavefronts_all must contain the same "
                f"number of samples, but got {grid_gradients_all.shape[0]} "
                f"and {wavefronts_all.shape[0]}."
            )

    # Use the same test split that follows the configured training partition.
    start = int(cfg.n_train)
    stop = start + int(cfg.n_test)

    if wavefronts_all.shape[0] < stop:
        raise ValueError(
            f"Need at least {stop} samples for the configured split, "
            f"but got {wavefronts_all.shape[0]}."
        )

    U_test = wavefronts_all[start:stop]
    D_sensor_test = sensor_gradients_all[start:stop]

    D_grid_test = (
        grid_gradients_all[start:stop]
        if grid_gradients_all is not None
        else None
    )

    return U_test, D_sensor_test, D_grid_test


def get_sensor_coords(
        cfg: EvalConfig,
) -> np.ndarray:
    """
    Load normalized sensor coordinates in the ordering expected by model inputs.

    Args:
        cfg: Shared evaluation configuration containing ``branch_grid_path``.

    Returns:
        Float32 sensor-coordinate array with shape ``(P_sensor, 2)``.

    Notes:
        The coordinate transform uses ``flip_y=True`` to match the convention
        used during DeepONet branch-input preparation and model training.
    """
    return np.asarray(
        load_branch_sensor_grid(
            cfg.branch_grid_path,
            use_flag=False,
            flip_y=True,
        ),
        dtype=np.float32,
    )


def get_grid_inputs(
        sensor_gradients,
        grid_gradients,
        cfg: EvalConfig,
) -> np.ndarray:
    """
    Prepare regular-grid gradient inputs compatible with FNO and Poisson models.

    Depending on ``cfg.mode``, the function either uses supplied regular-grid
    gradients directly or interpolates sparse sensor gradients onto a regular
    output grid.

    Args:
        sensor_gradients: Sparse sensor gradients with shape
            ``(N, P_sensor, 2)``. Used when ``cfg.mode == "sensor"``.
        grid_gradients: Optional precomputed regular-grid gradients with shape
            ``(N, p, 2)`` or ``(N, grid_size, grid_size, 2)``. Required when
            ``cfg.mode == "grid"``.
        cfg: Shared evaluation configuration.

    Returns:
        Float32 regular-grid gradients with shape:

            ``(N, grid_size, grid_size, 2)``

        where the final dimension stores:

            ``[..., 0] = dU/dx``
            ``[..., 1] = dU/dy``

    Raises:
        ValueError: If regular-grid input mode is selected without supplied
            grid gradients, or if provided grid gradients have an invalid
            shape.

    Notes:
        In sensor mode, interpolation behavior is controlled by
        ``cfg.sensor_to_grid_interp``.
    """
    grid_size = int(cfg.grid_size)

    if cfg.mode == "grid":
        # Use precomputed regular-grid gradients directly.
        if grid_gradients is None:
            raise ValueError(
                "mode='grid' requires precomputed grid_gradients."
            )

        X = np.asarray(
            grid_gradients,
            dtype=np.float32,
        )

        # Restore spatial dimensions when gradients are supplied in flattened
        # form: (N, grid_size * grid_size, 2).
        if X.ndim == 3:
            expected_p = grid_size * grid_size

            if X.shape[1:] != (expected_p, 2):
                raise ValueError(
                    "Flattened grid gradients must have shape "
                    f"(N, {expected_p}, 2), but got {X.shape}."
                )

            X = X.reshape(
                X.shape[0],
                grid_size,
                grid_size,
                2,
            )

        # Require a standard channels-last regular-grid representation.
        if X.ndim != 4 or X.shape[1:] != (
                grid_size,
                grid_size,
                2,
        ):
            raise ValueError(
                "grid_gradients must have shape "
                f"(N, {grid_size}, {grid_size}, 2), but got {X.shape}."
            )

        return X

    # In sensor mode, map irregular sensor derivatives onto the regular grid.
    sensor_gradients = np.asarray(
        sensor_gradients,
        dtype=np.float32,
    )

    sensor_coords = get_sensor_coords(cfg)

    return interpolate_sensor_derivatives_to_grid(
        derivatives_all=sensor_gradients,
        sensor_coords=sensor_coords,
        grid_size=grid_size,
        method=str(cfg.sensor_to_grid_interp),
    ).astype(np.float32)
