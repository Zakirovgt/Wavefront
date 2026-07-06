from __future__ import annotations

from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np

from wavefront.data.interpolation import (
    interpolate_sensor_derivatives_to_grid,
)
from wavefront.data.sensors import load_branch_sensor_grid
from wavefront.metamodel.gradmap import (
    deeponet_predict_grad_grid,
)
from wavefront.training.fno import apply_fno

InterpolationMethod = Literal[
    "linear",
    "nearest",
    "cubic",
]


def _get_model_from_state(state: dict):
    """Return a callable model from a supported loaded-state key."""
    for key in ("model_eval_fn", "model_fn", "model"):
        model = state.get(key)

        if model is not None:
            return model

    raise KeyError(
        "State must contain one of: "
        "'model_eval_fn', 'model_fn', or 'model'."
    )


def _as_sensor_gradient_batch(
        sensor_gradients,
) -> np.ndarray:
    """
    Convert sparse gradients to a batch with shape (B, P_sensor, 2).

    A single sample with shape (P_sensor, 2) is promoted to a batch of size 1.
    """
    sensor_gradients = np.asarray(
        sensor_gradients,
        dtype=np.float32,
    )

    if sensor_gradients.ndim == 2:
        sensor_gradients = sensor_gradients[None, ...]

    if (
            sensor_gradients.ndim != 3
            or sensor_gradients.shape[-1] != 2
    ):
        raise ValueError(
            "sensor_gradients must have shape (P_sensor, 2) or "
            f"(B, P_sensor, 2), got {sensor_gradients.shape}."
        )

    return sensor_gradients


def _as_gradient_grid_batch(
        gradient_grids,
        grid_size: int,
) -> np.ndarray:
    """
    Convert regular-grid gradients to shape (B, grid_size, grid_size, 2).

    Accepted inputs:
        (grid_size, grid_size, 2)
        (B, grid_size, grid_size, 2)
        (grid_size ** 2, 2)
        (B, grid_size ** 2, 2)
    """
    gradient_grids = np.asarray(
        gradient_grids,
        dtype=np.float32,
    )

    n_points = grid_size * grid_size

    if gradient_grids.ndim == 2:
        if gradient_grids.shape != (n_points, 2):
            raise ValueError(
                "Flattened gradient input must have shape "
                f"({n_points}, 2), got {gradient_grids.shape}."
            )

        return gradient_grids.reshape(
            1,
            grid_size,
            grid_size,
            2,
        )

    if gradient_grids.ndim == 3:
        if gradient_grids.shape[1:] != (n_points, 2):
            raise ValueError(
                "Batched flattened gradient input must have shape "
                f"(B, {n_points}, 2), got {gradient_grids.shape}."
            )

        return gradient_grids.reshape(
            gradient_grids.shape[0],
            grid_size,
            grid_size,
            2,
        )

    if gradient_grids.ndim == 4:
        expected_shape = (
            grid_size,
            grid_size,
            2,
        )

        if gradient_grids.shape[1:] != expected_shape:
            raise ValueError(
                "Grid gradient input must have shape "
                f"(B, {grid_size}, {grid_size}, 2), "
                f"got {gradient_grids.shape}."
            )

        return gradient_grids

    raise ValueError(
        "gradient_grids must have shape (P, 2), (B, P, 2), "
        f"({grid_size}, {grid_size}, 2), or "
        f"(B, {grid_size}, {grid_size}, 2)."
    )


def _as_wavefront_batch(
        prediction,
        grid_size: int,
) -> np.ndarray:
    """Convert an FNO prediction to shape (B, grid_size, grid_size)."""
    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    if prediction.ndim == 4:
        if prediction.shape[-1] == 1:
            prediction = prediction[..., 0]
        elif prediction.shape[1] == 1:
            prediction = prediction[:, 0, ...]

    if prediction.ndim == 2:
        n_points = grid_size * grid_size

        if prediction.shape[1] != n_points:
            raise ValueError(
                "Flattened FNO output must have shape "
                f"(B, {n_points}), got {prediction.shape}."
            )

        prediction = prediction.reshape(
            prediction.shape[0],
            grid_size,
            grid_size,
        )

    if prediction.ndim != 3:
        raise ValueError(
            "FNO output must have shape "
            f"(B, {grid_size}, {grid_size}), got {prediction.shape}."
        )

    if prediction.shape[1:] != (grid_size, grid_size):
        raise ValueError(
            "FNO output has an unexpected spatial shape: "
            f"{prediction.shape}."
        )

    return prediction


def _grid_size_from_fno_state(
        state: dict,
) -> int:
    """Read the FNO output-grid resolution from its saved configuration."""
    args = state.get("args")

    if args is None:
        raise KeyError(
            "Standalone FNO state does not contain saved 'args'."
        )

    return int(
        getattr(
            args,
            "grid_size",
            getattr(args, "nx", 24),
        )
    )


def _sensor_coords_from_fno_state(
        state: dict,
) -> np.ndarray:
    """
    Load the sensor geometry expected by a standalone FNO run.

    The original FNO trainer stores branch_grid_path in its training arguments.
    The loaded CSV defines the coordinate order used for interpolation.
    """
    args = state.get("args")

    if args is None:
        raise KeyError(
            "Standalone FNO state does not contain saved 'args'."
        )

    branch_grid_path = getattr(
        args,
        "branch_grid_path",
        None,
    )

    if not branch_grid_path:
        raise KeyError(
            "FNO training arguments do not contain branch_grid_path. "
            "Provide sensor coordinates explicitly in a future custom "
            "inference wrapper or retrain with this path recorded."
        )

    return np.asarray(
        load_branch_sensor_grid(
            Path(branch_grid_path),
            use_flag=False,
            flip_y=True,
        ),
        dtype=np.float32,
    )


def predict_fno_from_gradient_grids(
        state: dict,
        gradient_grids,
) -> np.ndarray:
    """
    Predict normalized wavefronts from normalized regular-grid gradients.

    Args:
        state: Loaded standalone FNO state.
        gradient_grids: Gradients with shape (H, W, 2), (B, H, W, 2),
            (P, 2), or (B, P, 2).

    Returns:
        Normalized predicted wavefronts with shape (B, H, W).
    """
    grid_size = _grid_size_from_fno_state(state)

    gradient_grids = _as_gradient_grid_batch(
        gradient_grids,
        grid_size=grid_size,
    )

    model_fn = _get_model_from_state(state)
    params = state["params"]

    prediction = apply_fno(
        model_fn,
        params,
        jnp.asarray(
            gradient_grids,
            dtype=jnp.float32,
        ),
        rng=None,
        training=False,
    )

    return _as_wavefront_batch(
        prediction,
        grid_size=grid_size,
    )


def predict_fno_from_sensor_gradients(
    state: dict,
    sensor_gradients,
    *,
    interpolation_method: InterpolationMethod | None = None,
    sensor_coords=None,
) -> dict[str, np.ndarray]:
    """
    Predict normalized wavefronts from normalized sparse sensor gradients.

    The sparse gradients are interpolated to the regular grid expected by FNO.

    Args:
        state:
            Loaded standalone FNO state.

        sensor_gradients:
            Normalized sparse gradients with shape ``(P_sensor, 2)`` or
            ``(B, P_sensor, 2)``.

        interpolation_method:
            Optional interpolation method. When omitted, uses the method
            stored in the FNO training configuration.

        sensor_coords:
            Optional normalized sensor coordinates with shape
            ``(P_sensor, 2)``.

            When provided, these coordinates override the
            ``branch_grid_path`` stored in the training artifact. This is
            useful when a trained run is moved to another machine or when the
            original CSV path no longer exists.

    Returns:
        A dictionary containing:

            wavefronts:
                Predicted normalized wavefronts with shape ``(B, H, W)``.

            input_gradient_grids:
                Interpolated FNO inputs with shape ``(B, H, W, 2)``.
    """
    sensor_gradients = _as_sensor_gradient_batch(
        sensor_gradients,
    )

    grid_size = _grid_size_from_fno_state(state)

    if sensor_coords is None:
        sensor_coords = _sensor_coords_from_fno_state(state)
    else:
        sensor_coords = np.asarray(
            sensor_coords,
            dtype=np.float32,
        )

        if sensor_coords.ndim != 2 or sensor_coords.shape[1] != 2:
            raise ValueError(
                "sensor_coords must have shape (P_sensor, 2), "
                f"got {sensor_coords.shape}."
            )

    if sensor_gradients.shape[1] != sensor_coords.shape[0]:
        raise ValueError(
            "Sensor count does not match the FNO training geometry: "
            f"input has {sensor_gradients.shape[1]}, "
            f"model expects {sensor_coords.shape[0]}."
        )

    args = state.get("args")
    method = interpolation_method or getattr(
        args,
        "sensor_to_grid_interp",
        "linear",
    )

    input_gradient_grids = (
        interpolate_sensor_derivatives_to_grid(
            derivatives_all=sensor_gradients,
            sensor_coords=sensor_coords,
            grid_size=grid_size,
            method=str(method),
        )
    ).astype(np.float32)

    wavefronts = predict_fno_from_gradient_grids(
        state=state,
        gradient_grids=input_gradient_grids,
    )

    return {
        "wavefronts": wavefronts,
        "input_gradient_grids": input_gradient_grids,
    }


def _grid_size_from_metamodel_state(
        state: dict,
) -> int:
    """Read the grid resolution from the Stage-1 DeepONet configuration."""
    deeponet_state = state.get("deeponet")

    if deeponet_state is None:
        raise KeyError(
            "Metamodel state must contain a 'deeponet' state."
        )

    cfg = deeponet_state.get("cfg")

    if cfg is None:
        raise KeyError(
            "Stage-1 DeepONet state does not contain its configuration."
        )

    return int(cfg.grid_size)


def predict_metamodel_from_sensor_gradients(
        state: dict,
        sensor_gradients,
) -> dict[str, np.ndarray]:
    """
    Predict normalized wavefronts from normalized sparse sensor gradients.

    The metamodel performs:

        sensor gradients
        -> Stage-1 DeepONet gradient map
        -> Stage-2 FNO
        -> wavefront

    Args:
        state: Loaded two-stage or three-stage metamodel state.
        sensor_gradients: Shape (P_sensor, 2) or (B, P_sensor, 2).

    Returns:
        A dictionary containing:

            wavefronts:
                Predicted normalized wavefronts with shape (B, H, W).

            predicted_gradient_grids:
                Stage-1 predicted gradients with shape (B, H, W, 2).
    """
    sensor_gradients = _as_sensor_gradient_batch(
        sensor_gradients,
    )

    deeponet_state = state["deeponet"]
    fno_state = state["fno"]

    grid_size = _grid_size_from_metamodel_state(state)

    expected_sensor_coords = deeponet_state.get(
        "branch_sensor_coords",
    )

    if expected_sensor_coords is not None:
        if sensor_gradients.shape[1] != expected_sensor_coords.shape[0]:
            raise ValueError(
                "Sensor count does not match the Stage-1 DeepONet geometry: "
                f"input has {sensor_gradients.shape[1]}, "
                f"model expects {expected_sensor_coords.shape[0]}."
            )

    deeponet_model = _get_model_from_state(deeponet_state)
    deeponet_params = deeponet_state["params"]

    predicted_gradient_grids = deeponet_predict_grad_grid(
        model_fn=deeponet_model,
        params=deeponet_params,
        grad_sensor_batch=jnp.asarray(
            sensor_gradients,
            dtype=jnp.float32,
        ),
        grid_size=grid_size,
        rng=None,
    )

    predicted_gradient_grids = np.asarray(
        predicted_gradient_grids,
        dtype=np.float32,
    )

    wavefronts = predict_fno_from_gradient_grids(
        state=fno_state,
        gradient_grids=predicted_gradient_grids,
    )

    return {
        "wavefronts": wavefronts,
        "predicted_gradient_grids": predicted_gradient_grids,
    }
