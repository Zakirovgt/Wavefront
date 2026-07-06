from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from wavefront.baselines.poisson import (
    poisson_baseline_from_deriv_grid,
)
from wavefront.inference.deeponet import (
    predict_batch_on_grid,
)
from wavefront.metamodel.gradmap import (
    deeponet_predict_grad_grid,
)
from wavefront.training.fno import apply_fno


def _get_model_from_state(
        state: dict,
):
    """
    Return a callable model stored under a supported state-dictionary key.

    Args:
        state: Loaded model state. The function searches for a model under the
            following keys, in order:

            - ``"model_eval_fn"``
            - ``"model_fn"``
            - ``"model"``

    Returns:
        Callable model object associated with the loaded state.

    Raises:
        KeyError: If no supported model key is present or contains a usable
            value.
    """
    for key in ("model_eval_fn", "model_fn", "model"):
        model = state.get(key)

        if model is not None:
            return model

    raise KeyError(
        "State must contain one of: "
        "'model_eval_fn', 'model_fn', or 'model'."
    )


def _as_gradient_points(
        gradient,
        grid_size: int,
) -> np.ndarray:
    """
    Convert one gradient sample into a pointwise ``(P, 2)`` representation.

    Supported input layouts are:

        - Sparse sensor gradients: ``(P_sensor, 2)``.
        - Flattened regular-grid gradients: ``(grid_size**2, 2)``.
        - Regular-grid gradients: ``(grid_size, grid_size, 2)``.
        - Flat derivative vector: ``(2 * P,)``.

    Args:
        gradient: Gradient sample in one supported sparse, flattened, or
            regular-grid layout.
        grid_size: Number of spatial points along one regular-grid axis.

    Returns:
        Float32 pointwise gradient array with shape ``(P, 2)``, where:

            ``[..., 0] = dU/dx``
            ``[..., 1] = dU/dy``

    Raises:
        ValueError: If the gradient layout is unsupported or a flat vector has
            an odd number of values.
    """
    gradient = np.asarray(
        gradient,
        dtype=np.float32,
    )

    # Restore derivative pairs when a single flat vector is provided.
    if gradient.ndim == 1:
        if gradient.size % 2 != 0:
            raise ValueError(
                "A flattened gradient vector must contain an even number of "
                f"values, but got shape {gradient.shape}."
            )

        return gradient.reshape(-1, 2)

    # Preserve sparse or already flattened pointwise gradients.
    if gradient.ndim == 2 and gradient.shape[-1] == 2:
        return gradient

    # Flatten a channel-last regular grid into spatial query points.
    if gradient.ndim == 3 and gradient.shape[-1] == 2:
        if gradient.shape[:2] != (grid_size, grid_size):
            raise ValueError(
                "A regular-grid gradient must have shape "
                f"({grid_size}, {grid_size}, 2), but got {gradient.shape}."
            )

        return gradient.reshape(-1, 2)

    raise ValueError(
        "Gradient input must have shape (P, 2), (grid, grid, 2), "
        f"or a flat vector, but got {gradient.shape}."
    )


def as_gradient_grid(
        gradient,
        grid_size: int,
) -> np.ndarray:
    """
    Convert one gradient sample into a channel-last regular-grid layout.

    Supported input layouts are:

        - ``(grid_size**2, 2)``
        - ``(2, grid_size**2)``
        - ``(grid_size, grid_size, 2)``
        - ``(2, grid_size, grid_size)``

    Args:
        gradient: Gradient sample in a supported flattened or regular-grid
            format.
        grid_size: Number of spatial points along each output-grid axis.

    Returns:
        Float32 gradient field with shape:

            ``(grid_size, grid_size, 2)``

    Raises:
        ValueError: If the supplied gradient does not match a supported grid
            representation.

    Notes:
        Channel-first inputs are converted to the channels-last convention used
        by FNO and Poisson reconstruction helpers.
    """
    gradient = np.asarray(
        gradient,
        dtype=np.float32,
    )

    expected_points = grid_size * grid_size

    # Convert flattened pointwise derivative pairs to a channel-last grid.
    if gradient.ndim == 2:
        if gradient.shape == (expected_points, 2):
            return gradient.reshape(
                grid_size,
                grid_size,
                2,
            )

        # Convert flattened channel-first derivatives to channels-last layout.
        if gradient.shape == (2, expected_points):
            return np.moveaxis(
                gradient.reshape(
                    2,
                    grid_size,
                    grid_size,
                ),
                0,
                -1,
            )

    # Preserve channels-last grids and convert channel-first grids.
    if gradient.ndim == 3:
        if gradient.shape == (grid_size, grid_size, 2):
            return gradient

        if gradient.shape == (2, grid_size, grid_size):
            return np.moveaxis(
                gradient,
                0,
                -1,
            )

    raise ValueError(
        "Gradient grid must have shape "
        f"({expected_points}, 2), "
        f"({grid_size}, {grid_size}, 2), or "
        f"(2, {grid_size}, {grid_size}); "
        f"got {gradient.shape}."
    )


def select_deeponet_input(
        state: dict,
        sensor_gradient,
        grid_gradient=None,
) -> np.ndarray:
    """
    Select the branch input required by a standalone DeepONet model.

    A sensor-mode DeepONet receives sparse sensor gradients directly. A
    regular-grid DeepONet receives gradients flattened into pointwise
    ``(P, 2)`` format.

    Args:
        state: Loaded standalone DeepONet state containing its saved argument
            namespace under ``"args"`` when available.
        sensor_gradient: Sparse gradient data for one sample.
        grid_gradient: Optional regular-grid gradient data for one sample.
            Required when the DeepONet was trained in regular-grid mode.

    Returns:
        Float32 DeepONet branch input array in the layout expected by the
        loaded model.

    Raises:
        ValueError: If a regular-grid DeepONet is selected but no
            ``grid_gradient`` is supplied.
    """
    args = state.get("args")
    data_mode = getattr(args, "data_mode", "sensor")

    # Regular-grid DeepONets expect one derivative pair per grid point.
    if data_mode in {"grid", "regular_grid"}:
        if grid_gradient is None:
            raise ValueError(
                "A DeepONet trained in regular_grid mode requires a "
                "regular-grid gradient input."
            )

        grid_size = int(
            getattr(
                args,
                "grid_size",
                getattr(args, "nx", 24),
            )
        )

        return _as_gradient_points(
            grid_gradient,
            grid_size=grid_size,
        )

    # Sensor-mode DeepONets consume sparse measurements without interpolation.
    return np.asarray(
        sensor_gradient,
        dtype=np.float32,
    )


def predict_deeponet_sample(
        state: dict,
        gradient_input,
        grid_size: int,
) -> np.ndarray:
    """
    Reconstruct one wavefront with a standalone DeepONet.

    Args:
        state: Loaded standalone DeepONet state containing the model and
            restored parameter tree.
        gradient_input: Sparse or regular-grid gradient input for one sample.
        grid_size: Number of spatial points along each output-wavefront axis.

    Returns:
        Float32 reconstructed wavefront with shape:

            ``(grid_size, grid_size)``

    Notes:
        The input is normalized into pointwise ``(P, 2)`` form before being
        passed to the batched DeepONet grid-inference helper.
    """
    model_fn = _get_model_from_state(state)
    params = state["params"]

    # Normalize sparse or grid-shaped derivatives into a pointwise layout.
    gradient_points = _as_gradient_points(
        gradient_input,
        grid_size=grid_size,
    )

    prediction = predict_batch_on_grid(
        model_fn=model_fn,
        params=params,
        grad_sensor_batch=gradient_points[None, ...],
        p_test=grid_size * grid_size,
    )

    return np.asarray(
        prediction["pred_grid"][0],
        dtype=np.float32,
    )


def predict_fno_sample(
        state: dict,
        gradient_grid,
        grid_size: int,
) -> np.ndarray:
    """
    Reconstruct one wavefront with a standalone FNO.

    Args:
        state: Loaded standalone FNO state containing the model and restored
            parameter tree.
        gradient_grid: Gradient field for one sample in any layout accepted by
            ``as_gradient_grid``.
        grid_size: Number of spatial points along each output-wavefront axis.

    Returns:
        Float32 reconstructed wavefront with shape:

            ``(grid_size, grid_size)``

    Raises:
        ValueError: If the FNO output does not have the expected scalar-field
            layout.
    """
    model_fn = _get_model_from_state(state)
    params = state["params"]

    # Convert the derivative input to the channels-last grid layout required by
    # the FNO.
    gradient_grid = as_gradient_grid(
        gradient_grid,
        grid_size=grid_size,
    )

    prediction = apply_fno(
        model_fn,
        params,
        jnp.asarray(
            gradient_grid[None, ...],
            dtype=jnp.float32,
        ),
        rng=None,
        training=False,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    # Remove an optional singleton output-channel dimension.
    if prediction.ndim == 4 and prediction.shape[-1] == 1:
        prediction = prediction[..., 0]

    if prediction.shape != (1, grid_size, grid_size):
        raise ValueError(
            "FNO prediction must have shape "
            f"(1, {grid_size}, {grid_size}), but got {prediction.shape}."
        )

    return prediction[0]


def predict_metamodel_gradient_sample(
        state: dict,
        sensor_gradient,
        grid_size: int,
) -> np.ndarray:
    """
    Predict one regular-grid gradient field with Stage-1 DeepONet.

    Args:
        state: Loaded two-stage or three-stage metamodel state.
        sensor_gradient: Sparse sensor gradients with shape (P_sensor, 2).
        grid_size: Number of grid points along each spatial dimension.

    Returns:
        Gradient field with shape (grid_size, grid_size, 2).
    """
    deeponet_state = state["deeponet"]

    deeponet_model = _get_model_from_state(deeponet_state)
    deeponet_params = deeponet_state["params"]

    prediction = deeponet_predict_grad_grid(
        model_fn=deeponet_model,
        params=deeponet_params,
        grad_sensor_batch=jnp.asarray(
            np.asarray(sensor_gradient, dtype=np.float32)[None, ...],
            dtype=jnp.float32,
        ),
        grid_size=grid_size,
        rng=None,
    )

    prediction = np.asarray(prediction, dtype=np.float32)

    if prediction.shape != (1, grid_size, grid_size, 2):
        raise ValueError(
            "Stage-1 DeepONet prediction must have shape "
            f"(1, {grid_size}, {grid_size}, 2), got {prediction.shape}."
        )

    return prediction[0]


def predict_metamodel_sample(
        state: dict,
        sensor_gradient,
        grid_size: int,
) -> np.ndarray:
    """
    Reconstruct one wavefront with a DeepONet-to-FNO metamodel.

    Stage 1 maps sparse sensor gradients to a regular-grid gradient field.
    Stage 2 maps that predicted field to the wavefront.
    """
    predicted_gradient_grid = predict_metamodel_gradient_sample(
        state=state,
        sensor_gradient=sensor_gradient,
        grid_size=grid_size,
    )

    return predict_fno_sample(
        state=state["fno"],
        gradient_grid=predicted_gradient_grid,
        grid_size=grid_size,
    )


def predict_poisson_sample(
        gradient_grid,
        grid_size: int,
        *,
        use_circular_mask: bool = True,
) -> np.ndarray:
    """
    Reconstruct one wavefront with the least-squares Poisson baseline.

    Args:
        gradient_grid: Gradient field for one sample in any layout accepted by
            ``as_gradient_grid``.
        grid_size: Number of spatial points along each regular-grid axis.
        use_circular_mask: Whether to restrict reconstruction to the circular
            aperture mask used by the baseline.

    Returns:
        Float32 reconstructed wavefront with shape:

            ``(grid_size, grid_size)``
    """
    gradient_grid = as_gradient_grid(
        gradient_grid,
        grid_size=grid_size,
    )

    result = poisson_baseline_from_deriv_grid(
        deriv_grid=gradient_grid,
        U_true=None,
        grid_size=grid_size,
        use_circular_mask=use_circular_mask,
    )

    return np.asarray(
        result["U_pred"],
        dtype=np.float32,
    )
