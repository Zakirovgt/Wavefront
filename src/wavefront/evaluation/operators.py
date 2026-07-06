from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from wavefront.baselines.poisson import (
    poisson_baseline_from_deriv_grid,
)
from wavefront.evaluation.config import EvalConfig
from wavefront.evaluation.data import get_grid_inputs
from wavefront.evaluation.deeponet import get_error
from wavefront.evaluation.timing import (
    benchmark_inference_deeponet,
    benchmark_inference_fno,
    benchmark_inference_poisson,
)
from wavefront.metamodel.gradmap import (
    deeponet_predict_grad_grid,
)
from wavefront.training.fno import apply_fno
from wavefront.training.fno_losses import fno_rel_l2


def _get_model_from_state(
        state: dict,
):
    """
    Return the model module stored under a supported training-state key.

    Args:
        state: Model-state dictionary. Supported model keys are checked in the
            following order:

            - ``"model_eval_fn"``
            - ``"model_fn"``
            - ``"model"``

    Returns:
        The first non-None model object found in the state dictionary.

    Raises:
        KeyError: If none of the supported model keys are present or contain a
            usable model object.

    Notes:
        Different training routines expose their Flax modules under slightly
        different key names. This helper provides one consistent lookup path
        for the evaluation functions.
    """
    for key in ("model_eval_fn", "model_fn", "model"):
        model = state.get(key)

        if model is not None:
            return model

    raise KeyError(
        "State must contain one of: "
        "'model_eval_fn', 'model_fn', or 'model'."
    )


def _as_wavefront_grid(
        wavefronts,
        grid_size: int,
) -> np.ndarray:
    """
    Convert flattened or grid-shaped wavefronts to regular-grid form.

    Args:
        wavefronts: Wavefront array with shape ``(N, p)`` or
            ``(N, grid_size, grid_size)``, where
            ``p = grid_size * grid_size``.
        grid_size: Number of points along each spatial axis.

    Returns:
        Float32 wavefront array with shape:

            ``(N, grid_size, grid_size)``

    Raises:
        ValueError: If the input does not match either supported layout.
    """
    wavefronts = np.asarray(
        wavefronts,
        dtype=np.float32,
    )

    # Restore spatial structure when wavefronts are stored as flattened grids.
    if wavefronts.ndim == 2:
        expected_p = grid_size * grid_size

        if wavefronts.shape[1] != expected_p:
            raise ValueError(
                f"Flattened wavefronts must contain {expected_p} values, "
                f"but got shape {wavefronts.shape}."
            )

        return wavefronts.reshape(
            wavefronts.shape[0],
            grid_size,
            grid_size,
        )

    # Preserve an already grid-shaped wavefront representation.
    if wavefronts.ndim == 3 and wavefronts.shape[1:] == (
            grid_size,
            grid_size,
    ):
        return wavefronts

    raise ValueError(
        "Wavefronts must have shape "
        f"(N, {grid_size * grid_size}) or "
        f"(N, {grid_size}, {grid_size}), "
        f"but got {wavefronts.shape}."
    )


def _relative_l2_per_sample(
        prediction: np.ndarray,
        target: np.ndarray,
) -> np.ndarray:
    """
    Compute one relative L2 reconstruction error for each full wavefront.

    Args:
        prediction: Predicted wavefronts with shape ``(N, H, W)``.
        target: Reference wavefronts with shape ``(N, H, W)``.

    Returns:
        One-dimensional array with shape ``(N,)`` containing:

            ||prediction - target||_2 / (||target||_2 + 1e-8)
    """
    difference = prediction - target

    difference_norm = np.sqrt(
        np.sum(difference ** 2, axis=(1, 2))
    )

    target_norm = np.sqrt(
        np.sum(target ** 2, axis=(1, 2))
    ) + 1e-8

    return difference_norm / target_norm


def _metric_summary(
        errors: np.ndarray,
        benchmark: dict,
) -> dict:
    """
    Create a consistent metrics dictionary for one reconstruction operator.

    Args:
        errors: Per-sample relative L2 errors.
        benchmark: Inference timing dictionary for the same operator.

    Returns:
        Dictionary containing mean and median relative L2 error, evaluated
        sample count, and the supplied benchmark results.
    """
    errors = np.asarray(
        errors,
        dtype=np.float64,
    ).reshape(-1)

    return {
        "mean_rel_l2": float(np.mean(errors)),
        "median_rel_l2": float(np.median(errors)),
        "n": int(errors.size),
        "bench": benchmark,
    }


def evaluate_deeponet_state(
        state: dict,
        U_test,
        D_sensor_test,
        D_grid_test,
        cfg: EvalConfig,
) -> dict:
    """
    Evaluate a standalone DeepONet model on the shared held-out test split.

    The branch input source is selected from the model configuration:

        - Sensor mode:
            Uses sparse sensor gradients directly.

        - Regular-grid mode:
            Uses regular-grid gradients as the DeepONet branch input.

    Args:
        state: DeepONet training-state dictionary containing model and
            parameter entries.
        U_test: Held-out wavefront targets with shape ``(N, p)`` or
            ``(N, grid_size, grid_size)``.
        D_sensor_test: Held-out sparse sensor gradients with shape
            ``(N, P_sensor, 2)``.
        D_grid_test: Optional held-out regular-grid derivatives with shape
            ``(N, p, 2)`` or ``(N, grid_size, grid_size, 2)``.
        cfg: Shared evaluation configuration.

    Returns:
        Dictionary containing mean and median relative L2 error, evaluated
        sample count, and batched DeepONet inference timings.

    Raises:
        ValueError: If no samples are available or a regular-grid DeepONet is
            evaluated without regular-grid derivative data.
    """
    model_fn = _get_model_from_state(state)
    params = state["params"]

    n = min(int(cfg.n_eval), len(U_test))

    if n < 1:
        raise ValueError(
            "No samples are available for DeepONet evaluation."
        )

    args = state.get("args")
    data_mode = getattr(args, "data_mode", cfg.mode)

    # A DeepONet trained with regular-grid branch inputs must receive regular
    # grid gradients at evaluation time.
    if data_mode in {"regular_grid", "grid"}:
        if D_grid_test is None:
            raise ValueError(
                "DeepONet trained in regular_grid mode requires D_grid_test."
            )

        branch_gradients = np.asarray(
            D_grid_test[:n],
            dtype=np.float32,
        )

        # Flatten grid-shaped branch gradients into the expected
        # (N, P_branch, 2) layout.
        if branch_gradients.ndim == 4:
            branch_gradients = branch_gradients.reshape(
                n,
                int(cfg.grid_size) ** 2,
                2,
            )
    else:
        # Sensor-mode DeepONet receives sparse gradients directly.
        branch_gradients = np.asarray(
            D_sensor_test[:n],
            dtype=np.float32,
        )

    # DeepONet error evaluation expects flattened wavefront targets.
    wavefronts = np.asarray(
        U_test[:n],
        dtype=np.float32,
    ).reshape(n, -1)

    errors = get_error(
        model_fn=model_fn,
        params=params,
        grad_sensor_all=jnp.asarray(branch_gradients),
        wavefront_true_all=jnp.asarray(wavefronts),
        idx=jnp.arange(n),
        p_err=int(cfg.deeponet_p_test),
        return_data=False,
    )

    benchmark = benchmark_inference_deeponet(
        model_fn=model_fn,
        params=params,
        grad_sensor_test=branch_gradients,
        p_test=int(cfg.deeponet_p_test),
        n=n,
        batch_size=int(cfg.bench_batch),
    )

    return _metric_summary(errors, benchmark)


def evaluate_fno_state(
        state: dict,
        U_test,
        D_sensor_test,
        D_grid_test,
        cfg: EvalConfig,
) -> dict:
    """
    Evaluate a standalone FNO model on the shared held-out test split.

    Sparse sensor gradients are interpolated to the regular grid when
    ``cfg.mode == "sensor"``. In regular-grid mode, precomputed gradients are
    used directly.

    Args:
        state: FNO training-state dictionary containing model and parameter
            entries.
        U_test: Held-out wavefront targets.
        D_sensor_test: Held-out sparse sensor gradients.
        D_grid_test: Optional held-out regular-grid derivatives.
        cfg: Shared evaluation configuration.

    Returns:
        Dictionary containing error statistics and batched FNO inference
        timing results.

    Raises:
        ValueError: If no held-out samples are available.
    """
    model_fn = _get_model_from_state(state)
    params = state["params"]

    n = min(int(cfg.n_eval), len(U_test))

    if n < 1:
        raise ValueError(
            "No samples are available for FNO evaluation."
        )

    # Build regular-grid FNO inputs from sparse or precomputed gradients.
    X = get_grid_inputs(
        sensor_gradients=D_sensor_test[:n],
        grid_gradients=(
            D_grid_test[:n]
            if D_grid_test is not None
            else None
        ),
        cfg=cfg,
    )

    # Normalize wavefront target layout to (N, H, W).
    Y = _as_wavefront_grid(
        U_test[:n],
        grid_size=int(cfg.grid_size),
    )

    # Compute one deterministic relative L2 error per held-out wavefront.
    errors = jax.vmap(
        lambda x, y: fno_rel_l2(
            model_fn,
            params,
            x,
            y,
        )
    )(
        jnp.asarray(X),
        jnp.asarray(Y),
    )

    benchmark = benchmark_inference_fno(
        model_fn=model_fn,
        params=params,
        X_test=X,
        n=n,
        batch_size=int(cfg.bench_batch),
    )

    return _metric_summary(errors, benchmark)


def evaluate_metamodel_state(
        state: dict,
        U_test,
        D_sensor_test,
        cfg: EvalConfig,
) -> dict:
    """
    Evaluate a two-model DeepONet-to-FNO metamodel on the common test split.

    Expected state structure:

        {
            "deeponet": {
                "model_fn": ...,
                "params": ...,
            },
            "fno": {
                "model_fn": ...,
                "params": ...,
            },
        }

    The evaluated reconstruction pipeline is:

        Sparse sensor gradients
            -> DeepONet gradient-map predictor
            -> regular-grid gradient field
            -> FNO
            -> reconstructed wavefront

    Args:
        state: Nested DeepONet and FNO state dictionaries.
        U_test: Held-out wavefront targets.
        D_sensor_test: Held-out sparse sensor gradients.
        cfg: Shared evaluation configuration.

    Returns:
        Dictionary containing relative L2 metrics and end-to-end inference
        timing results.

    Raises:
        ValueError: If the common test split contains no usable samples.
    """
    deeponet_state = state["deeponet"]
    fno_state = state["fno"]

    deeponet_model = _get_model_from_state(deeponet_state)
    deeponet_params = deeponet_state["params"]

    fno_model = _get_model_from_state(fno_state)
    fno_params = fno_state["params"]

    n = min(int(cfg.n_eval), len(U_test))

    if n < 1:
        raise ValueError(
            "No samples are available for metamodel evaluation."
        )

    sensor_gradients = np.asarray(
        D_sensor_test[:n],
        dtype=np.float32,
    )

    wavefronts = _as_wavefront_grid(
        U_test[:n],
        grid_size=int(cfg.grid_size),
    )

    errors = []
    batch_size = int(cfg.bench_batch)

    # Evaluate the complete DeepONet-to-FNO path in manageable batches.
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)

        # Stage 1: predict regular-grid gradient maps from sensor gradients.
        predicted_gradients = deeponet_predict_grad_grid(
            model_fn=deeponet_model,
            params=deeponet_params,
            grad_sensor_batch=jnp.asarray(
                sensor_gradients[start:end],
                dtype=jnp.float32,
            ),
            grid_size=int(cfg.grid_size),
            rng=None,
        )

        # Stage 2: reconstruct wavefronts from predicted gradient maps.
        prediction = apply_fno(
            fno_model,
            fno_params,
            predicted_gradients,
            rng=None,
            training=False,
        )

        prediction = np.asarray(
            prediction,
            dtype=np.float32,
        )

        # Remove an optional singleton scalar output channel.
        if prediction.ndim == 4 and prediction.shape[-1] == 1:
            prediction = prediction[..., 0]

        # Support flattened FNO output fields.
        if prediction.ndim == 2:
            prediction = prediction.reshape(
                prediction.shape[0],
                int(cfg.grid_size),
                int(cfg.grid_size),
            )

        errors.append(
            _relative_l2_per_sample(
                prediction,
                wavefronts[start:end],
            )
        )

    errors = np.concatenate(errors)

    # Warm up both stages before timing to avoid measuring JAX compilation.
    warmup_size = min(batch_size, n)

    warmup_gradients = deeponet_predict_grad_grid(
        model_fn=deeponet_model,
        params=deeponet_params,
        grad_sensor_batch=jnp.asarray(
            sensor_gradients[:warmup_size],
            dtype=jnp.float32,
        ),
        grid_size=int(cfg.grid_size),
        rng=None,
    )

    apply_fno(
        fno_model,
        fno_params,
        warmup_gradients,
        rng=None,
        training=False,
    ).block_until_ready()

    # Benchmark the full inference chain, including DeepONet grid-gradient
    # prediction and FNO wavefront reconstruction.
    start_time = time.perf_counter()

    processed = 0

    while processed < n:
        end = min(processed + batch_size, n)

        predicted_gradients = deeponet_predict_grad_grid(
            model_fn=deeponet_model,
            params=deeponet_params,
            grad_sensor_batch=jnp.asarray(
                sensor_gradients[processed:end],
                dtype=jnp.float32,
            ),
            grid_size=int(cfg.grid_size),
            rng=None,
        )

        apply_fno(
            fno_model,
            fno_params,
            predicted_gradients,
            rng=None,
            training=False,
        ).block_until_ready()

        processed = end

    total_seconds = float(time.perf_counter() - start_time)

    benchmark = {
        "n": n,
        "total_s": total_seconds,
        "ms_per_function": (
                1000.0 * total_seconds / max(1, n)
        ),
        "functions_per_s": (
                float(n) / max(1e-12, total_seconds)
        ),
    }

    return _metric_summary(errors, benchmark)


def evaluate_poisson_baseline(
        U_test,
        D_sensor_test,
        D_grid_test,
        cfg: EvalConfig,
) -> dict:
    """
    Evaluate the least-squares Poisson reconstruction baseline.

    Sparse sensor gradients are interpolated to the regular grid when required
    by the configured mode. Each gradient field is independently integrated
    into a wavefront estimate using the Poisson baseline.

    Args:
        U_test: Held-out wavefront targets.
        D_sensor_test: Held-out sparse sensor gradients.
        D_grid_test: Optional held-out regular-grid gradients.
        cfg: Shared evaluation configuration.

    Returns:
        Dictionary containing relative L2 metrics and sequential Poisson
        reconstruction timing results.

    Raises:
        ValueError: If no held-out samples are available.
    """
    n = min(int(cfg.n_eval), len(U_test))

    if n < 1:
        raise ValueError(
            "No samples are available for Poisson evaluation."
        )

    # Prepare regular-grid derivative inputs for Poisson integration.
    grid_gradients = get_grid_inputs(
        sensor_gradients=D_sensor_test[:n],
        grid_gradients=(
            D_grid_test[:n]
            if D_grid_test is not None
            else None
        ),
        cfg=cfg,
    )

    wavefronts = _as_wavefront_grid(
        U_test[:n],
        grid_size=int(cfg.grid_size),
    )

    errors = []

    # Reconstruct each wavefront independently from its gradient field.
    for index in range(n):
        result = poisson_baseline_from_deriv_grid(
            deriv_grid=grid_gradients[index],
            U_true=wavefronts[index],
            grid_size=int(cfg.grid_size),
            use_circular_mask=True,
        )

        errors.append(float(result["err"]))

    benchmark = benchmark_inference_poisson(
        deriv_grid_test=grid_gradients,
        n=n,
        grid_size=int(cfg.grid_size),
        use_circular_mask=True,
    )

    return _metric_summary(errors, benchmark)
