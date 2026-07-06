from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

import jax.numpy as jnp
import numpy as np

from wavefront.evaluation.benchmark_visualization import (
    save_benchmark_comparison,
    save_stage1_gradient_comparison,
)
from wavefront.evaluation.config import EvalConfig
from wavefront.evaluation.data import get_grid_inputs
from wavefront.evaluation.loaders import (
    CheckpointSelector,
    load_standalone_deeponet_state,
    load_standalone_fno_state,
    load_three_stage_metamodel_state,
    load_two_stage_metamodel_state,
    release_jax_memory,
)
from wavefront.evaluation.sample_predictions import (
    as_gradient_grid,
    predict_deeponet_sample,
    predict_fno_sample,
    predict_metamodel_sample,
    predict_poisson_sample,
    select_deeponet_input,
)
from wavefront.metamodel.gradmap import (
    deeponet_predict_grad_grid,
)

Stage1Source = Literal[
    "auto",
    "two_stage",
    "three_stage",
    "none",
]


def _record_failure(
        result: dict[str, Any],
        operator_name: str,
        error: Exception,
) -> None:
    """
    Record an operator-level visualization failure without discarding results.

    Args:
        result: Visualization result dictionary updated in place.
        operator_name: Name of the operator that failed during loading or
            prediction.
        error: Exception raised by the failed operation.

    Side Effects:
        Adds or replaces an entry in ``result["failures"]`` containing the
        exception type and message.

    Notes:
        Tracebacks are intentionally omitted here to keep the visualization
        manifest compact. The original exception can still be re-raised by
        setting ``fail_fast=True``.
    """
    result.setdefault("failures", {})[operator_name] = {
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _resolve_stage1_source(
        requested: Stage1Source,
        *,
        two_stage_run: str | Path | None,
        three_stage_run: str | Path | None,
) -> str | None:
    """
    Select the metamodel used for the optional Stage-1 gradient diagnostic.

    Selection behavior:

        - ``"none"``:
            Disable the Stage-1 gradient diagnostic.

        - ``"two_stage"``:
            Use the two-stage metamodel when its run directory is available.

        - ``"three_stage"``:
            Use the three-stage metamodel when its run directory is available.

        - ``"auto"``:
            Prefer the three-stage metamodel; otherwise use the two-stage
            metamodel when available.

    Args:
        requested: Requested diagnostic source policy.
        two_stage_run: Optional two-stage pipeline output directory.
        three_stage_run: Optional three-stage pipeline output directory.

    Returns:
        ``"two_stage"``, ``"three_stage"``, or None when no suitable
        diagnostic source is available.

    Raises:
        ValueError: If ``requested`` is not one of the supported values.
    """
    if requested == "none":
        return None

    if requested == "two_stage":
        return (
            "two_stage"
            if two_stage_run is not None
            else None
        )

    if requested == "three_stage":
        return (
            "three_stage"
            if three_stage_run is not None
            else None
        )

    if requested != "auto":
        raise ValueError(
            f"Unknown stage1_source={requested!r}. "
            "Expected 'auto', 'two_stage', 'three_stage', or 'none'."
        )

    # Prefer the final three-stage pipeline when both metamodel variants are
    # available, because it includes standalone FNO pretraining before joint
    # fine-tuning.
    if three_stage_run is not None:
        return "three_stage"

    if two_stage_run is not None:
        return "two_stage"

    return None


def _validate_sample_indices(
        sample_indices,
        n_available: int,
) -> list[int]:
    """
    Validate, convert, sort, and deduplicate held-out sample indices.

    Args:
        sample_indices: Requested sample indices relative to the held-out test
            split.
        n_available: Number of samples in the held-out split.

    Returns:
        Sorted list of unique valid integer indices.

    Raises:
        ValueError: If no sample indices are supplied.
        IndexError: If any requested index lies outside the held-out split.
    """
    if not sample_indices:
        raise ValueError(
            "At least one visualization sample index is required."
        )

    indices = sorted(
        {
            int(index)
            for index in sample_indices
        }
    )

    for index in indices:
        if index < 0 or index >= n_available:
            raise IndexError(
                f"Sample index {index} is outside the held-out test split "
                f"with {n_available} samples."
            )

    return indices


def _predict_metamodel_gradient_sample(
        state: dict,
        sensor_gradient,
        grid_size: int,
) -> np.ndarray:
    """
    Predict a regular-grid gradient field using a metamodel's Stage-1 DeepONet.

    Args:
        state: Nested metamodel state containing a ``"deeponet"`` component
            with model and parameter entries.
        sensor_gradient: Sparse sensor-gradient measurements for one sample
            with shape ``(P_sensor, 2)``.
        grid_size: Number of regular-grid points along each spatial axis.

    Returns:
        Float32 predicted gradient field with shape:

            ``(grid_size, grid_size, 2)``

    Notes:
        This helper is used only for diagnostic visualization. Full metamodel
        wavefront reconstruction remains delegated to
        ``predict_metamodel_sample``.
    """
    deeponet_state = state["deeponet"]

    model_fn = (
            deeponet_state.get("model_eval_fn")
            or deeponet_state.get("model_fn")
            or deeponet_state.get("model")
    )

    if model_fn is None:
        raise KeyError(
            "Metamodel DeepONet state must contain one of: "
            "'model_eval_fn', 'model_fn', or 'model'."
        )

    params = deeponet_state["params"]

    predicted_gradient = deeponet_predict_grad_grid(
        model_fn=model_fn,
        params=params,
        grad_sensor_batch=jnp.asarray(
            np.asarray(
                sensor_gradient,
                dtype=np.float32,
            )[None, ...],
            dtype=jnp.float32,
        ),
        grid_size=grid_size,
        rng=None,
    )

    return np.asarray(
        predicted_gradient[0],
        dtype=np.float32,
    )


def _run_loaded_prediction(
        *,
        result: dict[str, Any],
        operator_key: str,
        loader: Callable[[], dict],
        callback: Callable[[dict], None],
        fail_fast: bool,
) -> None:
    """
    Load one model state, generate requested predictions, and release it.

    Args:
        result: Visualization result dictionary updated in place.
        operator_key: Report key identifying the loaded operator.
        loader: Zero-argument callable that returns a loaded state dictionary.
        callback: Function that receives the loaded state and adds predictions
            to the shared result structures.
        fail_fast: Whether to re-raise any loading or prediction exception.

    Side Effects:
        Records an operator failure when ``fail_fast=False``. The loaded JAX
        state is always released before returning.
    """
    state = None

    try:
        state = loader()
        callback(state)

    except Exception as error:
        if fail_fast:
            raise

        _record_failure(
            result=result,
            operator_name=operator_key,
            error=error,
        )

    finally:
        # Keep accelerator memory bounded by retaining only one learned state
        # at a time.
        release_jax_memory(
            state,
            clear_compilation_cache=True,
        )


def run_sequential_sample_visualization(
        *,
        cfg: EvalConfig,
        U_test,
        D_sensor_test,
        D_grid_test,
        output_dir: str | Path,
        sample_indices,
        deeponet_run: str | Path | None = None,
        fno_run: str | Path | None = None,
        two_stage_run: str | Path | None = None,
        three_stage_run: str | Path | None = None,
        checkpoint: CheckpointSelector = "best",
        error_mode: str = "global",
        solution_percentiles: tuple[float, float] = (0.0, 100.0),
        error_percentile: float = 100.0,
        separate_poisson_scale: bool = False,
        save_stage1_gradient: bool = True,
        stage1_source: Stage1Source = "auto",
        dpi: int = 180,
        fail_fast: bool = False,
) -> dict[str, Any]:
    """
    Generate sample-level benchmark figures without retaining learned models.

    Every selected operator reconstructs the same held-out samples. Learned
    model states are loaded, used, and released sequentially to reduce CPU,
    GPU, or TPU memory pressure.

    Sample indices are relative to the supplied held-out test split, rather
    than to the complete synthetic dataset.

    Args:
        cfg: Shared evaluation configuration.
        U_test: Held-out target wavefronts.
        D_sensor_test: Held-out sparse sensor-gradient measurements.
        D_grid_test: Optional held-out regular-grid gradient fields.
        output_dir: Root directory where the ``figures`` directory is created.
        sample_indices: Test-split-relative sample indices to visualize.
        deeponet_run: Optional standalone DeepONet run directory.
        fno_run: Optional standalone FNO run directory.
        two_stage_run: Optional two-stage metamodel run directory.
        three_stage_run: Optional three-stage metamodel run directory.
        checkpoint: Saved checkpoint selector, either ``"best"`` or
            ``"last"``.
        error_mode: Error-map normalization strategy passed to
            ``save_benchmark_comparison``.
        solution_percentiles: Percentile range for the shared solution color
            scale.
        error_percentile: Percentile used to set robust difference and error
            color limits.
        separate_poisson_scale: Whether the Poisson baseline may use a
            separate error scale from learned methods.
        save_stage1_gradient: Whether to save a Stage-1 gradient diagnostic
            when a metamodel and reference grid gradients are available.
        stage1_source: Metamodel selected for the Stage-1 diagnostic.
        dpi: Resolution of saved PNG figures.
        fail_fast: Whether to raise immediately when an operator fails.

    Returns:
        Dictionary containing selected indices, saved figure metadata, optional
        Stage-1 diagnostic metadata, failure records, and the resolved
        diagnostic source.

    Notes:
        Poisson predictions are generated first because they do not require a
        learned state. Learned operators are evaluated one at a time.
    """
    output_dir = Path(output_dir)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    U_test = np.asarray(
        U_test,
        dtype=np.float32,
    )

    D_sensor_test = np.asarray(
        D_sensor_test,
        dtype=np.float32,
    )

    if D_grid_test is not None:
        D_grid_test = np.asarray(
            D_grid_test,
            dtype=np.float32,
        )

    # Validate indices against the already selected held-out split.
    indices = _validate_sample_indices(
        sample_indices,
        n_available=len(U_test),
    )

    grid_size = int(cfg.grid_size)

    # FNO and Poisson operate on regular-grid derivatives. In sensor mode,
    # those derivatives are built by interpolating sparse sensor measurements.
    selected_sensor_gradients = D_sensor_test[indices]

    selected_grid_gradients = (
        D_grid_test[indices]
        if D_grid_test is not None
        else None
    )

    operator_grid_inputs = get_grid_inputs(
        sensor_gradients=selected_sensor_gradients,
        grid_gradients=selected_grid_gradients,
        cfg=cfg,
    )

    # Store method predictions independently for every requested sample.
    predictions: dict[int, dict[str, np.ndarray]] = {
        index: {}
        for index in indices
    }

    # Store Stage-1 DeepONet gradient maps only when they are requested for a
    # diagnostic comparison.
    stage1_predictions: dict[int, np.ndarray] = {}

    diagnostic_source = _resolve_stage1_source(
        stage1_source,
        two_stage_run=two_stage_run,
        three_stage_run=three_stage_run,
    )

    result: dict[str, Any] = {
        "sample_indices": indices,
        "figures": {},
        "failures": {},
        "stage1_source": diagnostic_source,
    }

    # Poisson requires no checkpoint, so generate its predictions first.
    try:
        for position, index in enumerate(indices):
            predictions[index]["Poisson"] = predict_poisson_sample(
                gradient_grid=operator_grid_inputs[position],
                grid_size=grid_size,
                use_circular_mask=True,
            )

    except Exception as error:
        if fail_fast:
            raise

        _record_failure(
            result=result,
            operator_name="poisson",
            error=error,
        )

    if deeponet_run is not None:

        def add_deeponet_predictions(
                state: dict,
        ) -> None:
            """Generate standalone DeepONet predictions for all samples."""
            for index in indices:
                grid_gradient = (
                    D_grid_test[index]
                    if D_grid_test is not None
                    else None
                )

                # Choose sparse or regular-grid branch data based on the
                # loaded DeepONet training configuration.
                branch_input = select_deeponet_input(
                    state=state,
                    sensor_gradient=D_sensor_test[index],
                    grid_gradient=grid_gradient,
                )

                predictions[index]["DeepONet"] = (
                    predict_deeponet_sample(
                        state=state,
                        gradient_input=branch_input,
                        grid_size=grid_size,
                    )
                )

        _run_loaded_prediction(
            result=result,
            operator_key="deeponet",
            loader=lambda: load_standalone_deeponet_state(
                deeponet_run,
                checkpoint=checkpoint,
            ),
            callback=add_deeponet_predictions,
            fail_fast=fail_fast,
        )

    if fno_run is not None:

        def add_fno_predictions(
                state: dict,
        ) -> None:
            """Generate standalone FNO predictions for all samples."""
            for position, index in enumerate(indices):
                predictions[index]["FNO"] = predict_fno_sample(
                    state=state,
                    gradient_grid=operator_grid_inputs[position],
                    grid_size=grid_size,
                )

        _run_loaded_prediction(
            result=result,
            operator_key="fno",
            loader=lambda: load_standalone_fno_state(
                fno_run,
                checkpoint=checkpoint,
            ),
            callback=add_fno_predictions,
            fail_fast=fail_fast,
        )

    if two_stage_run is not None:

        def add_two_stage_predictions(
                state: dict,
        ) -> None:
            """
            Generate two-stage wavefront predictions and optional Stage-1 maps.
            """
            for index in indices:
                if diagnostic_source == "two_stage":
                    # Reuse the Stage-1 gradient prediction both for the
                    # diagnostic figure and Stage-2 FNO reconstruction.
                    predicted_gradient = (
                        _predict_metamodel_gradient_sample(
                            state=state,
                            sensor_gradient=D_sensor_test[index],
                            grid_size=grid_size,
                        )
                    )

                    stage1_predictions[index] = predicted_gradient

                    predictions[index]["Two-stage"] = (
                        predict_fno_sample(
                            state=state["fno"],
                            gradient_grid=predicted_gradient,
                            grid_size=grid_size,
                        )
                    )
                else:
                    predictions[index]["Two-stage"] = (
                        predict_metamodel_sample(
                            state=state,
                            sensor_gradient=D_sensor_test[index],
                            grid_size=grid_size,
                        )
                    )

        _run_loaded_prediction(
            result=result,
            operator_key="two_stage",
            loader=lambda: load_two_stage_metamodel_state(
                two_stage_run,
                checkpoint=checkpoint,
            ),
            callback=add_two_stage_predictions,
            fail_fast=fail_fast,
        )

    if three_stage_run is not None:

        def add_three_stage_predictions(
                state: dict,
        ) -> None:
            """
            Generate three-stage wavefront predictions and optional Stage-1 maps.
            """
            for index in indices:
                if diagnostic_source == "three_stage":
                    # Reuse the Stage-1 output for both visualization and the
                    # downstream FNO wavefront reconstruction.
                    predicted_gradient = (
                        _predict_metamodel_gradient_sample(
                            state=state,
                            sensor_gradient=D_sensor_test[index],
                            grid_size=grid_size,
                        )
                    )

                    stage1_predictions[index] = predicted_gradient

                    predictions[index]["Three-stage"] = (
                        predict_fno_sample(
                            state=state["fno"],
                            gradient_grid=predicted_gradient,
                            grid_size=grid_size,
                        )
                    )
                else:
                    predictions[index]["Three-stage"] = (
                        predict_metamodel_sample(
                            state=state,
                            sensor_gradient=D_sensor_test[index],
                            grid_size=grid_size,
                        )
                    )

        _run_loaded_prediction(
            result=result,
            operator_key="three_stage",
            loader=lambda: load_three_stage_metamodel_state(
                three_stage_run,
                checkpoint=checkpoint,
            ),
            callback=add_three_stage_predictions,
            fail_fast=fail_fast,
        )

    # Save complete wavefront comparisons and, when available, Stage-1
    # gradient diagnostics for every requested held-out sample.
    for position, index in enumerate(indices):
        if not predictions[index]:
            result["figures"][str(index)] = {
                "error": (
                    "No predictions were created for this sample."
                )
            }
            continue

        comparison_path = (
                figures_dir
                / f"sample_{index:04d}_comparison.png"
        )

        comparison = save_benchmark_comparison(
            sample_index=index,
            wavefront_true=U_test[index],
            predictions=predictions[index],
            output_path=comparison_path,
            grid_size=grid_size,
            error_mode=error_mode,
            solution_percentiles=solution_percentiles,
            error_percentile=error_percentile,
            separate_poisson_scale=separate_poisson_scale,
            dpi=int(dpi),
        )

        entry: dict[str, Any] = {
            "comparison": comparison,
        }

        # A Stage-1 diagnostic requires predicted DeepONet grid gradients and
        # clean reference grid gradients for the same sample.
        if (
                save_stage1_gradient
                and index in stage1_predictions
                and D_grid_test is not None
        ):
            gradient_path = (
                    figures_dir
                    / f"sample_{index:04d}_stage1_gradients.png"
            )

            entry["stage1_gradient"] = (
                save_stage1_gradient_comparison(
                    sample_index=index,
                    reference_gradient=as_gradient_grid(
                        D_grid_test[index],
                        grid_size=grid_size,
                    ),
                    interpolated_gradient=operator_grid_inputs[position],
                    deeponet_gradient=stage1_predictions[index],
                    output_path=gradient_path,
                    grid_size=grid_size,
                    reference_label="Reference gradient",
                    dpi=int(dpi),
                )
            )

        result["figures"][str(index)] = entry

    return result
