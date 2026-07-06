from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable

from wavefront.evaluation.config import EvalConfig
from wavefront.evaluation.data import take_common_test
from wavefront.evaluation.loaders import (
    CheckpointSelector,
    load_standalone_deeponet_state,
    load_standalone_fno_state,
    load_three_stage_metamodel_state,
    load_two_stage_metamodel_state,
    release_jax_memory,
)
from wavefront.evaluation.operators import (
    evaluate_deeponet_state,
    evaluate_fno_state,
    evaluate_metamodel_state,
    evaluate_poisson_baseline,
)
from wavefront.evaluation.reporting import (
    add_result,
    create_report,
)


def _record_failure(
        report: dict,
        operator_name: str,
        error: Exception,
) -> None:
    """
    Record an operator failure in a serializable benchmark-report entry.

    Args:
        report: Benchmark report dictionary updated in place.
        operator_name: Name of the operator that failed during loading or
            evaluation.
        error: Exception raised by the failed operation.

    Side Effects:
        Adds or replaces an entry in ``report["failures"]`` containing the
        exception type, message, and formatted traceback.

    Notes:
        This helper is used when ``fail_fast=False`` so one unavailable or
        incompatible model does not prevent evaluation of the remaining
        operators.
    """
    report.setdefault("failures", {})[operator_name] = {
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }


def _evaluate_loaded_operator(
        report: dict,
        operator_name: str,
        loader: Callable[[], dict],
        evaluator: Callable[[dict], dict],
        *,
        fail_fast: bool,
) -> None:
    """
    Load, evaluate, record, and release exactly one learned operator.

    The loaded state is always released in the ``finally`` block, including
    when model restoration or evaluation raises an exception. This prevents
    several large JAX model states from remaining in accelerator memory during
    sequential benchmarking.

    Args:
        report: Benchmark report dictionary updated in place.
        operator_name: Name used for the report result or failure entry.
        loader: Zero-argument callable that restores and returns one operator
            state dictionary.
        evaluator: Callable that accepts the loaded state and returns the
            corresponding metrics dictionary.
        fail_fast: Whether to re-raise a loading or evaluation exception
            immediately. When False, the error is recorded and evaluation
            continues with the next operator.

    Side Effects:
        On success, stores metrics in ``report["results"]``. On failure with
        ``fail_fast=False``, stores diagnostic details in
        ``report["failures"]``. In all cases, attempts to release the loaded
        state and clear JAX caches before returning.
    """
    state = None

    try:
        # Restore exactly one model state before evaluating it.
        state = loader()

        # Evaluate the restored operator on the common held-out split.
        result = evaluator(state)

        # Store the completed metric summary under the requested operator name.
        add_result(
            report,
            operator_name,
            result,
        )

    except Exception as error:
        if fail_fast:
            raise

        # Preserve the exception details while allowing remaining operators to
        # be evaluated independently.
        _record_failure(
            report=report,
            operator_name=operator_name,
            error=error,
        )

    finally:
        # Always release Python references and request JAX cache cleanup.
        release_jax_memory(
            state,
            clear_compilation_cache=True,
        )


def run_sequential_benchmark(
        cfg: EvalConfig,
        wavefronts_all,
        sensor_gradients_all,
        grid_gradients_all=None,
        *,
        deeponet_run: str | Path | None = None,
        fno_run: str | Path | None = None,
        two_stage_run: str | Path | None = None,
        three_stage_run: str | Path | None = None,
        checkpoint: CheckpointSelector = "best",
        fail_fast: bool = False,
) -> dict:
    """
    Benchmark available reconstruction operators one at a time.

    Every operator is evaluated on the same held-out test split. The routine
    first evaluates the parameter-free Poisson baseline, then loads each
    requested learned operator individually, evaluates it, and releases its
    state before loading the next one.

    This sequencing is useful on GPU or TPU systems where simultaneously
    retaining DeepONet, FNO, and joint metamodel states may exceed available
    device memory.

    Supported learned operators are:

        - Standalone DeepONet.
        - Standalone FNO.
        - Two-stage DeepONet-to-FNO metamodel.
        - Three-stage jointly fine-tuned DeepONet-to-FNO metamodel.

    Args:
        cfg: Shared benchmark configuration defining the common train/test
            split, input mode, grid resolution, evaluation count, and batch
            size.
        wavefronts_all: Complete wavefront target dataset. The first axis must
            index samples.
        sensor_gradients_all: Complete sparse sensor-gradient dataset with
            sample dimension first.
        grid_gradients_all: Optional complete regular-grid gradient dataset.
            It is required when evaluating operators configured for direct
            regular-grid input mode.
        deeponet_run: Optional standalone DeepONet training-run directory.
        fno_run: Optional standalone FNO training-run directory.
        two_stage_run: Optional two-stage metamodel run directory.
        three_stage_run: Optional three-stage metamodel run directory.
        checkpoint: Checkpoint selector passed to model loaders. Supported
            values are ``"best"`` and ``"last"``.
        fail_fast: Whether to stop immediately when an operator cannot be
            loaded or evaluated. When False, failures are added to
            ``report["failures"]`` and remaining operators continue.

    Returns:
        Benchmark report dictionary containing:

            cfg:
                Serialized evaluation configuration.

            available:
                Availability metadata for each requested operator and input
                source.

            results:
                Completed metric summaries keyed by operator name.

            failures:
                Loading or evaluation errors keyed by operator name.

    Notes:
        Operators that are not requested through their corresponding run-path
        argument are omitted from evaluation. The Poisson baseline is always
        attempted because it requires no learned checkpoint.
    """
    # Select one identical held-out split for every operator comparison.
    U_test, D_sensor_test, D_grid_test = take_common_test(
        wavefronts_all=wavefronts_all,
        sensor_gradients_all=sensor_gradients_all,
        grid_gradients_all=grid_gradients_all,
        cfg=cfg,
    )

    # Initialize the report and document which operators and gradient sources
    # were available at benchmark time.
    report = create_report(cfg)

    report["available"] = {
        "poisson": True,
        "deeponet": deeponet_run is not None,
        "fno": fno_run is not None,
        "two_stage": two_stage_run is not None,
        "three_stage": three_stage_run is not None,
        "grid_gradients_available": D_grid_test is not None,
        "checkpoint": checkpoint,
    }

    report["failures"] = {}

    # Evaluate Poisson first because it does not require loading a learned
    # model state or allocating accelerator memory for neural-network weights.
    try:
        poisson_result = evaluate_poisson_baseline(
            U_test=U_test,
            D_sensor_test=D_sensor_test,
            D_grid_test=D_grid_test,
            cfg=cfg,
        )

        add_result(
            report,
            "poisson",
            poisson_result,
        )

    except Exception as error:
        if fail_fast:
            raise

        _record_failure(
            report=report,
            operator_name="poisson",
            error=error,
        )

    # Load and evaluate each requested learned operator sequentially.
    if deeponet_run is not None:
        _evaluate_loaded_operator(
            report=report,
            operator_name="deeponet",
            loader=lambda: load_standalone_deeponet_state(
                deeponet_run,
                checkpoint=checkpoint,
            ),
            evaluator=lambda state: evaluate_deeponet_state(
                state=state,
                U_test=U_test,
                D_sensor_test=D_sensor_test,
                D_grid_test=D_grid_test,
                cfg=cfg,
            ),
            fail_fast=fail_fast,
        )

    if fno_run is not None:
        _evaluate_loaded_operator(
            report=report,
            operator_name="fno",
            loader=lambda: load_standalone_fno_state(
                fno_run,
                checkpoint=checkpoint,
            ),
            evaluator=lambda state: evaluate_fno_state(
                state=state,
                U_test=U_test,
                D_sensor_test=D_sensor_test,
                D_grid_test=D_grid_test,
                cfg=cfg,
            ),
            fail_fast=fail_fast,
        )

    if two_stage_run is not None:
        _evaluate_loaded_operator(
            report=report,
            operator_name="two_stage",
            loader=lambda: load_two_stage_metamodel_state(
                two_stage_run,
                checkpoint=checkpoint,
            ),
            evaluator=lambda state: evaluate_metamodel_state(
                state=state,
                U_test=U_test,
                D_sensor_test=D_sensor_test,
                cfg=cfg,
            ),
            fail_fast=fail_fast,
        )

    if three_stage_run is not None:
        _evaluate_loaded_operator(
            report=report,
            operator_name="three_stage",
            loader=lambda: load_three_stage_metamodel_state(
                three_stage_run,
                checkpoint=checkpoint,
            ),
            evaluator=lambda state: evaluate_metamodel_state(
                state=state,
                U_test=U_test,
                D_sensor_test=D_sensor_test,
                cfg=cfg,
            ),
            fail_fast=fail_fast,
        )

    return report
