from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import jax
import pandas as pd

from wavefront.inference.artifacts import load_artifacts
from wavefront.inference.csv import CSVColumns, to_deriv_array
from wavefront.inference.deeponet import (
    predict_batch_on_grid,
    save_predictions,
)
from wavefront.models.factory import setup_deeponet


def build_eval_model_from_config(
        cfg: dict,
        key=None,
):
    """
    Build an inference-ready DeepONet model from a saved args_inference.json
    configuration dictionary.

    Args:
        cfg: Model configuration dictionary loaded from args_inference.json.
        key: Optional JAX PRNG key used to initialize a model with the same
            parameter structure as the saved checkpoint. When None, a key is
            created from the saved seed value, or zero if no seed is present.

    Returns:
        A tuple containing:

            args:
                Namespace built from the saved configuration and updated with
                any required compatibility defaults.

            model_fn:
                Flax model module object suitable for use with apply_net and
                other helpers that call model.apply(...).

    Notes:
        This function initializes the model only to reconstruct its parameter
        structure. The returned initialized parameters are discarded because
        inference uses the parameter pytree restored from the saved artifacts.

        For the current ModernDeepONet implementation, do not attempt to
        access legacy attributes such as model.branch_layers or
        model.trunk_layers.

        The returned value must be the Flax module object itself rather than
        a jitted model.apply function, because downstream helpers invoke
        model.apply(...) internally.
    """
    if key is None:
        key = jax.random.PRNGKey(cfg.get("seed", 0))

    # Recreate the training configuration as an attribute-style namespace.
    args = Namespace(**cfg)

    # Support older saved configurations that used p_sensors instead of
    # n_sensors.
    if not hasattr(args, "n_sensors") and hasattr(args, "p_sensors"):
        args.n_sensors = args.p_sensors

    # Fill in defaults required by the current model setup when they are
    # absent from a legacy configuration file.
    if not hasattr(args, "branch_input_features"):
        args.branch_input_features = 2

    if not hasattr(args, "trunk_input_features"):
        args.trunk_input_features = 2

    # Initialize the model architecture. The returned parameter tree is not
    # used because trained parameters are restored separately from artifacts.
    args, model, model_fn, _ = setup_deeponet(args, key)

    return args, model_fn


def load_predictor(
        run_dir: str | Path,
):
    """
    Load a trained DeepONet predictor from a training run directory.

    Args:
        run_dir: Path to the training run directory that contains an
            ``artifacts`` subdirectory.

    Returns:
        Dictionary containing:

            args:
                Inference configuration namespace.

            model_fn:
                Reconstructed Flax DeepONet model module.

            params:
                Restored trained parameter pytree.

            branch_sensor_coords:
                Optional normalized sensor-coordinate array saved during
                training. It is None when no sensor layout was stored.

    Notes:
        The function expects the following path layout:

            run_dir/
                artifacts/
                    args_inference.json
                    params/
                    branch_sensor_coords.npy  # optional
    """
    run_dir = Path(run_dir).expanduser().resolve()

    # Load saved configuration, trained parameters, and optional sensor layout.
    artifacts = load_artifacts(run_dir / "artifacts")

    # Rebuild the model architecture to match the saved parameter tree.
    args, model_fn = build_eval_model_from_config(
        artifacts.model_config
    )

    return {
        "args": args,
        "model_fn": model_fn,
        "params": artifacts.params,
        "branch_sensor_coords": artifacts.branch_sensor_coords,
    }


def predict_from_dataframes(
        predictor,
        meas_df: pd.DataFrame,
        ref_df: pd.DataFrame | None = None,
        align_by: str = "row",
        flag_policy: str = "ignore",
        output_dir: str | Path | None = None,
        prefix: str = "sample",
):
    """
    Predict a wavefront from measurement and optional reference DataFrames.

    The measurement DataFrame is converted into a sensor-gradient array,
    validated against the sensor count expected by the trained DeepONet, and
    evaluated on the model's regular output grid.

    Args:
        predictor: Dictionary returned by load_predictor.
        meas_df: Measurement DataFrame containing sensor slope columns.
        ref_df: Optional reference DataFrame defining the expected sensor
            ordering when alignment by sensor ID or coordinates is required.
        align_by: Sensor alignment strategy passed to to_deriv_array:
            - "row": Preserve measurement row order.
            - "N": Align using sensor identifiers.
            - "XY": evaluated on the model's regular output grid.

    Args:
        predictor: Dictionary returned by load_predictor.
        meas_df: Measurement DataFrame containing sensor slope columns.
        ref_df: Optional reference DataFrame defining the expected sensor
            ordering when alignment by Align using rounded x/y coordinates.
            - "auto": Prefer sensor IDs and otherwise use coordinates.
        flag_policy: Policy for flagged sensor measurements, passed to
            to_deriv_array.
        output_dir: Optional output directory for saved prediction files.
            When None, predictions are returned without writing files.
        prefix: Filename prefix used when saving output artifacts.

    Returns:
        Dictionary returned by predict_batch_on_grid, containing flattened and
        grid-shaped wavefront predictions together with evaluation coordinates.

    Raises:
        ValueError: If the extracted gradient array does not have the sensor
            count expected by the trained DeepONet.

    Notes:
        A DeepONet trained with a fixed branch input dimension cannot accept a
        different number of sensors at inference time. For unlabeled CSV files,
        the typical choice is:

            align_by="row"
            flag_policy="ignore"
    """
    # Convert measurement-table slopes into the standard shape:
    # (P_sensor, 2), where columns represent [dU/dx, dU/dy].
    grad_sensor_one = to_deriv_array(
        meas=meas_df,
        ref=ref_df,
        align_by=align_by,
        flag_policy=flag_policy,
        cols=CSVColumns(),
    )

    # Read the expected sensor count, supporting either modern n_sensors or
    # legacy p_sensors configuration names.
    expected_p = int(
        getattr(
            predictor["args"],
            "n_sensors",
            predictor["args"].p_sensors,
        )
    )

    # The number of sensors must match the architecture used during training.
    if grad_sensor_one.shape != (expected_p, 2):
        raise ValueError(
            "The model expects grad_sensor_one with shape "
            f"({expected_p}, 2), but the CSV produced "
            f"{grad_sensor_one.shape}. A DeepONet cannot change its number "
            "of sensors at inference time. Check align_by, reference_csv, "
            "and flag_policy. For unlabeled CSV files, the usual settings "
            "are align_by='row' and flag_policy='ignore'."
        )

    # Add a batch dimension and reconstruct the wavefront on the regular grid.
    pred = predict_batch_on_grid(
        model_fn=predictor["model_fn"],
        params=predictor["params"],
        grad_sensor_batch=grad_sensor_one[None, ...],
        p_test=int(predictor["args"].p_test),
    )

    # Optionally save NumPy arrays, CSV files, and rendered images.
    if output_dir is not None:
        save_predictions(
            pred,
            output_dir=output_dir,
            prefix=prefix,
        )

    return pred


def predict_from_csv(
        predictor,
        measurement_csv: str | Path,
        reference_csv: str | Path | None = None,
        align_by: str = "row",
        flag_policy: str = "ignore",
        output_dir: str | Path | None = None,
        prefix: str = "sample",
        read_csv_kwargs: dict | None = None,
):
    """
    Load sensor-gradient measurements from CSV files and predict a wavefront.

    Args:
        predictor: Dictionary returned by load_predictor.
        measurement_csv: Path to the CSV file containing measured sensor
            slopes.
        reference_csv: Optional CSV path defining the reference sensor layout
            and desired ordering.
        align_by: Sensor alignment strategy passed to predict_from_dataframes.
        flag_policy: Flag-handling policy passed to predict_from_dataframes.
        output_dir: Optional directory for saved prediction outputs.
        prefix: Filename prefix used for saved outputs.
        read_csv_kwargs: Optional keyword arguments forwarded to pandas.read_csv
            for both measurement_csv and reference_csv.

    Returns:
        Dictionary returned by predict_from_dataframes.

    Notes:
        When reference_csv is omitted, the measurement CSV row order is used
        unless another alignment mode is explicitly requested.
    """
    # Avoid mutable default arguments while allowing custom pandas CSV options.
    read_csv_kwargs = read_csv_kwargs or {}

    # Load measurement data and an optional reference sensor-layout table.
    meas_df = pd.read_csv(
        measurement_csv,
        **read_csv_kwargs,
    )

    ref_df = (
        pd.read_csv(
            reference_csv,
            **read_csv_kwargs,
        )
        if reference_csv is not None
        else None
    )

    return predict_from_dataframes(
        predictor=predictor,
        meas_df=meas_df,
        ref_df=ref_df,
        align_by=align_by,
        flag_policy=flag_policy,
        output_dir=output_dir,
        prefix=prefix,
    )
