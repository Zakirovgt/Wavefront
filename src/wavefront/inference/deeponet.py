from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wavefront.training.deeponet import apply_net_tasks


def build_uniform_eval_grid(
        p_test: int,
) -> tuple[jnp.ndarray, int]:
    """
    Construct a uniform square evaluation grid over [-1, 1] x [-1, 1].

    Args:
        p_test: Total number of evaluation points. This value must be a
            perfect square so the points can form a square grid.

    Returns:
        A tuple containing:

            coords:
                JAX array with shape (p_test, 2), where each row contains
                one Cartesian coordinate pair [x, y].

            side:
                Number of grid points along each spatial dimension.

    Raises:
        ValueError: If p_test is not a perfect square.

    Notes:
        The returned coordinates are ordered according to the flattened
        output of ``jnp.meshgrid``.
    """
    side = int(np.sqrt(p_test))

    if side * side != p_test:
        raise ValueError(
            f"p_test={p_test} must be a perfect square."
        )

    # Create uniformly spaced coordinates over the normalized domain.
    xy = jnp.linspace(-1.0, 1.0, side)

    # Construct the Cartesian mesh and flatten it into coordinate pairs.
    xg, yg = jnp.meshgrid(xy, xy)

    coords = jnp.stack(
        [xg.ravel(), yg.ravel()],
        axis=1,
    )

    return coords.astype(jnp.float32), side


def predict_batch_on_grid(
        model_fn,
        params,
        grad_sensor_batch,
        p_test: int,
):
    """
    Reconstruct wavefronts on a uniform grid from a batch of sensor gradients.

    Each sample contains sensor-space gradient measurements:

        [dU/dx, dU/dy]

    The sensor gradients are flattened into DeepONet branch inputs, while the
    trunk network is evaluated on a shared regular grid over [-1, 1] x [-1, 1].

    Args:
        model_fn: Flax DeepONet model module.
        params: Model parameter pytree.
        grad_sensor_batch: Sensor-gradient array with shape
            (B, P_sensor, 2), or a single sample with shape (P_sensor, 2).
        p_test: Number of regular-grid query points. Must be a perfect square.

    Returns:
        Dictionary containing:

            pred_flat:
                Predicted wavefront values with shape (B, p_test).

            pred_grid:
                Predicted wavefronts reshaped to shape
                (B, side, side).

            coords:
                NumPy coordinate array with shape (p_test, 2).

            side:
                Grid resolution along each spatial axis.

    Raises:
        ValueError: If grad_sensor_batch does not have shape
            (B, P_sensor, 2) after optional batch expansion, or if the model
            output cannot be interpreted as one scalar wavefront value per
            query point.
    """
    grad_sensor_batch = jnp.asarray(
        grad_sensor_batch,
        dtype=jnp.float32,
    )

    # Add a batch dimension when a single sensor-gradient field is provided.
    if grad_sensor_batch.ndim == 2:
        grad_sensor_batch = grad_sensor_batch[None, ...]

    # Each sensor must provide exactly two gradient components:
    # [dU/dx, dU/dy].
    if (
            grad_sensor_batch.ndim != 3
            or grad_sensor_batch.shape[-1] != 2
    ):
        raise ValueError(
            "Expected grad_sensor_batch with shape "
            f"(B, P_sensor, 2), but got {grad_sensor_batch.shape}."
        )

    B = grad_sensor_batch.shape[0]

    # Build the common regular evaluation grid used for all samples.
    coords, side = build_uniform_eval_grid(p_test)

    # Flatten sensor gradient values into DeepONet branch-network inputs:
    #
    # (B, P_sensor, 2) -> (B, 2 * P_sensor)
    branch_inputs = grad_sensor_batch.reshape(B, -1)

    # Reuse the same spatial query coordinates for every input sample:
    #
    # (p_test, 2) -> (B, p_test, 2)
    coords_batch = jnp.broadcast_to(
        coords[None, :, :],
        (B, coords.shape[0], 2),
    )

    # Evaluate all tasks and all query points in a vectorized batch.
    pred = apply_net_tasks(
        model_fn,
        params,
        branch_inputs,
        coords_batch,
    )

    # Convert scalar output channels into shape (B, p_test).
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[..., 0]
    elif pred.ndim == 2:
        pass
    else:
        raise ValueError(
            "Expected wavefront output with shape "
            f"(B, {p_test}) or (B, {p_test}, 1), "
            f"but got {pred.shape}."
        )

    # Convert predictions to NumPy for downstream saving and visualization.
    pred = np.asarray(pred, dtype=np.float32)

    # Restore each flattened wavefront into a square spatial grid.
    pred_grid = pred.reshape(B, side, side)

    return {
        "pred_flat": pred,
        "pred_grid": pred_grid,
        "coords": np.asarray(coords),
        "side": side,
    }


def save_predictions(
        pred_dict,
        output_dir,
        prefix: str = "sample",
        save_csv: bool = True,
        save_npy: bool = True,
        save_png: bool = True,
):
    """
    Save predicted wavefronts in NumPy, CSV, and image formats.

    Args:
        pred_dict: Dictionary returned by ``predict_batch_on_grid`` containing
            ``pred_flat``, ``pred_grid``, and ``coords``.
        output_dir: Directory where prediction artifacts will be written.
        prefix: Prefix used for output filenames.
        save_csv: Whether to save one coordinate/value CSV file per sample.
        save_npy: Whether to save complete batched prediction arrays as NPY
            files.
        save_png: Whether to save one rendered wavefront image per sample.

    Side Effects:
        Creates output_dir if it does not exist.

        When save_npy=True, writes:

            {prefix}_all_pred_flat.npy
            {prefix}_all_pred_grid.npy
            {prefix}_coords.npy

        When save_csv=True, writes one file per sample:

            {prefix}_0000_pred.csv
            {prefix}_0001_pred.csv
            ...

        When save_png=True, writes one image per sample:

            {prefix}_0000_pred.png
            {prefix}_0001_pred.png
            ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract expected prediction arrays from the model-output dictionary.
    pred_flat = pred_dict["pred_flat"]
    pred_grid = pred_dict["pred_grid"]
    coords = pred_dict["coords"]

    B = pred_flat.shape[0]

    # Save complete batched arrays for efficient loading in later workflows.
    if save_npy:
        np.save(
            output_dir / f"{prefix}_all_pred_flat.npy",
            pred_flat,
        )

        np.save(
            output_dir / f"{prefix}_all_pred_grid.npy",
            pred_grid,
        )

        np.save(
            output_dir / f"{prefix}_coords.npy",
            coords,
        )

    # Save individual sample artifacts.
    for i in range(B):
        sample_name = f"{prefix}_{i:04d}"

        if save_csv:
            # Store predicted values together with their corresponding x/y
            # evaluation coordinates.
            df_pred = pd.DataFrame(
                {
                    "x": coords[:, 0],
                    "y": coords[:, 1],
                    "U_pred": pred_flat[i],
                }
            )

            df_pred.to_csv(
                output_dir / f"{sample_name}_pred.csv",
                index=False,
            )

        if save_png:
            # Render the reconstructed wavefront on the normalized spatial
            # domain [-1, 1] x [-1, 1].
            plt.figure(figsize=(6, 5))

            plt.imshow(
                pred_grid[i],
                extent=[-1, 1, 1, -1],
                origin="upper",
                cmap="magma",
            )

            plt.colorbar(label="U_pred")
            plt.title(sample_name)
            plt.tight_layout()

            plt.savefig(
                output_dir / f"{sample_name}_pred.png",
                dpi=150,
            )

            plt.close()
