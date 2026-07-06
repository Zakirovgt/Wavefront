import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from wavefront.evaluation.deeponet import get_error
from wavefront.training.fno import apply_fno


def visualize(
        args,
        model_fn,
        params,
        result_dir,
        step,
        grad_sensor,
        wavefront_true,
        idx,
        test: bool = False,
):
    """
    Visualize a DeepONet wavefront reconstruction for one dataset sample.

    The function evaluates the model, compares the predicted wavefront against
    the ground truth, and saves a three-panel image containing:

        1. Ground-truth wavefront.
        2. Predicted wavefront.
        3. Pointwise prediction error.

    Args:
        args: Configuration object expected to contain p_test, the number of
            evaluation points in the regular output grid.
        model_fn: Flax DeepONet model module.
        params: Model parameter pytree.
        result_dir: Root directory where visualization files are saved.
        step: Training step used to organize output directories.
        grad_sensor: Sensor-gradient dataset used as DeepONet branch input.
        wavefront_true: Ground-truth wavefront dataset.
        idx: Index of the sample to visualize.
        test: Whether the selected sample belongs to the test split. Used only
            for the figure title.

    Side Effects:
        Saves the figure to:

            {result_dir}/vis/{step:06d}/{idx}/U_comparison.png
    """
    # Evaluate the selected sample and retrieve both its relative error and
    # predicted wavefront values.
    err, pred = get_error(
        model_fn,
        params,
        grad_sensor,
        wavefront_true,
        idx,
        args.p_test,
        return_data=True,
    )

    # Infer the regular square-grid resolution from the flattened target.
    side = int(np.sqrt(wavefront_true.shape[1]))

    true_grid = np.asarray(wavefront_true[idx]).reshape(side, side)
    pred_grid = np.asarray(pred).reshape(side, side)

    # Compute the pointwise reconstruction error.
    diff = pred_grid - true_grid

    # Create a separate directory for this training step and sample index.
    plot_dir = os.path.join(
        result_dir,
        f"vis/{step:06d}/{idx}/",
    )
    os.makedirs(plot_dir, exist_ok=True)

    # Use identical display limits for the true and predicted wavefronts.
    vmin = min(true_grid.min(), pred_grid.min())
    vmax = max(true_grid.max(), pred_grid.max())

    # Use symmetric display limits around zero for the error map.
    vmax_diff = max(np.abs(diff).max(), 1e-8)

    fig = plt.figure(figsize=(15, 5))

    # Ground-truth wavefront.
    plt.subplot(1, 3, 1)
    plt.imshow(
        true_grid,
        vmin=vmin,
        vmax=vmax,
        extent=[-1, 1, 1, -1],
        origin="upper",
        cmap="magma",
    )
    plt.colorbar()
    plt.title("True U")

    # DeepONet reconstruction.
    plt.subplot(1, 3, 2)
    plt.imshow(
        pred_grid,
        vmin=vmin,
        vmax=vmax,
        extent=[-1, 1, 1, -1],
        origin="upper",
        cmap="magma",
    )
    plt.colorbar()
    plt.title("Predicted U")

    # Pointwise difference between prediction and reference.
    plt.subplot(1, 3, 3)
    plt.imshow(
        diff,
        extent=[-1, 1, 1, -1],
        origin="upper",
        cmap="seismic",
        vmin=-vmax_diff,
        vmax=vmax_diff,
    )
    plt.colorbar()
    plt.title("U difference")

    split_name = "Test" if test else "Train"

    fig.suptitle(
        f"{split_name} sample {idx}, rel L2: {float(err):.3e}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(plot_dir, "U_comparison.png")
    )

    plt.close(fig)


def _visualize_fno(
        model_fn,
        params,
        X_test,
        Y_test,
        result_dir,
        step,
        idx,
        args,
):
    """
    Visualize an FNO wavefront reconstruction for one test sample.

    The function runs FNO inference for a selected regular-grid input and
    saves a three-panel comparison of the reference wavefront, predicted
    wavefront, and pointwise error.

    Args:
        model_fn: Flax FNO model module.
        params: Model parameter pytree.
        X_test: FNO input dataset with shape (N, H, W, C).
        Y_test: Ground-truth wavefront dataset with shape (N, H, W), or a
            flattened equivalent representation.
        result_dir: Root directory where visualization files are saved.
        step: Training step used to organize output directories.
        idx: Index of the test sample to visualize.
        args: Configuration object retained for interface consistency.

    Side Effects:
        Saves the figure to:

            {result_dir}/vis/{step:06d}/{idx}/U_comparison.png
    """
    # Preserve the batch dimension expected by the FNO:
    # (H, W, C) -> (1, H, W, C).
    x = jnp.asarray(
        X_test[idx:idx + 1],
        dtype=jnp.float32,
    )

    # Run FNO inference without dropout.
    y_pred = apply_fno(
        model_fn,
        params,
        x,
        rng=None,
        training=False,
    )

    # Remove optional singleton output-channel and batch dimensions:
    # (1, H, W, 1) -> (H, W).
    if y_pred.ndim == 4 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    if y_pred.ndim == 3:
        y_pred = y_pred[0]

    # Retrieve the corresponding ground-truth wavefront.
    y_true = np.asarray(Y_test[idx])
    y_pred = np.asarray(y_pred)

    # Support flattened target or prediction arrays by restoring their square
    # regular-grid representation.
    if y_true.ndim == 1:
        g = int(np.sqrt(y_true.shape[0]))
        y_true = y_true.reshape(g, g)

    if y_pred.ndim == 1:
        g = int(np.sqrt(y_pred.shape[0]))
        y_pred = y_pred.reshape(g, g)

    # Compute the pointwise error and global relative L2 reconstruction error.
    diff = y_pred - y_true

    rel_err = float(
        np.linalg.norm(diff)
        / (np.linalg.norm(y_true) + 1e-8)
    )

    # Use shared limits for direct visual comparison between reference and
    # prediction, and symmetric limits for the error map.
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    vmax_diff = max(np.abs(diff).max(), 1e-8)

    # Create the output directory for this step and sample.
    plot_dir = os.path.join(
        result_dir,
        f"vis/{step:06d}/{idx}/",
    )
    os.makedirs(plot_dir, exist_ok=True)

    fig = plt.figure(figsize=(15, 5))

    # Ground-truth wavefront.
    plt.subplot(1, 3, 1)
    plt.imshow(
        y_true,
        vmin=vmin,
        vmax=vmax,
        extent=[-1, 1, 1, -1],
        origin="upper",
        cmap="magma",
    )
    plt.colorbar()
    plt.title("True U")

    # FNO reconstruction.
    plt.subplot(1, 3, 2)
    plt.imshow(
        y_pred,
        vmin=vmin,
        vmax=vmax,
        extent=[-1, 1, 1, -1],
        origin="upper",
        cmap="magma",
    )
    plt.colorbar()
    plt.title("FNO predicted U")

    # Pointwise difference between FNO prediction and reference.
    plt.subplot(1, 3, 3)
    plt.imshow(
        diff,
        extent=[-1, 1, 1, -1],
        origin="upper",
        cmap="seismic",
        vmin=-vmax_diff,
        vmax=vmax_diff,
    )
    plt.colorbar()
    plt.title("U difference")

    fig.suptitle(
        f"FNO test sample {idx}, rel L2: {rel_err:.3e}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(plot_dir, "U_comparison.png")
    )

    plt.close(fig)
