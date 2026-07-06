from __future__ import annotations

from argparse import Namespace

import numpy as np

from wavefront.metamodel.gradmap import deeponet_predict_grad_grid
from wavefront.training.fno_trainer import main_routine_fno


def train_fno_on_deeponet_outputs(
        fno_args: Namespace,
        deeponet_state: dict,
        grad_sensor_noisy: np.ndarray,
        wavefront_true: np.ndarray,
        batch_infer: int = 64,
):
    """
    Train an FNO using gradient fields predicted by a pretrained DeepONet.

    This function implements the second stage of a two-stage reconstruction
    pipeline:

        Noisy sensor gradients
            -> DeepONet gradient-map model
            -> predicted regular-grid gradients
            -> FNO
            -> reconstructed wavefront

    The DeepONet is evaluated over all supplied sensor-gradient samples in
    batches. Its predicted clean gradient maps are then passed to the existing
    FNO training routine as regular-grid inputs.

    Args:
        fno_args: FNO training configuration namespace. Its data_mode,
            n_train, n_test, and grid_size attributes are updated before
            launching the FNO training routine.
        deeponet_state: Dictionary returned by ``train_deeponet_gradmap``.
            It must contain:

                model_fn:
                    Trained DeepONet model module.

                params:
                    Selected DeepONet parameter pytree.

        grad_sensor_noisy: Noisy sensor-gradient data with shape
            ``(N, P_sensor, 2)``.

        wavefront_true: Ground-truth wavefront targets with shape
            ``(N, p)`` or ``(N, grid_size, grid_size)``.

        batch_infer: Number of sensor-gradient functions processed by the
            DeepONet at once during regular-grid gradient prediction.

    Returns:
        Dictionary returned by ``main_routine_fno`` containing the trained FNO
        model, parameters, test data, artifact paths, and related metadata.

    Notes:
        The DeepONet model is used in inference mode only. Its parameters are
        not updated during the FNO training stage.
    """
    # Resolve the regular-grid resolution from the FNO configuration, falling
    # back to nx or 24 when grid_size is not explicitly available.
    grid_size = int(
        getattr(
            fno_args,
            "grid_size",
            getattr(fno_args, "nx", 24),
        )
    )

    grad_sensor_noisy = np.asarray(
        grad_sensor_noisy,
        dtype=np.float32,
    )

    N = grad_sensor_noisy.shape[0]

    # Predict regular-grid gradient fields for all sensor-gradient samples.
    # Batched inference prevents unnecessarily large memory allocations.
    preds = []

    for start in range(0, N, int(batch_infer)):
        end = min(N, start + int(batch_infer))

        grad_grid_pred = deeponet_predict_grad_grid(
            model_fn=deeponet_state["model_fn"],
            params=deeponet_state["params"],
            grad_sensor_batch=grad_sensor_noisy[start:end],
            grid_size=grid_size,
            rng=None,
        )

        preds.append(
            np.asarray(
                grad_grid_pred,
                dtype=np.float32,
            )
        )

    # Combine batch outputs into the regular-grid FNO input layout:
    #
    # (N, grid_size, grid_size, 2)
    grad_grid_pred_all = np.concatenate(
        preds,
        axis=0,
    )

    # Route the predicted gradient maps through the standard FNO pipeline as
    # regular-grid derivative inputs.
    fno_args.data_mode = "regular_grid"

    fno_args.n_train = int(
        getattr(fno_args, "n_train", N)
    )

    fno_args.n_test = int(
        getattr(fno_args, "n_test", 0)
    )

    fno_args.grid_size = int(
        getattr(fno_args, "grid_size", grid_size)
    )

    return main_routine_fno(
        args=fno_args,
        grad_sensor_data=None,
        wavefront_true_data=wavefront_true,
        grad_grid_data=grad_grid_pred_all,
    )
