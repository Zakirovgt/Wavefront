import jax
import jax.numpy as jnp
import numpy as np

from wavefront.data.deeponet_data import make_test_task
from wavefront.training.deeponet import apply_net


def get_error(
        model_fn,
        params,
        grad_sensor_all,
        wavefront_true_all,
        idx,
        p_err: int = 24 * 24,
        return_data: bool = False,
):
    """
    Compute the relative L2 reconstruction error for one or more DeepONet tasks.

    Args:
        model_fn: Flax DeepONet model module.
        params: Model parameter pytree.
        grad_sensor_all: Sensor-gradient dataset with shape
            (N, P_sensor, 2).
        wavefront_true_all: Ground-truth wavefront dataset containing
            p_err values per sample.
        idx: Index of one sample, or a one-dimensional collection of sample
            indices. When multiple indices are supplied, an error value is
            computed for each selected sample.
        p_err: Number of regular-grid evaluation points. This must match the
            number of wavefront target values in each selected example.
        return_data: Whether to return the predicted wavefront values together
            with the relative L2 error. This option applies to single-sample
            evaluation.

    Returns:
        For a single index:

            err:
                Scalar relative L2 error.

        When return_data=True for a single index:

            (err, pred)

            where pred has shape (p_err,).

        For a one-dimensional collection of indices:

            errs:
                Array containing one relative L2 error per selected index.

    Raises:
        ValueError: If the model output cannot be interpreted as a scalar
            prediction at every evaluation point.

    Notes:
        The relative L2 error is computed as:

            ||s_true - pred||_2 / (||s_true||_2 + 1e-8)

        For a single task, the branch input is repeated across all coordinate
        queries so that the model evaluates the same sensor measurement at
        every regular-grid location.
    """
    # Support evaluating several dataset examples at once.
    #
    # Each selected index is evaluated independently through jax.vmap.
    if (
            isinstance(idx, (list, tuple, jnp.ndarray, np.ndarray))
            and getattr(idx, "ndim", 1) == 1
    ):
        errs = jax.vmap(
            lambda i: get_error(
                model_fn,
                params,
                grad_sensor_all,
                wavefront_true_all,
                i,
                p_err,
                False,
            )
        )(jnp.array(idx))

        return errs

    # Build the branch input, regular-grid coordinate queries, and reference
    # wavefront values for one dataset example.
    u_test, y_test, s_test = make_test_task(
        grad_sensor_all,
        wavefront_true_all,
        idx,
        p_err,
    )

    # Separate the query-coordinate tensor into x and y components.
    x = y_test[:, 0]
    y = y_test[:, 1]

    # Repeat the same branch input for every spatial query point.
    #
    # u_test: (branch_dim,)
    # u_rep:  (p_err, branch_dim)
    u_rep = jnp.broadcast_to(
        u_test[None, :],
        (x.shape[0], u_test.shape[0]),
    )

    # Evaluate the DeepONet at all regular-grid coordinates.
    pred = apply_net(
        model_fn,
        params,
        u_rep,
        x,
        y,
        training=False,
    )

    # Convert a single-output column vector from shape (P, 1) to shape (P,).
    if pred.ndim == 2 and pred.shape[-1] == 1:
        pred = pred[:, 0]
    elif pred.ndim != 1:
        raise ValueError(
            "Expected model output with shape (P,) or (P, 1), "
            f"but received {pred.shape}."
        )

    # Flatten the reference field to match the one-dimensional prediction.
    s_true = s_test.reshape(-1)

    # Compute the relative L2 reconstruction error.
    err = jnp.linalg.norm(s_true - pred) / (
            jnp.linalg.norm(s_true) + 1e-8
    )

    if return_data:
        return err, pred

    return err
