from functools import partial

import jax
import jax.numpy as jnp

from wavefront.training.fno import (
    apply_fno,
    rel_l2_field_loss,
)


def loss_res_fno(
        model_fn,
        params,
        batch,
        rng,
        grid_size=None,
):
    """
    Compute the gradient-matching residual loss for an FNO prediction.

    The first two input channels are interpreted as the target spatial
    derivatives:

        x[..., 0] = dU/dx
        x[..., 1] = dU/dy

    The predicted wavefront is differentiated numerically on the regular grid
    using centered finite differences, and the resulting gradient field is
    compared against the input gradient channels.

    Args:
        model_fn: Flax FNO model module.
        params: Model parameter pytree.
        batch: Tuple containing:

            x:
                Input tensor with shape (B, H, W, C), where C >= 2.

            _y_true:
                Unused target tensor retained for compatibility with the
                standard batch structure.

        rng: JAX PRNG key used for stochastic model components such as dropout.
        grid_size: Optional grid resolution used to determine spatial spacing.
            When None, the height and width of the predicted field are used.

    Returns:
        Scalar relative L2 loss between target and predicted gradient fields.

    Raises:
        ValueError: If the input tensor does not have shape (B, H, W, C)
            with at least two channels, or if the predicted wavefront does not
            have shape (B, H, W) after removing an optional singleton channel.

    Notes:
        Centered differences are computed using periodic-style `jnp.roll`
        operations. Boundary derivatives are then explicitly set to zero to
        avoid wraparound artifacts at the edges.
    """
    x, _y_true = batch
    x = jnp.asarray(x)

    # The first two channels must contain the target gradient components.
    if x.ndim != 4 or x.shape[-1] < 2:
        raise ValueError(
            f"Expected x with shape (B, H, W, C>=2), but got {x.shape}."
        )

    gx_true = x[..., 0]
    gy_true = x[..., 1]

    # Evaluate the FNO in training mode.
    U_pred = apply_fno(
        model_fn,
        params,
        x,
        rng=rng,
        training=True,
    )

    # Remove a final singleton output channel when present:
    # (B, H, W, 1) -> (B, H, W).
    if U_pred.ndim == 4 and U_pred.shape[-1] == 1:
        U_pred = U_pred[..., 0]

    if U_pred.ndim != 3:
        raise ValueError(
            f"Expected U_pred with shape (B, H, W), but got {U_pred.shape}."
        )

    H, W = U_pred.shape[1], U_pred.shape[2]

    # The spatial domain is assumed to span [-1, 1] along both axes.
    #
    # If grid_size is supplied, it is used for both spatial dimensions to
    # preserve the original square-grid assumption.
    dx = (
        2.0 / (W - 1)
        if grid_size is None
        else 2.0 / (int(grid_size) - 1)
    )

    dy = (
        2.0 / (H - 1)
        if grid_size is None
        else 2.0 / (int(grid_size) - 1)
    )

    # Approximate dU/dx with centered finite differences along the width axis.
    dU_dx = (
                    jnp.roll(U_pred, -1, axis=2)
                    - jnp.roll(U_pred, 1, axis=2)
            ) / (2.0 * dx)

    # Approximate dU/dy with centered finite differences along the height axis.
    dU_dy = (
                    jnp.roll(U_pred, -1, axis=1)
                    - jnp.roll(U_pred, 1, axis=1)
            ) / (2.0 * dy)

    # Remove artificial periodic wraparound derivatives at grid boundaries.
    dU_dx = dU_dx.at[:, :, 0].set(0.0)
    dU_dx = dU_dx.at[:, :, -1].set(0.0)

    dU_dy = dU_dy.at[:, 0, :].set(0.0)
    dU_dy = dU_dy.at[:, -1, :].set(0.0)

    # Combine x/y derivative components into gradient fields:
    # (B, H, W, 2).
    grad_true = jnp.stack([gx_true, gy_true], axis=-1)
    grad_pred = jnp.stack([dU_dx, dU_dy], axis=-1)

    return rel_l2_field_loss(grad_true, grad_pred)


def loss_fno(
        model_fn,
        params,
        batch,
        rng,
        res_weight: float = 0.0,
):
    """
    Compute the combined FNO training loss.

    The main loss compares predicted and target wavefront fields. Optionally,
    a gradient-matching residual loss is added.

    Args:
        model_fn: Flax FNO model module.
        params: Model parameter pytree.
        batch: Tuple containing:

            x:
                Input tensor with shape (B, H, W, C).

            y_true:
                Target wavefront tensor, typically with shape (B, H, W) or
                (B, H, W, 1).

        rng: JAX PRNG key used for training-time stochastic operations.
        res_weight: Weight assigned to the gradient-matching residual term.
            Set to 0.0 to use only the wavefront reconstruction loss.

    Returns:
        Scalar combined loss:

            reconstruction_loss + res_weight * gradient_loss

        When res_weight is zero, only reconstruction_loss is returned.
    """
    x, y_true = batch

    # Predict the wavefront field in training mode.
    y_pred = apply_fno(
        model_fn,
        params,
        x,
        rng=rng,
        training=True,
    )

    # Main wavefront reconstruction loss.
    l_ic = rel_l2_field_loss(y_true, y_pred)

    # Optionally enforce consistency between input gradients and derivatives
    # of the predicted wavefront.
    if float(res_weight) > 0.0:
        l_r = loss_res_fno(
            model_fn,
            params,
            batch,
            rng,
            grid_size=x.shape[1],
        )

        return l_ic + float(res_weight) * l_r

    return l_ic


@partial(jax.jit, static_argnums=(0,))
def fno_rel_l2(
        model_fn,
        params,
        x,
        y_true,
):
    """
    Compute the relative L2 error of an FNO prediction during evaluation.

    Args:
        model_fn: Flax FNO model module.
        params: Model parameter pytree.
        x: Input tensor with shape (H, W, C) or (B, H, W, C).
        y_true: Target wavefront with shape (H, W), (H, W, 1),
            (B, H, W), or a shape compatible with the FNO output.

    Returns:
        Scalar relative L2 error:

            ||y_true - y_pred||_2 / (||y_true||_2 + 1e-12)

    Notes:
        Single examples without an explicit batch dimension are automatically
        expanded into batch size one.
    """
    # Add a batch dimension when evaluating a single grid sample.
    if x.ndim == 3:
        x = x[None, ...]

    if y_true.ndim == 2:
        y_true = y_true[None, ...]

    # Run the model in evaluation mode without dropout.
    y_pred = apply_fno(
        model_fn,
        params,
        x,
        rng=None,
        training=False,
    )

    return jnp.linalg.norm(y_true - y_pred) / (
            jnp.linalg.norm(y_true) + 1e-12
    )
