import jax
import jax.numpy as jnp

from wavefront.training.deeponet import (
    apply_net,
    rel_l2_batch_loss,
)


def loss_ics(
        model_fn,
        params,
        ics_batch,
        rng,
):
    """
    Compute the supervised initial-condition loss for scalar wavefront values.

    Args:
        model_fn: Flax DeepONet model module.
        params: Model parameter pytree.
        ics_batch: Supervised pointwise batch structured as:

            ((u, y), outputs)

            where:
                u: Branch inputs with shape (B, branch_dim).
                y: Coordinate queries with shape (B, 2).
                outputs: Scalar wavefront targets with shape (B,) or (B, 1).

        rng: JAX PRNG key used for dropout when training is enabled.

    Returns:
        Scalar global relative L2 loss between predicted and target wavefront
        values.

    Raises:
        ValueError: If the model output does not represent a single scalar
            field per queried coordinate.
    """
    inputs, outputs = ics_batch
    u, y = inputs

    # Split coordinate pairs into independent x and y tensors.
    x = y[:, 0]
    y_ = y[:, 1]

    # Evaluate the model in training mode.
    pred = apply_net(
        model_fn,
        params,
        u,
        x,
        y_,
        rng=rng,
        training=True,
    )

    # Normalize scalar predictions and targets to column-vector format.
    if pred.ndim == 1:
        pred = pred[:, None]

    if outputs.ndim == 1:
        outputs = outputs[:, None]

    # This loss expects a scalar wavefront prediction at every query point.
    if pred.shape[-1] != 1:
        raise ValueError(
            f"Expected model output with shape (B, 1), but received {pred.shape}."
        )

    # Flatten any remaining target dimensions into a single output dimension.
    if outputs.shape[-1] != 1:
        outputs = outputs.reshape(outputs.shape[0], -1)

    return rel_l2_batch_loss(outputs, pred)


def loss_res(
        model_fn,
        params,
        batch,
        rng,
):
    """
    Compute a gradient-matching residual loss using automatic differentiation.

    The model predicts a scalar wavefront U(x, y). Its spatial derivatives are
    computed through vector-Jacobian products and compared with the provided
    gradient targets:

        dU/dx
        dU/dy

    Args:
        model_fn: Flax DeepONet model module.
        params: Model parameter pytree.
        batch: Derivative-supervision batch structured as:

            ((u, y), outputs)

            where:
                u: Branch inputs with shape (B, branch_dim).
                y: Coordinate queries with shape (B, 2).
                outputs: Gradient targets with shape (B, 2), containing:
                    outputs[:, 0] = dU/dx
                    outputs[:, 1] = dU/dy

        rng: JAX PRNG key used for dropout during model evaluation.

    Returns:
        Scalar global relative L2 loss between target and predicted gradients.

    Notes:
        This implementation assumes that every batch element can be evaluated
        independently with respect to its own coordinate pair. The vector of
        ones supplied to jax.vjp extracts gradients of the summed batch output,
        which is equivalent to per-sample derivatives when batch samples do
        not interact inside the network.
    """
    inputs, outputs = batch
    u, y = inputs

    # Split coordinate pairs into x and y components.
    x = y[:, 0]
    y_ = y[:, 1]

    # Extract target gradient components.
    g1_true = outputs[:, 0]
    g2_true = outputs[:, 1]

    # Evaluate once to preserve the expected prediction/output conventions.
    s_pred = apply_net(
        model_fn,
        params,
        u,
        x,
        y_,
        rng=rng,
    )

    def extract_U(out):
        """
        Extract the scalar wavefront channel from model output.

        Supported output layouts:
            - (B,): Direct scalar output.
            - (B, 1): Single scalar output channel.
            - (B, C), C >= 3: Wavefront stored in channel index 2.

        Returns:
            Scalar wavefront tensor with shape (B,).
        """
        if out.ndim == 1:
            return out

        if out.shape[1] == 1:
            return out[:, 0]

        if out.shape[1] >= 3:
            return out[:, 2]

        raise ValueError(
            f"loss_res received an unexpected number of outputs: {out.shape[1]}."
        )

    def U_of_x(x_):
        """
        Evaluate the scalar wavefront while varying x and keeping y fixed.
        """
        return extract_U(
            apply_net(
                model_fn,
                params,
                u,
                x_,
                y_,
                rng=rng,
            )
        )

    def U_of_y(y__):
        """
        Evaluate the scalar wavefront while varying y and keeping x fixed.
        """
        return extract_U(
            apply_net(
                model_fn,
                params,
                u,
                x,
                y__,
                rng=rng,
            )
        )

    # Construct cotangent vectors for vector-Jacobian products.
    v_x = jnp.ones_like(U_of_x(x))
    v_y = jnp.ones_like(U_of_y(y_))

    # Differentiate the scalar wavefront prediction with respect to each
    # coordinate axis.
    dU_dx = jax.vjp(U_of_x, x)[1](v_x)[0]
    dU_dy = jax.vjp(U_of_y, y_)[1](v_y)[0]

    # Stack gradient components into shape (B, 2).
    grad_true = jnp.stack([g1_true, g2_true], axis=1)
    grad_pred = jnp.stack([dU_dx, dU_dy], axis=1)

    return rel_l2_batch_loss(grad_true, grad_pred)


def loss_fn(
        model_fn,
        params,
        ics_batch,
        res_batch,
        res_weight,
        rng,
):
    """
    Compute the combined supervised and gradient-matching training loss.

    Args:
        model_fn: Flax DeepONet model module.
        params: Model parameter pytree.
        ics_batch: Pointwise wavefront-value supervision batch.
        res_batch: Pointwise derivative supervision batch.
        res_weight: Weight applied to the gradient-matching residual term.
        rng: JAX PRNG key used for stochastic model components such as dropout.

    Returns:
        Combined scalar loss:

            loss_ics + res_weight * loss_res
    """
    return (
            loss_ics(
                model_fn,
                params,
                ics_batch,
                rng=rng,
            )
            + res_weight
            * loss_res(
        model_fn,
        params,
        res_batch,
        rng=rng,
    )
    )
