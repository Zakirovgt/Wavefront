"""Core DeepONet training and inference primitives."""

from functools import partial

import jax
import jax.numpy as jnp
import optax

from wavefront.training.common import _tree_l2_norm
from wavefront.training.precision import _MP_ENABLED, _maybe_mp


def rel_l2_batch_loss(y_true, y_pred, eps=1e-12):
    """
    Compute the global relative L2 error over an entire pointwise batch.

    Suitable for DeepONet pointwise batches:
        y_true: shape (B, 1) or (B,)
        y_pred: shape (B, 1) or (B,)
    """
    y_true = jnp.asarray(y_true).reshape(-1)
    y_pred = jnp.asarray(y_pred).reshape(-1)

    numerator = jnp.sqrt(jnp.sum((y_pred - y_true) ** 2) + eps)
    denominator = jnp.sqrt(jnp.sum(y_true ** 2) + eps)

    return numerator / denominator


@partial(jax.jit)
def mse(y_true, y_pred):
    """Compute the mean squared error between targets and predictions."""
    return jnp.mean(jnp.square(y_true - y_pred))


@partial(jax.jit)
def mse_single(y_pred):
    """Compute the mean squared value of a single prediction tensor."""
    return jnp.mean(jnp.square(y_pred))


@partial(jax.jit, static_argnums=(0,))
def apply_net(model, params, u, x, y, rng=None, training=False):
    """
    Apply a DeepONet model to branch inputs and coordinate inputs.

    Mixed precision is applied to inputs when enabled. The output is converted
    back to float32 before returning to keep losses, metrics, and logging stable.
    """
    u = _maybe_mp(u)
    x = _maybe_mp(x)
    y = _maybe_mp(y)

    if rng is None:
        out = model.apply(
            {"params": params},
            u,
            x,
            y,
            training=training,
        )
    else:
        out = model.apply(
            {"params": params},
            u,
            x,
            y,
            training=training,
            rngs={"dropout": rng},
        )

    # Keep downstream loss computation, metrics, and logging in float32.
    if _MP_ENABLED:
        out = out.astype(jnp.float32)

    return out


@partial(jax.jit, static_argnums=(0,))
def apply_net_task(model_fn, params, branch_input, coords, rng=None):
    """
    Evaluate one DeepONet task at multiple spatial coordinates.

    Args:
        model_fn: DeepONet model instance.
        params: Model parameters.
        branch_input: Sensor-gradient representation with shape (branch_dim,).
        coords: Evaluation coordinates with shape (p, 2).
        rng: Optional dropout random key.

    Returns:
        Predictions with shape (p,) or (p, num_outputs).
    """
    x = coords[:, 0]
    y = coords[:, 1]

    def eval_one(xi, yi):
        # The model expects a batch dimension, even for a single point.
        branch_one = branch_input[None, :]  # Shape: (1, branch_dim)
        x_one = jnp.asarray([xi])  # Shape: (1,)
        y_one = jnp.asarray([yi])  # Shape: (1,)

        out = apply_net(model_fn, params, branch_one, x_one, y_one, rng=rng)
        return out[0]

    return jax.vmap(eval_one)(x, y)


@partial(jax.jit, static_argnums=(0,))
def apply_net_tasks(model_fn, params, branch_inputs, coords_batch, rng=None):
    """
    Evaluate multiple DeepONet tasks at batches of spatial coordinates.

    Args:
        model_fn: DeepONet model instance.
        params: Model parameters.
        branch_inputs: Branch inputs with shape (B, branch_dim).
        coords_batch: Coordinate batches with shape (B, p, 2).
        rng: Optional dropout random key.

    Returns:
        Predictions with shape (B, p) or (B, p, num_outputs).
    """
    return jax.vmap(
        lambda u, coords: apply_net_task(model_fn, params, u, coords, rng=rng)
    )(branch_inputs, coords_batch)


@partial(jax.jit, static_argnums=(0, 1, 2))
def step(
        optimizer,
        loss_fn,
        model_fn,
        opt_state,
        params_step,
        ics_batch,
        res_batch,
        res_weight,
        rng,
):
    """
    Perform one DeepONet optimization step.

    The loss function combines supervised initial-condition data and the
    gradient-residual term, then parameters are updated using Optax.
    """
    loss, grads = jax.value_and_grad(loss_fn, argnums=1)(
        model_fn,
        params_step,
        ics_batch,
        res_batch,
        res_weight,
        rng,
    )

    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        params=params_step,
    )
    params_step = optax.apply_updates(params_step, updates)

    return loss, _tree_l2_norm(grads), params_step, opt_state
