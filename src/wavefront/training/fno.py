from functools import partial

import jax
import jax.numpy as jnp
import optax

from wavefront.training.common import _tree_l2_norm
from wavefront.training.precision import _MP_ENABLED, _maybe_mp


def rel_l2_field_loss(y_true, y_pred, eps=1e-12):
    """
    Compute the mean relative L2 error across samples in a batch.

    Suitable for FNO outputs:
        y_true: shape (B, H, W) or (B, H, W, 1)
        y_pred: shape (B, H, W) or (B, H, W, 1)

    Returns:
        mean_i ||pred_i - true_i|| / ||true_i||
    """
    y_true = jnp.asarray(y_true)
    y_pred = jnp.asarray(y_pred)

    # Remove a trailing singleton channel dimension when present.
    if y_true.ndim == 4 and y_true.shape[-1] == 1:
        y_true = y_true[..., 0]

    if y_pred.ndim == 4 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    # Flatten each spatial field while preserving the batch dimension.
    y_true = y_true.reshape(y_true.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)

    num = jnp.sqrt(jnp.sum((y_pred - y_true) ** 2, axis=1) + eps)
    den = jnp.sqrt(jnp.sum(y_true ** 2, axis=1) + eps)

    return jnp.mean(num / den)


@partial(jax.jit, static_argnums=(0,))
def apply_fno(model, params, x, rng=None, training=False):
    """
    Apply an FNO model to a batch of regular-grid inputs.

    Inputs are converted to the mixed-precision dtype when enabled. Outputs
    are converted back to float32 to keep losses, metrics, and logging stable.
    """
    x = _maybe_mp(x)

    kwargs = {}
    if rng is not None:
        kwargs["rngs"] = {"dropout": rng}

    out = model.apply(
        {"params": params},
        x,
        **kwargs,
    )

    # Return a scalar field instead of a one-channel field when applicable.
    if out.shape[-1] == 1:
        out = jnp.squeeze(out, axis=-1)

    # Keep downstream loss computation, metrics, and logging in float32.
    if _MP_ENABLED:
        out = out.astype(jnp.float32)

    return out


@partial(jax.jit, static_argnums=(0, 1, 2))
def step_fno(optimizer, loss_fn, model_fn, opt_state, params_step, batch, rng):
    """Perform one FNO optimization step using the supplied batch."""
    loss, grads = jax.value_and_grad(loss_fn, argnums=1)(
        model_fn,
        params_step,
        batch,
        rng,
    )

    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        params=params_step,
    )

    params_step = optax.apply_updates(params_step, updates)

    return loss, _tree_l2_norm(grads), params_step, opt_state
