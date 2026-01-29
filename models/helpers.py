import jax.numpy as jnp
import jax
import optax
from functools import partial

def _tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    # sqrt(∑_i ||g_i||^2)
    return jnp.sqrt(sum([jnp.vdot(x, x) for x in leaves]))

@partial(jax.jit)
def mse(y_true, y_pred):
    return jnp.mean(jnp.square(y_true - y_pred))

@partial(jax.jit)
def mse_single(y_pred):
    return jnp.mean(jnp.square(y_pred))

@partial(jax.jit, static_argnums=(0,))
def apply_net(model_fn, params, branch_input, *trunk_in, rng=None):
    # Define forward pass for normal DeepOnet that takes series of trunk inputs and stacks them
    if len(trunk_in) == 1:
        trunk_input = trunk_in[0]
    else:
        trunk_input = jnp.stack(trunk_in, axis=-1)
    kwargs = {}
    if rng is not None:
        kwargs['rngs'] = {'dropout': rng}
    else:
        kwargs['rngs'] = {'dropout': jax.random.PRNGKey(0)}  # dummy key, safe

    out = model_fn(params, branch_input, trunk_input, **kwargs)
    # Reshape to vector for single output for easier gradient computation
    if out.shape[1]==1:
        out = jnp.squeeze(out, axis=1)
    return out

@partial(jax.jit, static_argnums=(0, ))
def apply_net_sep(model_fn, params, branch_input, *trunk_in):
    # Define forward pass for separable DeepONet that takes series of trunk inputs
    out = model_fn(params, branch_input, *trunk_in)
    return out

@partial(jax.jit, static_argnums=(0, 1, 2))
def step(optimizer, loss_fn, model_fn,
         opt_state, params_step,
         ics_batch, res_batch, res_weight, rng):
    loss, grads = jax.value_and_grad(loss_fn, argnums=1)(
        model_fn, params_step, ics_batch, res_batch, res_weight, rng
    )
    updates, opt_state = optimizer.update(grads, opt_state, params=params_step)
    params_step = optax.apply_updates(params_step, updates)
    return loss, _tree_l2_norm(grads), params_step, opt_state