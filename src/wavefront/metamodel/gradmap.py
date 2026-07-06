from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

from wavefront.data.batching import DataGenerator
from wavefront.data.sensors import load_branch_sensor_grid
from wavefront.models.deeponet import ModernDeepONet
from wavefront.training.common import _tree_l2_norm
from wavefront.training.deeponet import (
    apply_net,
    apply_net_tasks,
    rel_l2_batch_loss,
)
from wavefront.training.schedules import make_lr_schedule


@dataclass
class DeepONetGradMapConfig:
    """
    Configuration for training a DeepONet that maps sensor gradients to a
    clean regular-grid gradient field.
    """

    grid_size: int = 24
    n_train: int = 10_000
    n_test: int = 1_000

    batch_size: int = 4096
    steps: int = 150_000
    lr: float = 5e-4
    weight_decay: float = 1e-4
    log_iter: int = 250
    eval_iter: int = 1000
    eval_batch: int = 128
    seed: int = 1234

    # Learning-rate schedule configuration.
    lr_scheduler: str = "warmup_cosine"
    warmup_steps: int | None = None
    warmup_frac: float = 0.05
    end_lr_factor: float = 0.01

    # ModernDeepONet architecture configuration.
    basis_dim: int = 128
    branch_hidden_dim: int = 256
    trunk_hidden_dim: int = 256
    branch_num_layers: int = 4
    trunk_num_layers: int = 4
    trunk_num_frequencies: int = 8
    trunk_max_freq: float = 12.0
    dropout_rate: float = 0.0


def _build_uniform_grid_coords(
        grid_size: int,
) -> jnp.ndarray:
    """
    Construct flattened Cartesian coordinates for a uniform square grid.

    Args:
        grid_size: Number of points along each spatial axis.

    Returns:
        Coordinate array with shape (grid_size * grid_size, 2), covering
        the normalized domain [-1, 1] x [-1, 1].
    """
    xy = jnp.linspace(-1.0, 1.0, int(grid_size))
    xg, yg = jnp.meshgrid(xy, xy)

    return jnp.stack(
        [xg.ravel(), yg.ravel()],
        axis=1,
    ).astype(jnp.float32)


def setup_deeponet_gradmap(
        cfg: DeepONetGradMapConfig,
        n_sensors: int,
        key,
):
    """
    Build and initialize a ModernDeepONet for gradient-map approximation.

    The model learns the mapping:

        Sensor gradients (g_x, g_y)
            -> clean regular-grid gradients (g_x, g_y)

    Args:
        cfg: DeepONet gradient-map training configuration.
        n_sensors: Number of input sensor locations.
        key: JAX PRNG key used for model parameter initialization.

    Returns:
        A tuple containing:

            model:
                Initialized ModernDeepONet module.

            params:
                Flax parameter pytree.
    """
    model = ModernDeepONet(
        basis_dim=int(cfg.basis_dim),
        num_outputs=2,
        branch_hidden_dim=int(cfg.branch_hidden_dim),
        trunk_hidden_dim=int(cfg.trunk_hidden_dim),
        branch_num_layers=int(cfg.branch_num_layers),
        trunk_num_layers=int(cfg.trunk_num_layers),

        # Output-specific branch features are required for multi-output
        # DeepONet prediction.
        split_branch=True,
        split_trunk=False,

        trunk_num_frequencies=int(cfg.trunk_num_frequencies),
        trunk_max_freq=float(cfg.trunk_max_freq),
        dropout_rate=float(cfg.dropout_rate),
    )

    # The branch network receives flattened [g_x, g_y] values from all sensors.
    u_dummy = jnp.ones(
        (1, int(n_sensors) * 2),
        dtype=jnp.float32,
    )

    # The trunk network receives one spatial coordinate pair per sample.
    x_dummy = jnp.ones((1,), dtype=jnp.float32)
    y_dummy = jnp.ones((1,), dtype=jnp.float32)

    variables = model.init(
        key,
        u_dummy,
        x_dummy,
        y_dummy,
        training=True,
    )

    params = variables["params"]

    total_params = sum(
        leaf.size
        for leaf in jax.tree_util.tree_leaves(params)
    )

    print("--- deeponet_gradmap_summary ---")
    print(f"total params: {total_params}")
    print("--- deeponet_gradmap_summary ---")

    return model, params


def loss_deeponet_gradmap(
        model_fn,
        params,
        batch,
        rng,
):
    """
    Compute the supervised relative L2 loss for gradient-map prediction.

    Args:
        model_fn: Flax ModernDeepONet module.
        params: Model parameter pytree.
        batch: Training batch structured as:

            ((u, coords), target_grad)

            where:
                u:
                    Flattened branch inputs with shape (B, branch_dim).

                coords:
                    Spatial query points with shape (B, 2).

                target_grad:
                    Target gradients with shape (B, 2).

        rng: JAX PRNG key used for dropout during training.

    Returns:
        Scalar relative L2 loss between predicted and target gradient vectors.
    """
    (u, coords), target_grad = batch

    # Extract Cartesian coordinate components.
    x = coords[:, 0]
    y = coords[:, 1]

    pred = apply_net(
        model_fn,
        params,
        u,
        x,
        y,
        rng=rng,
        training=True,
    )

    # Both prediction and target have shape (B, 2).
    return rel_l2_batch_loss(target_grad, pred)


@partial(jax.jit, static_argnums=(0, 1, 2))
def step_supervised(
        optimizer,
        loss_fn,
        model_fn,
        opt_state,
        params_step,
        batch,
        rng,
):
    """
    Perform one supervised optimization step.

    Args:
        optimizer: Optax optimizer transformation.
        loss_fn: Supervised loss function.
        model_fn: Flax model module.
        opt_state: Current optimizer state.
        params_step: Current model parameter pytree.
        batch: Supervised training batch.
        rng: JAX PRNG key passed to the loss function.

    Returns:
        A tuple containing:

            loss:
                Scalar loss before updates.

            grad_norm:
                Global L2 norm of the gradient pytree.

            params_step:
                Updated model parameters.

            opt_state:
                Updated optimizer state.
    """
    loss, grads = jax.value_and_grad(
        loss_fn,
        argnums=1,
    )(
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

    params_step = optax.apply_updates(
        params_step,
        updates,
    )

    return (
        loss,
        _tree_l2_norm(grads),
        params_step,
        opt_state,
    )


def deeponet_predict_grad_grid(
        model_fn,
        params,
        grad_sensor_batch,
        grid_size: int,
        rng=None,
):
    """
    Predict clean gradient fields on a regular grid from sensor gradients.

    Args:
        model_fn: Flax ModernDeepONet module.
        params: Model parameter pytree.
        grad_sensor_batch: Sensor gradient array with shape
            (B, P_sensor, 2), or a single sample with shape (P_sensor, 2).
        grid_size: Number of points along each axis of the output grid.
        rng: Optional JAX PRNG key used for dropout.

    Returns:
        Predicted regular-grid gradients with shape:

            (B, grid_size, grid_size, 2)

        The final dimension contains:

            [..., 0] = predicted dU/dx
            [..., 1] = predicted dU/dy

    Raises:
        ValueError: If the final sensor-gradient dimension is not 2 or if
            the model output does not have shape (B, p, 2).
    """
    grad_sensor_batch = jnp.asarray(
        grad_sensor_batch,
        dtype=jnp.float32,
    )

    # Add a batch dimension for a single sensor-gradient example.
    if grad_sensor_batch.ndim == 2:
        grad_sensor_batch = grad_sensor_batch[None, ...]

    B, P, C = grad_sensor_batch.shape

    if C != 2:
        raise ValueError(
            "Expected final dimension 2 for (g_x, g_y), "
            f"but got {grad_sensor_batch.shape}."
        )

    coords = _build_uniform_grid_coords(grid_size)
    p = coords.shape[0]

    # Flatten [g_x, g_y] values across all sensors for the branch network.
    branch_inputs = grad_sensor_batch.reshape(B, -1)

    # Reuse the same regular query grid for every batch element.
    coords_batch = jnp.broadcast_to(
        coords[None, :, :],
        (B, p, 2),
    )

    pred = apply_net_tasks(
        model_fn,
        params,
        branch_inputs,
        coords_batch,
        rng=rng,
    )

    if pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError(
            "DeepONet gradient-map output must have shape (B, p, 2), "
            f"but got {pred.shape}."
        )

    return pred.reshape(
        B,
        grid_size,
        grid_size,
        2,
    )


def train_deeponet_gradmap(
        cfg: DeepONetGradMapConfig,
        grad_sensor_noisy: np.ndarray,
        grad_grid_clean: np.ndarray,
        branch_grid_path: str = "1.csv",
):
    """
    Train a DeepONet to map noisy sensor gradients to clean grid gradients.

    This is intended as the first stage of a two-stage reconstruction pipeline:

        Noisy sensor gradients
            -> DeepONet
            -> clean gradients on a regular spatial grid

    Args:
        cfg: Training configuration.
        grad_sensor_noisy: Noisy sensor derivatives with shape
            (N, P_sensor, 2).
        grad_grid_clean: Clean regular-grid gradient targets with shape
            (N, p, 2), where p = grid_size * grid_size.
        branch_grid_path: CSV path defining the sensor layout expected by the
            branch network.

    Returns:
        Dictionary containing:

            model_fn:
                Trained ModernDeepONet module.

            params:
                Best parameter pytree according to validation error, when a
                test split is provided. Otherwise, the final parameters.

            last_params:
                Final parameter pytree after the last training step.

            best_e:
                Best full-grid validation relative L2 error.

            last_e:
                Most recently computed validation relative L2 error.

            branch_sensor_coords:
                Normalized sensor-coordinate array.

            cfg:
                Original training configuration.
    """
    key = jax.random.PRNGKey(int(cfg.seed))
    key_model, key_data = jax.random.split(key)

    # Verify that the supplied sensor-gradient data uses the same number of
    # sensors as the layout defined by the branch-grid CSV.
    branch_sensor_coords = load_branch_sensor_grid(
        branch_grid_path,
        use_flag=False,
        flip_y=True,
    )

    expected_p = int(branch_sensor_coords.shape[0])

    grad_sensor_noisy = np.asarray(
        grad_sensor_noisy,
        dtype=np.float32,
    )

    grad_grid_clean = np.asarray(
        grad_grid_clean,
        dtype=np.float32,
    )

    if (
            grad_sensor_noisy.ndim != 3
            or grad_sensor_noisy.shape[-1] != 2
    ):
        raise ValueError(
            "grad_sensor_noisy must have shape (N, P_sensor, 2), "
            f"but got {grad_sensor_noisy.shape}."
        )

    if grad_sensor_noisy.shape[1] != expected_p:
        raise ValueError(
            f"grad_sensor_noisy contains P={grad_sensor_noisy.shape[1]} "
            f"sensors, but {branch_grid_path} defines {expected_p} sensors."
        )

    grid_size = int(cfg.grid_size)
    p = grid_size * grid_size

    if grad_grid_clean.shape[1:] != (p, 2):
        raise ValueError(
            f"grad_grid_clean must have shape (N, {p}, 2) for "
            f"grid_size={grid_size}, but got {grad_grid_clean.shape}."
        )

    N_train = int(cfg.n_train)
    N_test = int(getattr(cfg, "n_test", 0))
    N_needed = N_train + N_test

    if (
            grad_sensor_noisy.shape[0] < N_needed
            or grad_grid_clean.shape[0] < N_needed
    ):
        raise ValueError(
            f"Not enough samples for the requested split: need {N_needed}, "
            f"got grad_sensor_noisy={grad_sensor_noisy.shape[0]} and "
            f"grad_grid_clean={grad_grid_clean.shape[0]}."
        )

    # Training split.
    grad_sensor_train = grad_sensor_noisy[:N_train]
    grad_grid_train = grad_grid_clean[:N_train]

    # Validation/test split.
    if N_test > 0:
        grad_sensor_test = grad_sensor_noisy[
                           N_train:N_train + N_test
                           ]

        grad_grid_test = grad_grid_clean[
                         N_train:N_train + N_test
                         ]
    else:
        grad_sensor_test = None
        grad_grid_test = None

    # Create pointwise DeepONet tasks over a shared uniform output grid.
    u_tasks = jnp.asarray(
        grad_sensor_train,
        dtype=jnp.float32,
    ).reshape(N_train, -1)

    coords = _build_uniform_grid_coords(grid_size)

    coords_tasks = jnp.broadcast_to(
        coords[None, :, :],
        (N_train, p, 2),
    )

    s_tasks = jnp.asarray(
        grad_grid_train,
        dtype=jnp.float32,
    )

    gen = DataGenerator(
        u_tasks,
        coords_tasks,
        s_tasks,
        int(cfg.batch_size),
        key_data,
    )

    it = iter(gen)

    model, params = setup_deeponet_gradmap(
        cfg,
        n_sensors=expected_p,
        key=key_model,
    )

    lr_scheduler = make_lr_schedule(
        lr=float(cfg.lr),
        steps=int(cfg.steps),
        scheduler=str(cfg.lr_scheduler),
        warmup_steps=cfg.warmup_steps,
        warmup_frac=float(cfg.warmup_frac),
        end_lr_factor=float(cfg.end_lr_factor),
    )

    optimizer = optax.adamw(
        learning_rate=lr_scheduler,
        weight_decay=float(cfg.weight_decay),
    )

    opt_state = optimizer.init(params)

    best_e = np.inf
    best_params = params
    last_e = np.nan

    pbar = tqdm.trange(int(cfg.steps))

    for step_i in pbar:
        batch = next(it)

        # Generate a distinct dropout key for this optimization step.
        key, drop_key = jax.random.split(key)

        loss_val, grad_norm, params, opt_state = step_supervised(
            optimizer,
            loss_deeponet_gradmap,
            model,
            opt_state,
            params,
            batch,
            drop_key,
        )

        # Compute full-grid validation error at the configured interval.
        if (
                grad_sensor_test is not None
                and N_test > 0
                and step_i % int(cfg.eval_iter) == 0
        ):
            last_e = eval_deeponet_gradmap_full_grid_error(
                model_fn=model,
                params=params,
                grad_sensor=grad_sensor_test,
                grad_grid_true=grad_grid_test,
                grid_size=grid_size,
                batch_size=int(cfg.eval_batch),
            )

            if last_e < best_e:
                best_e = last_e
                best_params = params

        # Display current training metrics in the progress bar.
        if step_i % int(cfg.log_iter) == 0:
            postfix = {
                "l": f"{float(loss_val):.2e}",
                "g": f"{float(grad_norm):.2e}",
            }

            if N_test > 0:
                postfix["e"] = f"{float(last_e):.2e}"
                postfix["best_e"] = f"{float(best_e):.2e}"

            pbar.set_postfix(postfix)

    return {
        "model_fn": model,

        # Use the best validation checkpoint for the next pipeline stage.
        "params": best_params if N_test > 0 else params,

        # Preserve the final checkpoint as well.
        "last_params": params,

        "best_e": best_e,
        "last_e": last_e,
        "branch_sensor_coords": branch_sensor_coords,
        "cfg": cfg,
    }


def eval_deeponet_gradmap_full_grid_error(
        model_fn,
        params,
        grad_sensor: np.ndarray,
        grad_grid_true: np.ndarray,
        grid_size: int = 24,
        batch_size: int = 128,
):
    """
    Evaluate full-grid DeepONet gradient-map reconstruction error.

    The evaluated mapping is:

        Noisy sensor gradients
            -> DeepONet
            -> clean regular-grid gradients

    Args:
        model_fn: Flax ModernDeepONet module.
        params: Model parameter pytree.
        grad_sensor: Sensor gradients with shape (N, P_sensor, 2).
        grad_grid_true: Target grid gradients with shape (N, p, 2), or
            (N, grid_size, grid_size, 2).
        grid_size: Number of grid points along each spatial axis.
        batch_size: Number of complete functions evaluated per batch.

    Returns:
        Mean relative L2 error across all functions:

            mean(||G_pred - G_true||_2 / ||G_true||_2)

    Raises:
        ValueError: If the target gradient array does not match one of the
            supported regular-grid layouts.
    """
    grad_sensor = np.asarray(
        grad_sensor,
        dtype=np.float32,
    )

    grad_grid_true = np.asarray(
        grad_grid_true,
        dtype=np.float32,
    )

    N = grad_sensor.shape[0]
    p = grid_size * grid_size

    # Flatten a grid-shaped target representation when necessary.
    if grad_grid_true.ndim == 4:
        grad_grid_true = grad_grid_true.reshape(N, p, 2)

    if grad_grid_true.shape != (N, p, 2):
        raise ValueError(
            f"grad_grid_true must have shape (N, {p}, 2) or "
            f"(N, {grid_size}, {grid_size}, 2), but got "
            f"{grad_grid_true.shape}."
        )

    errs = []

    # Evaluate in batches to limit memory use for large test sets.
    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)

        pred = deeponet_predict_grad_grid(
            model_fn=model_fn,
            params=params,
            grad_sensor_batch=grad_sensor[start:end],
            grid_size=grid_size,
            rng=None,
        )

        pred = np.asarray(
            pred,
            dtype=np.float32,
        ).reshape(end - start, p, 2)

        true = grad_grid_true[start:end]

        diff = pred - true

        # Compute one relative L2 error per full gradient field.
        diff_norm = np.sqrt(
            np.sum(diff ** 2, axis=(1, 2))
        )

        true_norm = np.sqrt(
            np.sum(true ** 2, axis=(1, 2))
        ) + 1e-8

        batch_err = diff_norm / true_norm
        errs.append(batch_err)

    return float(np.mean(np.concatenate(errs)))
