from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

from wavefront.metamodel.gradmap import deeponet_predict_grad_grid
from wavefront.training.common import _tree_l2_norm
from wavefront.training.fno import apply_fno, rel_l2_field_loss
from wavefront.training.schedules import make_lr_schedule


@partial(jax.jit, static_argnums=(0, 1, 2, 7))
def step_joint(
        optimizer,
        model_deeponet,
        model_fno,
        opt_state,
        params_step,
        batch,
        rng,
        grid_size: int,
):
    """
    Perform one end-to-end optimization step for a DeepONet–FNO pipeline.

    The joint model follows this sequence:

        Noisy sensor gradients
            -> DeepONet
            -> predicted regular-grid gradients
            -> FNO
            -> reconstructed wavefront

    The loss is computed only on the final reconstructed wavefront. Gradients
    therefore propagate backward through the FNO and then through the
    DeepONet, updating both parameter trees simultaneously.

    Args:
        optimizer: Optax optimizer transformation for the combined parameter
            pytree.
        model_deeponet: Flax DeepONet model that maps sensor gradients to
            regular-grid gradient fields.
        model_fno: Flax FNO model that maps grid gradients to wavefronts.
        opt_state: Current Optax state for both parameter groups.
        params_step: Nested parameter pytree with the structure:

            {
                "deeponet": deeponet_params,
                "fno": fno_params,
            }

        batch: Tuple containing:

            grad_sensor:
                Noisy sensor gradients with shape (B, P_sensor, 2).

            wavefront_true:
                Ground-truth wavefronts with shape
                (B, grid_size, grid_size).

        rng: JAX PRNG key used for FNO dropout during training.
        grid_size: Number of regular-grid points along each spatial axis.

    Returns:
        A tuple containing:

            loss:
                Scalar relative L2 wavefront reconstruction loss.

            grad_norm:
                Global L2 norm of the nested gradient pytree.

            params_step:
                Updated DeepONet and FNO parameters.

            opt_state:
                Updated optimizer state.
    """

    def _loss(params):
        grad_sensor, wavefront_true = batch

        # Stage 1: map irregular noisy sensor gradients onto a regular grid.
        grad_grid_pred = deeponet_predict_grad_grid(
            model_fn=model_deeponet,
            params=params["deeponet"],
            grad_sensor_batch=grad_sensor,
            grid_size=grid_size,
            rng=None,
        )

        # Stage 2: reconstruct the scalar wavefront from predicted gradients.
        wavefront_pred = apply_fno(
            model_fno,
            params["fno"],
            grad_grid_pred,
            rng=rng,
            training=True,
        )

        # Remove a singleton scalar output-channel dimension when present:
        #
        # (B, H, W, 1) -> (B, H, W)
        if wavefront_pred.ndim == 4 and wavefront_pred.shape[-1] == 1:
            wavefront_pred = wavefront_pred[..., 0]

        # Support flattened FNO outputs when needed:
        #
        # (B, H * W) -> (B, H, W)
        if wavefront_pred.ndim == 2:
            wavefront_pred = wavefront_pred.reshape(
                wavefront_pred.shape[0],
                grid_size,
                grid_size,
            )

        # Supervise only the final wavefront reconstruction.
        return rel_l2_field_loss(
            wavefront_true,
            wavefront_pred,
        )

    loss, grads = jax.value_and_grad(_loss)(params_step)

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


def eval_joint_deeponet_fno_error(
        model_deeponet,
        model_fno,
        params,
        grad_sensor: np.ndarray,
        wavefront_true: np.ndarray,
        grid_size: int = 24,
        batch_size: int = 32,
):
    """
    Evaluate the complete DeepONet-to-FNO reconstruction pipeline.

    The evaluated mapping is:

        Noisy sensor gradients
            -> DeepONet gradient-map prediction
            -> FNO wavefront reconstruction

    The returned metric is the mean relative L2 error over complete wavefront
    fields:

        mean(||U_pred - U_true||_2 / ||U_true||_2)

    Args:
        model_deeponet: Flax DeepONet gradient-map model.
        model_fno: Flax FNO wavefront-reconstruction model.
        params: Nested parameter pytree containing ``"deeponet"`` and
            ``"fno"`` subtrees.
        grad_sensor: Sensor gradients with shape (N, P_sensor, 2).
        wavefront_true: Ground-truth wavefronts with shape (N, p) or
            (N, grid_size, grid_size).
        grid_size: Number of points along each output-grid dimension.
        batch_size: Number of full wavefront functions evaluated at once.

    Returns:
        Mean relative L2 reconstruction error across all supplied samples.
    """
    grad_sensor = np.asarray(
        grad_sensor,
        dtype=np.float32,
    )

    wavefront_true = np.asarray(
        wavefront_true,
        dtype=np.float32,
    )

    # Convert flattened wavefront targets to regular-grid form when needed.
    if wavefront_true.ndim == 2:
        wavefront_true = wavefront_true.reshape(
            wavefront_true.shape[0],
            grid_size,
            grid_size,
        )

    n_samples = grad_sensor.shape[0]
    errors = []

    # Evaluate in batches to bound device and host memory usage.
    for start in range(0, n_samples, batch_size):
        end = min(n_samples, start + batch_size)

        grad_batch = jnp.asarray(
            grad_sensor[start:end],
            dtype=jnp.float32,
        )

        true_batch = np.asarray(
            wavefront_true[start:end],
            dtype=np.float32,
        )

        # Predict regular-grid gradients with the DeepONet.
        grad_grid_pred = deeponet_predict_grad_grid(
            model_fn=model_deeponet,
            params=params["deeponet"],
            grad_sensor_batch=grad_batch,
            grid_size=grid_size,
            rng=None,
        )

        # Reconstruct wavefronts using deterministic FNO inference.
        wavefront_pred = apply_fno(
            model_fno,
            params["fno"],
            grad_grid_pred,
            rng=None,
            training=False,
        )

        pred = np.asarray(
            wavefront_pred,
            dtype=np.float32,
        )

        # Remove the optional scalar output channel:
        #
        # (B, H, W, 1) -> (B, H, W)
        if pred.ndim == 4 and pred.shape[-1] == 1:
            pred = pred[..., 0]

        # Restore grid structure for flattened predictions:
        #
        # (B, H * W) -> (B, H, W)
        if pred.ndim == 2:
            pred = pred.reshape(
                pred.shape[0],
                grid_size,
                grid_size,
            )

        difference = pred - true_batch

        # Compute one relative L2 error per complete wavefront.
        difference_norm = np.sqrt(
            np.sum(difference ** 2, axis=(1, 2))
        )

        true_norm = (
                np.sqrt(
                    np.sum(true_batch ** 2, axis=(1, 2))
                )
                + 1e-8
        )

        errors.append(difference_norm / true_norm)

    return float(np.mean(np.concatenate(errors)))


def finetune_joint_deeponet_fno(
        fno_state: dict,
        deeponet_state: dict,
        grad_sensor_noisy: np.ndarray,
        wavefront_true: np.ndarray,
        grad_sensor_test: np.ndarray | None = None,
        wavefront_test: np.ndarray | None = None,
        grid_size: int = 24,
        batch_size: int = 32,
        steps: int = 10_000,
        lr: float | None = None,
        weight_decay: float = 1e-6,
        seed: int = 0,
):
    """
    Jointly fine-tune pretrained DeepONet and FNO models end to end.

    DeepONet starts from the Stage-1 gradient-map solution, while the FNO
    starts from the Stage-2 wavefront-reconstruction solution. The DeepONet
    receives a learning rate ten times lower than the FNO so its learned
    sensor-to-grid mapping is adjusted conservatively during end-to-end
    training.

    Pipeline:

        Noisy sensor gradients
            -> pretrained DeepONet
            -> regular-grid gradient prediction
            -> pretrained FNO
            -> wavefront reconstruction loss

    Args:
        fno_state: Dictionary returned by Stage-2 FNO training. It must
            include ``model_fn``, ``params``, and ``args``.
        deeponet_state: Dictionary returned by Stage-1 DeepONet gradient-map
            training. It must include ``model_fn`` and ``params``.
        grad_sensor_noisy: Training sensor gradients with shape
            (N, P_sensor, 2).
        wavefront_true: Training wavefront targets with shape (N, p) or
            (N, grid_size, grid_size).
        grad_sensor_test: Optional held-out sensor-gradient samples.
        wavefront_test: Optional held-out wavefront targets. It must be
            supplied whenever grad_sensor_test is supplied.
        grid_size: Number of points along each regular-grid axis.
        batch_size: Number of wavefront functions sampled per optimization
            step.
        steps: Number of joint fine-tuning iterations.
        lr: Optional FNO peak learning rate. When None, the value is read from
            ``fno_state["args"].lr``. DeepONet uses one tenth of this rate.
        weight_decay: AdamW weight decay applied to both parameter groups.
        seed: JAX random seed used for mini-batch selection and dropout keys.

    Returns:
        Dictionary containing:

            deeponet:
                Updated DeepONet state with selected parameters.

            fno:
                Updated FNO state with selected parameters.

            fno_lr:
                Learning rate used for the FNO optimizer branch.

            deeponet_lr:
                Learning rate used for the DeepONet optimizer branch.

            best_e:
                Best held-out relative L2 error, or infinity when no
                validation data was supplied.

            last_e:
                Most recently computed held-out error, or NaN when no
                validation data was supplied.

            best_params:
                Selected nested parameter pytree.

            last_params:
                Final nested parameter pytree after the final update.

    Raises:
        ValueError: If input arrays have incompatible shapes, or only one of
            grad_sensor_test and wavefront_test is supplied.
    """
    grad_sensor_noisy = np.asarray(
        grad_sensor_noisy,
        dtype=np.float32,
    )

    wavefront_true = np.asarray(
        wavefront_true,
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

    # Support flattened wavefront targets.
    if wavefront_true.ndim == 2:
        wavefront_true = wavefront_true.reshape(
            wavefront_true.shape[0],
            grid_size,
            grid_size,
        )

    expected_wavefront_shape = (
        grad_sensor_noisy.shape[0],
        grid_size,
        grid_size,
    )

    if wavefront_true.shape != expected_wavefront_shape:
        raise ValueError(
            "wavefront_true must have shape "
            f"{expected_wavefront_shape}, but got {wavefront_true.shape}."
        )

    # Validation is enabled only when both held-out arrays are supplied.
    has_validation = (
            grad_sensor_test is not None
            and wavefront_test is not None
    )

    if (grad_sensor_test is None) != (wavefront_test is None):
        raise ValueError(
            "grad_sensor_test and wavefront_test must either both be "
            "provided or both be None."
        )

    if has_validation:
        grad_sensor_test = np.asarray(
            grad_sensor_test,
            dtype=np.float32,
        )

        wavefront_test = np.asarray(
            wavefront_test,
            dtype=np.float32,
        )

        # Support flattened held-out wavefront arrays.
        if wavefront_test.ndim == 2:
            wavefront_test = wavefront_test.reshape(
                wavefront_test.shape[0],
                grid_size,
                grid_size,
            )

        if (
                grad_sensor_test.ndim != 3
                or grad_sensor_test.shape[-1] != 2
        ):
            raise ValueError(
                "grad_sensor_test must have shape (N, P_sensor, 2), "
                f"but got {grad_sensor_test.shape}."
            )

        if wavefront_test.shape != (
                grad_sensor_test.shape[0],
                grid_size,
                grid_size,
        ):
            raise ValueError(
                "wavefront_test must have shape "
                f"(N, {grid_size}, {grid_size}), "
                f"but got {wavefront_test.shape}."
            )

    # Use separate random streams for training-batch sampling and FNO dropout.
    key = jax.random.PRNGKey(int(seed))
    key_data, key_opt = jax.random.split(key)

    n_train = grad_sensor_noisy.shape[0]

    grad_sensor_jax = jnp.asarray(
        grad_sensor_noisy,
        dtype=jnp.float32,
    )

    wavefront_jax = jnp.asarray(
        wavefront_true,
        dtype=jnp.float32,
    )

    @partial(jax.jit, static_argnums=(1,))
    def _sample_batch(
            key,
            current_batch_size: int,
    ):
        """
        Randomly sample complete wavefront functions with replacement.
        """
        indices = jax.random.randint(
            key,
            shape=(current_batch_size,),
            minval=0,
            maxval=n_train,
        )

        return (
            grad_sensor_jax[indices],
            wavefront_jax[indices],
        )

    # Combine both parameter pytrees so Optax can update them jointly.
    params = {
        "deeponet": deeponet_state["params"],
        "fno": fno_state["params"],
    }

    # Read the FNO training learning rate unless the caller overrides it.
    base_lr = float(
        getattr(fno_state["args"], "lr", 1e-3)
    )

    if lr is None:
        fno_lr = base_lr
        deeponet_lr = base_lr / 10.0
    else:
        fno_lr = float(lr)
        deeponet_lr = float(lr) / 10.0

    # Use five percent of joint-training steps for linear warmup.
    warmup_steps = max(
        1,
        int(0.05 * int(steps)),
    )

    deeponet_schedule = make_lr_schedule(
        lr=deeponet_lr,
        steps=int(steps),
        scheduler="warmup_cosine",
        warmup_steps=warmup_steps,
        end_lr_factor=0.01,
    )

    fno_schedule = make_lr_schedule(
        lr=fno_lr,
        steps=int(steps),
        scheduler="warmup_cosine",
        warmup_steps=warmup_steps,
        end_lr_factor=0.01,
    )

    # Assign independent AdamW schedules to each model parameter subtree.
    transforms = {
        "deeponet": optax.adamw(
            learning_rate=deeponet_schedule,
            weight_decay=float(weight_decay),
        ),
        "fno": optax.adamw(
            learning_rate=fno_schedule,
            weight_decay=float(weight_decay),
        ),
    }

    parameter_labels = {
        "deeponet": jax.tree_util.tree_map(
            lambda _: "deeponet",
            params["deeponet"],
        ),
        "fno": jax.tree_util.tree_map(
            lambda _: "fno",
            params["fno"],
        ),
    }

    optimizer = optax.multi_transform(
        transforms,
        parameter_labels,
    )

    opt_state = optimizer.init(params)

    model_deeponet = deeponet_state["model_fn"]
    model_fno = fno_state["model_fn"]

    best_error = np.inf
    last_error = np.nan
    best_params = params

    progress_bar = tqdm.trange(int(steps))

    for step_index in progress_bar:
        # Randomly choose a batch of complete sensor-gradient/wavefront pairs.
        key_data, sample_key = jax.random.split(key_data)

        batch = _sample_batch(
            sample_key,
            int(batch_size),
        )

        # Generate the FNO dropout key for this update.
        key_opt, dropout_key = jax.random.split(key_opt)

        loss_value, grad_norm, params, opt_state = step_joint(
            optimizer,
            model_deeponet,
            model_fno,
            opt_state,
            params,
            batch,
            dropout_key,
            int(grid_size),
        )

        # Evaluate the complete two-stage model every 1,000 steps when held-out
        # data is available.
        if has_validation and step_index % 1000 == 0:
            last_error = eval_joint_deeponet_fno_error(
                model_deeponet=model_deeponet,
                model_fno=model_fno,
                params=params,
                grad_sensor=grad_sensor_test,
                wavefront_true=wavefront_test,
                grid_size=grid_size,
                batch_size=batch_size,
            )

            if last_error < best_error:
                best_error = last_error
                best_params = params

        # Show optimization metrics every 250 iterations.
        if step_index % 250 == 0:
            postfix = {
                "loss": f"{float(loss_value):.2e}",
                "grad_norm": f"{float(grad_norm):.2e}",
            }

            if has_validation:
                postfix["error"] = f"{float(last_error):.2e}"
                postfix["best_error"] = (
                    f"{float(best_error):.2e}"
                )

            progress_bar.set_postfix(postfix)

    # Without held-out data, only the final parameters can be selected.
    selected_params = best_params if has_validation else params

    # Copy state dictionaries before replacing parameter references.
    deeponet_state = dict(deeponet_state)
    fno_state = dict(fno_state)

    # Expose selected parameters as the default model state.
    deeponet_state["params"] = selected_params["deeponet"]
    fno_state["params"] = selected_params["fno"]

    # Preserve the last-step checkpoint separately.
    deeponet_state["last_params"] = params["deeponet"]
    fno_state["last_params"] = params["fno"]

    return {
        "deeponet": deeponet_state,
        "fno": fno_state,
        "fno_lr": fno_lr,
        "deeponet_lr": deeponet_lr,
        "best_e": best_error,
        "last_e": last_error,
        "best_params": selected_params,
        "last_params": params,
    }
