import json
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
import orbax.checkpoint.args as ocp_args
import tqdm

from wavefront.data.batching import DataGenerator
from wavefront.data.deeponet_data import (
    generate_one_res_training_data,
    make_supervised_task,
)
from wavefront.data.sensors import load_branch_sensor_grid
from wavefront.evaluation.deeponet import get_error
from wavefront.evaluation.timing import benchmark_inference_deeponet
from wavefront.evaluation.visualization import visualize
from wavefront.models.factory import setup_deeponet
from wavefront.training.deeponet import step
from wavefront.training.deeponet_losses import (
    loss_fn,
    loss_ics,
    loss_res,
)
from wavefront.training.precision import set_mixed_precision


def main_routine_deeponet(
        args,
        grad_sensor_data=None,
        wavefront_true_data=None,
):
    """
    Train a ModernDeepONet model for wavefront reconstruction.

    The model learns to reconstruct a scalar wavefront field from sensor-space
    gradient measurements. Training combines:

        - A supervised wavefront-value loss on regular query coordinates.
        - A gradient-matching residual loss at sensor coordinates.

    Args:
        args: Configuration object containing data, model, optimizer, training,
            visualization, checkpointing, and output settings.
        grad_sensor_data: Optional sensor-gradient data with shape
            (N, P_sensor, 2). When None, data is loaded from
            ``data/derivatives.npy``.
        wavefront_true_data: Optional wavefront target data. When None, data
            is loaded from ``data/U_true.npy``.

    Returns:
        Dictionary containing the trained model, selected best parameters,
        test data, sensor coordinates, configuration, and output directory.

    Notes:
        The best model is selected according to mean relative L2 error over
        the test split at logging iterations.
    """
    # Configure global mixed-precision behavior before model initialization.
    set_mixed_precision(
        enabled=bool(getattr(args, "mixed_precision", False)),
        dtype=str(getattr(args, "mp_dtype", "bfloat16")),
    )

    main_key = jax.random.PRNGKey(args.seed)

    # This training routine supports the standard DeepONet only.
    if args.separable:
        raise ValueError(
            "This routine requires a standard DeepONet, not a separable DeepONet."
        )

    # Load supplied arrays or fall back to the default on-disk dataset files.
    grad_sensor_all = (
        jnp.asarray(grad_sensor_data)
        if grad_sensor_data is not None
        else jnp.load("data/derivatives.npy")
    )

    wavefront_true_all = (
        jnp.asarray(wavefront_true_data, dtype=jnp.float32)
        if wavefront_true_data is not None
        else jnp.load("data/U_true.npy")
    )

    # Remove a trailing singleton output-channel dimension when it appears in
    # one of the expected input layouts.
    for drop in [(4, 1), (3, 1), (3,)]:
        if (
                wavefront_true_all.ndim == len(drop)
                and wavefront_true_all.shape[-1] == 1
        ):
            wavefront_true_all = wavefront_true_all[..., 0]

    # Flatten regular-grid targets into one wavefront vector per sample.
    if wavefront_true_all.ndim == 3:
        wavefront_true_all = wavefront_true_all.reshape(
            wavefront_true_all.shape[0],
            -1,
        )

    if wavefront_true_all.ndim != 2:
        raise ValueError(
            f"Unexpected wavefront shape: {wavefront_true_all.shape}"
        )

    N, P_data, _ = grad_sensor_all.shape

    # Determine the coordinate layout associated with the branch input.
    #
    # In regular-grid mode, input gradients are expected at every point of a
    # regular grid. In sensor mode, physical sensor coordinates are loaded
    # from the configured CSV file.
    if getattr(args, "data_mode", "sensor") in ("regular_grid", "grid"):
        g = int(args.grid_size)

        xy = jnp.linspace(
            -1.0,
            1.0,
            g,
            dtype=jnp.float32,
        )

        xg, yg = jnp.meshgrid(xy, xy)

        branch_sensor_coords = jnp.stack(
            [xg.ravel(), yg.ravel()],
            axis=1,
        )

        P_branch = branch_sensor_coords.shape[0]

        if P_data != P_branch:
            raise ValueError(
                f"DeepONet regular_grid expects P={P_branch}, "
                f"but received P={P_data}."
            )
    else:
        branch_sensor_coords = load_branch_sensor_grid(
            args.branch_grid_path,
            use_flag=False,
            flip_y=True,
        )

        P_branch = branch_sensor_coords.shape[0]

        if P_data != P_branch:
            raise ValueError(
                f"Sensor count mismatch: gradients contain {P_data} points, "
                f"while the CSV layout contains {P_branch}."
            )

    P = P_branch
    n_train, n_test = args.n_train, args.n_test

    if N < n_train + n_test:
        raise ValueError(
            f"Need at least {n_train + n_test} samples, but received {N}."
        )

    # No validation split is used. The first n_train samples are used for
    # training, followed by n_test examples for testing.
    grad_sensor_train = grad_sensor_all[:n_train]
    wavefront_true_train = jnp.asarray(
        wavefront_true_all[:n_train]
    )

    grad_sensor_test = grad_sensor_all[
                       n_train:n_train + n_test
                       ]
    wavefront_true_test = jnp.asarray(
        wavefront_true_all[n_train:n_train + n_test]
    )

    # Create independent random streams for pointwise batches, visualization,
    # model initialization, and other training operations.
    key = jax.random.PRNGKey(args.seed)

    key_ics, key_res, key_test, key_model, key_plot = jax.random.split(
        key,
        5,
    )

    # Construct regular-grid supervised tasks for wavefront-value training.
    u_ics, y_ics, s_ics = jax.vmap(
        make_supervised_task,
        in_axes=(0, 0, None),
    )(
        grad_sensor_train,
        wavefront_true_train,
        args.p_ics_train,
    )

    # Construct sensor-resolution tasks for gradient-matching residual training.
    sensor_coords_jax = jnp.asarray(
        branch_sensor_coords,
        dtype=jnp.float32,
    )

    u_res, y_res, s_res = jax.vmap(
        generate_one_res_training_data,
        in_axes=(0, None),
    )(
        grad_sensor_train,
        sensor_coords_jax,
    )

    # Create random pointwise batch generators for the supervised and
    # gradient-matching loss terms.
    ics_dataset = DataGenerator(
        u_ics,
        y_ics,
        s_ics,
        args.batch_size,
        key_ics,
    )

    res_dataset = DataGenerator(
        u_res,
        y_res,
        s_res,
        args.batch_size,
        key_res,
    )

    test_idx_all = jnp.arange(n_test)

    # Select test examples for periodic visualizations.
    n_vis_test = int(
        min(
            getattr(args, "n_vis_test", 1),
            n_test,
        )
    )

    test_idx_vis = jax.random.permutation(
        key_test,
        test_idx_all,
    )[:n_vis_test]

    # Allow explicit visualization indices to override random selection.
    sample_idx_vis = getattr(args, "sample_idx_vis", None)

    if sample_idx_vis is not None:
        if isinstance(sample_idx_vis, (int, np.integer)):
            test_idx_vis = jnp.array(
                [int(sample_idx_vis)],
                dtype=int,
            )
        else:
            test_idx_vis = jnp.array(
                list(sample_idx_vis),
                dtype=int,
            )

    # Keep architecture settings synchronized with the prepared input data.
    args.n_sensors = P
    args.branch_input_features = 2
    args.trunk_input_features = 2

    # Build and initialize the DeepONet model.
    args, model, model_fn, params = setup_deeponet(
        args,
        key_model,
    )

    # Keep a separate evaluation-model alias for compatibility with helper
    # functions that expect this name.
    model_eval_fn = model

    # Configure the requested learning-rate schedule.
    if args.lr_scheduler == "exponential_decay":
        lr_scheduler = optax.exponential_decay(
            args.lr,
            args.lr_schedule_steps,
            args.lr_decay_rate,
        )

    elif args.lr_scheduler == "constant":
        lr_scheduler = optax.constant_schedule(args.lr)

    elif args.lr_scheduler == "cosine":
        lr_scheduler = optax.cosine_decay_schedule(
            init_value=args.lr,
            decay_steps=args.steps,
            alpha=getattr(args, "lr_alpha", 0.05),
        )

    elif args.lr_scheduler == "warmup_cosine":
        warmup_steps = int(
            getattr(args, "warmup_steps", 1000)
        )
        alpha = float(
            getattr(args, "lr_alpha", 0.05)
        )

        warmup = optax.linear_schedule(
            init_value=0.0,
            end_value=args.lr,
            transition_steps=warmup_steps,
        )

        cosine = optax.cosine_decay_schedule(
            init_value=args.lr,
            decay_steps=max(args.steps - warmup_steps, 1),
            alpha=alpha,
        )

        lr_scheduler = optax.join_schedules(
            schedules=[warmup, cosine],
            boundaries=[warmup_steps],
        )

    else:
        raise ValueError(
            f"Unknown lr_scheduler: {args.lr_scheduler}"
        )

    # Initialize AdamW optimizer state.
    optimizer = optax.adamw(
        learning_rate=lr_scheduler,
        weight_decay=getattr(args, "weight_decay", 1e-4),
    )

    opt_state = optimizer.init(params)

    # Create a timestamped results directory for logs, figures, checkpoints,
    # trained parameters, and benchmark results.
    result_dir = os.path.join(
        os.getcwd(),
        args.result_dir,
        time.strftime("%Y%m%d-%H%M%S"),
    )

    os.makedirs(result_dir, exist_ok=True)

    # Configure checkpoint management when checkpointing is enabled.
    if args.checkpoint_iter > 0:
        options = ocp.CheckpointManagerOptions(
            max_to_keep=args.checkpoints_to_keep,
            save_interval_steps=args.checkpoint_iter,
            save_on_steps=[args.steps],
        )

        mngr = ocp.CheckpointManager(
            os.path.join(result_dir, "ckpt"),
            options=options,
            item_names=("params", "opt_state"),
        )
    else:
        mngr = None

    # Initialize data iterators and CSV training log.
    ics_iter = iter(ics_dataset)
    res_iter = iter(res_dataset)

    pbar = tqdm.trange(args.steps)

    log_file = open(
        os.path.join(result_dir, "log.csv"),
        "w",
    )

    log_file.write(
        "iter,loss,l_ic,l_r,res_weight,grad_norm,test_err,is_best\n"
    )

    # Store history for training curves.
    hist_iters = []
    hist_l_ic = []
    hist_l_r = []
    hist_l_rwr = []
    hist_grad_norm = []

    res_weight = args.res_weight0

    # Track parameters with the best mean test relative L2 error.
    best_test_err = float("inf")
    best_params = None
    best_it = -1

    for it in pbar:
        # Draw independent random pointwise supervised and residual batches.
        batch_ics = next(ics_iter)
        batch_res = next(res_iter)

        # Generate a dropout key for this optimization step.
        main_key, dropout_key, _ = jax.random.split(
            main_key,
            3,
        )

        # Perform one parameter-update step.
        loss_val, grad_norm, params, opt_state = step(
            optimizer,
            loss_fn,
            model_fn,
            opt_state,
            params,
            batch_ics,
            batch_res,
            res_weight,
            dropout_key,
        )

        # Log detailed losses and evaluate the current model periodically.
        if it % args.log_iter == 0:
            l_ic = loss_ics(
                model_fn,
                params,
                batch_ics,
                dropout_key,
            )

            l_r = loss_res(
                model_fn,
                params,
                batch_res,
                dropout_key,
            )

            test_err = get_error(
                model_eval_fn,
                params,
                grad_sensor_test,
                wavefront_true_test,
                test_idx_all,
                args.p_test,
            )

            # Aggregate per-sample errors into a mean test-set metric.
            if isinstance(test_err, jnp.ndarray):
                test_err = float(jnp.mean(test_err))
            else:
                test_err = float(test_err)

            is_best = 0

            # Preserve the current parameter pytree whenever the test metric
            # improves.
            if test_err < best_test_err:
                best_test_err = test_err
                best_params = params
                best_it = int(it)
                is_best = 1

            log_file.write(
                f"{it},{float(loss_val)},{float(l_ic)},{float(l_r)},"
                f"{float(res_weight)},{float(grad_norm)},"
                f"{test_err},{is_best}\n"
            )

            log_file.flush()

            pbar.set_postfix(
                {
                    "l": f"{loss_val:.2e}",
                    "l_ic": f"{l_ic:.2e}",
                    "l_r": f"{l_r:.2e}",
                    "test": f"{test_err:.2e}",
                }
            )

            hist_iters.append(it)
            hist_l_ic.append(float(l_ic))
            hist_l_r.append(float(l_r))
            hist_l_rwr.append(
                float(l_r) * float(res_weight)
            )
            hist_grad_norm.append(float(grad_norm))

        # Save visual comparisons at the requested interval.
        if (
                args.vis_iter > 0
                and (
                it == 1
                or (it + 1) % args.vis_iter == 0
        )
        ):
            for idx_vis in test_idx_vis:
                visualize(
                    args,
                    model_eval_fn,
                    params,
                    result_dir,
                    it,
                    grad_sensor_test,
                    wavefront_true_test,
                    int(idx_vis),
                    test=True,
                )

        # Save a checkpoint periodically when checkpoint management is enabled.
        if (it + 1) % args.checkpoint_iter == 0 and mngr is not None:
            mngr.save(
                it + 1,
                args=ocp_args.Composite(
                    params=ocp_args.StandardSave(params),
                    opt_state=ocp_args.StandardSave(opt_state),
                ),
            )

    log_file.close()

    # Save supervised, residual, weighted-residual loss curves.
    plt.figure(figsize=(8, 5))

    plt.plot(hist_iters, hist_l_ic, label="L_ic")
    plt.plot(
        hist_iters,
        hist_l_r,
        label="L_r (unweighted)",
    )
    plt.plot(
        hist_iters,
        hist_l_rwr,
        label="L_r * w_res",
    )

    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(result_dir, "loss_curves.png"),
        dpi=150,
    )

    plt.close()

    # Save the global gradient-norm history.
    plt.figure(figsize=(8, 5))

    plt.plot(
        hist_iters,
        hist_grad_norm,
        label="||grad||_2",
    )

    plt.xlabel("iteration")
    plt.ylabel("grad norm")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(result_dir, "grad_norm_curve.png"),
        dpi=150,
    )

    plt.close()

    def _jsonable(x):
        """
        Convert common Python, NumPy, and JAX objects into JSON-compatible
        values for configuration export.
        """
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x

        if isinstance(x, (list, tuple)):
            return [_jsonable(v) for v in x]

        if isinstance(x, dict):
            return {
                str(k): _jsonable(v)
                for k, v in x.items()
            }

        if isinstance(x, (np.ndarray, jnp.ndarray)):
            return np.asarray(x).tolist()

        return str(x)

    args_dump = {
        k: _jsonable(v)
        for k, v in vars(args).items()
    }

    # Save final and best parameter trees together with inference metadata.
    artifacts_dir = os.path.join(
        result_dir,
        "artifacts",
    )

    os.makedirs(artifacts_dir, exist_ok=True)

    params_last = params
    params_best = (
        best_params
        if best_params is not None
        else params
    )

    # Expose the best model parameters as the main output parameter tree.
    params = params_best

    ckp = ocp.StandardCheckpointer()

    ckp.save(
        os.path.join(artifacts_dir, "params_last"),
        params_last,
        force=True,
    )

    ckp.save(
        os.path.join(artifacts_dir, "params_best"),
        params_best,
        force=True,
    )

    ckp.save(
        os.path.join(artifacts_dir, "params"),
        params_best,
        force=True,
    )

    with open(
            os.path.join(
                artifacts_dir,
                "best_on_test.json",
            ),
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            {
                "best_it": int(best_it),
                "best_test_err": float(best_test_err),
            },
            f,
            indent=2,
        )

    with open(
            os.path.join(
                artifacts_dir,
                "args_inference.json",
            ),
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            args_dump,
            f,
            ensure_ascii=False,
            indent=2,
        )

    np.save(
        os.path.join(
            artifacts_dir,
            "branch_sensor_coords.npy",
        ),
        np.asarray(branch_sensor_coords),
    )

    # Benchmark inference using the selected best parameters.
    bench = {}

    try:
        bench["deeponet"] = benchmark_inference_deeponet(
            model_fn=model_eval_fn,
            params=params,
            grad_sensor_test=np.asarray(grad_sensor_test),
            p_test=int(args.p_test),
            n=int(getattr(args, "benchmark_n", 1000)),
            batch_size=int(
                getattr(args, "benchmark_batch", 64)
            ),
        )
    except Exception as e:
        bench["deeponet_error"] = str(e)

    with open(
            os.path.join(
                artifacts_dir,
                "inference_benchmark.json",
            ),
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            bench,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Store the trained state globally for notebook-based workflows and return
    # it to the caller for direct downstream use.
    trained_state = {
        "operator_type": "deeponet",
        "args": args,
        "params": params,
        "model": model,
        "model_fn": model_fn,
        "model_eval_fn": model_eval_fn,
        "result_dir": result_dir,
        "branch_sensor_coords": branch_sensor_coords,
        "grad_sensor_test": grad_sensor_test,
        "wavefront_true_test": wavefront_true_test,
    }

    globals()["TRAINED_STATE"] = trained_state

    return trained_state
