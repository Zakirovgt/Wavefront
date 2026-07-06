import json
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
import tqdm

from wavefront.data.batching import GridDataGenerator
from wavefront.data.fno_inputs import prepare_fno_arrays
from wavefront.data.sensors import load_branch_sensor_grid
from wavefront.evaluation.timing import benchmark_inference_fno
from wavefront.evaluation.visualization import _visualize_fno
from wavefront.models.factory import setup_fno
from wavefront.training.fno import (
    apply_fno,
    rel_l2_field_loss,
    step_fno,
)
from wavefront.training.fno_losses import (
    fno_rel_l2,
    loss_fno,
    loss_res_fno,
)
from wavefront.training.precision import set_mixed_precision


def main_routine_fno(
        args,
        grad_sensor_data=None,
        wavefront_true_data=None,
        grad_grid_data=None,
):
    """
    Train a Fourier Neural Operator for wavefront reconstruction.

    The FNO reconstructs a scalar wavefront field on a regular spatial grid
    using either:

        - Regular-grid gradient inputs.
        - Sensor gradients interpolated onto a regular grid.

    Training uses a supervised relative L2 reconstruction loss and can
    optionally include a gradient-matching residual loss.

    Args:
        args: Configuration object containing data, model, optimizer,
            visualization, and output settings.
        grad_sensor_data: Optional irregular sensor-gradient data, typically
            with shape (N, P_sensor, 2).
        wavefront_true_data: Optional ground-truth wavefront data.
        grad_grid_data: Optional regular-grid gradient fields. Required when
            args.data_mode is set to "regular_grid".

    Returns:
        Dictionary containing the trained FNO, selected best parameters,
        test data, optional sensor coordinates, configuration, and result path.

    Notes:
        The best parameter tree is selected according to mean test relative
        L2 error measured at logging iterations.
    """
    # Configure global mixed-precision behavior before model initialization.
    set_mixed_precision(
        enabled=bool(getattr(args, "mixed_precision", False)),
        dtype=str(getattr(args, "mp_dtype", "bfloat16")),
    )

    main_key = jax.random.PRNGKey(args.seed)

    # Sensor coordinates are needed only when sensor-space derivatives must be
    # interpolated onto the regular grid required by the FNO.
    branch_sensor_coords = (
        load_branch_sensor_grid(
            args.branch_grid_path,
            use_flag=False,
            flip_y=True,
        )
        if args.data_mode == "sensor"
        else None
    )

    # Prepare FNO inputs and wavefront targets in regular-grid layout.
    X_fno, Y_fno = prepare_fno_arrays(
        args=args,
        grad_sensor_data=grad_sensor_data,
        wavefront_true_data=wavefront_true_data,
        grad_grid_data=grad_grid_data,
        sensor_coords=branch_sensor_coords,
    )

    N = X_fno.shape[0]
    n_train, n_test = args.n_train, args.n_test

    if N < n_train + n_test:
        raise ValueError(
            f"Need at least {n_train + n_test} samples, but received {N}."
        )

    # No validation split is used. The full n_train partition is used for
    # training, followed by n_test samples for testing.
    X_train = X_fno[:n_train]
    Y_train = Y_fno[:n_train]

    X_test = X_fno[n_train:n_train + n_test]
    Y_test = Y_fno[n_train:n_train + n_test]

    # Create independent random streams for batching and model initialization.
    key = jax.random.PRNGKey(args.seed)
    key_data, key_model, _ = jax.random.split(key, 3)

    # Ensure the FNO is configured for scalar wavefront reconstruction.
    args.num_outputs = 1
    args.nx = int(args.nx)
    args.ny = int(args.ny)

    # Build and initialize the FNO model.
    args, model, model_fn, params = setup_fno(args, key_model)

    # Configure the requested learning-rate schedule.
    if args.lr_scheduler == "exponential_decay":
        lr_scheduler = optax.exponential_decay(
            args.lr,
            args.lr_schedule_steps,
            args.lr_decay_rate,
        )

    elif args.lr_scheduler == "constant":
        lr_scheduler = optax.constant_schedule(args.lr)

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

    # Initialize AdamW optimizer and its state.
    optimizer = optax.adamw(
        learning_rate=lr_scheduler,
        weight_decay=getattr(args, "weight_decay", 1e-4),
    )

    opt_state = optimizer.init(params)

    # Create a timestamped output directory for logs, figures, artifacts, and
    # inference benchmarks.
    result_dir = os.path.join(
        os.getcwd(),
        args.result_dir,
        time.strftime("%Y%m%d-%H%M%S"),
    )

    os.makedirs(result_dir, exist_ok=True)

    # Construct a random mini-batch generator for full regular-grid samples.
    train_gen = GridDataGenerator(
        X_train,
        Y_train,
        args.batch_size,
        key_data,
    )

    train_iter = iter(train_gen)

    # The gradient-consistency loss is optional for FNO training.
    fno_res_weight = float(
        getattr(args, "fno_res_weight", 0.0)
    )

    def loss_fno_total(model_fn, params, batch, rng):
        """
        Apply the configured combination of supervised and residual FNO losses.
        """
        return loss_fno(
            model_fn,
            params,
            batch,
            rng,
            res_weight=fno_res_weight,
        )

    # Initialize the CSV training log.
    log_file = open(
        os.path.join(result_dir, "log.csv"),
        "w",
    )

    log_file.write(
        "iter,loss,l_ic,l_r,grad_norm,test_err,is_best\n"
    )

    pbar = tqdm.trange(args.steps)

    # Store loss and gradient-norm history for post-training plots.
    hist_iters = []
    hist_loss = []
    hist_l_ic = []
    hist_l_r = []
    hist_l_rwr = []
    hist_grad_norm = []

    # Track the best test metric and its associated parameter tree.
    best_test_err = float("inf")
    best_params = None
    best_it = -1

    # Select test examples for periodic FNO visualization.
    n_test_fno = X_test.shape[0]

    n_vis_test_fno = int(
        min(
            getattr(args, "n_vis_test", 1),
            n_test_fno,
        )
    )

    key_vis_fno = jax.random.PRNGKey(
        int(args.seed) + 999
    )

    test_idx_vis_fno = jax.random.permutation(
        key_vis_fno,
        jnp.arange(n_test_fno),
    )[:n_vis_test_fno]

    # Allow explicitly supplied visualization indices to override random ones.
    sample_idx_vis_fno = getattr(args, "sample_idx_vis", None)

    if sample_idx_vis_fno is not None:
        if isinstance(sample_idx_vis_fno, (int, np.integer)):
            test_idx_vis_fno = jnp.array(
                [int(sample_idx_vis_fno)],
                dtype=int,
            )
        else:
            test_idx_vis_fno = jnp.array(
                list(sample_idx_vis_fno),
                dtype=int,
            )

    for it in pbar:
        # Draw one random mini-batch of complete regular-grid samples.
        batch = next(train_iter)

        # Generate an independent dropout key for this optimization step.
        main_key, dropout_key = jax.random.split(main_key)

        # Perform one FNO optimization step.
        loss_val, grad_norm, params, opt_state = step_fno(
            optimizer,
            loss_fno_total,
            model_fn,
            opt_state,
            params,
            batch,
            dropout_key,
        )

        # Compute detailed losses and the mean test error periodically.
        if it % args.log_iter == 0:
            # Compute the supervised and residual terms separately for logging.
            l_ic_val = loss_fno(
                model_fn,
                params,
                batch,
                dropout_key,
                res_weight=0.0,
            )

            l_r_val = (
                loss_res_fno(
                    model_fn,
                    params,
                    batch,
                    dropout_key,
                    grid_size=int(args.nx),
                )
                if fno_res_weight > 0.0
                else 0.0
            )

            # Evaluate all test examples and compute mean relative L2 error.
            test_errs = jax.vmap(
                lambda x, y: fno_rel_l2(
                    model_fn,
                    params,
                    x,
                    y,
                )
            )(X_test, Y_test)

            test_err = float(jnp.mean(test_errs))

            is_best = 0

            # Preserve parameters whenever the test metric improves.
            if test_err < best_test_err:
                best_test_err = test_err
                best_params = params
                best_it = int(it)
                is_best = 1

            log_file.write(
                f"{it},{float(loss_val)},{float(l_ic_val)},"
                f"{float(l_r_val)},{float(grad_norm)},"
                f"{test_err},{is_best}\n"
            )

            log_file.flush()

            pbar.set_postfix(
                {
                    "l": f"{loss_val:.2e}",
                    "l_ic": f"{float(l_ic_val):.2e}",
                    "l_r": f"{float(l_r_val):.2e}",
                    "test": f"{test_err:.2e}",
                }
            )

            hist_iters.append(it)
            hist_loss.append(float(loss_val))
            hist_l_ic.append(float(l_ic_val))
            hist_l_r.append(float(l_r_val))
            hist_l_rwr.append(
                float(l_r_val) * fno_res_weight
            )
            hist_grad_norm.append(float(grad_norm))

        # Save visual comparisons for selected test examples.
        if (
                args.vis_iter > 0
                and (
                it == 1
                or (it + 1) % args.vis_iter == 0
        )
        ):
            for idx_vis in test_idx_vis_fno:
                _visualize_fno(
                    model_fn,
                    params,
                    X_test,
                    Y_test,
                    result_dir,
                    it,
                    int(idx_vis),
                    args,
                )

    log_file.close()

    # Save reconstruction and residual-loss curves.
    plt.figure(figsize=(8, 5))

    plt.plot(
        hist_iters,
        hist_loss,
        label="total loss",
    )

    plt.plot(
        hist_iters,
        hist_l_ic,
        label="L_ic (supervised)",
    )

    if any(v > 0 for v in hist_l_r):
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

    # Save the global gradient-norm curve.
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

    # Create the artifact directory for parameters and inference metadata.
    artifacts_dir = os.path.join(
        result_dir,
        "artifacts",
    )

    os.makedirs(artifacts_dir, exist_ok=True)

    # Preserve both the last optimization state and the best test-set state.
    params_last = params
    params_best = (
        best_params
        if best_params is not None
        else params
    )

    # Expose best parameters as the default trained parameter tree.
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

    # Save the best test metric and the corresponding training iteration.
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

    # Mark the saved configuration as an FNO inference configuration.
    args.operator_type = "fno"

    # Convert non-JSON-native configuration values to strings.
    args_dump = {
        k: (
            str(v)
            if not isinstance(
                v,
                (
                    int,
                    float,
                    str,
                    bool,
                    list,
                    tuple,
                    type(None),
                ),
            )
            else v
        )
        for k, v in vars(args).items()
    }

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

    # Save the branch sensor layout only when sensor-space inputs were used.
    if branch_sensor_coords is not None:
        np.save(
            os.path.join(
                artifacts_dir,
                "branch_sensor_coords.npy",
            ),
            np.asarray(branch_sensor_coords),
        )

    # Benchmark inference using the best test-set parameters.
    bench = {}

    try:
        bench["fno"] = benchmark_inference_fno(
            model_fn=model_fn,
            params=params,
            X_test=np.asarray(X_test),
            n=int(getattr(args, "benchmark_n", 1000)),
            batch_size=int(
                getattr(args, "benchmark_batch", 64)
            ),
        )
    except Exception as e:
        bench["fno_error"] = str(e)

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

    # Store the final trained state globally for notebook workflows and return
    # it to the caller for downstream evaluation or inference.
    trained_state = {
        "operator_type": "fno",
        "args": args,
        "params": params,
        "model": model,
        "model_fn": model_fn,
        "result_dir": result_dir,
        "branch_sensor_coords": branch_sensor_coords,
        "X_test": X_test,
        "Y_test": Y_test,
    }

    globals()["TRAINED_STATE"] = trained_state

    return trained_state
