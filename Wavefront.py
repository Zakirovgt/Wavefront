import os
import argparse
import time
import shutil
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import tqdm
import optax
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
import orbax.checkpoint.args as ocp_args
import jax.tree_util as jtu
from models import mse, mse_single, apply_net, step
from models.deeponet import DeepONet
from models.setup_model import setup_deeponet


# Class for iterating over batches
class DataGenerator:
    def __init__(self, u, y, s, batch_size, gen_key):
        self.u = u
        self.y = y
        self.s = s
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = gen_key

    def __iter__(self):
        return self

    def __next__(self):
        self.key, subkey = jax.random.split(self.key)
        return self.__data_generation(subkey)

    @partial(jax.jit, static_argnums=(0,))
    def __data_generation(self, key_i):
        idx = jax.random.choice(key_i, self.N, (self.batch_size,), replace=False)
        u = self.u[idx, :]
        y = self.y[idx, :]
        s = self.s[idx, :]
        return (u, y), s


# Initialization for 12x12 g1(x,y) and g2(x,y)
def generate_one_ics_training_data(g1_vals, g2_vals, U_vals, p=144):
    # Branch input shape: [g1_1, g2_1, g1_2, g2_2, ..., g1_p, g2_p]

    branch_input = jnp.stack([g1_vals, g2_vals], axis=1).ravel()  # Size (2*p,)

    # Create a fixed grid matching the g1, g2 measurement dimensions:
    side = int(jnp.sqrt(p))
    xy = jnp.linspace(-1, 1, side)
    xg, yg = jnp.meshgrid(xy, xy)
    coords = jnp.stack([xg.ravel(), yg.ravel()], axis=1)  # Size (p,2)

    # Prepared targets used for comparison in loss_ics:
    targets = jnp.stack([g1_vals, g2_vals, U_vals], axis=1)  # Size (p,3)

    # Flatten into a single array (DeepONet-specific):
    u = jnp.tile(branch_input[None, :], (p, 1))  # (p, 2*p)

    return u, coords, targets


def generate_one_res_training_data(deriv_vals, p=576):
    # deriv_vals: (P, 2) = [g1, g2] on the sensor grid
    # assume p == P (res-grid coincides with the ICS-grid)
    g1_vals = deriv_vals[:, 0]
    g2_vals = deriv_vals[:, 1]
    branch_input = deriv_vals.ravel()  # (2*P,)

    side = int(jnp.sqrt(p))
    x = jnp.linspace(-1, 1, side)
    y = jnp.linspace(-1, 1, side)
    xm, ym = jnp.meshgrid(x, y)
    coords = jnp.stack([xm.ravel(), ym.ravel()], axis=1)  # (p,2)

    u = jnp.tile(branch_input[None, :], (p, 1))  # (p, 2*P)

    # Store the "true" g1,g2 at the points of the same grid in s_res
    s = jnp.stack([g1_vals, g2_vals, jnp.zeros_like(g1_vals)], axis=1)  # (p,3)

    return u, coords, s


# Test generator: same as ICS data
def generate_one_test_data(usol, U_true, idx, p_test=144):
    # usol[idx] has shape (p_sensors, 2) = (144,2) = [g1_vals, g2_vals]
    g1_vals = usol[idx, :, 0]
    g2_vals = usol[idx, :, 1]
    U_vals = U_true[idx, :]
    return generate_one_ics_training_data(g1_vals, g2_vals, U_vals, p_test)


def loss_ics(model_fn, params, ics_batch, rng):
    inputs, outputs = ics_batch
    u, y = inputs  # u - branch, y - coordinates

    x = y[:, 0]
    y_ = y[:, 1]

    s_pred = apply_net(model_fn, params, u, x, y_, rng=rng)  # (batch_size, num_outputs) or (batch_size,)

    # convert to 2D if it came out as a vector
    if s_pred.ndim == 1:
        s_pred = s_pred[:, None]

    # num_outputs = 1 -> model predicts only U
    if s_pred.shape[1] == 1:
        loss_U = mse(outputs[:, 2], s_pred[:, 0])
        return loss_U
    else:
        loss_g1 = mse(outputs[:, 0].flatten(), s_pred[:, 0])  # g1
        loss_g2 = mse(outputs[:, 1].flatten(), s_pred[:, 1])  # g2
        loss_U = mse(outputs[:, 2], s_pred[:, 2])  # U
        return 0.1 * loss_g1 + 0.1 * loss_g2 + 1.0 * loss_U


def loss_res(model_fn, params, batch, rng):
    inputs, outputs = batch
    u, y = inputs
    x = y[:, 0]
    y_ = y[:, 1]

    # Forward pass
    s_pred = apply_net(model_fn, params, u, x, y_, rng=rng)

    # Determine the number of outputs
    if s_pred.ndim == 1:
        num_out = 1
        s_pred = s_pred[:, None]  # (N,) -> (N,1)
    else:
        num_out = s_pred.shape[1]

    v_x = jnp.ones_like(x)
    v_y = jnp.ones_like(y_)

    if num_out == 1:
        # Model predicts only U
        g1_true = outputs[:, 0]
        g2_true = outputs[:, 1]

        # Functions for vjp: return 1D vector U(x,y)
        def U_of_x(x_):
            # apply_net -> (N,) when num_outputs=1
            return apply_net(model_fn, params, u, x_, y_, rng=rng)

        def U_of_y(y__):
            return apply_net(model_fn, params, u, x, y__, rng=rng)

        dU_dx = jax.vjp(U_of_x, x)[1](v_x)[0]
        dU_dy = jax.vjp(U_of_y, y_)[1](v_y)[0]

        res_g1 = dU_dx - g1_true
        res_g2 = dU_dy - g2_true

    else:
        # Multi-output mode: model predicts [g1, g2, U]
        g1_pred = s_pred[:, 0]
        g2_pred = s_pred[:, 1]

        def U_of_x(x_):
            out = apply_net(model_fn, params, u, x_, y_, rng=rng)
            # out: (N, num_out) -> select U
            return out[:, 2]

        def U_of_y(y__):
            out = apply_net(model_fn, params, u, x, y__, rng=rng)
            return out[:, 2]

        dU_dx = jax.vjp(U_of_x, x)[1](v_x)[0]
        dU_dy = jax.vjp(U_of_y, y_)[1](v_y)[0]

        res_g1 = dU_dx - g1_pred
        res_g2 = dU_dy - g2_pred

    return mse_single(res_g1) + mse_single(res_g2)


def tree_sqnorm(tree):
    # ||g||_2^2 for a pytree
    return sum(jnp.sum(x ** 2) for x in jtu.tree_leaves(tree))


def compute_res_weight_ntk(model_fn, params, ics_batch, res_batch, rng):
    # Partial losses without mixing
    def L_ics_only(p):
        return loss_ics(model_fn, p, ics_batch, rng)

    def L_res_only(p):
        return loss_res(model_fn, p, res_batch, rng)

    g_ics = jax.grad(L_ics_only)(params)
    g_res = jax.grad(L_res_only)(params)

    tr_ics = tree_sqnorm(g_ics)  # ~ Tr K_ics * (something)
    tr_res = tree_sqnorm(g_res)  # ~ Tr K_res * (something)

    # NTK weight: want average speeds to be comparable
    # λ_res ≈ Tr(K_ics) / Tr(K_res)
    ratio = tr_ics / (tr_res + 1e-12)

    # Clip a bit so it does not blow up to 1e10
    ratio = jnp.clip(ratio, 1e-3, 1e3)

    return float(ratio)


# Final loss: boundary error weight is 0 => removal of this functional
def loss_fn(model_fn, params, ics_batch, res_batch, res_weight, rng):
    loss_ics_i = loss_ics(model_fn, params, ics_batch, rng=rng)
    loss_res_i = loss_res(model_fn, params, res_batch, rng=rng)
    loss_value = 1 * loss_ics_i + res_weight * loss_res_i
    return loss_value


def get_error(model_fn, params, deriv, U_true, idx, p_err=144, return_data=False):
    # Index vector -> compute error for each, but WITHOUT returning s_pred
    if isinstance(idx, (list, tuple, jnp.ndarray)) and getattr(idx, "ndim", 1) == 1:
        errs = jax.vmap(
            lambda i: get_error(model_fn, params, deriv, U_true, i, p_err, False)
        )(jnp.array(idx))
        return errs  # (len(idx),)

    u_test, y_test, s_test = generate_one_test_data(deriv, U_true, idx, p_err)
    x_test = y_test[:, 0]
    y_test_ = y_test[:, 1]

    s_pred = apply_net(model_fn, params, u_test, x_test, y_test_)

    # convert to 2D
    if s_pred.ndim == 1:
        s_pred = s_pred[:, None]

    u_true = s_test[:, 2]

    if s_pred.shape[1] == 1:
        u_pred = s_pred[:, 0]
    else:
        u_pred = s_pred[:, 2]

    err_u = jnp.linalg.norm(u_true - u_pred) / jnp.linalg.norm(u_true)

    if return_data:
        return err_u, s_pred
    else:
        return err_u


def visualize(args, model_fn, params, result_dir, epoch, usol, U_true, idx, test=False):
    # Error and predictions
    err_u, s_pred = get_error(
        model_fn, params, usol, U_true, idx,
        args.p_test, return_data=True
    )

    # Grid sizes
    P_usol = usol.shape[1]
    side_usol = int(np.sqrt(P_usol))
    side_pred = int(np.sqrt(args.p_test))

    # True U on the coarse grid
    U_true_grid = np.asarray(U_true[idx]).reshape(side_usol, side_usol)

    # Require p_test and P to match
    assert side_pred == side_usol, "p_test must match the number of sensors P"

    num_out = s_pred.shape[1]

    plot_dir = os.path.join(result_dir, f'vis/{epoch:06d}/{idx}/')
    os.makedirs(plot_dir, exist_ok=True)

    # Branch on number of outputs: >1 -> plot g1,g2; ==1 -> no
    if num_out > 1:
        # True g1,g2 (coarse grid)
        g1_true = np.asarray(usol[idx, :, 0]).reshape(side_usol, side_usol)
        g2_true = np.asarray(usol[idx, :, 1]).reshape(side_usol, side_usol)

        # Predicted g1,g2,U (p_test points)
        g1_pred = np.asarray(s_pred[:, 0]).reshape(side_pred, side_pred)
        g2_pred = np.asarray(s_pred[:, 1]).reshape(side_pred, side_pred)
        U_pred = np.asarray(s_pred[:, 2]).reshape(side_pred, side_pred)

        # L2 error for U
        u_error = float(
            np.linalg.norm(U_pred - U_true_grid) /
            np.linalg.norm(U_true_grid)
        )

        # Limits for colormaps
        vmin_g1, vmax_g1 = np.min(g1_true), np.max(g1_true)
        vmin_g2, vmax_g2 = np.min(g2_true), np.max(g2_true)
        vmin_U, vmax_U = np.min(U_true_grid), np.max(U_true_grid)

        # Axes for extent
        t = np.linspace(-1, 1, side_usol)
        z = np.linspace(-1, 1, side_usol)

        # Figure for g1/g2
        fig = plt.figure(figsize=(18, 12))

        plt.subplot(2, 3, 1)
        plt.imshow(
            g1_true, vmin=vmin_g1, vmax=vmax_g1, interpolation='nearest',
            extent=(t.min(), t.max(), z.max(), z.min()),
            origin='upper', aspect='auto', cmap='viridis'
        )
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('$g_1$ (True)')

        plt.subplot(2, 3, 2)
        plt.imshow(
            g1_pred, vmin=vmin_g1, vmax=vmax_g1, interpolation='nearest',
            extent=(t.min(), t.max(), z.max(), z.min()),
            origin='upper', aspect='auto', cmap='viridis'
        )
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title(r'$\hat{g}_1$ (Pred)')

        plt.subplot(2, 3, 3)
        diff_g1 = g1_pred - g1_true
        vmax = np.abs(diff_g1).max()
        plt.imshow(
            diff_g1, interpolation='nearest',
            extent=(t.min(), t.max(), z.max(), z.min()),
            origin='upper', aspect='auto', cmap='seismic',
            vmax=vmax, vmin=-vmax
        )
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title(r'$g_1 - \hat{g}_1$')

        plt.subplot(2, 3, 4)
        plt.imshow(
            g2_true, vmin=vmin_g2, vmax=vmax_g2, interpolation='nearest',
            extent=(t.min(), t.max(), z.max(), z.min()),
            origin='upper', aspect='auto', cmap='plasma'
        )
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('$g_2$ (True)')

        plt.subplot(2, 3, 5)
        plt.imshow(
            g2_pred, vmin=vmin_g2, vmax=vmax_g2, interpolation='nearest',
            extent=(t.min(), t.max(), z.max(), z.min()),
            origin='upper', aspect='auto', cmap='plasma'
        )
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title(r'$\hat{g}_2$ (Pred)')

        plt.subplot(2, 3, 6)
        diff_g2 = g2_pred - g2_true
        vmax = np.abs(diff_g2).max()
        plt.imshow(
            diff_g2, interpolation='nearest',
            extent=(t.min(), t.max(), z.max(), z.min()),
            origin='upper', aspect='auto', cmap='PuOr',
            vmax=vmax, vmin=-vmax
        )
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title(r'$g_2 - \hat{g}_2$')

        plt.suptitle(f'{"Test" if test else "Train"} Sample, U L2 Error: {u_error:.3e}')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'g1_g2_comparison.png'))
        plt.close(fig)

    else:
        # num_outputs == 1: model outputs only U
        U_pred = np.asarray(s_pred[:, 0]).reshape(side_pred, side_pred)
        u_error = float(
            np.linalg.norm(U_pred - U_true_grid) /
            np.linalg.norm(U_true_grid)
        )
        vmin_U, vmax_U = np.min(U_true_grid), np.max(U_true_grid)

    # Figure for U (common for both modes)
    fig_U = plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(
        U_true_grid, vmin=vmin_U, vmax=vmax_U,
        extent=(-1, 1, 1, -1), origin='upper', cmap='magma'
    )
    plt.colorbar()
    plt.title('True U')

    plt.subplot(1, 3, 2)
    plt.imshow(
        U_pred, vmin=vmin_U, vmax=vmax_U,
        extent=(-1, 1, 1, -1), origin='upper', cmap='magma'
    )
    plt.colorbar()
    plt.title('Predicted U')

    plt.subplot(1, 3, 3)
    diff = U_pred - U_true_grid
    v_otnos = max(vmax_U, -vmin_U)
    plt.imshow(
        diff,
        extent=(-1, 1, 1, -1), origin='upper', cmap='seismic',
        vmax=v_otnos, vmin=-v_otnos
    )
    plt.colorbar()
    plt.title('Difference')

    fig_U.suptitle(f'{"Test" if test else "Train"} Sample, U L2 Error: {u_error:.3e}')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'U_comparison.png'))
    plt.close(fig_U)


def main_routine(args):
    main_key = jax.random.PRNGKey(args.seed)
    if args.separable:
        raise ValueError('Needs normal DeepONet, not separable DeepONet')

    # Assume `data/derivatives.npy` is shape (N_samples, P, 2), where
    #   derivatives[i, j, 0] = g1 at j-th grid point of sample i
    #   derivatives[i, j, 1] = g2 at j-th grid point of sample i
    deriv = jnp.load('data/derivatives.npy')
    U_true_all = jnp.load('data/U_true.npy')  # (N, P)
    N, P, _ = deriv.shape

    n_train = args.n_train
    n_test = args.n_test
    if N < n_train + n_test:
        raise ValueError(f'Need at least {n_train + n_test} samples, got {N}')

    # Split into train / test sets
    train_deriv = deriv[:n_train]
    test_deriv = deriv[n_train:n_train + n_test]
    U_train_flat = U_true_all[:n_train]
    U_train_flat = jnp.array(U_train_flat)

    # Create keys for pseudo-randomness
    key = jax.random.PRNGKey(args.seed)
    key_ics, key_res, key_test, key_model, key_plot = jax.random.split(key, 5)

    # Read g1, g2 separately, initialize ICS data
    g1_train = train_deriv[..., 0]
    g2_train = train_deriv[..., 1]

    u_ics, y_ics, s_ics = jax.vmap(generate_one_ics_training_data,
                                   in_axes=(0, 0, 0, None))(
        g1_train, g2_train, U_train_flat, args.p_ics_train
    )
    # Convert formats from (n_train, p, ...) to (n_train*p, ...)
    u_ics = u_ics.reshape(-1, u_ics.shape[-1])
    y_ics = y_ics.reshape(-1, y_ics.shape[-1])
    s_ics = s_ics.reshape(-1, s_ics.shape[-1])

    u_res, y_res, s_res = jax.vmap(generate_one_res_training_data,
                                   in_axes=(0, None))(
        train_deriv, args.p_res
    )

    u_res = u_res.reshape(-1, u_res.shape[-1])
    y_res = y_res.reshape(-1, y_res.shape[-1])
    s_res = s_res.reshape(-1, s_res.shape[-1])

    # Wrap with a generator for SGD
    ics_dataset = DataGenerator(u_ics, y_ics, s_ics, args.batch_size, key_ics)
    res_dataset = DataGenerator(u_res, y_res, s_res, args.batch_size, key_res)

    # Preprocessing for visualization
    test_range = jnp.arange(n_test)
    test_idx_all = jax.random.choice(key_test, test_range, (n_test,), replace=False)

    n_vis_test = getattr(args, "n_vis_test", 1)
    n_vis_test = int(min(n_vis_test, n_test))
    test_idx_vis = test_idx_all[:n_vis_test]

    # If sample_idx_vis is explicitly set — it takes priority
    sample_idx_vis = getattr(args, "sample_idx_vis", None)
    if sample_idx_vis is not None:
        # option 1: single index (int)
        if isinstance(sample_idx_vis, (int, np.integer)):
            test_idx_vis = jnp.array([int(sample_idx_vis)], dtype=int)

        # option 2: list / tuple of indices
        elif isinstance(sample_idx_vis, (list, tuple)):
            test_idx_vis = jnp.array(sample_idx_vis, dtype=int)

        # option 3: np.ndarray / jnp.ndarray
        elif isinstance(sample_idx_vis, (np.ndarray, jnp.ndarray)):
            test_idx_vis = jnp.array(sample_idx_vis, dtype=int)

        else:
            raise TypeError(
                f"sample_idx_vis of type {type(sample_idx_vis)} is not supported. "
                f"Use int, list, tuple, np.ndarray or jnp.ndarray."
            )

    # Set architecture parameters
    args.n_sensors = P  # Number of points in the branch input
    args.branch_input_features = 2  # (g1,g2)
    args.trunk_input_features = 2  # (x,y)
    args.p_test = args.p_test  # 144, test on the same grid

    args.split_branch = True
    args, model, model_fn, params = setup_deeponet(args, key_model)
    model_eval = DeepONet(
        branch_layers=model.branch_layers,
        trunk_layers=model.trunk_layers,
        split_branch=model.split_branch,
        split_trunk=model.split_trunk,
        stacked=model.stacked,
        output_dim=model.output_dim,
        dropout_rate=model.dropout_rate,
        is_training=False,
    )

    model_eval_fn = jax.jit(model_eval.apply)

    # Initialize optimizer
    if args.lr_scheduler == 'exponential_decay':
        lr_scheduler = optax.exponential_decay(
            args.lr, args.lr_schedule_steps, args.lr_decay_rate
        )
    elif args.lr_scheduler == 'constant':
        lr_scheduler = optax.constant_schedule(args.lr)
    else:
        raise ValueError(f"learning rate scheduler {args.lr_scheduler} not implemented.")

    weight_decay = 1e-4
    optimizer = optax.adamw(learning_rate=lr_scheduler, weight_decay=weight_decay)

    opt_state = optimizer.init(params)
    result_dir = os.path.join(
        os.getcwd(),
        args.result_dir,
        time.strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(result_dir, exist_ok=True)

    if os.path.exists(os.path.join(result_dir, 'vis')):
        shutil.rmtree(os.path.join(result_dir, 'vis'))

    # Create logs
    if args.checkpoint_iter > 0:
        options = ocp.CheckpointManagerOptions(
            max_to_keep=args.checkpoints_to_keep,
            save_interval_steps=args.checkpoint_iter,
            save_on_steps=[args.epochs],
        )
        mngr = ocp.CheckpointManager(
            os.path.join(result_dir, 'ckpt'),
            options=options,
            item_names=('params', 'opt_state'),
        )

    else:
        mngr = None
    # Training
    ics_iter = iter(ics_dataset)
    res_iter = iter(res_dataset)

    pbar = tqdm.trange(args.epochs)

    log_file_path = os.path.join(result_dir, 'log.csv')
    log_file = open(log_file_path, 'w')
    log_file.write('iter,loss,l_ic,l_r,res_weight,grad_norm,test_err\n')

    # Arrays for plotting
    hist_iters = []
    hist_l_ic = []
    hist_l_rwr = []
    hist_grad_norm = []
    ntk_update_every = 1000
    res_weight = args.res_weight0

    for it in pbar:
        batch_ics = next(ics_iter)
        batch_res = next(res_iter)
        main_key, dropout_key, ntk_key = jax.random.split(main_key, 3)
        # 1) Periodically recompute the NTK weight
        if it % ntk_update_every == 0:
            ntk_ratio = compute_res_weight_ntk(
                model_fn, params, batch_ics, batch_res, ntk_key
            )
            # Final weight for the residual
            res_weight = args.res_weight0 * ntk_ratio

        # 2) Standard optimization step
        loss_val, grad_norm, params, opt_state = step(
            optimizer, loss_fn, model_fn,
            opt_state, params,
            batch_ics, batch_res, res_weight, dropout_key
        )

        # Logging
        if it % args.log_iter == 0:
            l_ic = loss_ics(model_fn, params, batch_ics, dropout_key)
            l_r = loss_res(model_fn, params, batch_res, dropout_key)

            test_err = get_error(
                model_eval_fn, params, test_deriv,
                U_true_all[n_train:n_train + n_test],
                test_idx_all, args.p_test, return_data=False
            )

            if isinstance(test_err, tuple):
                test_err = test_err[0]

            if isinstance(test_err, (jax.Array, jnp.ndarray, np.ndarray)):
                test_err = float(jnp.mean(test_err))
            else:
                test_err = float(test_err)

            log_file.write(
                f"{it},"
                f"{float(loss_val)},"
                f"{float(l_ic)},"
                f"{float(l_r)},"
                f"{float(res_weight)},"
                f"{float(grad_norm)},"
                f"{test_err}\n"
            )
            log_file.flush()

            pbar.set_postfix({'l': f'{loss_val:.2e}',
                              'l_ic': f'{l_ic:.2e}',
                              'l_r': f'{l_r:.2e}',
                              'e': f'{test_err:.2e}'})
            hist_iters.append(it)
            hist_l_ic.append(float(l_ic))
            hist_l_rwr.append(float(l_r) * float(res_weight))
            hist_grad_norm.append(float(grad_norm))

        # Save png
        if args.vis_iter > 0 and it == 1:
            for idx_vis in test_idx_vis:
                visualize(
                    args, model_eval_fn, params, result_dir, it,
                    test_deriv, U_true_all[n_train:n_train + n_test],
                    int(idx_vis), test=True
                )

        if args.vis_iter > 0 and (it + 1) % args.vis_iter == 0:
            for idx_vis in test_idx_vis:
                visualize(
                    args, model_eval_fn, params, result_dir, it,
                    test_deriv, U_true_all[n_train:n_train + n_test],
                    int(idx_vis), test=True
                )

        if (it + 1) % 4000 == 0:
            print()
            # You can add saving weights here
        if mngr is not None and (it + 1) % args.checkpoint_iter == 0:
            mngr.save(
                it + 1,
                args=ocp_args.Composite(
                    # wrap each pytree in StandardSave
                    params=ocp_args.StandardSave(params),
                    opt_state=ocp_args.StandardSave(opt_state),
                )
            )
    log_file.close()
    plt.figure(figsize=(8, 5))
    plt.plot(hist_iters, hist_l_ic, label='L_ic')
    plt.plot(hist_iters, hist_l_rwr, label='L_r * w_res')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(result_dir, 'loss_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(hist_iters, hist_grad_norm, label='||grad||_2')
    plt.xlabel('iteration')
    plt.ylabel('grad norm')
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'grad_norm_curve.png'), dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepONet locally")

    # data
    parser.add_argument("--deriv_path", type=str, default="data/derivatives.npy")
    parser.add_argument("--utrue_path", type=str, default="data/U_true.npy")

    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=50)

    parser.add_argument("--p_res", type=int, default=24 * 24)
    parser.add_argument("--p_test", type=int, default=24 * 24)
    parser.add_argument("--p_ics_train", type=int, default=24 * 24)
    parser.add_argument("--batch_size", type=int, default=1000)

    # model
    parser.add_argument("--separable", action="store_true", default=False)
    parser.add_argument("--stacked_do", action="store_true", default=False)
    parser.add_argument("--r", type=int, default=0)
    parser.add_argument("--num_outputs", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--branch_layers", type=int, nargs="+", default=[100, 100, 150, 150, 100])
    parser.add_argument("--trunk_layers", type=int, nargs="+", default=[100, 100, 150, 100, 100])
    parser.add_argument("--split_branch", action="store_true", default=True)
    parser.add_argument("--split_trunk", action="store_true", default=False)

    # training
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=150000)

    parser.add_argument("--res_weight0", type=float, default=0.3)
    parser.add_argument("--lr_scheduler", type=str, default="exponential_decay",
                        choices=["constant", "exponential_decay"])
    parser.add_argument("--lr_schedule_steps", type=int, default=10000)
    parser.add_argument("--lr_decay_rate", type=float, default=0.8)

    # logging/checkpoint
    parser.add_argument("--result_dir", type=str, default="results/gradient_recon/")
    parser.add_argument("--log_iter", type=int, default=100)
    parser.add_argument("--vis_iter", type=int, default=4000)
    parser.add_argument("--checkpoint_iter", type=int, default=4000)
    parser.add_argument("--checkpoints_to_keep", type=int, default=3)

    parser.add_argument("--n_vis_test", type=int, default=3)
    parser.add_argument("--sample_idx_vis", type=int, nargs="+", default=[20, 21, 22])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.deriv_path = args.deriv_path
    args.utrue_path = args.utrue_path
    main_routine(args)
