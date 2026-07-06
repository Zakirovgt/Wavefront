import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from wavefront.data.sensors import load_sensor_coords_from_csv
from wavefront.physics.distortions import butterworth
from wavefront.physics.gradients import central_diff_2d
from wavefront.physics.grid import block_average_downsample, build_grid
from wavefront.physics.spiral import spiral_wavefront_at_points
from wavefront.physics.zernike import (
    get_basis_indices_same_order,
    zernike_wavefront_at_points,
)


def generate_zernike_span_dataset(
        num_basis: int = 65,
        eval_size: int = 24,
        sensor_coords: np.ndarray = None,
        num_train: int = 500,
        num_test: int = 50,
        apply_blur: bool = False,
        sigma_pix: float = 1.0,
        seed: int = 42,
        strong_first_max_n: int = 15,
        first_amp: float = 10.0,
        other_amp: float = 1.0,
        h: float = 1e-3,
):
    """
    Generate a dataset of normalized wavefronts represented by Zernike modes.

    Each sample is constructed from:
        - One relatively strong low-order Zernike mode or mode pair.
        - Two additional weaker higher-order modes.

    The function evaluates the wavefront on a regular grid and computes its
    spatial derivatives both at sensor locations and across the evaluation grid.

    Args:
        num_basis: Number of Zernike basis functions to use.
        eval_size: Resolution of the square evaluation grid.
        sensor_coords: Sensor coordinates with shape (P_sensor, 2).
        num_train: Number of training samples.
        num_test: Number of test samples.
        apply_blur: Reserved argument. Blurring is intentionally unsupported
            for analytic Zernike wavefronts with derivative targets.
        sigma_pix: Reserved blur standard deviation in pixels.
        seed: Random seed.
        strong_first_max_n: Number of initial basis modes eligible for the
            strongest Zernike contribution.
        first_amp: Maximum absolute amplitude of the dominant mode.
        other_amp: Maximum absolute amplitude of weaker additional modes.
        h: Central-difference step size used for derivative estimation.

    Returns:
        A tuple containing:

            U_array_norm:
                Normalized wavefronts with shape (N, P_eval).

            derivatives_norm:
                Normalized sensor derivatives with shape
                (N, P_sensor, 2).

            derivatives_grid:
                Normalized derivatives on the regular evaluation grid with
                shape (N, P_eval, 2).

            U_mean:
                Per-sample mean wavefront values inside the aperture,
                with shape (N, 1).

            U_std:
                Per-sample wavefront standard deviations inside the aperture,
                with shape (N, 1).
    """
    if sensor_coords is None:
        raise ValueError("sensor_coords must be provided")

    if apply_blur:
        raise ValueError(
            "For analytic Zernike wavefronts with sensor derivatives, "
            "apply_blur should remain False."
        )

    N = num_train + num_test

    # Build the regular Cartesian grid and unit-disk aperture mask.
    xs, ys, X, Y, R, Theta, inside, step = build_grid(eval_size)
    P_eval = eval_size * eval_size

    # Construct the ordered Zernike basis and a lookup table from (n, m)
    # pairs to their corresponding basis indices.
    inds = get_basis_indices_same_order(num_basis)
    nm_to_idx = {nm: i for i, nm in enumerate(inds)}

    rng = np.random.default_rng(seed)

    U_list = []
    deriv_list = []
    deriv_grid_list = []

    # Limit the dominant mode to an initial subset of the basis.
    strong_pool = min(strong_first_max_n, num_basis)

    # Flatten the regular evaluation grid for pointwise wavefront evaluation.
    eval_x = X.ravel()
    eval_y = Y.ravel()

    for _ in range(N):
        coeffs = np.zeros(num_basis, dtype=np.float32)

        # Select one dominant basis mode, excluding piston mode at index 0.
        first_idx = rng.integers(1, strong_pool)
        n0, m0 = inds[first_idx]
        used = [first_idx]

        # For non-axisymmetric modes, also activate the matching sine/cosine
        # partner with the same radial order and opposite azimuthal sign.
        if m0 != 0:
            sym_idx = nm_to_idx[(n0, -m0)]
            used.append(sym_idx)

            coeffs[first_idx] = rng.uniform(-first_amp, first_amp)
            coeffs[sym_idx] = rng.uniform(-first_amp, first_amp)
        else:
            coeffs[first_idx] = rng.uniform(-first_amp, first_amp)

        # Select two weaker modes from the remaining higher-order basis pool.
        tail_pool = np.arange(strong_pool, num_basis)
        remaining = np.setdiff1d(tail_pool, used)
        other_two = rng.choice(remaining, size=2, replace=False)

        coeffs[other_two] = rng.uniform(
            -other_amp,
            other_amp,
            size=2,
        )

        def eval_fn(x, y):
            return zernike_wavefront_at_points(
                coeffs,
                inds,
                x,
                y,
            )

        # Evaluate the wavefront on the regular grid.
        U_vals = eval_fn(eval_x, eval_y).reshape(eval_size, eval_size)
        U_vals[~inside] = 0.0

        # Compute dU/dx and dU/dy at sensor locations.
        deriv_sensor = central_diff_2d(
            eval_fn,
            sensor_coords,
            h=h,
        )

        # Compute derivatives at every point of the regular evaluation grid.
        eval_coords = np.stack(
            [eval_x, eval_y],
            axis=1,
        ).astype(np.float32)

        deriv_grid = central_diff_2d(
            eval_fn,
            eval_coords,
            h=h,
        )

        deriv_grid[~inside.ravel()] = 0.0

        # Compute normalization statistics using only values inside the
        # circular aperture.
        U_flat = U_vals.ravel()
        inside_flat = inside.ravel()
        U_inside = U_flat[inside_flat]

        U_mean = U_inside.mean(keepdims=True).astype(np.float32)
        U_std = (
                U_inside.std(keepdims=True) + 1e-8
        ).astype(np.float32)

        # Normalize the wavefront only within the aperture. Values outside
        # remain zero.
        U_norm = np.zeros_like(U_flat, dtype=np.float32)
        U_norm[inside_flat] = (
                                      U_flat[inside_flat] - U_mean
                              ) / U_std

        # Derivatives scale by the same standard deviation as the field.
        deriv_norm = deriv_sensor / U_std.reshape(1, 1)
        deriv_grid_norm = deriv_grid / U_std.reshape(1, 1)

        U_list.append(U_norm.astype(np.float32))
        deriv_list.append(deriv_norm.astype(np.float32))
        deriv_grid_list.append(deriv_grid_norm.astype(np.float32))

    U_array = np.asarray(U_list, dtype=np.float32)
    derivatives = np.asarray(deriv_list, dtype=np.float32)
    derivatives_grid = np.asarray(
        deriv_grid_list,
        dtype=np.float32,
    )

    # Recompute per-sample statistics from the normalized output layout for
    # compatibility with the common dataset return format.
    U_inside = U_array[:, inside.ravel()]
    U_mean = U_inside.mean(axis=1, keepdims=True)
    U_std = U_inside.std(axis=1, keepdims=True) + 1e-8

    # U_array has already been normalized sample by sample above. These aliases
    # keep the return names consistent with the other dataset generators.
    U_array_norm = U_array.astype(np.float32)
    derivatives_norm = derivatives.astype(np.float32)

    return (
        U_array_norm.astype(np.float32),
        derivatives_norm.astype(np.float32),
        derivatives_grid.astype(np.float32),
        U_mean.astype(np.float32),
        U_std.astype(np.float32),
    )


def generate_spiral_span_dataset(
        eval_size: int = 24,
        sensor_coords: np.ndarray = None,
        num_train: int = 500,
        num_test: int = 50,
        apply_blur: bool = False,
        sigma_pix: float = 1.0,
        seed: int = 42,
        kappa_range=(1.0, 4.0),
        n_range=(1, 10),
        amp: float = 10.0,
        nonsmooth: bool = False,
        h: float = 1e-3,
        smooth_mode: str = "ridge",
):
    """
    Generate a dataset of normalized spiral wavefronts.

    Each sample uses randomly sampled radial phase growth, winding number,
    and global rotation angle. Wavefront values and spatial derivatives are
    evaluated at sensor positions and on a regular Cartesian grid.

    Args:
        eval_size: Resolution of the square evaluation grid.
        sensor_coords: Sensor coordinates with shape (P_sensor, 2).
        num_train: Number of training samples.
        num_test: Number of test samples.
        apply_blur: Reserved argument. Blurring is intentionally unsupported
            for analytic spiral wavefronts with derivative targets.
        sigma_pix: Reserved blur standard deviation in pixels.
        seed: Random seed.
        kappa_range: Inclusive range of radial phase-growth coefficients.
        n_range: Inclusive integer range of spiral winding numbers.
        amp: Amplitude multiplier for the generated spiral wavefront.
        nonsmooth: Compatibility argument passed to the spiral generator.
        h: Central-difference step size used for derivative estimation.
        smooth_mode: Spiral field transformation mode, such as "ridge" or
            "cos".

    Returns:
        A tuple containing normalized wavefronts, normalized derivatives at
        sensor positions, normalized derivatives on the regular grid, and
        per-sample normalization statistics.
    """
    if sensor_coords is None:
        raise ValueError("sensor_coords must be provided")

    if apply_blur:
        raise ValueError(
            "For analytic spiral wavefronts with sensor derivatives, "
            "apply_blur should remain False."
        )

    rng = np.random.default_rng(seed)
    N = num_train + num_test

    # Build the regular grid and unit-disk aperture mask.
    xs, ys, X, Y, R, Theta, inside, step = build_grid(eval_size)
    inside_flat = inside.ravel()

    eval_x = X.ravel()
    eval_y = Y.ravel()

    U_list = []
    deriv_list = []
    deriv_grid_list = []
    mean_list = []
    std_list = []

    for _ in range(N):
        # Sample the spiral parameters independently for each example.
        kappa = rng.uniform(*kappa_range)
        n = int(rng.integers(n_range[0], n_range[1] + 1))
        rotation_angle = float(rng.uniform(0.0, 2.0 * np.pi))

        def eval_fn(x, y):
            return spiral_wavefront_at_points(
                x,
                y,
                kappa=kappa,
                n=n,
                rotation_angle=rotation_angle,
                amp=amp,
                nonsmooth=nonsmooth,
                smooth_mode=smooth_mode,
            )

        # Evaluate the wavefront on the regular grid.
        U_vals = eval_fn(eval_x, eval_y).reshape(eval_size, eval_size)
        U_vals[~inside] = 0.0

        # Estimate spatial derivatives at sensor locations.
        deriv_sensor = central_diff_2d(
            eval_fn,
            sensor_coords,
            h=h,
        )

        # Estimate derivatives at each regular grid point.
        eval_coords = np.stack(
            [eval_x, eval_y],
            axis=1,
        ).astype(np.float32)

        deriv_grid = central_diff_2d(
            eval_fn,
            eval_coords,
            h=h,
        )

        deriv_grid[~inside_flat] = 0.0

        # Compute per-sample normalization statistics inside the aperture.
        U_flat = U_vals.ravel()
        U_inside = U_flat[inside_flat]

        U_mean = np.array(
            [U_inside.mean()],
            dtype=np.float32,
        )
        U_std = np.array(
            [U_inside.std() + 1e-8],
            dtype=np.float32,
        )

        # Normalize the wavefront while preserving zeros outside the aperture.
        U_norm = np.zeros_like(U_flat, dtype=np.float32)
        U_norm[inside_flat] = (
                                      U_flat[inside_flat] - U_mean[0]
                              ) / U_std[0]

        # Normalize derivatives using the same field standard deviation.
        deriv_norm = deriv_sensor / U_std[0]
        deriv_grid_norm = deriv_grid / U_std[0]

        U_list.append(U_norm.astype(np.float32))
        deriv_list.append(deriv_norm.astype(np.float32))
        deriv_grid_list.append(deriv_grid_norm.astype(np.float32))
        mean_list.append(U_mean)
        std_list.append(U_std)

    return (
        np.asarray(U_list, dtype=np.float32),
        np.asarray(deriv_list, dtype=np.float32),
        np.asarray(deriv_grid_list, dtype=np.float32),
        np.asarray(mean_list, dtype=np.float32),
        np.asarray(std_list, dtype=np.float32),
    )


def generate_distortion_span_dataset(
        coarse_size: int = 24,
        sensor_coords: np.ndarray = None,
        num_train: int = 500,
        num_test: int = 50,
        mean: float = 0.0,
        std: float = 10.0,
        butter_N: int = 10,
        butter_fc_factor: float = 0.1,
        apply_blur: bool = False,
        sigma_pix: float = 1.0,
        seed: int = 42,
        superres: int = 8,
        fd_h: float = None,
):
    """
    Generate smooth random distortion wavefronts from filtered noise.

    A high-resolution white-noise field is low-pass filtered using a
    Butterworth filter, optionally blurred, masked to the unit disk, and then
    downsampled to the requested coarse resolution.

    Spatial derivatives are computed on the high-resolution grid and
    interpolated both to sensor coordinates and the coarse regular grid.

    Args:
        coarse_size: Resolution of the final coarse evaluation grid.
        sensor_coords: Sensor coordinates with shape (P_sensor, 2).
        num_train: Number of training samples.
        num_test: Number of test samples.
        mean: Mean of the high-resolution white-noise source.
        std: Standard deviation of the high-resolution white-noise source.
        butter_N: Butterworth low-pass filter order.
        butter_fc_factor: Cutoff-frequency factor relative to coarse_size.
        apply_blur: Whether to apply an additional Gaussian blur.
        sigma_pix: Gaussian blur standard deviation in high-resolution pixels.
        seed: Random seed.
        superres: Integer high-resolution multiplier relative to coarse_size.
        fd_h: Grid spacing for numerical derivatives. When None, the
            high-resolution grid spacing is used.

    Returns:
        A tuple containing normalized coarse wavefronts, normalized sensor
        derivatives, normalized regular-grid derivatives, and per-sample
        normalization statistics.
    """
    if sensor_coords is None:
        raise ValueError("sensor_coords must be provided")

    rng = np.random.default_rng(seed)
    N = num_train + num_test

    # Build the target coarse grid.
    xs_c, ys_c, X_c, Y_c, R_c, Theta_c, inside_c, step_c = build_grid(
        coarse_size
    )
    inside_c_flat = inside_c.ravel()

    # Build the high-resolution grid used for random field generation and
    # derivative computation.
    hr_size = coarse_size * superres
    xs_h, ys_h, X_h, Y_h, R_h, Theta_h, inside_h, step_h = build_grid(
        hr_size
    )

    if fd_h is None:
        fd_h = step_h

    # The cutoff frequency is defined relative to the coarse grid size.
    fc_h = butter_fc_factor * coarse_size

    # RegularGridInterpolator expects coordinates ordered as (y, x), while
    # sensor_coords are stored in the conventional (x, y) order.
    sensor_pts_yx = np.stack(
        [sensor_coords[:, 1], sensor_coords[:, 0]],
        axis=1,
    ).astype(np.float32)

    sensor_inside = (
            sensor_coords[:, 0] ** 2
            + sensor_coords[:, 1] ** 2
            <= 1.0
    )

    # Construct coarse-grid coordinates for derivative interpolation.
    eval_coords_c = np.stack(
        [X_c.ravel(), Y_c.ravel()],
        axis=1,
    ).astype(np.float32)

    eval_pts_yx = np.stack(
        [eval_coords_c[:, 1], eval_coords_c[:, 0]],
        axis=1,
    ).astype(np.float32)

    U_list = []
    deriv_list = []
    deriv_grid_list = []
    mean_list = []
    std_list = []

    for _ in range(N):
        # Step 1: Draw white Gaussian noise on the high-resolution grid.
        Z_h = rng.normal(
            loc=mean,
            scale=std,
            size=(hr_size, hr_size),
        ).astype(np.float32)

        # Step 2: Apply the low-pass Butterworth filter.
        U_h = butterworth(
            butter_N,
            fc_h,
            Z_h,
        ).astype(np.float32)

        # Step 3: Optionally apply a Gaussian blur.
        if apply_blur:
            U_h = gaussian_filter(
                U_h,
                sigma=sigma_pix,
                mode="constant",
                cval=0.0,
            ).astype(np.float32)

        # Step 4: Enforce the circular aperture mask.
        U_h[~inside_h] = 0.0

        # Step 5: Compute numerical derivatives on the high-resolution grid.
        #
        # Axis 0 corresponds to y, while axis 1 corresponds to x.
        dU_dy_h, dU_dx_h = np.gradient(
            U_h,
            fd_h,
            fd_h,
            edge_order=2,
        )

        dU_dx_h = dU_dx_h.astype(np.float32)
        dU_dy_h = dU_dy_h.astype(np.float32)

        dU_dx_h[~inside_h] = 0.0
        dU_dy_h[~inside_h] = 0.0

        # Step 6: Construct interpolators for the high-resolution gradients.
        interp_gx = RegularGridInterpolator(
            (ys_h, xs_h),
            dU_dx_h,
            bounds_error=False,
            fill_value=0.0,
        )

        interp_gy = RegularGridInterpolator(
            (ys_h, xs_h),
            dU_dy_h,
            bounds_error=False,
            fill_value=0.0,
        )

        # Step 7: Interpolate gradients at sensor coordinates.
        gx_sensor = interp_gx(sensor_pts_yx).astype(np.float32)
        gy_sensor = interp_gy(sensor_pts_yx).astype(np.float32)

        deriv_sensor = np.stack(
            [gx_sensor, gy_sensor],
            axis=1,
        ).astype(np.float32)

        deriv_sensor[~sensor_inside] = 0.0

        # Step 8: Interpolate gradients on the regular coarse grid.
        gx_grid = interp_gx(eval_pts_yx).astype(np.float32)
        gy_grid = interp_gy(eval_pts_yx).astype(np.float32)

        deriv_grid = np.stack(
            [gx_grid, gy_grid],
            axis=1,
        ).astype(np.float32)

        deriv_grid[~inside_c_flat] = 0.0

        # Step 9: Downsample the high-resolution wavefront using block averages.
        U_c = block_average_downsample(
            U_h,
            superres,
        ).astype(np.float32)

        U_c[~inside_c] = 0.0

        # Compute per-sample normalization statistics inside the aperture.
        U_flat = U_c.ravel()
        U_inside = U_flat[inside_c_flat]

        U_mean = np.array(
            [U_inside.mean()],
            dtype=np.float32,
        )
        U_std = np.array(
            [U_inside.std() + 1e-8],
            dtype=np.float32,
        )

        # Normalize the coarse wavefront and derivative targets.
        U_norm = np.zeros_like(U_flat, dtype=np.float32)
        U_norm[inside_c_flat] = (
                                        U_flat[inside_c_flat] - U_mean[0]
                                ) / U_std[0]

        deriv_norm = deriv_sensor / U_std[0]
        deriv_grid_norm = deriv_grid / U_std[0]

        U_list.append(U_norm.astype(np.float32))
        deriv_list.append(deriv_norm.astype(np.float32))
        deriv_grid_list.append(deriv_grid_norm.astype(np.float32))
        mean_list.append(U_mean)
        std_list.append(U_std)

    return (
        np.asarray(U_list, dtype=np.float32),
        np.asarray(deriv_list, dtype=np.float32),
        np.asarray(deriv_grid_list, dtype=np.float32),
        np.asarray(mean_list, dtype=np.float32),
        np.asarray(std_list, dtype=np.float32),
    )


def _split_counts(total: int, fractions):
    """
    Split an integer total across three groups using fractional weights.

    The returned integer counts are guaranteed to sum exactly to total.

    Args:
        total: Total number of samples to distribute.
        fractions: Three non-negative relative weights.

    Returns:
        List of three integer counts:
            [count_zernike, count_spiral, count_distortion]

    Notes:
        The procedure is:

            1. Normalize the supplied fractions.
            2. Take the floor of each fractional allocation.
            3. Distribute any remaining samples to the largest fractional
               remainders.
    """
    fr = np.array(fractions, dtype=float)
    fr = np.clip(fr, 0.0, None)

    if fr.sum() <= 0:
        raise ValueError("Sum of fractions must be greater than zero.")

    fr = fr / fr.sum()

    raw = fr * total
    base = np.floor(raw).astype(int)
    remainder = int(total - base.sum())

    if remainder > 0:
        frac_parts = raw - base
        order = np.argsort(-frac_parts)

        for i in range(remainder):
            base[order[i]] += 1

    return base.tolist()


def generate_mixed_span_dataset(
        coarse_size: int = 24,
        num_train: int = 500,
        num_test: int = 50,
        # Fractions of Zernike, spiral, and distortion samples in the training set.
        frac_zernike: float = 1 / 3,
        frac_spiral: float = 1 / 3,
        frac_distortion: float = 1 / 3,
        # Optional equivalent fractions for the test set.
        frac_zernike_test=None,
        frac_spiral_test=None,
        frac_distortion_test=None,
        # Zernike parameters.
        num_basis: int = 65,
        strong_first_max_n: int = 15,
        first_amp: float = 10.0,
        other_amp: float = 1.0,
        # Spiral parameters.
        spiral_kappa_range=(1.0, 4.0),
        spiral_n_range=(1, 10),
        spiral_amp: float = 10.0,
        spiral_nonsmooth: bool = False,
        # Distortion parameters.
        dist_mean: float = 0.0,
        dist_std: float = 10.0,
        butter_N: int = 10,
        butter_fc_factor: float = 0.1,
        # Noise parameters.
        with_noise: bool = False,
        noise_percentage: float = 0.0,
        noise_lambda: float = 0.0,
        # Shared settings.
        apply_blur: bool = True,
        sigma_pix: float = 1.0,
        seed: int = 42,
        save_dir: str = "data",
        sensor_csv_path: str = "1.csv",
        sensor_target_radius: float = 0.9,
        fd_h: float = 1e-3,
):
    """
    Generate a mixed dataset of Zernike, spiral, and random distortion fields.

    The first num_train samples form the training split, and the final
    num_test samples form the test split. Each split is shuffled independently
    after its requested proportions of wavefront classes are generated.

    This function expects load_sensor_coords_from_csv to be available in the
    surrounding project scope.

    Args:
        coarse_size: Resolution of the output wavefront grid.
        num_train: Number of training samples.
        num_test: Number of test samples.
        frac_zernike: Training fraction allocated to Zernike wavefronts.
        frac_spiral: Training fraction allocated to spiral wavefronts.
        frac_distortion: Training fraction allocated to distortions.
        frac_zernike_test: Optional Zernike fraction for the test split.
        frac_spiral_test: Optional spiral fraction for the test split.
        frac_distortion_test: Optional distortion fraction for the test split.
        num_basis: Number of Zernike modes.
        strong_first_max_n: Number of low-order modes eligible for the
            strongest Zernike contribution.
        first_amp: Maximum absolute dominant Zernike coefficient.
        other_amp: Maximum absolute weaker Zernike coefficient.
        spiral_kappa_range: Range of spiral radial-growth coefficients.
        spiral_n_range: Integer range of spiral winding numbers.
        spiral_amp: Spiral field amplitude.
        spiral_nonsmooth: Compatibility flag passed to spiral generation.
        dist_mean: Mean of random distortion source noise.
        dist_std: Standard deviation of random distortion source noise.
        butter_N: Butterworth filter order for random distortions.
        butter_fc_factor: Butterworth cutoff scaling factor.
        with_noise: Whether to inject Poisson-like gradient noise.
        noise_percentage: Fraction of all samples selected for noise injection.
        noise_lambda: Noise scale relative to gradient standard deviation.
        apply_blur: Shared blur parameter retained for configuration
            compatibility.
        sigma_pix: Blur standard deviation in pixels.
        seed: Random seed.
        save_dir: Directory for optional NumPy output files. Set to None to
            disable saving.
        sensor_csv_path: Path to the sensor-coordinate CSV file.
        sensor_target_radius: Reserved sensor-radius configuration parameter.
        fd_h: Finite-difference step size used by analytic generators.

    Returns:
        A tuple containing:

            U_all:
                Normalized wavefronts with shape (N_total, P_eval).

            derivatives_all:
                Sensor derivatives, optionally noisy, with shape
                (N_total, P_sensor, 2).

            derivatives_grid_clean_all:
                Clean regular-grid derivative targets with shape
                (N_total, P_eval, 2).

            derivatives_grid_noisy_all:
                Regular-grid derivative targets after optional noise injection,
                with shape (N_total, P_eval, 2).

            U_mean_all:
                Per-sample normalization means with shape (N_total, 1).

            U_std_all:
                Per-sample normalization standard deviations with shape
                (N_total, 1).

            labels_all:
                Integer class labels with shape (N_total,), where:
                    0 = Zernike
                    1 = Spiral
                    2 = Distortion
    """
    sensor_coords = load_sensor_coords_from_csv(
        sensor_csv_path,
        use_flag=False,
        flip_y=True,
    )

    # Build the coarse evaluation-grid mask.
    xs, ys, X, Y, R, Theta, inside, step = build_grid(coarse_size)
    inside_flat = inside.ravel()

    N_total = num_train + num_test

    if N_total <= 0:
        raise ValueError("num_train + num_test must be greater than zero.")

    rng = np.random.default_rng(seed)

    # Reuse training proportions for testing when separate test proportions
    # are not provided.
    if frac_zernike_test is None:
        frac_zernike_test = frac_zernike

    if frac_spiral_test is None:
        frac_spiral_test = frac_spiral

    if frac_distortion_test is None:
        frac_distortion_test = frac_distortion

    train_counts = _split_counts(
        num_train,
        [
            frac_zernike,
            frac_spiral,
            frac_distortion,
        ],
    )

    test_counts = _split_counts(
        num_test,
        [
            frac_zernike_test,
            frac_spiral_test,
            frac_distortion_test,
        ],
    )

    # Create lists that will hold independent train/test blocks before they
    # are concatenated and shuffled.
    U_train_list = []
    D_train_list = []
    Dg_train_list = []

    U_test_list = []
    D_test_list = []
    Dg_test_list = []

    mean_train_list = []
    std_train_list = []

    mean_test_list = []
    std_test_list = []

    labels_train_list = []
    labels_test_list = []

    # -------------------------------------------------------------------------
    # ZERNIKE WAVEFRONTS
    # -------------------------------------------------------------------------
    n_train_z, n_test_z = train_counts[0], test_counts[0]

    if n_train_z + n_test_z > 0:
        N_z = n_train_z + n_test_z
        z_seed = int(rng.integers(0, 1_000_000_000))

        U_z, D_z, Dg_z, mean_z, std_z = generate_zernike_span_dataset(
            num_basis=num_basis,
            eval_size=coarse_size,
            sensor_coords=sensor_coords,
            num_train=N_z,
            num_test=0,
            apply_blur=False,
            sigma_pix=sigma_pix,
            seed=z_seed,
            strong_first_max_n=strong_first_max_n,
            first_amp=first_amp,
            other_amp=other_amp,
            h=fd_h,
        )

        if n_train_z > 0:
            U_train_list.append(U_z[:n_train_z])
            D_train_list.append(D_z[:n_train_z])
            Dg_train_list.append(Dg_z[:n_train_z])
            mean_train_list.append(mean_z[:n_train_z])
            std_train_list.append(std_z[:n_train_z])

            labels_train_list.append(
                np.full(
                    n_train_z,
                    0,
                    dtype=np.int64,
                )
            )

        if n_test_z > 0:
            start = n_train_z
            end = n_train_z + n_test_z

            U_test_list.append(U_z[start:end])
            D_test_list.append(D_z[start:end])
            Dg_test_list.append(Dg_z[start:end])
            mean_test_list.append(mean_z[start:end])
            std_test_list.append(std_z[start:end])

            labels_test_list.append(
                np.full(
                    n_test_z,
                    0,
                    dtype=np.int64,
                )
            )

    # -------------------------------------------------------------------------
    # SPIRAL WAVEFRONTS
    # -------------------------------------------------------------------------
    n_train_s, n_test_s = train_counts[1], test_counts[1]

    if n_train_s + n_test_s > 0:
        N_s = n_train_s + n_test_s
        s_seed = int(rng.integers(0, 1_000_000_000))

        U_s, D_s, Dg_s, mean_s, std_s = generate_spiral_span_dataset(
            eval_size=coarse_size,
            sensor_coords=sensor_coords,
            num_train=N_s,
            num_test=0,
            apply_blur=False,
            sigma_pix=sigma_pix,
            seed=s_seed,
            kappa_range=spiral_kappa_range,
            n_range=spiral_n_range,
            amp=spiral_amp,
            nonsmooth=spiral_nonsmooth,
            h=fd_h,
            smooth_mode="ridge",
        )

        if n_train_s > 0:
            U_train_list.append(U_s[:n_train_s])
            D_train_list.append(D_s[:n_train_s])
            Dg_train_list.append(Dg_s[:n_train_s])
            mean_train_list.append(mean_s[:n_train_s])
            std_train_list.append(std_s[:n_train_s])

            labels_train_list.append(
                np.full(
                    n_train_s,
                    1,
                    dtype=np.int64,
                )
            )

        if n_test_s > 0:
            start = n_train_s
            end = n_train_s + n_test_s

            U_test_list.append(U_s[start:end])
            D_test_list.append(D_s[start:end])
            Dg_test_list.append(Dg_s[start:end])
            mean_test_list.append(mean_s[start:end])
            std_test_list.append(std_s[start:end])

            labels_test_list.append(
                np.full(
                    n_test_s,
                    1,
                    dtype=np.int64,
                )
            )

    # -------------------------------------------------------------------------
    # RANDOM DISTORTION WAVEFRONTS
    # -------------------------------------------------------------------------
    n_train_d, n_test_d = train_counts[2], test_counts[2]

    if n_train_d + n_test_d > 0:
        N_d = n_train_d + n_test_d
        d_seed = int(rng.integers(0, 1_000_000_000))

        U_d, D_d, Dg_d, mean_d, std_d = generate_distortion_span_dataset(
            coarse_size=coarse_size,
            sensor_coords=sensor_coords,
            num_train=N_d,
            num_test=0,
            mean=dist_mean,
            std=dist_std,
            butter_N=butter_N,
            butter_fc_factor=butter_fc_factor,
            apply_blur=False,
            sigma_pix=sigma_pix,
            seed=d_seed,
            superres=8,
            # When None, the high-resolution grid spacing is used.
            fd_h=None,
        )

        if n_train_d > 0:
            U_train_list.append(U_d[:n_train_d])
            D_train_list.append(D_d[:n_train_d])
            Dg_train_list.append(Dg_d[:n_train_d])
            mean_train_list.append(mean_d[:n_train_d])
            std_train_list.append(std_d[:n_train_d])

            labels_train_list.append(
                np.full(
                    n_train_d,
                    2,
                    dtype=np.int64,
                )
            )

        if n_test_d > 0:
            start = n_train_d
            end = n_train_d + n_test_d

            U_test_list.append(U_d[start:end])
            D_test_list.append(D_d[start:end])
            Dg_test_list.append(Dg_d[start:end])
            mean_test_list.append(mean_d[start:end])
            std_test_list.append(std_d[start:end])

            labels_test_list.append(
                np.full(
                    n_test_d,
                    2,
                    dtype=np.int64,
                )
            )

    # Concatenate source-specific blocks within the training split.
    P_sensor = sensor_coords.shape[0]

    if num_train > 0:
        U_train = np.concatenate(U_train_list, axis=0)
        D_train = np.concatenate(D_train_list, axis=0)
        Dg_train = np.concatenate(Dg_train_list, axis=0)
        mean_train = np.concatenate(mean_train_list, axis=0)
        std_train = np.concatenate(std_train_list, axis=0)
        labels_train = np.concatenate(labels_train_list, axis=0)
    else:
        U_train = np.empty(
            (0, coarse_size * coarse_size),
            dtype=np.float32,
        )
        D_train = np.empty(
            (0, P_sensor, 2),
            dtype=np.float32,
        )
        Dg_train = np.empty(
            (0, coarse_size * coarse_size, 2),
            dtype=np.float32,
        )
        mean_train = np.empty(
            (0, 1),
            dtype=np.float32,
        )
        std_train = np.empty(
            (0, 1),
            dtype=np.float32,
        )
        labels_train = np.empty(
            (0,),
            dtype=np.int64,
        )

    # Concatenate source-specific blocks within the test split.
    if num_test > 0:
        U_test = np.concatenate(U_test_list, axis=0)
        D_test = np.concatenate(D_test_list, axis=0)
        Dg_test = np.concatenate(Dg_test_list, axis=0)
        mean_test = np.concatenate(mean_test_list, axis=0)
        std_test = np.concatenate(std_test_list, axis=0)
        labels_test = np.concatenate(labels_test_list, axis=0)
    else:
        U_test = np.empty(
            (0, coarse_size * coarse_size),
            dtype=np.float32,
        )
        D_test = np.empty(
            (0, P_sensor, 2),
            dtype=np.float32,
        )
        Dg_test = np.empty(
            (0, coarse_size * coarse_size, 2),
            dtype=np.float32,
        )
        mean_test = np.empty(
            (0, 1),
            dtype=np.float32,
        )
        std_test = np.empty(
            (0, 1),
            dtype=np.float32,
        )
        labels_test = np.empty(
            (0,),
            dtype=np.int64,
        )

    # Verify that the split sizes exactly match the requested counts.
    assert U_train.shape[0] == num_train
    assert U_test.shape[0] == num_test

    # Shuffle each split independently so that wavefront types are mixed.
    if num_train > 0:
        perm_train = rng.permutation(num_train)

        U_train = U_train[perm_train]
        D_train = D_train[perm_train]
        Dg_train = Dg_train[perm_train]
        mean_train = mean_train[perm_train]
        std_train = std_train[perm_train]
        labels_train = labels_train[perm_train]

    if num_test > 0:
        perm_test = rng.permutation(num_test)

        U_test = U_test[perm_test]
        D_test = D_test[perm_test]
        Dg_test = Dg_test[perm_test]
        mean_test = mean_test[perm_test]
        std_test = std_test[perm_test]
        labels_test = labels_test[perm_test]

    # Merge training and test partitions into the final output arrays.
    U_all = np.concatenate([U_train, U_test], axis=0)
    derivatives_all = np.concatenate(
        [D_train, D_test],
        axis=0,
    )
    derivatives_grid_all = np.concatenate(
        [Dg_train, Dg_test],
        axis=0,
    )
    U_mean_all = np.concatenate(
        [mean_train, mean_test],
        axis=0,
    )
    U_std_all = np.concatenate(
        [std_train, std_test],
        axis=0,
    )
    labels_all = np.concatenate(
        [labels_train, labels_test],
        axis=0,
    )

    # Keep an immutable clean copy before optional noise injection.
    derivatives_grid_clean_all = derivatives_grid_all.copy()
    derivatives_grid_noisy_all = derivatives_grid_all.copy()

    # Build masks for valid sensor and regular-grid points inside the aperture.
    grid_mask = inside_flat.astype(bool)

    sensor_mask = (
            sensor_coords[:, 0] ** 2
            + sensor_coords[:, 1] ** 2
            <= 1.0
    )

    # Optionally inject Poisson-like zero-mean noise into a randomly selected
    # subset of samples. Noise is applied only within the circular aperture.
    if with_noise and noise_percentage > 0 and noise_lambda > 0:
        N_total = U_all.shape[0]
        n_noisy = int(noise_percentage * N_total)

        if n_noisy > 0:
            noisy_indices = rng.choice(
                N_total,
                size=n_noisy,
                replace=False,
            )

            for idx in noisy_indices:
                # Add noise to sensor-space gradients.
                deriv = derivatives_all[idx]
                active = sensor_mask

                grad_vals = deriv[active].reshape(-1)
                std_grad = grad_vals.std() + 1e-8
                scale = noise_lambda * std_grad

                n_pts = int(active.sum())

                noise = (
                                rng.poisson(
                                    lam=1.0,
                                    size=(n_pts, 2),
                                ).astype(np.float32)
                                - 1.0
                        ) * scale

                derivatives_all[idx, active, :] += noise

                # Add independent noise to regular-grid gradients.
                deriv_g = derivatives_grid_noisy_all[idx]
                active_g = grid_mask

                grad_g_vals = deriv_g[active_g].reshape(-1)
                std_grad_g = grad_g_vals.std() + 1e-8
                scale_g = noise_lambda * std_grad_g

                n_grid = int(active_g.sum())

                noise_g = (
                                  rng.poisson(
                                      lam=1.0,
                                      size=(n_grid, 2),
                                  ).astype(np.float32)
                                  - 1.0
                          ) * scale_g

                derivatives_grid_noisy_all[idx, active_g, :] += noise_g

    # Optionally save generated arrays and normalization metadata to disk.
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        np.save(
            os.path.join(save_dir, "U_true.npy"),
            U_all.astype(np.float32),
        )

        np.save(
            os.path.join(save_dir, "derivatives.npy"),
            derivatives_all.astype(np.float32),
        )

        np.save(
            os.path.join(
                save_dir,
                "derivatives_grid_24.npy",
            ),
            derivatives_grid_clean_all.astype(np.float32),
        )

        np.save(
            os.path.join(
                save_dir,
                "derivatives_grid_24_noisy.npy",
            ),
            derivatives_grid_noisy_all.astype(np.float32),
        )

        np.savez(
            os.path.join(save_dir, "normalization.npz"),
            U_mean=U_mean_all.astype(np.float32),
            U_std=U_std_all.astype(np.float32),
            num_train=num_train,
            num_test=num_test,
            labels=labels_all.astype(np.int64),
        )

    return (
        U_all,
        derivatives_all,
        derivatives_grid_clean_all,
        derivatives_grid_noisy_all,
        U_mean_all,
        U_std_all,
        labels_all,
    )
