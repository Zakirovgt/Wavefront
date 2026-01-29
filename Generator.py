import numpy as np
import os
from math import factorial, sqrt, pi
from scipy.ndimage import gaussian_filter


# Common grid [-1,1] x [-1,1] and circular pupil mask

def build_grid(coarse_size: int):
    xs = np.linspace(-1, 1, coarse_size)
    ys = np.linspace(-1, 1, coarse_size)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    R = np.sqrt(X ** 2 + Y ** 2)
    Theta = np.arctan2(Y, X)
    inside = (R <= 1.0)
    step = 2.0 / (coarse_size - 1)
    return X, Y, R, Theta, inside, step


# ZERNIKE BASIS + ZERNIKE WAVEFRONT GENERATOR
def zernike_radial(n: int, m_abs: int, r: np.ndarray):
    # Radial polynomial R_n^m(r). If invalid (n < |m| or parity mismatch) -> zeros.
    if (n < m_abs) or ((n - m_abs) % 2 != 0):
        return np.zeros_like(r, dtype=float)

    out = np.zeros_like(r, dtype=float)
    nm_half = (n - m_abs) // 2
    for k in range(nm_half + 1):
        num = factorial(n - k)
        den = factorial(k) * factorial((n + m_abs) // 2 - k) * factorial(nm_half - k)
        out += ((-1) ** k) * num / den * (r ** (n - 2 * k))
    return out


def zernike_value(n: int, m: int, R: np.ndarray, Theta: np.ndarray):
    # Orthonormalized Zernike on the unit disk (Noll-like normalization with pi in denom).
    m_abs = abs(m)
    Rnm = zernike_radial(n, m_abs, R)

    if m == 0:
        norm = sqrt((n + 1) / pi)
        ang = 1.0
    elif m > 0:
        norm = sqrt(2 * (n + 1) / pi)
        ang = np.cos(m * Theta)
    else:
        norm = sqrt(2 * (n + 1) / pi)
        ang = np.sin(m_abs * Theta)

    Z = norm * Rnm * ang
    return Z.astype(np.float32)


def get_basis_indices_same_order(num_terms: int):
    # Build (n,m) list in a stable "same order" sweep:
    # n = 0,1,2,...; for each n: m in {0,2,4,...} or {1,3,5,...} depending on parity
    # and for m>0 we append both +m and -m consecutively.
    idx = []
    count = 0
    n = 0
    while count < num_terms:
        m_vals = (range(0, n + 1, 2) if n % 2 == 0 else range(1, n + 1, 2))
        for m in m_vals:
            if count >= num_terms:
                break
            if m == 0:
                idx.append((n, 0))
                count += 1
            else:
                idx.append((n, +m))
                count += 1
                if count < num_terms:
                    idx.append((n, -m))
                    count += 1
        n += 1
    return idx  # length == num_terms


def generate_zernike_span_dataset(
        num_basis: int = 65,
        coarse_size: int = 24,
        num_train: int = 500,
        num_test: int = 50,
        apply_blur: bool = True,
        sigma_pix: float = 1.0,
        seed: int = 42,
        strong_first_max_n: int = 15,
        first_amp: float = 10.0,  # U(-10,10) for the main index
        other_amp: float = 1.0    # U(-1,1) for two additional indices
):
    """
    Zernike wavefront generator.

    Returns:
      U_array_norm:       (N, P)
      derivatives_norm:   (N, P, 2)
      U_mean:             (N, 1)
      U_std:              (N, 1)

    Normalization is computed ONLY over points inside the unit disk.
    Outside the disk we keep zeros (so the padding stays "clean").
    """
    N = num_train + num_test

    X, Y, R, Theta, inside, step = build_grid(coarse_size)
    P = coarse_size * coarse_size

    inds = get_basis_indices_same_order(num_basis)

    # Precompute basis on the grid: Z_mat[i] = Z_i flattened (with outside-disk = 0).
    Z_mat = np.empty((num_basis, P), dtype=np.float32)
    Rf, Tf = R.ravel(), Theta.ravel()
    for i, (n, m) in enumerate(inds):
        Zi = zernike_value(n, m, Rf, Tf)
        Zi = Zi.reshape(coarse_size, coarse_size)
        Zi[~inside] = 0.0
        Z_mat[i] = Zi.ravel()

    # Map (n,m) -> basis index (needed to quickly find symmetric partner for m != 0)
    nm_to_idx = {nm: i for i, nm in enumerate(inds)}

    rng = np.random.default_rng(seed)
    U_list, deriv_list = [], []

    # "Strong" pool: early modes are more likely to dominate (except piston is excluded below)
    strong_pool = min(strong_first_max_n, num_basis)

    for _ in range(N):
        coeffs = np.zeros(num_basis, dtype=np.float32)

        # Main index: NOT the piston (index 0), so start from 1.
        first_idx = rng.integers(1, strong_pool)
        n0, m0 = inds[first_idx]
        used = [first_idx]

        # If m != 0, also activate the symmetric mode (n, -m) independently.
        # This gives you a "pair" with strong amplitudes, typical for real aberrations.
        if m0 != 0:
            sym_idx = nm_to_idx[(n0, -m0)]
            used.append(sym_idx)
            coeffs[first_idx] = rng.uniform(-first_amp, first_amp)
            coeffs[sym_idx] = rng.uniform(-first_amp, first_amp)
        else:
            coeffs[first_idx] = rng.uniform(-first_amp, first_amp)

        # Two extra modes from the higher part of the basis:
        # we avoid collisions with the main (and its symmetric partner if present).
        tail_pool = np.arange(strong_pool, num_basis)
        remaining = np.setdiff1d(tail_pool, used)
        other_two = rng.choice(remaining, size=2, replace=False)
        coeffs[other_two] = rng.uniform(-other_amp, other_amp, size=2)

        # U as a linear combination of Zernike basis
        U_vals = coeffs @ Z_mat
        U_img = U_vals.reshape(coarse_size, coarse_size)

        # Optional blur: mimics smoothing / limited spatial bandwidth
        if apply_blur:
            U_img = gaussian_filter(U_img, sigma=sigma_pix, mode="constant", cval=0.0)

        # Numerical gradients (note: np.gradient returns (d/dy, d/dx))
        gy_img, gx_img = np.gradient(U_img, step, step)

        # Apply the disk mask and cast to float32
        U_img = U_img.astype(np.float32)
        gx_img = gx_img.astype(np.float32)
        gy_img = gy_img.astype(np.float32)

        U_img[~inside] = 0.0
        gx_img[~inside] = 0.0
        gy_img[~inside] = 0.0

        U_list.append(U_img.ravel())
        deriv_list.append(
            np.stack([gx_img.ravel(), gy_img.ravel()], axis=-1).astype(np.float32)
        )

    U_array = np.asarray(U_list, dtype=np.float32)          # (N, P)
    derivatives = np.asarray(deriv_list, dtype=np.float32)  # (N, P, 2)

    # Normalize ONLY on the disk
    inside_flat = inside.ravel()
    U_inside = U_array[:, inside_flat]  # (N, P_inside)

    U_mean = U_inside.mean(axis=1, keepdims=True)           # (N, 1)
    U_std = U_inside.std(axis=1, keepdims=True) + 1e-8      # avoid div-by-zero

    # Normalize U on disk; keep outside as 0
    U_array_norm = np.zeros_like(U_array, dtype=np.float32)
    U_array_norm[:, inside_flat] = (U_array[:, inside_flat] - U_mean) / U_std

    # Normalize slopes: g -> g / sigma (same scaling as U)
    scale = 1.0 / U_std  # (N, 1)
    scale_3d = scale.reshape(-1, 1, 1)
    derivatives_norm = derivatives * scale_3d

    return (
        U_array_norm.astype(np.float32),
        derivatives_norm.astype(np.float32),
        U_mean.astype(np.float32),
        U_std.astype(np.float32),
    )


# SPIRAL FIELD + GENERATOR ([-1,1] x [-1,1])
def spiral_field(
        X: np.ndarray,
        Y: np.ndarray,
        kappa: float,
        n: int,
        rotation_angle: float = 0.0,
        nonsmooth: bool = False,
):
    """
    Spiral-like field on a grid [-1,1]x[-1,1].

    Smooth variant:
        phase = kappa * r + n * (theta - pi/2) + rotation_angle
        U = phase mod (2*pi)

    Non-smooth variant:
        temp2 = phase mod (2*pi)
        k = |phase // (2*pi)|
        if k is odd  -> temp2
        else         -> 2*pi - temp2

    (So it "folds" the phase every other turn, producing a kinked pattern.)
    """
    xz = X
    yz = Y

    r = np.sqrt(xz * xz + yz * yz)
    theta = np.arctan2(yz, xz)

    phase = kappa * r + n * (theta - np.pi / 2.0) + rotation_angle

    if not nonsmooth:
        U = np.mod(phase, 2.0 * np.pi)
    else:
        temp2 = np.mod(phase, 2.0 * np.pi)
        k = np.abs(phase // (2.0 * np.pi))
        mask_odd = (k % 2 == 1)
        U = np.where(mask_odd, temp2, 2.0 * np.pi - temp2)

    return U.astype(np.float32)


def generate_spiral_span_dataset(
        coarse_size: int = 24,
        num_train: int = 500,
        num_test: int = 50,
        apply_blur: bool = True,
        sigma_pix: float = 1.0,
        seed: int = 42,
        kappa_range=(1.0, 4.0),
        n_range=(1, 10),
        amp: float = 10.0,
        nonsmooth: bool = False,
):
    """
    Generator of wavefronts with a SINGLE spiral.

    Returns:
      U_array_norm:       (N, P)
      derivatives_norm:   (N, P, 2)
      U_mean:             (N, 1)
      U_std:              (N, 1)
    """
    rng = np.random.default_rng(seed)
    N = num_train + num_test

    X, Y, R, Theta, inside, step = build_grid(coarse_size)
    inside_flat = inside.ravel()

    U_list = []
    deriv_list = []

    for _ in range(N):
        kappa = rng.uniform(*kappa_range)
        n = int(rng.integers(n_range[0], n_range[1] + 1))
        rotation_angle = float(rng.uniform(0.0, 2.0 * np.pi))

        U_img = amp * spiral_field(
            X, Y,
            kappa=kappa,
            n=n,
            rotation_angle=rotation_angle,
            nonsmooth=nonsmooth,
        )

        if apply_blur:
            U_img = gaussian_filter(
                U_img,
                sigma=sigma_pix,
                mode="constant",
                cval=0.0,
            )

        gy_img, gx_img = np.gradient(U_img, step, step)

        U_img = U_img.astype(np.float32)
        gx_img = gx_img.astype(np.float32)
        gy_img = gy_img.astype(np.float32)

        U_img[~inside] = 0.0
        gx_img[~inside] = 0.0
        gy_img[~inside] = 0.0

        U_list.append(U_img.ravel())
        deriv_list.append(
            np.stack([gx_img.ravel(), gy_img.ravel()], axis=-1).astype(np.float32)
        )

    U_array = np.asarray(U_list, dtype=np.float32)
    derivatives = np.asarray(deriv_list, dtype=np.float32)

    # Disk-only normalization
    U_inside = U_array[:, inside_flat]

    U_mean = U_inside.mean(axis=1, keepdims=True)
    U_std = U_inside.std(axis=1, keepdims=True) + 1e-8

    U_array_norm = np.zeros_like(U_array, dtype=np.float32)
    U_array_norm[:, inside_flat] = (U_array[:, inside_flat] - U_mean) / U_std

    scale = 1.0 / U_std
    scale_3d = scale.reshape(-1, 1, 1)
    derivatives_norm = derivatives * scale_3d

    return (
        U_array_norm.astype(np.float32),
        derivatives_norm.astype(np.float32),
        U_mean.astype(np.float32),
        U_std.astype(np.float32),
    )


# ATMOSPHERIC DISTORTIONS: WHITE NOISE + BUTTERWORTH FILTER

def butterworth(N: int, fc: float, z: np.ndarray):
    """
    2D Butterworth low-pass filter.

    N  - filter order
    fc - cutoff frequency in "pixel" units

    Note: implemented with explicit loops (clear but not the fastest).
    For speed you could vectorize dist computation, but this version matches the "straight" style.
    """
    N1, N2 = z.shape
    fft_z = np.fft.fft2(z)
    fft_z_shifted = np.fft.fftshift(fft_z)

    butterworth_filter = np.ones((N1, N2))
    for i in range(N1):
        for j in range(N2):
            dist = np.sqrt((i - (N1 / 2 + 1)) ** 2 + (j - (N2 / 2 + 1)) ** 2)
            butterworth_filter[i, j] = 1.0 / (1.0 + (dist / fc) ** (2 * N))

    filtered_fft_z_shifted = fft_z_shifted * butterworth_filter
    filtered_fft_z = np.fft.ifftshift(filtered_fft_z_shifted)
    filtered_z = np.real(np.fft.ifft2(filtered_fft_z))
    return filtered_z


def generate_distortion_span_dataset(
        coarse_size: int = 24,
        num_train: int = 500,
        num_test: int = 50,
        mean: float = 0.0,
        std: float = 10.0,
        butter_N: int = 10,
        butter_fc_factor: float = 0.1,  # fc ~ butter_fc_factor * coarse_size
        apply_blur: bool = False,
        sigma_pix: float = 1.0,
        seed: int = 42,
):
    """
    Generator of "atmospheric distortions":
      - white Gaussian noise
      - 2D Butterworth low-pass
      - (optional) extra Gaussian blur
      - disk-only normalization

    Returns:
      U_array_norm:       (N, P)
      derivatives_norm:   (N, P, 2)
      U_mean:             (N, 1)
      U_std:              (N, 1)
    """
    rng = np.random.default_rng(seed)
    N = num_train + num_test

    X, Y, R, Theta, inside, step = build_grid(coarse_size)
    inside_flat = inside.ravel()

    U_list = []
    deriv_list = []

    fc = butter_fc_factor * coarse_size

    for _ in range(N):
        # White noise
        Z = rng.normal(loc=mean, scale=std, size=(coarse_size, coarse_size)).astype(np.float32)

        # Butterworth low-pass
        U_img = butterworth(butter_N, fc, Z)

        if apply_blur:
            U_img = gaussian_filter(
                U_img,
                sigma=sigma_pix,
                mode="constant",
                cval=0.0,
            )

        gy_img, gx_img = np.gradient(U_img, step, step)

        U_img = U_img.astype(np.float32)
        gx_img = gx_img.astype(np.float32)
        gy_img = gy_img.astype(np.float32)

        U_img[~inside] = 0.0
        gx_img[~inside] = 0.0
        gy_img[~inside] = 0.0

        U_list.append(U_img.ravel())
        deriv_list.append(
            np.stack([gx_img.ravel(), gy_img.ravel()], axis=-1).astype(np.float32)
        )

    U_array = np.asarray(U_list, dtype=np.float32)
    derivatives = np.asarray(deriv_list, dtype=np.float32)

    U_inside = U_array[:, inside_flat]

    U_mean = U_inside.mean(axis=1, keepdims=True)
    U_std = U_inside.std(axis=1, keepdims=True) + 1e-8

    U_array_norm = np.zeros_like(U_array, dtype=np.float32)
    U_array_norm[:, inside_flat] = (U_array[:, inside_flat] - U_mean) / U_std

    scale = 1.0 / U_std
    scale_3d = scale.reshape(-1, 1, 1)
    derivatives_norm = derivatives * scale_3d

    return (
        U_array_norm.astype(np.float32),
        derivatives_norm.astype(np.float32),
        U_mean.astype(np.float32),
        U_std.astype(np.float32),
    )


# MIXED GENERATOR: ZERNIKE + SPIRALS + DISTORTIONS

def _split_counts(total: int, fractions):
    """
    Split 'total' according to 'fractions' (len == 3),
    ensuring the integer counts sum EXACTLY to total.

    Implementation detail:
    - take floor for each bucket
    - distribute the remaining samples to the largest fractional parts
    """
    fr = np.array(fractions, dtype=float)
    fr = np.clip(fr, 0.0, None)
    if fr.sum() <= 0:
        raise ValueError("Sum of fractions must be > 0.")
    fr = fr / fr.sum()
    raw = fr * total
    base = np.floor(raw).astype(int)
    remainder = int(total - base.sum())
    if remainder > 0:
        frac_parts = raw - base
        order = np.argsort(-frac_parts)
        for i in range(remainder):
            base[order[i]] += 1
    return base.tolist()  # [count_z, count_spiral, count_dist]


def generate_mixed_span_dataset(
        coarse_size: int = 24,
        num_train: int = 500,
        num_test: int = 50,
        # fractions of types in train/test (computed separately, same idea)
        frac_zernike: float = 1 / 3,
        frac_spiral: float = 1 / 3,
        frac_distortion: float = 1 / 3,
        # Zernike params
        frac_zernike_test=None,
        frac_spiral_test=None,
        frac_distortion_test=None,
        num_basis: int = 65,
        strong_first_max_n: int = 15,
        first_amp: float = 10.0,
        other_amp: float = 1.0,
        # Spiral params
        spiral_kappa_range=(1.0, 4.0),
        spiral_n_range=(1, 10),
        spiral_amp: float = 10.0,
        spiral_nonsmooth: bool = False,
        # Distortion params
        dist_mean: float = 0.0,
        dist_std: float = 10.0,
        butter_N: int = 10,
        butter_fc_factor: float = 0.1,
        # common
        apply_blur: bool = True,
        sigma_pix: float = 1.0,
        seed: int = 42,
        save_dir: str = "data",
):
    """
    Mixed dataset with the same structure as Zernikes:

      - U_all:            (N, P)
      - derivatives_all:  (N, P, 2)
      - U_mean_all:       (N, 1)
      - U_std_all:        (N, 1)
      - labels_all:       (N,)   0=Zernike, 1=Spiral, 2=Distortion

    First num_train samples are train, last num_test are test.

    frac_zernike, frac_spiral, frac_distortion are train fractions.
    frac_zernike_test, frac_spiral_test, frac_distortion_test are test fractions
        (if None, we reuse train fractions).
    """
    N_total = num_train + num_test
    if N_total <= 0:
        raise ValueError("num_train + num_test must be > 0.")

    rng = np.random.default_rng(seed)

    # If test fractions are not provided, reuse train fractions
    if frac_zernike_test is None:
        frac_zernike_test = frac_zernike
    if frac_spiral_test is None:
        frac_spiral_test = frac_spiral
    if frac_distortion_test is None:
        frac_distortion_test = frac_distortion

    train_counts = _split_counts(num_train, [frac_zernike, frac_spiral, frac_distortion])
    test_counts = _split_counts(num_test, [frac_zernike_test, frac_spiral_test, frac_distortion_test])

    # Lists for building train/test blocks
    U_train_list, D_train_list = [], []
    U_test_list, D_test_list = [], []
    mean_train_list, std_train_list = [], []
    mean_test_list, std_test_list = [], []
    labels_train_list, labels_test_list = [], []

    # ---------- ZERNIKE ----------
    n_train_z, n_test_z = train_counts[0], test_counts[0]
    if n_train_z + n_test_z > 0:
        N_z = n_train_z + n_test_z
        z_seed = int(rng.integers(0, 1_000_000_000))
        U_z, D_z, mean_z, std_z = generate_zernike_span_dataset(
            num_basis=num_basis,
            coarse_size=coarse_size,
            num_train=N_z,
            num_test=0,
            apply_blur=apply_blur,
            sigma_pix=sigma_pix,
            seed=z_seed,
            strong_first_max_n=strong_first_max_n,
            first_amp=first_amp,
            other_amp=other_amp,
        )
        if n_train_z > 0:
            U_train_list.append(U_z[:n_train_z])
            D_train_list.append(D_z[:n_train_z])
            mean_train_list.append(mean_z[:n_train_z])
            std_train_list.append(std_z[:n_train_z])
            labels_train_list.append(np.full(n_train_z, 0, dtype=np.int64))
        if n_test_z > 0:
            start = n_train_z
            end = n_train_z + n_test_z
            U_test_list.append(U_z[start:end])
            D_test_list.append(D_z[start:end])
            mean_test_list.append(mean_z[start:end])
            std_test_list.append(std_z[start:end])
            labels_test_list.append(np.full(n_test_z, 0, dtype=np.int64))

    # ---------- SPIRALS ----------
    n_train_s, n_test_s = train_counts[1], test_counts[1]
    if n_train_s + n_test_s > 0:
        N_s = n_train_s + n_test_s
        s_seed = int(rng.integers(0, 1_000_000_000))
        U_s, D_s, mean_s, std_s = generate_spiral_span_dataset(
            coarse_size=coarse_size,
            num_train=N_s,
            num_test=0,
            apply_blur=apply_blur,
            sigma_pix=sigma_pix,
            seed=s_seed,
            kappa_range=spiral_kappa_range,
            n_range=spiral_n_range,
            amp=spiral_amp,
            nonsmooth=spiral_nonsmooth,
        )
        if n_train_s > 0:
            U_train_list.append(U_s[:n_train_s])
            D_train_list.append(D_s[:n_train_s])
            mean_train_list.append(mean_s[:n_train_s])
            std_train_list.append(std_s[:n_train_s])
            labels_train_list.append(np.full(n_train_s, 1, dtype=np.int64))
        if n_test_s > 0:
            start = n_train_s
            end = n_train_s + n_test_s
            U_test_list.append(U_s[start:end])
            D_test_list.append(D_s[start:end])
            mean_test_list.append(mean_s[start:end])
            std_test_list.append(std_s[start:end])
            labels_test_list.append(np.full(n_test_s, 1, dtype=np.int64))

    # ---------- DISTORTIONS ----------
    n_train_d, n_test_d = train_counts[2], test_counts[2]
    if n_train_d + n_test_d > 0:
        N_d = n_train_d + n_test_d
        d_seed = int(rng.integers(0, 1_000_000_000))
        U_d, D_d, mean_d, std_d = generate_distortion_span_dataset(
            coarse_size=coarse_size,
            num_train=N_d,
            num_test=0,
            mean=dist_mean,
            std=dist_std,
            butter_N=butter_N,
            butter_fc_factor=butter_fc_factor,
            apply_blur=False,  # Butterworth is usually enough on its own
            sigma_pix=sigma_pix,
            seed=d_seed,
        )
        if n_train_d > 0:
            U_train_list.append(U_d[:n_train_d])
            D_train_list.append(D_d[:n_train_d])
            mean_train_list.append(mean_d[:n_train_d])
            std_train_list.append(std_d[:n_train_d])
            labels_train_list.append(np.full(n_train_d, 2, dtype=np.int64))
        if n_test_d > 0:
            start = n_train_d
            end = n_train_d + n_test_d
            U_test_list.append(U_d[start:end])
            D_test_list.append(D_d[start:end])
            mean_test_list.append(mean_d[start:end])
            std_test_list.append(std_d[start:end])
            labels_test_list.append(np.full(n_test_d, 2, dtype=np.int64))

    # Concatenate train/test across types
    if num_train > 0:
        U_train = np.concatenate(U_train_list, axis=0)
        D_train = np.concatenate(D_train_list, axis=0)
        mean_train = np.concatenate(mean_train_list, axis=0)
        std_train = np.concatenate(std_train_list, axis=0)
        labels_train = np.concatenate(labels_train_list, axis=0)
    else:
        U_train = np.empty((0, coarse_size * coarse_size), dtype=np.float32)
        D_train = np.empty((0, coarse_size * coarse_size, 2), dtype=np.float32)
        mean_train = np.empty((0, 1), dtype=np.float32)
        std_train = np.empty((0, 1), dtype=np.float32)
        labels_train = np.empty((0,), dtype=np.int64)

    if num_test > 0:
        U_test = np.concatenate(U_test_list, axis=0)
        D_test = np.concatenate(D_test_list, axis=0)
        mean_test = np.concatenate(mean_test_list, axis=0)
        std_test = np.concatenate(std_test_list, axis=0)
        labels_test = np.concatenate(labels_test_list, axis=0)
    else:
        U_test = np.empty((0, coarse_size * coarse_size), dtype=np.float32)
        D_test = np.empty((0, coarse_size * coarse_size, 2), dtype=np.float32)
        mean_test = np.empty((0, 1), dtype=np.float32)
        std_test = np.empty((0, 1), dtype=np.float32)
        labels_test = np.empty((0,), dtype=np.int64)

    # Sanity-check: should match requested split exactly
    assert U_train.shape[0] == num_train
    assert U_test.shape[0] == num_test

    # Shuffle within train/test (so types are mixed)
    if num_train > 0:
        perm_train = rng.permutation(num_train)
        U_train = U_train[perm_train]
        D_train = D_train[perm_train]
        mean_train = mean_train[perm_train]
        std_train = std_train[perm_train]
        labels_train = labels_train[perm_train]

    if num_test > 0:
        perm_test = rng.permutation(num_test)
        U_test = U_test[perm_test]
        D_test = D_test[perm_test]
        mean_test = mean_test[perm_test]
        std_test = std_test[perm_test]
        labels_test = labels_test[perm_test]

    # Merge train + test into one big array
    U_all = np.concatenate([U_train, U_test], axis=0)
    derivatives_all = np.concatenate([D_train, D_test], axis=0)
    U_mean_all = np.concatenate([mean_train, mean_test], axis=0)
    U_std_all = np.concatenate([std_train, std_test], axis=0)
    labels_all = np.concatenate([labels_train, labels_test], axis=0)

    # Save (optional)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "U_true.npy"), U_all.astype(np.float32))
        np.save(os.path.join(save_dir, "derivatives.npy"), derivatives_all.astype(np.float32))
        np.savez(
            os.path.join(save_dir, "normalization.npz"),
            U_mean=U_mean_all.astype(np.float32),
            U_std=U_std_all.astype(np.float32),
            num_train=num_train,
            num_test=num_test,
            labels=labels_all.astype(np.int64),
        )

    return (U_all, derivatives_all, U_mean_all, U_std_all, labels_all)


if __name__ == "__main__":
    U_all, deriv_all, U_mean_all, U_std_all, labels_all = generate_mixed_span_dataset(
        coarse_size=24,
        num_train=500,
        num_test=50,
        frac_zernike=0.5,
        frac_spiral=0.3,
        frac_distortion=0.2,
        frac_zernike_test=None,
        frac_spiral_test=None,
        frac_distortion_test=None,
        apply_blur=True,
        sigma_pix=1.0,
        seed=42,
    )

    print("U_all:", U_all.shape)
    print("derivatives_all:", deriv_all.shape)
    print("U_mean_all:", U_mean_all.shape)
    print("U_std_all:", U_std_all.shape)
    print("labels_all:", labels_all.shape, " (0=Zernike, 1=Spiral, 2=Distortion)")
    res = np.vstack([labels_all[-50:], np.arange(50)]).T
    print(res)
