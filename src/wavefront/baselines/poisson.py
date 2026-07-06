import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def poisson_reconstruct_from_gradients_lsq(
        gx,
        gy,
        dx=None,
        dy=None,
        mask=None,
        anchor_weight=1.0,
        maxiter=10000,
):
    """
    Reconstruct a scalar field U from its spatial gradients using
    least-squares gradient integration.

    Args:
        gx: Array with shape (H, W) containing horizontal derivatives dU/dx
            on a regular grid.
        gy: Array with shape (H, W) containing vertical derivatives dU/dy
            on a regular grid.
        dx: Grid spacing along the x-axis. If None, the domain is assumed to
            span [-1, 1] along the width dimension.
        dy: Grid spacing along the y-axis. If None, the domain is assumed to
            span [-1, 1] along the height dimension.
        mask: Optional boolean array with shape (H, W). When provided,
            reconstruction is performed only at points where mask is True.
        anchor_weight: Weight of the mean-zero constraint used to remove the
            additive-constant ambiguity in the reconstructed field.
        maxiter: Maximum number of iterations for scipy.sparse.linalg.lsmr.

    Returns:
        Reconstructed scalar field U with shape (H, W) and dtype float32.

    Notes:
        The reconstruction solves a sparse least-squares problem based on
        finite-difference constraints along valid horizontal and vertical
        grid edges.

        Horizontal constraints:

            (U[i, j + 1] - U[i, j]) / dx = gx_edge

        Vertical constraints:

            (U[i + 1, j] - U[i, j]) / dy = gy_edge

        Since gradients determine U only up to an additive constant, a final
        mean-zero constraint is added:

            mean(U) = 0
    """
    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)

    H, W = gx.shape
    assert gy.shape == (H, W)

    # Assume a normalized spatial domain of [-1, 1] when grid spacings are
    # not supplied explicitly.
    if dx is None:
        dx = 2.0 / (W - 1)

    if dy is None:
        dy = 2.0 / (H - 1)

    # Use the full grid when no aperture or region mask is provided.
    if mask is None:
        mask = np.ones((H, W), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    # Assign one unknown index to each valid point inside the mask.
    idx_map = -np.ones((H, W), dtype=int)
    points = np.argwhere(mask)

    for k, (i, j) in enumerate(points):
        idx_map[i, j] = k

    n_unknowns = len(points)

    # Sparse least-squares system components.
    rows = []
    cols = []
    vals = []
    rhs = []

    row = 0

    def add_equation(indices, coefficients, b):
        """
        Add one sparse linear equation to the least-squares system.

        Args:
            indices: Indices of unknowns participating in this equation.
            coefficients: Corresponding linear coefficients.
            b: Right-hand-side target value.
        """
        nonlocal row

        for ind, coef in zip(indices, coefficients):
            rows.append(row)
            cols.append(ind)
            vals.append(coef)

        rhs.append(b)
        row += 1

    # Add horizontal edge constraints:
    #
    #     (U[i, j + 1] - U[i, j]) / dx = gx
    #
    # The edge gradient is approximated by averaging the derivative values at
    # its two endpoints.
    for i in range(H):
        for j in range(W - 1):
            if not (mask[i, j] and mask[i, j + 1]):
                continue

            a = idx_map[i, j]
            b = idx_map[i, j + 1]

            gx_edge = 0.5 * (gx[i, j] + gx[i, j + 1])

            add_equation(
                indices=[a, b],
                coefficients=[-1.0 / dx, 1.0 / dx],
                b=gx_edge,
            )

    # Add vertical edge constraints:
    #
    #     (U[i + 1, j] - U[i, j]) / dy = gy
    #
    # The edge gradient is approximated by averaging the derivative values at
    # its two endpoints.
    for i in range(H - 1):
        for j in range(W):
            if not (mask[i, j] and mask[i + 1, j]):
                continue

            a = idx_map[i, j]
            b = idx_map[i + 1, j]

            gy_edge = 0.5 * (gy[i, j] + gy[i + 1, j])

            add_equation(
                indices=[a, b],
                coefficients=[-1.0 / dy, 1.0 / dy],
                b=gy_edge,
            )

    # Fix the additive constant ambiguity by enforcing a zero-mean solution.
    mean_indices = np.arange(n_unknowns)
    mean_coeffs = np.full(
        n_unknowns,
        anchor_weight / n_unknowns,
    )

    add_equation(
        indices=mean_indices,
        coefficients=mean_coeffs,
        b=0.0,
    )

    # Build the sparse design matrix in compressed sparse row format.
    A = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(row, n_unknowns),
    ).tocsr()

    rhs = np.asarray(rhs, dtype=np.float64)

    # Solve the sparse least-squares problem.
    sol = spla.lsmr(
        A,
        rhs,
        atol=1e-10,
        btol=1e-10,
        maxiter=maxiter,
    )[0]

    # Scatter the solved values back into the original grid layout.
    U = np.zeros((H, W), dtype=np.float64)

    for k, (i, j) in enumerate(points):
        U[i, j] = sol[k]

    return U.astype(np.float32)


def rel_l2_error_no_align(
        U_true,
        U_pred,
        mask=None,
        eps: float = 1e-12,
):
    """
    Compute the relative L2 reconstruction error without offset alignment.

    Args:
        U_true: Reference scalar field.
        U_pred: Predicted scalar field with shape compatible with U_true.
        mask: Optional boolean mask selecting values used in the error metric.
        eps: Small positive constant for numerical stability.

    Returns:
        Relative L2 error:

            ||U_true - U_pred||_2 / (||U_true||_2 + eps)

    Notes:
        No constant offset is removed from U_pred before evaluation. This is
        appropriate when both fields are already expressed in a consistent
        zero-mean or otherwise aligned reference frame.
    """
    U_true = np.asarray(U_true, dtype=np.float64)
    U_pred = np.asarray(U_pred, dtype=np.float64)

    # Restrict evaluation to the requested masked region when provided.
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        t = U_true[mask]
        p = U_pred[mask]
    else:
        t = U_true.reshape(-1)
        p = U_pred.reshape(-1)

    err = np.linalg.norm(t - p) / (np.linalg.norm(t) + eps)

    return float(err)


def poisson_baseline_from_deriv_grid(
        deriv_grid,
        U_true=None,
        grid_size: int = 24,
        use_circular_mask: bool = False,
):
    """
    Reconstruct a wavefront from a regular-grid gradient field using the
    least-squares Poisson integration baseline.

    Args:
        deriv_grid: Gradient array with shape (grid_size, grid_size, 2), or a
            flattened array with shape (grid_size * grid_size, 2). The final
            dimension contains:

                deriv_grid[..., 0] = dU/dx
                deriv_grid[..., 1] = dU/dy

        U_true: Optional reference wavefront. When provided, the relative L2
            reconstruction error is calculated.
        grid_size: Resolution of the square regular grid.
        use_circular_mask: Whether to restrict reconstruction and error
            computation to the unit circular aperture.

    Returns:
        Dictionary containing:

            U_pred:
                Reconstructed wavefront with shape (grid_size, grid_size).

            mask:
                Circular aperture mask when use_circular_mask=True;
                otherwise None.

        When U_true is supplied, the dictionary additionally contains:

            err:
                Relative L2 error without offset alignment.

            offset:
                Fixed at 0.0 for compatibility with baseline result formats.

            U_pred_aligned:
                Same as U_pred because no alignment is performed.

    Notes:
        This function is intended as a baseline for reconstructing scalar
        fields from predicted or measured regular-grid derivatives.
    """
    D = np.asarray(deriv_grid, dtype=np.float32)

    # Convert flattened gradient values into a 2D grid when necessary.
    if D.ndim == 2 and D.shape[-1] == 2:
        D = D.reshape(grid_size, grid_size, 2)

    # Extract horizontal and vertical gradient components.
    gx = D[..., 0]
    gy = D[..., 1]

    # Optionally construct a unit-disk pupil mask over [-1, 1] x [-1, 1].
    if use_circular_mask:
        xy = np.linspace(-1, 1, grid_size)
        X, Y = np.meshgrid(xy, xy)

        mask = X ** 2 + Y ** 2 <= 1.0
    else:
        mask = None

    # Reconstruct the wavefront from the gradient field.
    U_pred = poisson_reconstruct_from_gradients_lsq(
        gx,
        gy,
        mask=mask,
    )

    result = {
        "U_pred": U_pred,
        "mask": mask,
    }

    # Optionally evaluate reconstruction quality against the reference field.
    if U_true is not None:
        U_true_grid = np.asarray(U_true).reshape(
            grid_size,
            grid_size,
        )

        err = rel_l2_error_no_align(
            U_true_grid,
            U_pred,
            mask=mask,
        )

        result["err"] = err
        result["offset"] = 0.0
        result["U_pred_aligned"] = U_pred

    return result
