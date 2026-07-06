import numpy as np


def central_diff_2d(
        eval_fn,
        coords: np.ndarray,
        h: float = 1e-3,
) -> np.ndarray:
    """
    Estimate the 2D spatial gradient of a scalar function using central differences.

    Args:
        eval_fn: Callable with signature eval_fn(x, y) that returns function
            values U at coordinate arrays x and y. Both x and y are expected
            to have shape (P,).
        coords: Coordinate array with shape (P, 2), where:
            coords[:, 0] contains x coordinates,
            coords[:, 1] contains y coordinates.
        h: Finite-difference step size.

    Returns:
        Gradient array with shape (P, 2), where:
            grads[:, 0] = dU/dx
            grads[:, 1] = dU/dy

    Notes:
        Central differences approximate derivatives as:

            dU/dx ≈ [U(x + h, y) - U(x - h, y)] / (2h)

            dU/dy ≈ [U(x, y + h) - U(x, y - h)] / (2h)

        Smaller values of h reduce truncation error but may increase numerical
        error due to floating-point precision.
    """
    x = coords[:, 0]
    y = coords[:, 1]

    # Estimate the derivative with respect to x while keeping y fixed.
    gx = (
                 eval_fn(x + h, y) - eval_fn(x - h, y)
         ) / (2.0 * h)

    # Estimate the derivative with respect to y while keeping x fixed.
    gy = (
                 eval_fn(x, y + h) - eval_fn(x, y - h)
         ) / (2.0 * h)

    # Combine both partial derivatives into gradient vectors:
    # (P,) + (P,) -> (P, 2)
    return np.stack([gx, gy], axis=-1).astype(np.float32)
