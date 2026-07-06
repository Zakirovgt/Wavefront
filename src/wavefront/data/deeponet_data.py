import jax.numpy as jnp


def make_supervised_task(
        grad_sensor,
        wavefront_true,
        p: int = 24 * 24,
):
    """
    Construct one pointwise supervised-learning task for a DeepONet.

    The sensor gradient values are flattened into a branch-network input. The
    target wavefront is evaluated on a regular square grid over [-1, 1] x [-1, 1].

    Args:
        grad_sensor: Sensor gradient measurements, typically with shape
            (P_sensor, 2), where the final dimension contains [dU/dx, dU/dy].
        wavefront_true: Target wavefront values. The input must contain
            exactly p values after flattening.
        p: Total number of regular-grid target points. It must be a perfect
            square so that the points can be arranged into a square grid.

    Returns:
        A tuple containing:

            branch_input:
                Flattened sensor-gradient input with shape
                (P_sensor * 2,).

            coords:
                Regular query coordinates with shape (p, 2).

            targets:
                Wavefront targets with shape (p, 1).

    Raises:
        ValueError: If p is not a perfect square or if wavefront_true does
            not contain exactly p values.
    """
    # Flatten all sensor gradient components into the branch-network input.
    branch_input = grad_sensor.ravel()

    # Infer the side length of the square regular target grid.
    side = int(jnp.sqrt(p))

    if side * side != p:
        raise ValueError(
            f"p={p} must be a perfect square."
        )

    # Build a normalized Cartesian grid over [-1, 1] x [-1, 1].
    xy = jnp.linspace(-1, 1, side)
    xg, yg = jnp.meshgrid(xy, xy)

    # Flatten the regular grid into coordinate pairs with shape (p, 2).
    coords = jnp.stack(
        [xg.ravel(), yg.ravel()],
        axis=1,
    )

    # Store scalar wavefront values as a column vector.
    targets = jnp.asarray(
        wavefront_true,
        dtype=jnp.float32,
    ).reshape(-1, 1)

    if targets.shape[0] != p:
        raise ValueError(
            f"Expected a target with p={p} values, "
            f"but received shape {targets.shape}."
        )

    return branch_input, coords, targets


def generate_one_res_training_data(
        deriv_vals,
        sensor_coords,
):
    """
    Construct a sensor-resolution training task from derivative measurements.

    This helper uses sensor coordinates as query locations and creates a
    three-channel target consisting of:

        [dU/dx, dU/dy, 0]

    The final zero channel may be useful when downstream code expects a
    three-component target vector at every sensor location.

    Args:
        deriv_vals: Sensor derivative values with shape (P_sensor, 2), where:
            deriv_vals[:, 0] contains dU/dx,
            deriv_vals[:, 1] contains dU/dy.

        sensor_coords: Sensor coordinate array with shape (P_sensor, 2).

    Returns:
        A tuple containing:

            branch_input:
                Flattened derivative data with shape (P_sensor * 2,).

            coords:
                Sensor coordinates with shape (P_sensor, 2).

            s:
                Three-component target values with shape (P_sensor, 3):
                [dU/dx, dU/dy, 0].
    """
    # Extract the two spatial derivative components.
    g1_vals = deriv_vals[:, 0]
    g2_vals = deriv_vals[:, 1]

    # Flatten all sensor derivatives into a branch-network input vector.
    branch_input = deriv_vals.ravel()

    # Convert coordinates to a JAX float32 array for downstream use.
    coords = jnp.asarray(
        sensor_coords,
        dtype=jnp.float32,
    )

    # Add a zero-valued third component at every sensor location.
    s = jnp.stack(
        [
            g1_vals,
            g2_vals,
            jnp.zeros_like(g1_vals),
        ],
        axis=1,
    )

    return branch_input, coords, s


def make_test_task(
        grad_sensor_all,
        wavefront_true_all,
        idx,
        p_test: int = 24 * 24,
):
    """
    Extract and construct a single supervised test task from dataset arrays.

    Args:
        grad_sensor_all: Sensor gradient dataset with shape
            (N, P_sensor, 2).
        wavefront_true_all: Wavefront dataset with shape (N, p_test), or an
            equivalent shape that contains p_test values per sample.
        idx: Index of the requested test sample.
        p_test: Number of target grid points for the extracted task.

    Returns:
        The same tuple returned by make_supervised_task:
            (branch_input, coords, targets).
    """
    return make_supervised_task(
        grad_sensor_all[idx],
        wavefront_true_all[idx],
        p_test,
    )
