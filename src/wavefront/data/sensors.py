import jax.numpy as jnp
import numpy as np
import pandas as pd


def sensor_affine_transform(
        X,
        Y,
        flip_y: bool = True,
):
    """
    Transform sensor pixel coordinates into normalized Cartesian coordinates.

    The transformation applies independent affine scaling and translation along
    the x and y axes. The y-axis can optionally be flipped to convert between
    image-coordinate and Cartesian-coordinate conventions.

    Args:
        X: Sensor x-coordinates in the original pixel coordinate system.
        Y: Sensor y-coordinates in the original pixel coordinate system.
        flip_y: Whether to invert the y-axis. This is typically needed when
            image coordinates increase downward while Cartesian coordinates
            increase upward.

    Returns:
        A tuple containing:

            x:
                Transformed x-coordinates with dtype float32.

            y:
                Transformed y-coordinates with dtype float32.

    Notes:
        The affine mapping is:

            x = ax * (X - cx) + tx

        When flip_y=True:

            y = -ay * (Y - cy) + ty

        Otherwise:

            y = ay * (Y - cy) + ty
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    # Reference center in the original sensor/image coordinate system.
    cx = 590.0
    cy = 559.0

    # Independent coordinate scaling factors.
    ax = 0.002442159383
    ay = 0.002619718310

    # Translation offsets in the normalized coordinate system.
    tx = 0.02
    ty = 0.02

    # Transform the horizontal coordinate.
    x = ax * (X - cx) + tx

    # Transform the vertical coordinate, optionally converting from an
    # image-style downward-positive axis to an upward-positive Cartesian axis.
    if flip_y:
        y = -ay * (Y - cy) + ty
    else:
        y = ay * (Y - cy) + ty

    return x.astype(np.float32), y.astype(np.float32)


def load_sensor_coords_from_csv(
        csv_path: str,
        use_flag: bool = False,
        flip_y: bool = True,
):
    """
    Load sensor coordinates from a CSV file and transform them to normalized
    Cartesian coordinates.

    The CSV file must contain columns named "X" and "Y". If requested and if
    the column exists, rows can also be filtered using the "Flag(0/1)" column.

    Args:
        csv_path: Path to the CSV file containing sensor coordinates.
        use_flag: Whether to keep only rows where "Flag(0/1)" equals 1.
            Filtering is applied only if that column exists in the file.
        flip_y: Whether to invert the y-axis during affine transformation.

    Returns:
        Sensor coordinates with shape (N, 2) and dtype float32, where:

            coords[:, 0] contains normalized x-coordinates.
            coords[:, 1] contains normalized y-coordinates.

    Raises:
        ValueError: If the CSV file does not contain both required "X" and
            "Y" columns.

    Side Effects:
        Prints:
            - The maximum radial distance of transformed sensor coordinates.
            - The percentage of sensor points located inside the unit disk.
    """
    # Load the complete sensor-coordinate table.
    df = pd.read_csv(csv_path)

    # Optionally retain only active or valid sensor locations.
    if use_flag and "Flag(0/1)" in df.columns:
        df = df[df["Flag(0/1)"] == 1].copy()

    # Validate the required source-coordinate columns.
    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError(
            "The CSV file must contain both 'X' and 'Y' columns."
        )

    # Read sensor locations in the original pixel coordinate system.
    X = df["X"].to_numpy(dtype=np.float32)
    Y = df["Y"].to_numpy(dtype=np.float32)

    # Map pixel coordinates into the normalized model coordinate system.
    x, y = sensor_affine_transform(
        X,
        Y,
        flip_y=flip_y,
    )

    # Store coordinates as (number_of_sensors, 2).
    coords = np.stack([x, y], axis=1).astype(np.float32)

    # Report how the transformed layout fits within the unit circular pupil.
    rr = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)

    print(f"max radius = {rr.max():.6f}")
    print(f"points inside unit disk = {(rr <= 1.0).mean() * 100:.2f}%")

    return coords


def load_branch_sensor_grid(
        csv_path: str,
        use_flag: bool = False,
        flip_y: bool = True,
):
    """
    Load the sensor layout used as a branch-network input grid.

    This function loads pixel-space sensor coordinates from a CSV file,
    optionally filters them using the "Flag(0/1)" column, transforms the
    coordinates into the normalized Cartesian coordinate system, and returns
    them as a JAX float32 array.

    Args:
        csv_path: Path to the CSV file containing sensor coordinates.
        use_flag: Whether to retain only rows with "Flag(0/1)" equal to 1.
            This filter is applied only when the column exists.
        flip_y: Whether to invert the y-axis during coordinate transformation.

    Returns:
        JAX array with shape (P_sensor, 2) and dtype jnp.float32, where:

            coords[:, 0] contains normalized x-coordinates.
            coords[:, 1] contains normalized y-coordinates.

    Raises:
        ValueError: If the CSV file does not contain both "X" and "Y" columns.
        ValueError: If fewer than two sensor points remain after optional
            filtering.

    Notes:
        The returned array can be used as the spatial sensor layout for a
        branch network, interpolation routine, or sensor-based DeepONet input
        pipeline.
    """
    # Load the full sensor-coordinate table from the CSV file.
    df = pd.read_csv(csv_path)

    # Validate that the required pixel-coordinate columns are available.
    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError(
            "The CSV file must contain both 'X' and 'Y' columns."
        )

    # Optionally keep only rows marked as active or valid.
    if use_flag and "Flag(0/1)" in df.columns:
        df = df[df["Flag(0/1)"] == 1].copy()

    # Read coordinates in the original pixel coordinate system.
    x_pix = df["X"].to_numpy(dtype=np.float32)
    y_pix = df["Y"].to_numpy(dtype=np.float32)

    # A spatial sensor layout needs at least two points to be meaningful.
    if len(x_pix) < 2:
        raise ValueError(
            "The branch sensor grid contains too few points after filtering."
        )

    # Convert pixel coordinates to normalized Cartesian coordinates.
    x, y = sensor_affine_transform(
        x_pix,
        y_pix,
        flip_y=flip_y,
    )

    # Arrange coordinates in the standard shape: (P_sensor, 2).
    coords = np.stack([x, y], axis=1)

    # Return a JAX array for direct use in JAX/Flax training pipelines.
    return jnp.asarray(coords, dtype=jnp.float32)
