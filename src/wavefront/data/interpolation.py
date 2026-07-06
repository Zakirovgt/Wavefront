import hashlib

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

from wavefront.physics.grid import build_grid

# Cache for precomputed Delaunay triangulations and barycentric interpolation
# weights. The cache avoids repeating expensive geometry calculations when the
# same sensor layout and output grid resolution are used multiple times.
_SENSOR_TO_GRID_INTERP_CACHE = {}


def _sensor_grid_cache_key(
        sensor_coords: np.ndarray,
        grid_size: int,
):
    """
    Create a stable cache key for a sensor layout and target grid resolution.

    Args:
        sensor_coords: Sensor coordinates with shape (P_sensor, 2).
        grid_size: Resolution of the square target grid.

    Returns:
        A tuple containing:
            - A BLAKE2 hash of the contiguous float32 coordinate data.
            - The coordinate array shape.
            - The integer grid size.

    Notes:
        Including both the byte-level hash and shape prevents cache collisions
        between sensor layouts with different coordinate values or dimensions.
    """
    coords = np.ascontiguousarray(sensor_coords.astype(np.float32))

    digest = hashlib.blake2b(
        coords.view(np.uint8),
        digest_size=16,
    ).hexdigest()

    return digest, coords.shape, int(grid_size)


def _precompute_sensor_to_grid_linear(
        sensor_coords: np.ndarray,
        grid_size: int,
):
    """
    Precompute linear interpolation from irregular sensor locations to a
    regular grid over [-1, 1] x [-1, 1].

    The interpolation uses a Delaunay triangulation of sensor locations.
    Each valid target grid point is represented as a weighted combination of
    the three vertices of its containing triangle.

    Args:
        sensor_coords: Sensor coordinates with shape (P_sensor, 2).
        grid_size: Resolution of the square target grid.

    Returns:
        Dictionary containing:

            vertices:
                Integer array with shape (G, 3), containing the indices of
                the three sensor vertices for each target-grid triangle.

            weights:
                Float array with shape (G, 3), containing barycentric
                interpolation weights for each target point.

            valid:
                Boolean array with shape (G,). A point is valid only when it
                lies inside both the sensor-coordinate convex hull and the
                unit circular aperture.

            inside:
                Boolean unit-disk mask with shape (grid_size, grid_size).

            grid_size:
                Target grid resolution.

    Notes:
        Target points outside the convex hull do not belong to a Delaunay
        simplex and remain invalid. Points outside the unit disk are also
        marked invalid, even when they lie inside the convex hull.
    """
    xs, ys, X, Y, R, Theta, inside, step = build_grid(grid_size)

    # Use float64 internally for Delaunay geometry calculations.
    pts = np.asarray(sensor_coords, dtype=np.float64)

    # Flatten the regular Cartesian target grid into shape (G, 2).
    target = np.stack(
        [X.ravel(), Y.ravel()],
        axis=1,
    ).astype(np.float64)

    # Build the Delaunay triangulation of irregular sensor locations.
    tri = Delaunay(pts)

    # For each target point, find the index of the containing simplex.
    # A value of -1 means that the point lies outside the convex hull.
    simplex = tri.find_simplex(target)

    G = target.shape[0]

    # Initialize arrays for all target points. Invalid entries remain zero.
    vertices = np.zeros((G, 3), dtype=np.int64)
    weights = np.zeros((G, 3), dtype=np.float32)

    valid = simplex >= 0

    if np.any(valid):
        s = simplex[valid]

        # Compute barycentric coordinates inside each containing simplex.
        #
        # tri.transform has the affine transform for each simplex:
        #   bary_01 = transform @ (point - offset)
        #   bary_2  = 1 - bary_0 - bary_1
        transform = tri.transform[s, :2]
        delta = target[valid] - tri.transform[s, 2]

        bary_01 = np.einsum(
            "mij,mj->mi",
            transform,
            delta,
        )

        bary_2 = 1.0 - bary_01.sum(axis=1, keepdims=True)

        # Combine all three barycentric weights.
        w = np.concatenate(
            [bary_01, bary_2],
            axis=1,
        )

        # Store triangle vertex indices and interpolation weights.
        vertices[valid] = tri.simplices[s]
        weights[valid] = w.astype(np.float32)

    # Values outside the unit circular aperture are explicitly treated as
    # invalid, even if Delaunay interpolation is geometrically possible.
    valid = valid & inside.ravel()

    return {
        "vertices": vertices,
        "weights": weights,
        "valid": valid,
        "inside": inside,
        "grid_size": grid_size,
    }


def _get_sensor_to_grid_linear_cache(
        sensor_coords: np.ndarray,
        grid_size: int,
):
    """
    Retrieve or construct cached linear sensor-to-grid interpolation data.

    Args:
        sensor_coords: Sensor coordinates with shape (P_sensor, 2).
        grid_size: Resolution of the square target grid.

    Returns:
        Dictionary returned by _precompute_sensor_to_grid_linear.
    """
    key = _sensor_grid_cache_key(sensor_coords, grid_size)

    if key not in _SENSOR_TO_GRID_INTERP_CACHE:
        _SENSOR_TO_GRID_INTERP_CACHE[key] = (
            _precompute_sensor_to_grid_linear(
                sensor_coords=sensor_coords,
                grid_size=grid_size,
            )
        )

    return _SENSOR_TO_GRID_INTERP_CACHE[key]


def interpolate_sensor_derivatives_to_grid(
        derivatives_all: np.ndarray,
        sensor_coords: np.ndarray,
        grid_size: int = 24,
        method: str = "linear",
):
    """
    Interpolate sensor-space gradient fields onto a regular Cartesian grid.

    Args:
        derivatives_all: Gradient values with shape (N, P_sensor, 2), where:
            derivatives_all[..., 0] contains dU/dx values.
            derivatives_all[..., 1] contains dU/dy values.

        sensor_coords: Irregular sensor coordinates with shape (P_sensor, 2).

        grid_size: Resolution of the square output grid.

        method: Interpolation strategy:
            - "linear": Fast precomputed Delaunay triangulation with
              barycentric interpolation weights.
            - "cubic": Cubic interpolation through scipy.interpolate.griddata.

    Returns:
        Interpolated gradient fields with shape:

            (N, grid_size, grid_size, 2)

        Values outside the unit circular aperture are set to zero.

    Raises:
        ValueError: If derivatives_all does not have shape
            (N, P_sensor, 2), if the sensor count does not match
            sensor_coords, or if method is unsupported.

    Notes:
        The "linear" option is generally much faster for repeated use with the
        same sensor layout because the Delaunay geometry and barycentric
        weights are cached.

        The "cubic" option calls scipy.griddata separately for each sample and
        may be useful when comparing against interpolation-based baselines,
        such as Poisson reconstruction or an FNO that operates on regular
        grids.
    """
    derivatives_all = np.asarray(
        derivatives_all,
        dtype=np.float32,
    )

    sensor_coords = np.asarray(
        sensor_coords,
        dtype=np.float32,
    )

    if derivatives_all.ndim != 3 or derivatives_all.shape[-1] != 2:
        raise ValueError(
            "derivatives_all must have shape (N, P_sensor, 2)."
        )

    P_sensor = sensor_coords.shape[0]

    if derivatives_all.shape[1] != P_sensor:
        raise ValueError(
            "Sensor count mismatch: "
            f"derivatives_all.shape[1]={derivatives_all.shape[1]}, "
            f"sensor_coords.shape[0]={P_sensor}."
        )

    method = str(method).lower()

    # Construct the target regular grid.
    xs, ys, X, Y, R, Theta, inside, step = build_grid(int(grid_size))

    target = np.stack(
        [X.ravel(), Y.ravel()],
        axis=1,
    ).astype(np.float32)

    inside_flat = inside.ravel()

    N = derivatives_all.shape[0]
    G = grid_size * grid_size

    if method == "linear":
        # Retrieve Delaunay vertices and barycentric weights from the cache.
        cache = _get_sensor_to_grid_linear_cache(
            sensor_coords,
            grid_size,
        )

        vertices = cache["vertices"]
        weights = cache["weights"]
        valid = cache["valid"]

        # Allocate flattened output before restoring the 2D grid shape.
        out_flat = np.zeros(
            (N, G, 2),
            dtype=np.float32,
        )

        # Keep only valid target points inside both the convex hull and aperture.
        v = vertices[valid]  # Shape: (G_valid, 3)
        w = weights[valid]  # Shape: (G_valid, 3)

        # Interpolate each gradient component independently using barycentric
        # weights over the three sensor vertices of the containing triangle.
        for c in range(2):
            vals = derivatives_all[:, :, c]  # Shape: (N, P_sensor)

            interp_vals = (
                    vals[:, v[:, 0]] * w[None, :, 0]
                    + vals[:, v[:, 1]] * w[None, :, 1]
                    + vals[:, v[:, 2]] * w[None, :, 2]
            )

            out_flat[:, valid, c] = interp_vals

        # Enforce zero values outside the circular pupil.
        out_flat[:, ~inside_flat, :] = 0.0

        out = out_flat.reshape(
            N,
            grid_size,
            grid_size,
            2,
        )

        return out.astype(np.float32)

    elif method == "cubic":
        # Allocate flattened output before restoring the 2D grid shape.
        out_flat = np.zeros(
            (N, G, 2),
            dtype=np.float32,
        )

        # scipy.griddata accepts values with shape (P_sensor, 2) and returns
        # both derivative components at once with shape (G, 2).
        for i in range(N):
            interp = griddata(
                points=sensor_coords,
                values=derivatives_all[i],
                xi=target,
                method="cubic",
                fill_value=0.0,
            ).astype(np.float32)

            # Explicitly zero locations outside the unit circular aperture.
            interp[~inside_flat] = 0.0

            out_flat[i] = interp

        out = out_flat.reshape(
            N,
            grid_size,
            grid_size,
            2,
        )

        return out.astype(np.float32)

    else:
        raise ValueError(
            f"Unknown interpolation method: {method!r}. "
            "Use 'linear' or 'cubic'."
        )
