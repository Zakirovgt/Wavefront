import numpy as np


def spiral_phase(
        X: np.ndarray,
        Y: np.ndarray,
        kappa: float,
        n: int,
        rotation_angle: float = 0.0,
        eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute the unwrapped phase of a spiral wavefront.

    Args:
        X: x-coordinate array with shape (...,).
        Y: y-coordinate array with shape (...,).
        kappa: Radial phase-growth coefficient. Larger values produce more
            tightly spaced radial spiral features.
        n: Azimuthal winding number. This controls the angular phase variation.
        rotation_angle: Additional global phase rotation in radians.
        eps: Small positive constant added inside the radial square root for
            numerical stability near the origin.

    Returns:
        Unwrapped phase array with the same shape as X and Y.

    Notes:
        The phase is defined as:

            phi(r, theta) =
                kappa * r
                + n * (theta - pi / 2)
                + rotation_angle

        where:

            r     = sqrt(X^2 + Y^2)
            theta = arctan2(Y, X)

        No modulo operation is applied, so the returned phase remains
        continuous apart from the inherent angular branch cut of arctan2.
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    # Compute polar coordinates from Cartesian coordinates.
    r = np.sqrt(X * X + Y * Y + eps)
    theta = np.arctan2(Y, X)

    # Construct the radial-plus-azimuthal spiral phase.
    phi = kappa * r + n * (theta - np.pi / 2.0) + rotation_angle

    return phi.astype(np.float32)


def spiral_field(
        X: np.ndarray,
        Y: np.ndarray,
        kappa: float,
        n: int,
        rotation_angle: float = 0.0,
        nonsmooth: bool = False,
        smooth_mode: str = "ridge",
) -> np.ndarray:
    """
    Generate a smooth spiral field without applying phase wrapping modulo 2*pi.

    Args:
        X: x-coordinate array with shape (...,).
        Y: y-coordinate array with shape (...,).
        kappa: Radial phase-growth coefficient.
        n: Azimuthal winding number.
        rotation_angle: Additional global phase rotation in radians.
        nonsmooth: Reserved argument for compatibility with alternative field
            generation modes. It is not used in the current implementation.
        smooth_mode: Output transformation applied to the spiral phase:
            - "cos": Returns U = cos(phase), with values in [-1, 1].
            - "ridge": Returns U = 0.5 * (1 + cos(phase)), with values in [0, 1].

    Returns:
        Smooth spiral field with the same shape as X and Y.

    Raises:
        ValueError: If smooth_mode is not "cos" or "ridge".

    Notes:
        The underlying phase is:

            phase =
                kappa * r
                + n * (theta - pi / 2)
                + rotation_angle

        The "ridge" mode shifts and rescales the cosine pattern so that
        bright spiral ridges have values close to 1 and dark regions have
        values close to 0.
    """
    # Keep local aliases for consistency with the Cartesian coordinate notation.
    xz = X
    yz = Y

    # Convert Cartesian coordinates to polar coordinates.
    r = np.sqrt(xz * xz + yz * yz)
    theta = np.arctan2(yz, xz)

    # Compute the continuous spiral phase.
    phase = kappa * r + n * (theta - np.pi / 2.0) + rotation_angle

    # Convert phase into the requested smooth scalar field.
    if smooth_mode == "cos":
        U = np.cos(phase)
    elif smooth_mode == "ridge":
        U = 0.5 * (1.0 + np.cos(phase))
    else:
        raise ValueError(f"Unknown smooth_mode: {smooth_mode}")

    return U.astype(np.float32)


def spiral_wavefront_at_points(
        x: np.ndarray,
        y: np.ndarray,
        kappa: float,
        n: int,
        rotation_angle: float = 0.0,
        amp: float = 10.0,
        nonsmooth: bool = False,
        smooth_mode: str = "ridge",
) -> np.ndarray:
    """
    Evaluate a scaled spiral wavefront at arbitrary Cartesian coordinates.

    Args:
        x: x-coordinate array with shape (...,).
        y: y-coordinate array with shape (...,).
        kappa: Radial phase-growth coefficient.
        n: Azimuthal winding number.
        rotation_angle: Additional global phase rotation in radians.
        amp: Amplitude multiplier applied to the generated spiral field.
        nonsmooth: Reserved argument passed to spiral_field. It is not used by
            the current smooth field implementation.
        smooth_mode: Field transformation mode passed to spiral_field.

    Returns:
        Wavefront values with the same shape as x and y.

    Notes:
        The wavefront is evaluated only inside the unit circular aperture:

            x^2 + y^2 <= 1

        Values outside this disk are set explicitly to zero.
    """
    # Convert coordinate inputs to a consistent floating-point representation.
    X = np.asarray(x, dtype=np.float32)
    Y = np.asarray(y, dtype=np.float32)

    # Generate and scale the smooth spiral field.
    U = amp * spiral_field(
        X,
        Y,
        kappa=kappa,
        n=n,
        rotation_angle=rotation_angle,
        nonsmooth=nonsmooth,
        smooth_mode=smooth_mode,
    )

    # Apply the unit-disk pupil mask.
    r = np.sqrt(X ** 2 + Y ** 2)
    U[r > 1.0] = 0.0

    return U.astype(np.float32)
