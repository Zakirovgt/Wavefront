import jax.numpy as jnp
import numpy as np

from wavefront.data.interpolation import (
    interpolate_sensor_derivatives_to_grid,
)


def prepare_fno_arrays(
        args,
        grad_sensor_data,
        wavefront_true_data,
        grad_grid_data=None,
        sensor_coords=None,
):
    """
    Prepare input and target arrays for training or evaluating an FNO model.

    The function converts wavefront targets into a regular 2D grid format and
    obtains FNO input channels from either:

        - Precomputed regular-grid derivatives.
        - Sensor-space derivatives interpolated onto a regular grid.

    Args:
        args: Configuration object expected to contain:
            - nx: Grid size along the x-axis.
            - ny: Grid size along the y-axis.
            - data_mode: Either "regular_grid" or "sensor".
            - sensor_to_grid_interp: Optional interpolation method used when
              data_mode is "sensor". Defaults to "linear".

        grad_sensor_data: Sensor gradient data. Expected shape is typically:

            (N, P_sensor, 2)

            where the final dimension contains [dU/dx, dU/dy].

        wavefront_true_data: Ground-truth wavefront data. Supported shapes:

            (N, nx * ny)
            (N, nx, ny)
            (N, nx, ny, 1)

        grad_grid_data: Optional regular-grid derivative data. Required when
            args.data_mode is "regular_grid". Supported shapes include:

            (N, nx * ny, 2)
            (N, nx, ny, 2)

        sensor_coords: Optional sensor coordinates with shape (P_sensor, 2).
            Required when args.data_mode is "sensor".

    Returns:
        A tuple:

            X_fno:
                FNO input array with shape (N, nx, ny, C), converted to a
                JAX array.

            U_all:
                Target wavefront array with shape (N, nx, ny), converted to
                a JAX array.

    Raises:
        ValueError: If the wavefront shape is unsupported, required derivative
            data is missing, derivative channels are invalid, sensor
            coordinates are missing in sensor mode, or data_mode is unknown.

    Side Effects:
        Updates args.in_channels to match the number of channels in X_fno.
    """
    # Convert wavefront targets to float32 NumPy arrays before reshaping.
    U_all = np.asarray(wavefront_true_data, dtype=np.float32)

    # Convert flattened wavefronts into regular 2D fields.
    if U_all.ndim == 2:
        U_all = U_all.reshape(
            U_all.shape[0],
            args.nx,
            args.ny,
        )

    # Remove a trailing singleton output-channel dimension when present.
    elif U_all.ndim == 3 and U_all.shape[-1] == 1:
        U_all = U_all[..., 0]

    # A valid target array must now have shape (N, nx, ny).
    elif U_all.ndim != 3:
        raise ValueError(
            f"Unexpected U_all shape: {U_all.shape}"
        )

    if args.data_mode == "regular_grid":
        # In regular-grid mode, derivatives are expected to already be aligned
        # with the FNO spatial grid.
        if grad_grid_data is None:
            raise ValueError(
                "grad_grid_data must be provided for FNO when "
                "data_mode='regular_grid'."
            )

        X_fno = np.asarray(
            grad_grid_data,
            dtype=np.float32,
        )

        # Convert flattened regular-grid gradients:
        #
        # (N, nx * ny, 2) -> (N, nx, ny, 2)
        if X_fno.ndim == 3:
            X_fno = X_fno.reshape(
                X_fno.shape[0],
                args.nx,
                args.ny,
                2,
            )

        # The final channel dimension stores the gradient components:
        # [dU/dx, dU/dy].
        if X_fno.shape[-1] != 2:
            raise ValueError(
                "Expected the final gradient dimension to have size 2 "
                f"for gx and gy, but got shape {X_fno.shape}."
            )

    elif args.data_mode == "sensor":
        # In sensor mode, interpolate irregular sensor gradients onto the
        # regular grid required by the FNO.
        if sensor_coords is None:
            raise ValueError(
                "sensor_coords must be provided for FNO when "
                "data_mode='sensor' so sensor derivatives can be "
                "interpolated onto the regular grid."
            )

        X_fno = interpolate_sensor_derivatives_to_grid(
            derivatives_all=np.asarray(
                grad_sensor_data,
                dtype=np.float32,
            ),
            sensor_coords=np.asarray(
                sensor_coords,
                dtype=np.float32,
            ),
            grid_size=args.nx,
            method=str(
                getattr(
                    args,
                    "sensor_to_grid_interp",
                    "linear",
                )
            ),
        )

    else:
        raise ValueError(
            f"Unknown data_mode: {args.data_mode}"
        )

    # Keep model configuration synchronized with the prepared input tensor.
    args.in_channels = int(X_fno.shape[-1])

    # Convert final arrays to JAX arrays for downstream JAX/Flax pipelines.
    return jnp.asarray(X_fno), jnp.asarray(U_all)
