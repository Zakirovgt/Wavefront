import jax.numpy as jnp

# Global mixed-precision configuration shared by neural-operator modules.
_MP_ENABLED = False
_MP_DTYPE = jnp.bfloat16


def set_mixed_precision(
        enabled: bool,
        dtype: str = "bfloat16",
):
    """
    Configure global mixed-precision behavior for neural operator models.

    Args:
        enabled: Whether mixed precision should be enabled globally.
        dtype: Low-precision floating-point type to use when enabled.
            Supported values are:
                - "bfloat16": Recommended in most cases because it usually
                  does not require loss scaling.
                - "float16": Can reduce memory use further, but may require
                  loss scaling for numerically stable training.

    Raises:
        ValueError: If dtype is not "bfloat16" or "float16".

    Notes:
        This function updates module-level global state. Any code that calls
        _maybe_mp after this configuration is changed will cast tensors to the
        selected low-precision dtype when mixed precision is enabled.
    """
    global _MP_ENABLED, _MP_DTYPE

    # Store the global mixed-precision activation flag.
    _MP_ENABLED = bool(enabled)

    # Select the target lower-precision JAX dtype.
    if dtype == "bfloat16":
        _MP_DTYPE = jnp.bfloat16
    elif dtype == "float16":
        _MP_DTYPE = jnp.float16
    else:
        raise ValueError(
            f"Unknown mixed precision dtype: {dtype!r}. "
            "Use 'bfloat16' or 'float16'."
        )


def _maybe_mp(x):
    """
    Cast an array to the configured mixed-precision dtype when enabled.

    Args:
        x: Input JAX array or array-like object supporting .astype().

    Returns:
        The input converted to _MP_DTYPE when mixed precision is enabled;
        otherwise, the original input is returned unchanged.
    """
    return x.astype(_MP_DTYPE) if _MP_ENABLED else x
