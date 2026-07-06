from functools import lru_cache

import numpy as np


def butterworth(
        N: int,
        fc: float,
        z: np.ndarray,
) -> np.ndarray:
    """
    Apply a 2D low-pass Butterworth filter in the frequency domain.

    The frequency-domain filter is cached by input shape, filter order, and
    cutoff frequency. Repeated calls with identical settings therefore avoid
    rebuilding the filter mask.

    Args:
        N: Butterworth filter order. Larger values create a sharper transition
            between passed and attenuated frequency components.
        fc: Cutoff frequency measured in frequency-grid pixels.
        z: Real-valued input array with shape (H, W).

    Returns:
        Filtered real-valued array with shape (H, W) and dtype float32.

    Notes:
        The processing steps are:

            1. Compute the 2D Fourier transform of the input.
            2. Shift the zero-frequency component to the spectrum center.
            3. Multiply by the cached low-pass Butterworth filter.
            4. Undo the frequency shift.
            5. Apply the inverse Fourier transform.

        The real part of the inverse transform is returned because the input
        and filter are real-valued, and any remaining imaginary component is
        expected to be numerical round-off noise.
    """
    z = np.asarray(z, dtype=np.float32)

    # Retrieve a cached frequency-domain filter or construct it on the first
    # call for this combination of shape, order, and cutoff frequency.
    butterworth_filter = _cached_butterworth_filter(
        z.shape,
        int(N),
        float(fc),
    )

    # Transform the input image into the 2D Fourier domain.
    fft_z = np.fft.fft2(z)

    # Move the zero-frequency component from the corner to the center.
    fft_z_shifted = np.fft.fftshift(fft_z)

    # Suppress high-frequency components according to the low-pass filter.
    filtered_fft_z_shifted = fft_z_shifted * butterworth_filter

    # Restore the original FFT coefficient arrangement.
    filtered_fft_z = np.fft.ifftshift(filtered_fft_z_shifted)

    # Transform back to the spatial domain.
    filtered_z = np.real(np.fft.ifft2(filtered_fft_z))

    return filtered_z.astype(np.float32)


@lru_cache(maxsize=32)
def _cached_butterworth_filter(
        shape,
        order: int,
        fc: float,
) -> np.ndarray:
    """
    Construct and cache a centered 2D low-pass Butterworth filter.

    Args:
        shape: Spatial array shape as (H, W).
        order: Butterworth filter order.
        fc: Cutoff frequency measured in frequency-grid pixels.

    Returns:
        Frequency-domain filter mask with shape (H, W) and dtype float32.

    Notes:
        The filter is defined as:

            H(D) = 1 / (1 + (D / fc)^(2 * order))

        where D is the distance from the center of the shifted Fourier
        spectrum. Low-frequency components near the center have values close
        to 1, while high-frequency components are increasingly attenuated.
    """
    H, W = shape

    # Construct row and column index grids.
    i = np.arange(H, dtype=np.float32)[:, None]
    j = np.arange(W, dtype=np.float32)[None, :]

    # Compute the Euclidean distance from the center of the shifted spectrum.
    #
    # The center convention below matches the original implementation.
    dist = np.sqrt(
        (i - (H / 2 + 1)) ** 2
        + (j - (W / 2 + 1)) ** 2
    )

    # Low-pass Butterworth frequency response.
    filt = 1.0 / (1.0 + (dist / fc) ** (2 * order))

    return filt.astype(np.float32)
