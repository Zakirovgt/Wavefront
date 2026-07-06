import jax.numpy as jnp
from flax import linen as nn


def _compl_mul2d(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Perform a batched complex-valued multiplication in the Fourier domain.

    Args:
        a: Complex-valued input Fourier coefficients with shape
            (B, X, Y, Cin), where:
                B   = batch size,
                X/Y = retained Fourier-mode dimensions,
                Cin = number of input channels.

        b: Complex-valued learned spectral weights with shape
            (Cin, Cout, X, Y).

    Returns:
        Complex-valued output Fourier coefficients with shape
        (B, X, Y, Cout).

    Notes:
        This computes a channel-wise learned linear transformation for each
        retained Fourier mode independently.
    """
    return jnp.einsum("bxyi,ioxy->bxyo", a, b)


class SpectralConv2d(nn.Module):
    """
    Two-dimensional spectral convolution layer used by Fourier Neural Operators.

    The layer:
        1. Transforms spatial inputs into the Fourier domain using rfft2.
        2. Keeps only a selected number of low-frequency Fourier modes.
        3. Applies learned complex-valued weights to those modes.
        4. Transforms the result back into physical space with irfft2.

    Only low-frequency modes are learned explicitly. The remaining Fourier
    coefficients are set to zero, which acts as a spectral truncation.
    """

    in_channels: int
    out_channels: int
    modes1: int  # Number of retained Fourier modes along the x-axis.
    modes2: int  # Number of retained Fourier modes along the y-axis / rFFT axis.

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply spectral convolution.

        Args:
            x: Real-valued input tensor with shape (B, Nx, Ny, Cin), where:
                B   = batch size,
                Nx  = grid size along the x-axis,
                Ny  = grid size along the y-axis,
                Cin = number of input channels.

        Returns:
            Real-valued output tensor with shape (B, Nx, Ny, Cout).
        """
        B, Nx, Ny, Cin = x.shape

        assert Cin == self.in_channels, (
            f"Cin mismatch: got {Cin}, expected {self.in_channels}"
        )
        assert self.modes1 <= Nx, (
            f"modes1={self.modes1} > Nx={Nx}"
        )
        assert self.modes2 <= (Ny // 2 + 1), (
            f"modes2={self.modes2} > Ny//2+1={Ny // 2 + 1}"
        )

        # Apply a real-valued 2D FFT over spatial dimensions.
        #
        # Input:
        #   x:    (B, Nx, Ny, Cin), real
        #
        # Output:
        #   x_ft: (B, Nx, Ny // 2 + 1, Cin), complex
        #
        # Since the input is real-valued, rfft2 stores only the non-redundant
        # part of the spectrum along the final spatial axis.
        x_ft = jnp.fft.rfft2(x, axes=(1, 2))

        # Store complex weights as separate real and imaginary parameters.
        # This is often more optimizer-friendly than registering complex-valued
        # parameters directly.
        scale = 1.0 / max(1, self.in_channels * self.out_channels)

        # Learned weights for positive x-frequency modes.
        #
        # Shape:
        #   (Cin, Cout, modes1, modes2)
        w1_re = self.param(
            "w1_re",
            nn.initializers.normal(stddev=scale),
            (self.in_channels, self.out_channels, self.modes1, self.modes2),
            jnp.float32,
        )
        w1_im = self.param(
            "w1_im",
            nn.initializers.normal(stddev=scale),
            (self.in_channels, self.out_channels, self.modes1, self.modes2),
            jnp.float32,
        )

        # Learned weights for negative x-frequency modes.
        #
        # Shape:
        #   (Cin, Cout, modes1, modes2)
        w2_re = self.param(
            "w2_re",
            nn.initializers.normal(stddev=scale),
            (self.in_channels, self.out_channels, self.modes1, self.modes2),
            jnp.float32,
        )
        w2_im = self.param(
            "w2_im",
            nn.initializers.normal(stddev=scale),
            (self.in_channels, self.out_channels, self.modes1, self.modes2),
            jnp.float32,
        )

        # Reconstruct complex-valued spectral weights.
        w1 = w1_re + 1j * w1_im
        w2 = w2_re + 1j * w2_im

        # Allocate the output Fourier tensor.
        #
        # All unfilled Fourier coefficients remain zero, so the layer only
        # learns transformations for the selected low-frequency modes.
        out_ft = jnp.zeros(
            (B, Nx, Ny // 2 + 1, self.out_channels),
            dtype=x_ft.dtype,
        )

        # Low-frequency block 1:
        # Positive x-frequency indices [0:modes1].
        #
        # a1:   (B, modes1, modes2, Cin)
        # out1: (B, modes1, modes2, Cout)
        a1 = x_ft[:, :self.modes1, :self.modes2, :]
        out1 = _compl_mul2d(a1, w1)

        out_ft = out_ft.at[
                 :, :self.modes1, :self.modes2, :
                 ].set(out1)

        # Low-frequency block 2:
        # Negative x-frequency indices [-modes1:].
        #
        # Separate weights are used for negative x frequencies because the
        # Fourier coefficients are not generally symmetric after channel mixing.
        #
        # a2:   (B, modes1, modes2, Cin)
        # out2: (B, modes1, modes2, Cout)
        a2 = x_ft[:, -self.modes1:, :self.modes2, :]
        out2 = _compl_mul2d(a2, w2)

        out_ft = out_ft.at[
                 :, -self.modes1:, :self.modes2, :
                 ].set(out2)

        # Transform the truncated, learned Fourier representation back into
        # real-valued physical space.
        #
        # Output shape:
        #   (B, Nx, Ny, Cout)
        y = jnp.fft.irfft2(out_ft, s=(Nx, Ny), axes=(1, 2))

        return y


class FNOBlock2d(nn.Module):
    """
    A single 2D Fourier Neural Operator block.

    Each block combines:
        - A global spectral convolution in Fourier space.
        - A local pointwise 1x1 convolution in physical space.
        - GELU activation.
        - Optional dropout.

    The spectral path captures long-range global interactions, while the
    pointwise convolution mixes channels locally at each grid location.
    """

    width: int
    modes1: int
    modes2: int
    dropout_rate: float = 0.0
    is_training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input feature tensor with shape (B, Nx, Ny, width).

        Returns:
            Output feature tensor with shape (B, Nx, Ny, width).
        """
        init = nn.initializers.glorot_normal()

        # Global interaction path:
        # spectral convolution mixes information across the full spatial domain.
        x1 = SpectralConv2d(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
        )(x)

        # Local interaction path:
        # a 1x1 convolution mixes channels independently at every grid point.
        x2 = nn.Conv(
            self.width,
            kernel_size=(1, 1),
            kernel_init=init,
            name="pwconv",
        )(x)

        # Combine global and local feature transformations.
        x = x1 + x2
        x = nn.activation.gelu(x)

        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(
                x,
                deterministic=not self.is_training,
            )

        return x


class FNO2d(nn.Module):
    """
    Standard 2D Fourier Neural Operator.

    The network maps a field defined on a 2D grid to another field defined on
    the same grid. It is commonly used for learning PDE solution operators,
    such as mappings from initial conditions, forcing fields, or material
    parameters to spatial solution fields.

    Input:
        (B, Nx, Ny, in_channels)

    Output:
        (B, Nx, Ny, out_channels)
    """

    modes1: int
    modes2: int
    width: int = 38
    depth: int = 4
    in_channels: int = 1
    out_channels: int = 1

    # When enabled, normalized x/y coordinates are concatenated as two
    # additional input channels before the lifting layer.
    use_grid: bool = True

    # Zero padding applied on the positive ends of both spatial axes.
    # Padding can help reduce artifacts for non-periodic domains.
    pad_size: int = 0

    dropout_rate: float = 0.0
    is_training: bool = True

    @staticmethod
    def _make_grid(
            Nx: int,
            Ny: int,
            dtype=jnp.float32,
    ) -> jnp.ndarray:
        """
        Construct normalized coordinate channels for a 2D grid.

        Args:
            Nx: Number of grid points along the x-axis.
            Ny: Number of grid points along the y-axis.
            dtype: Desired dtype of the output tensor.

        Returns:
            Coordinate tensor with shape (Nx, Ny, 2), where:
                grid[..., 0] contains x coordinates in [0, 1],
                grid[..., 1] contains y coordinates in [0, 1].
        """
        gx = jnp.linspace(0.0, 1.0, Nx, dtype=dtype)
        gy = jnp.linspace(0.0, 1.0, Ny, dtype=dtype)

        # X and Y each have shape (Nx, Ny).
        X, Y = jnp.meshgrid(gx, gy, indexing="ij")

        # Stack x/y coordinates into two feature channels.
        return jnp.stack([X, Y], axis=-1)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the FNO to a batch of 2D fields.

        Args:
            x: Input tensor with shape (B, Nx, Ny, in_channels).

        Returns:
            Output tensor with shape (B, Nx, Ny, out_channels).
        """
        init = nn.initializers.glorot_normal()

        B, Nx, Ny, Cin = x.shape

        assert Cin == self.in_channels, (
            f"Cin mismatch: got {Cin}, expected {self.in_channels}"
        )

        # Optionally append normalized spatial coordinates.
        #
        # Before concatenation:
        #   x:    (B, Nx, Ny, in_channels)
        #   grid: (B, Nx, Ny, 2)
        #
        # After concatenation:
        #   x:    (B, Nx, Ny, in_channels + 2)
        if self.use_grid:
            grid = self._make_grid(Nx, Ny, dtype=x.dtype)
            grid = jnp.broadcast_to(grid, (B, Nx, Ny, 2))
            x = jnp.concatenate([x, grid], axis=-1)

        # Lift input channels into a higher-dimensional latent representation.
        #
        # This is equivalent to a learnable pointwise linear projection.
        x = nn.Conv(
            self.width,
            kernel_size=(1, 1),
            kernel_init=init,
            name="lift",
        )(x)

        # Optional zero padding.
        #
        # Padding is useful when the target operator is not periodic, since
        # Fourier layers naturally assume periodicity in the transformed grid.
        #
        # Only the positive ends of the two spatial dimensions are padded.
        if self.pad_size > 0:
            p = self.pad_size
            x = jnp.pad(
                x,
                (
                    (0, 0),  # Batch dimension
                    (0, p),  # x-axis
                    (0, p),  # y-axis
                    (0, 0),  # Channel dimension
                ),
            )

        # Apply the stack of Fourier Neural Operator blocks.
        for i in range(self.depth):
            x = FNOBlock2d(
                width=self.width,
                modes1=self.modes1,
                modes2=self.modes2,
                dropout_rate=self.dropout_rate,
                is_training=self.is_training,
                name=f"fno_block_{i}",
            )(x)

        # Crop the result back to the original spatial resolution.
        if self.pad_size > 0:
            x = x[:, :Nx, :Ny, :]

        # Projection head:
        # width -> 2 * width -> out_channels.
        #
        # Both convolutions are 1x1, so this acts independently at each grid
        # point while using the learned latent features from the FNO stack.
        x = nn.Conv(
            self.width * 2,
            kernel_size=(1, 1),
            kernel_init=init,
            name="proj1",
        )(x)

        x = nn.activation.gelu(x)

        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(
                x,
                deterministic=not self.is_training,
            )

        x = nn.Conv(
            self.out_channels,
            kernel_size=(1, 1),
            kernel_init=init,
            name="proj2",
        )(x)

        return x


__all__ = [
    "FNO2d",
    "FNOBlock2d",
    "SpectralConv2d",
]
