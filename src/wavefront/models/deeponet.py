import jax.numpy as jnp
from flax import linen as nn


class FourierFeatures(nn.Module):
    num_frequencies: int = 8
    max_freq: float = 16.0
    include_input: bool = True

    @nn.compact
    def __call__(self, coords):
        """
        Apply Fourier feature encoding to 2D coordinates.

        Args:
            coords: Array with shape (..., 2), containing x/y coordinates
                expected to be normalized to the range [-1, 1].

        Returns:
            Encoded coordinates with shape:
                (..., 2 + 4 * num_frequencies), when include_input=True
                (..., 4 * num_frequencies), otherwise.

            Each input coordinate is expanded with sine and cosine features
            at geometrically spaced frequencies.
        """

        freqs = jnp.geomspace(
            1.0,
            self.max_freq,
            self.num_frequencies,
        )

        # Expand coordinates over the frequency axis:
        # coords: (..., 2)
        # freqs:  (num_frequencies,)
        # xb:     (..., num_frequencies, 2)
        xb = coords[..., None, :] * freqs[:, None] * jnp.pi

        # Flatten the frequency and coordinate dimensions:
        # (..., num_frequencies, 2) -> (..., 2 * num_frequencies)
        xb = xb.reshape(*coords.shape[:-1], -1)

        # Apply sine and cosine encodings:
        # (..., 2 * num_frequencies) -> (..., 4 * num_frequencies)
        encoded = jnp.concatenate(
            [jnp.sin(xb), jnp.cos(xb)],
            axis=-1,
        )

        # Optionally preserve the original x/y coordinates.
        if self.include_input:
            encoded = jnp.concatenate([coords, encoded], axis=-1)

        return encoded


class ResidualSwiGLUBlock(nn.Module):
    """
    Residual MLP block using SwiGLU-style gating.

    The block performs:
        LayerNorm (optional)
        -> two independent projections for gate and value
        -> SiLU(gate) * value
        -> output projection
        -> dropout (optional)
        -> residual addition when feature dimensions match
    """

    hidden_dim: int
    out_dim: int
    dropout_rate: float = 0.0
    use_layernorm: bool = True

    @nn.compact
    def __call__(self, x, training: bool = True):
        """
        Args:
            x: Input tensor with shape (..., input_dim).
            training: Whether the module is in training mode. Controls dropout.

        Returns:
            Tensor with shape (..., out_dim).
        """
        residual = x

        if self.use_layernorm:
            x = nn.LayerNorm()(x)

        # SwiGLU uses separate gate and value projections.
        gate = nn.Dense(self.hidden_dim)(x)
        value = nn.Dense(self.hidden_dim)(x)

        # Gated activation:
        # SiLU(gate) modulates the value projection element-wise.
        x = nn.silu(gate) * value
        x = nn.Dense(self.out_dim)(x)

        if self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate)(
                x,
                deterministic=not training,
            )

        # Add a residual connection only when input and output dimensions match.
        if residual.shape[-1] == x.shape[-1]:
            x = x + residual

        return x


class ResidualSwiGLUMLP(nn.Module):
    """
    Multi-layer residual SwiGLU MLP.

    An input projection maps features into hidden_dim, followed by a stack of
    residual SwiGLU blocks. A final projection maps the hidden representation
    to out_dim.
    """

    hidden_dim: int
    out_dim: int
    num_layers: int = 4
    expansion: int = 2
    dropout_rate: float = 0.0
    use_layernorm: bool = True

    @nn.compact
    def __call__(self, x, training: bool = True):
        """
        Args:
            x: Input tensor with shape (..., input_dim).
            training: Whether dropout should be active.

        Returns:
            Output tensor with shape (..., out_dim).
        """

        # Project the input into the model hidden dimension.
        x = nn.Dense(self.hidden_dim)(x)

        # Each residual block expands internally to:
        # expansion * hidden_dim, then projects back to hidden_dim.
        for _ in range(self.num_layers):
            x = ResidualSwiGLUBlock(
                hidden_dim=self.expansion * self.hidden_dim,
                out_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate,
                use_layernorm=self.use_layernorm,
            )(x, training=training)

        # Normalize the final hidden representation before the output head.
        if self.use_layernorm:
            x = nn.LayerNorm()(x)

        x = nn.Dense(self.out_dim)(x)

        return x


class SwiGLUBranchNet(nn.Module):
    """
    Branch network for DeepONet.

    The branch network receives a representation of the input function u
    (for example, sensor values or discretized initial/boundary conditions)
    and produces latent basis coefficients.
    """

    basis_dim: int
    num_outputs: int = 1
    split_branch: bool = False
    hidden_dim: int = 256
    num_layers: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, u, training: bool = True):
        """
        Args:
            u: Input function representation with shape (B, branch_dim).
            training: Whether dropout should be active.

        Returns:
            If split_branch=False:
                Tensor with shape (B, basis_dim).

            If split_branch=True:
                Tensor with shape (B, basis_dim * num_outputs), where each
                output channel receives a separate set of branch coefficients.
        """
        if self.split_branch:
            out_dim = self.basis_dim * self.num_outputs
        else:
            out_dim = self.basis_dim

        return ResidualSwiGLUMLP(
            hidden_dim=self.hidden_dim,
            out_dim=out_dim,
            num_layers=self.num_layers,
            expansion=2,
            dropout_rate=self.dropout_rate,
            use_layernorm=True,
        )(u, training=training)


class FourierSwiGLUTrunkNet(nn.Module):
    """
    Trunk network for DeepONet with Fourier coordinate encoding.

    The trunk network receives spatial coordinates and produces latent basis
    values. Fourier features help the network represent spatial functions with
    high-frequency structure more effectively.
    """

    basis_dim: int
    num_outputs: int = 1
    split_trunk: bool = False
    hidden_dim: int = 256
    num_layers: int = 4
    num_frequencies: int = 8
    max_freq: float = 16.0
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, coords, training: bool = True):
        """
        Args:
            coords: Coordinate tensor with shape (..., 2), containing x/y.
            training: Whether dropout should be active.

        Returns:
            If split_trunk=False:
                Tensor with shape (..., basis_dim).

            If split_trunk=True:
                Tensor with shape (..., basis_dim * num_outputs).
        """

        # Convert raw coordinates into multi-scale Fourier features.
        z = FourierFeatures(
            num_frequencies=self.num_frequencies,
            max_freq=self.max_freq,
            include_input=True,
        )(coords)

        # When the trunk is split, each output channel gets its own
        # basis_dim-dimensional trunk representation.
        if self.split_trunk:
            out_dim = self.basis_dim * self.num_outputs
        else:
            out_dim = self.basis_dim

        z = ResidualSwiGLUMLP(
            hidden_dim=self.hidden_dim,
            out_dim=out_dim,
            num_layers=self.num_layers,
            expansion=2,
            dropout_rate=self.dropout_rate,
            use_layernorm=True,
        )(z, training=training)

        return z


class ModernDeepONet(nn.Module):
    """
    DeepONet architecture with:
    - Residual SwiGLU MLPs for branch and trunk networks
    - Fourier features for spatial coordinates
    - Optional output-specific branch/trunk latent representations

    The model approximates an operator mapping an input function u and a
    spatial query location (x, y) to one or more output values.
    """

    basis_dim: int = 128
    num_outputs: int = 1

    branch_hidden_dim: int = 256
    trunk_hidden_dim: int = 256

    branch_num_layers: int = 4
    trunk_num_layers: int = 4

    split_branch: bool = True
    split_trunk: bool = False

    trunk_num_frequencies: int = 8
    trunk_max_freq: float = 16.0

    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, u, x, y, training: bool = True):
        """
        Evaluate the DeepONet at a batch of spatial query locations.

        Args:
            u: Branch input with shape (B, branch_dim). This represents the
                input function, for example values at sensor locations.
            x: x-coordinate tensor with shape (B,).
            y: y-coordinate tensor with shape (B,).
            training: Whether dropout should be active.

        Returns:
            If num_outputs=1:
                Tensor with shape (B, 1), representing U(x, y).

            If num_outputs=q:
                Tensor with shape (B, q), representing q output channels.
        """

        # Combine x and y coordinates into a single coordinate tensor.
        coords = jnp.stack([x, y], axis=-1)  # Shape: (B, 2)

        branch_out = SwiGLUBranchNet(
            basis_dim=self.basis_dim,
            num_outputs=self.num_outputs,
            split_branch=self.split_branch,
            hidden_dim=self.branch_hidden_dim,
            num_layers=self.branch_num_layers,
            dropout_rate=self.dropout_rate,
        )(u, training=training)

        trunk_out = FourierSwiGLUTrunkNet(
            basis_dim=self.basis_dim,
            num_outputs=self.num_outputs,
            split_trunk=self.split_trunk,
            hidden_dim=self.trunk_hidden_dim,
            num_layers=self.trunk_num_layers,
            num_frequencies=self.trunk_num_frequencies,
            max_freq=self.trunk_max_freq,
            dropout_rate=self.dropout_rate,
        )(coords, training=training)

        return self.merge_outputs(branch_out, trunk_out)

    def merge_outputs(self, branch_out, trunk_out):
        """
        Combine branch and trunk latent vectors through an inner product.

        The exact merge rule depends on whether the branch and/or trunk network
        uses separate latent representations for each output channel.

        Args:
            branch_out: Branch output tensor.
            trunk_out: Trunk output tensor.

        Returns:
            Model predictions with shape (B, num_outputs), except that a
            single-output model returns shape (B, 1).
        """

        B = branch_out.shape[0]
        p = self.basis_dim
        q = self.num_outputs

        # Standard single-output DeepONet inner product:
        # branch_out: (B, p)
        # trunk_out:  (B, p)
        # output:     (B, 1)
        if self.num_outputs == 1:
            return jnp.sum(branch_out * trunk_out, axis=-1, keepdims=True)

        if self.split_branch and not self.split_trunk:
            # Output-specific branch coefficients and a shared trunk basis:
            # branch_out: (B, q * p) -> (B, q, p)
            # trunk_out:  (B, p)
            # output:     (B, q)
            branch_out = branch_out.reshape(B, q, p)
            return jnp.einsum("bqp,bp->bq", branch_out, trunk_out)

        elif not self.split_branch and self.split_trunk:
            # Shared branch coefficients and output-specific trunk basis:
            # branch_out: (B, p)
            # trunk_out:  (B, q * p) -> (B, q, p)
            # output:     (B, q)
            trunk_out = trunk_out.reshape(B, q, p)
            return jnp.einsum("bp,bqp->bq", branch_out, trunk_out)

        elif self.split_branch and self.split_trunk:
            # Both branch and trunk have output-specific latent basis vectors:
            # branch_out: (B, q * p) -> (B, q, p)
            # trunk_out:  (B, q * p) -> (B, q, p)
            # output:     (B, q)
            branch_out = branch_out.reshape(B, q, p)
            trunk_out = trunk_out.reshape(B, q, p)
            return jnp.einsum("bqp,bqp->bq", branch_out, trunk_out)

        else:
            # With multiple outputs and no splitting, the model only produces
            # one shared scalar prediction. It is repeated across all channels.
            #
            # In most multi-output applications, split_branch=True and/or
            # split_trunk=True is preferable so each output can have its own
            # learnable latent representation.
            out_scalar = jnp.sum(branch_out * trunk_out, axis=-1, keepdims=True)
            return jnp.repeat(out_scalar, q, axis=-1)


__all__ = [
    "FourierFeatures",
    "ResidualSwiGLUBlock",
    "ResidualSwiGLUMLP",
    "SwiGLUBranchNet",
    "FourierSwiGLUTrunkNet",
    "ModernDeepONet",
]
