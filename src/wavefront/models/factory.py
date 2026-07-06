import jax
import jax.numpy as jnp

from wavefront.models.deeponet import ModernDeepONet
from wavefront.models.fno import FNO2d


def setup_deeponet(args, key):
    """
    Build and initialize a ModernDeepONet for wavefront reconstruction.

    This notebook currently supports only a single output field:

        Sensor measurements (g_x, g_y) -> reconstructed field U(x, y)

    The DeepONet branch network receives flattened sensor features, while the
    trunk network receives query coordinates (x, y) and predicts the field
    value at those locations.

    Args:
        args: Configuration object containing model and data parameters.
        key: JAX PRNG key used to initialize model parameters.

    Returns:
        A tuple:
            args: Updated configuration object, including total_params.
            model: Initialized ModernDeepONet module instance.
            model_fn: Alias of the model module for compatibility with code
                that expects a callable model object.
            params: Model parameter pytree.
    """

    # This implementation is configured for reconstructing one scalar field
    # U(x, y) at a time.
    args.num_outputs = int(getattr(args, "num_outputs", 1))

    if args.num_outputs != 1:
        raise ValueError(
            "Only num_outputs=1 is supported "
            "(reconstruction of a single field U)."
        )

    # For a single output channel, output-specific branch/trunk splitting is
    # unnecessary. The separable and branch-CNN variants are not used here.
    args.split_trunk = False
    args.split_branch = False
    args.separable = False

    model = ModernDeepONet(
        # The latent basis dimension used for the branch/trunk inner product.
        basis_dim=int(args.hidden_dim),

        # Fixed to one because this setup reconstructs a single scalar field.
        num_outputs=1,

        # Hidden widths of the branch and trunk residual SwiGLU MLPs.
        branch_hidden_dim=int(getattr(args, "branch_hidden_dim", 256)),
        trunk_hidden_dim=int(getattr(args, "trunk_hidden_dim", 256)),

        # Number of residual SwiGLU blocks in each subnetwork.
        branch_num_layers=int(getattr(args, "branch_num_layers", 4)),
        trunk_num_layers=int(getattr(args, "trunk_num_layers", 4)),

        # No output-specific latent representations are needed for one output.
        split_branch=False,
        split_trunk=False,

        # Fourier feature settings for the coordinate-based trunk network.
        trunk_num_frequencies=int(
            getattr(args, "trunk_num_frequencies", 8)
        ),
        trunk_max_freq=float(
            getattr(args, "trunk_max_freq", 16.0)
        ),

        # Dropout is active only when training=True during model application.
        dropout_rate=float(getattr(args, "dropout_rate", 0.0)),
    )

    # Create dummy tensors solely to initialize Flax parameter shapes.
    #
    # The branch input is assumed to be a flattened sensor representation:
    #   (batch_size, n_sensors * branch_input_features)
    #
    # For example, if each sensor has gradient components (g_x, g_y),
    # branch_input_features would typically be 2.
    u_dummy = jnp.ones(
        (
            1,
            int(args.n_sensors) * int(args.branch_input_features),
        ),
        dtype=jnp.float32,
    )

    # The trunk receives one x/y coordinate pair for each sample.
    x_dummy = jnp.ones((1,), dtype=jnp.float32)
    y_dummy = jnp.ones((1,), dtype=jnp.float32)

    # Initialize parameter collections using the provided PRNG key.
    variables = model.init(
        key,
        u_dummy,
        x_dummy,
        y_dummy,
        training=True,
    )

    params = variables["params"]

    # Count all trainable scalar parameters in the Flax pytree.
    args.total_params = sum(
        leaf.size
        for leaf in jax.tree_util.tree_leaves(params)
    )

    print("--- model_summary ---")
    print(f"total params: {args.total_params}")
    print("--- model_summary ---")

    # apply_net expects a model object and a separate parameter pytree.
    # The model is returned twice for compatibility with the existing pipeline.
    return args, model, model, params


def setup_fno(args, key):
    """
    Build and initialize a 2D Fourier Neural Operator.

    The model maps an input field defined on a regular 2D grid to one or more
    output fields on the same grid.

    Args:
        args: Configuration object containing FNO architecture and grid settings.
        key: JAX PRNG key used to initialize model parameters and, when enabled,
            initialize the dropout RNG stream.

    Returns:
        A tuple:
            args: Updated configuration object, including total_params.
            model: Initialized FNO2d module instance.
            model_fn: Alias of the model module for compatibility with code
                that expects a callable model object.
            params: Model parameter pytree.
    """

    model = FNO2d(
        # Number of low-frequency Fourier modes retained on each spatial axis.
        modes1=args.modes1,
        modes2=args.modes2,

        # Latent channel width and number of FNO blocks.
        width=args.width,
        depth=args.depth,

        # Number of input and output field channels.
        in_channels=args.in_channels,
        out_channels=args.num_outputs,

        # Whether normalized x/y coordinate channels are concatenated to input.
        use_grid=getattr(args, "use_grid", True),

        # Optional spatial padding for non-periodic domains.
        pad_size=getattr(args, "pad_size", 0),

        # Dropout configuration.
        dropout_rate=getattr(args, "dropout_rate", 0.0),
        is_training=getattr(args, "is_training", True),
    )

    # Create a dummy batch to infer parameter shapes during Flax initialization.
    #
    # Expected shape:
    #   (batch_size, nx, ny, in_channels)
    dummy = jnp.ones(
        (1, args.nx, args.ny, args.in_channels),
        dtype=jnp.float32,
    )

    # Separate initialization and dropout keys.
    # Flax requires a "dropout" RNG stream only when dropout is enabled.
    params_key, dropout_key = jax.random.split(key, 2)

    rngs = {"params": params_key}

    if getattr(args, "dropout_rate", 0.0) > 0.0:
        rngs["dropout"] = dropout_key

    # Initialize model parameters.
    variables = model.init(rngs, dummy)
    params = variables["params"]

    # Count all trainable scalar parameters in the model parameter pytree.
    args.total_params = sum(
        leaf.size
        for leaf in jax.tree_util.tree_leaves(params)
    )

    print("--- model_summary ---")
    print(f"total params: {args.total_params}")
    print("--- model_summary ---")

    # Keep a separate alias for compatibility with the surrounding training code.
    model_fn = model

    return args, model, model_fn, params
