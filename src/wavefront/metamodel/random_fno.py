from __future__ import annotations

import copy
from argparse import Namespace

import jax

from wavefront.models.factory import setup_fno


def setup_random_fno_state(
        fno_args: Namespace,
        seed: int = 0,
) -> dict:
    """
    Initialize an FNO state without standalone Stage-2 pretraining.

    The resulting model state is intended for direct use in joint end-to-end
    fine-tuning with a pretrained Stage-1 DeepONet gradient-map model.

    The FNO starts from random initialization and learns to reconstruct the
    wavefront through the complete pipeline:

        Sensor gradients
            -> pretrained DeepONet gradient mapper
            -> regular-grid gradient field
            -> randomly initialized FNO
            -> wavefront reconstruction

    Args:
        fno_args: FNO configuration namespace. A deep copy is made before
            model-specific inference settings are updated.
        seed: Random seed used to initialize FNO parameters.

    Returns:
        Dictionary containing:

            args:
                Copied and normalized FNO configuration namespace.

            model:
                Initialized Flax FNO model module.

            model_fn:
                Model function/module returned by ``setup_fno`` and used by
                training and inference helpers.

            params:
                Randomly initialized FNO parameter pytree.

    Notes:
        The function forces:

            args.data_mode = "regular_grid"
            args.in_channels = 2
            args.num_outputs = 1

        These settings match the expected joint-training interface, where the
        FNO receives a regular-grid gradient field with two channels:

            channel 0 = dU/dx
            channel 1 = dU/dy

        and predicts one scalar wavefront value per grid location.
    """
    # Avoid modifying the caller-owned configuration namespace.
    args = copy.deepcopy(fno_args)

    # Initialize a deterministic JAX random stream for FNO parameters.
    key = jax.random.PRNGKey(int(seed))

    # Configure the FNO for regular-grid gradient-to-wavefront reconstruction.
    args.data_mode = "regular_grid"
    args.in_channels = 2
    args.num_outputs = 1

    # Build the architecture and create a parameter pytree with the required
    # input/output structure.
    args, model, model_fn, params = setup_fno(
        args,
        key,
    )

    return {
        "args": args,
        "model": model,
        "model_fn": model_fn,
        "params": params,
    }
