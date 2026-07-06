import jax
import jax.numpy as jnp


def _tree_l2_norm(tree):
    """
    Compute the global L2 norm of all array leaves in a JAX pytree.

    Args:
        tree: A JAX pytree containing arrays, such as a gradient pytree,
            parameter pytree, or optimizer-state subtree.

    Returns:
        Scalar L2 norm of all values across every array leaf.

    Notes:
        The norm is computed as:

            sqrt(sum_i ||x_i||_2^2)

        where x_i is each array leaf in the pytree.

        For gradient trees, this is commonly used for global gradient-norm
        monitoring or gradient clipping.
    """
    # Flatten the pytree into its array leaves.
    leaves = jax.tree_util.tree_leaves(tree)

    # Compute:
    #
    #   sqrt(sum_i <x_i, x_i>)
    #
    # jnp.vdot handles real and complex arrays correctly by conjugating the
    # first argument when necessary.
    return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))
