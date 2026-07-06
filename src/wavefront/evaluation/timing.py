import time

import jax.numpy as jnp
import numpy as np

from wavefront.baselines.poisson import poisson_baseline_from_deriv_grid
from wavefront.training.deeponet import apply_net_tasks
from wavefront.training.fno import apply_fno


def benchmark_inference_deeponet(
        model_fn,
        params,
        grad_sensor_test,
        p_test,
        n: int = 1000,
        batch_size: int = 64,
):
    """
    Benchmark batched DeepONet inference over a set of sensor-gradient inputs.

    Each input sample is interpreted as one sensor-gradient field. The function
    reconstructs the wavefront at every point of a regular square query grid.

    Args:
        model_fn: Flax DeepONet model module.
        params: Model parameter pytree.
        grad_sensor_test: Sensor-gradient test data with shape
            (N_test, P_sensor, 2), or any shape that can be flattened into
            one branch input per sample.
        p_test: Number of spatial query points per reconstructed wavefront.
            This value must be a perfect square.
        n: Maximum number of test samples to benchmark.
        batch_size: Number of complete functions evaluated per inference batch.

    Returns:
        Dictionary containing:

            n:
                Number of benchmarked functions.

            total_s:
                Total measured inference time in seconds.

            ms_per_function:
                Average inference time per reconstructed function in
                milliseconds.

            functions_per_s:
                Average number of reconstructed functions per second.

            batch_size:
                Batch size used during benchmarking.

    Notes:
        One warm-up inference is performed before timing to trigger JAX
        compilation and avoid including compilation overhead in the measured
        runtime.
    """
    # Limit the benchmark to the number of available test examples.
    n = int(min(int(n), grad_sensor_test.shape[0]))

    # Construct the regular square coordinate grid over [-1, 1] x [-1, 1].
    side = int(np.sqrt(int(p_test)))

    xy = jnp.linspace(
        -1.0,
        1.0,
        side,
        dtype=jnp.float32,
    )

    xg, yg = jnp.meshgrid(xy, xy)

    coords = jnp.stack(
        [xg.ravel(), yg.ravel()],
        axis=1,
    )

    # Run one warm-up batch to trigger JAX compilation before timing.
    warm_B = min(batch_size, n)

    warm = jnp.asarray(
        grad_sensor_test[:warm_B],
        dtype=jnp.float32,
    )

    warm_branch = warm.reshape(warm_B, -1)

    warm_coords = jnp.broadcast_to(
        coords[None, :, :],
        (warm_B, coords.shape[0], 2),
    )

    apply_net_tasks(
        model_fn,
        params,
        warm_branch,
        warm_coords,
        rng=None,
    ).block_until_ready()

    # Measure batched inference time while synchronizing each device result.
    t0 = time.perf_counter()
    done = 0

    while done < n:
        end = min(done + batch_size, n)
        B = end - done

        batch = jnp.asarray(
            grad_sensor_test[done:end],
            dtype=jnp.float32,
        )

        # Flatten each sensor-gradient field into a DeepONet branch input.
        branch = batch.reshape(B, -1)

        # Reuse the same regular query grid for all functions in the batch.
        coords_b = jnp.broadcast_to(
            coords[None, :, :],
            (B, coords.shape[0], 2),
        )

        apply_net_tasks(
            model_fn,
            params,
            branch,
            coords_b,
            rng=None,
        ).block_until_ready()

        done = end

    t1 = time.perf_counter()
    total_s = float(t1 - t0)

    return {
        "n": n,
        "total_s": total_s,
        "ms_per_function": 1000.0 * total_s / max(1, n),
        "functions_per_s": float(n) / max(1e-12, total_s),
        "batch_size": int(batch_size),
    }


def benchmark_inference_fno(
        model_fn,
        params,
        X_test,
        n: int = 1000,
        batch_size: int = 64,
):
    """
    Benchmark batched FNO inference over regular-grid input fields.

    Args:
        model_fn: Flax FNO model module.
        params: Model parameter pytree.
        X_test: FNO test inputs with shape (N_test, H, W, C).
        n: Maximum number of test samples to benchmark.
        batch_size: Number of complete grid samples evaluated per batch.

    Returns:
        Dictionary containing:

            n:
                Number of benchmarked functions.

            total_s:
                Total measured inference time in seconds.

            ms_per_function:
                Average inference time per function in milliseconds.

            functions_per_s:
                Average number of functions processed per second.

            batch_size:
                Batch size used during benchmarking.

    Notes:
        A warm-up inference is executed before timing to exclude JAX compilation
        overhead from the benchmark result.
    """
    # Limit the benchmark to available test examples.
    n = int(min(int(n), X_test.shape[0]))

    # Trigger JAX compilation with a warm-up batch.
    warm_B = min(batch_size, n)

    apply_fno(
        model_fn,
        params,
        jnp.asarray(
            X_test[:warm_B],
            dtype=jnp.float32,
        ),
        rng=None,
        training=False,
    ).block_until_ready()

    # Measure batched inference time.
    t0 = time.perf_counter()
    done = 0

    while done < n:
        end = min(done + batch_size, n)

        apply_fno(
            model_fn,
            params,
            jnp.asarray(
                X_test[done:end],
                dtype=jnp.float32,
            ),
            rng=None,
            training=False,
        ).block_until_ready()

        done = end

    t1 = time.perf_counter()
    total_s = float(t1 - t0)

    return {
        "n": n,
        "total_s": total_s,
        "ms_per_function": 1000.0 * total_s / max(1, n),
        "functions_per_s": float(n) / max(1e-12, total_s),
        "batch_size": int(batch_size),
    }


def benchmark_inference_poisson(
        deriv_grid_test,
        n: int = 1000,
        grid_size: int = 24,
        use_circular_mask: bool = True,
):
    """
    Benchmark the least-squares Poisson reconstruction baseline.

    Each input sample contains a regular-grid gradient field. The benchmark
    reconstructs one scalar wavefront per sample using gradient integration.

    Args:
        deriv_grid_test: Gradient-field test data with shape
            (N_test, H, W, 2), or a flattened equivalent representation.
        n: Maximum number of test samples to benchmark.
        grid_size: Resolution of the square gradient grid.
        use_circular_mask: Whether reconstruction should be restricted to the
            unit circular aperture.

    Returns:
        Dictionary containing:

            n:
                Number of benchmarked reconstructions.

            total_s:
                Total measured reconstruction time in seconds.

            ms_per_function:
                Average reconstruction time per wavefront in milliseconds.

            functions_per_s:
                Average number of reconstructed wavefronts per second.

    Notes:
        One initial reconstruction is run before timing. Unlike JAX models,
        this baseline does not require JIT compilation, but the warm-up keeps
        the benchmarking procedure consistent with the neural-operator paths.
    """
    deriv_grid_test = np.asarray(deriv_grid_test)

    # Limit the benchmark to available derivative fields.
    n = int(min(int(n), deriv_grid_test.shape[0]))

    # Run one warm-up reconstruction before timing.
    poisson_baseline_from_deriv_grid(
        deriv_grid=deriv_grid_test[0],
        U_true=None,
        grid_size=int(grid_size),
        use_circular_mask=bool(use_circular_mask),
    )

    # Measure sequential Poisson reconstructions.
    t0 = time.perf_counter()

    for i in range(n):
        poisson_baseline_from_deriv_grid(
            deriv_grid=deriv_grid_test[i],
            U_true=None,
            grid_size=int(grid_size),
            use_circular_mask=bool(use_circular_mask),
        )

    t1 = time.perf_counter()
    total_s = float(t1 - t0)

    return {
        "n": n,
        "total_s": total_s,
        "ms_per_function": 1000.0 * total_s / max(1, n),
        "functions_per_s": float(n) / max(1e-12, total_s),
    }
