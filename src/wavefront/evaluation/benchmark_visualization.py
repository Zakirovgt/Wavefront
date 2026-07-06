from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


def _as_solution_grid(
        solution,
        grid_size: int,
) -> np.ndarray:
    """
    Convert one wavefront sample to a regular two-dimensional grid.

    Args:
        solution: Wavefront in either ``(grid_size, grid_size)`` form or a
            flattened representation containing ``grid_size**2`` values.
        grid_size: Number of spatial points along each wavefront axis.

    Returns:
        Float32 wavefront array with shape:

            ``(grid_size, grid_size)``

    Raises:
        ValueError: If the input cannot be interpreted as one wavefront on the
            requested regular grid.
    """
    solution = np.asarray(
        solution,
        dtype=np.float32,
    )
    solution = np.squeeze(solution)

    if solution.shape == (grid_size, grid_size):
        return solution

    if solution.size == grid_size * grid_size:
        return solution.reshape(
            grid_size,
            grid_size,
        )

    raise ValueError(
        "Wavefront must have shape "
        f"({grid_size}, {grid_size}) or contain "
        f"{grid_size * grid_size} values, but got {solution.shape}."
    )


def _as_gradient_grid(
        gradient,
        grid_size: int,
) -> np.ndarray:
    """
    Convert one gradient sample to a channels-last regular-grid layout.

    Supported layouts are:

        - ``(grid_size, grid_size, 2)``
        - ``(2, grid_size, grid_size)``
        - ``(grid_size**2, 2)``
        - ``(2, grid_size**2)``

    Args:
        gradient: Gradient field in one supported regular-grid layout.
        grid_size: Number of spatial points along each grid axis.

    Returns:
        Float32 gradient field with shape:

            ``(grid_size, grid_size, 2)``

        where the final axis stores the two derivative components.

    Raises:
        ValueError: If the supplied array does not match a supported gradient
            representation.
    """
    gradient = np.asarray(
        gradient,
        dtype=np.float32,
    )
    gradient = np.squeeze(gradient)

    # Preserve an already channels-last gradient grid.
    if gradient.shape == (grid_size, grid_size, 2):
        return gradient

    # Convert channels-first regular-grid derivatives to channels-last format.
    if gradient.shape == (2, grid_size, grid_size):
        return np.moveaxis(
            gradient,
            0,
            -1,
        )

    # Restore spatial structure from flattened pointwise derivative pairs.
    if gradient.shape == (grid_size * grid_size, 2):
        return gradient.reshape(
            grid_size,
            grid_size,
            2,
        )

    # Restore and transpose flattened channels-first derivatives.
    if gradient.shape == (2, grid_size * grid_size):
        return np.moveaxis(
            gradient.reshape(
                2,
                grid_size,
                grid_size,
            ),
            0,
            -1,
        )

    raise ValueError(
        "Gradient must have shape "
        f"({grid_size}, {grid_size}, 2), "
        f"({grid_size * grid_size}, 2), or a channel-first equivalent; "
        f"but got {gradient.shape}."
    )


def relative_error_map(
        prediction,
        target,
        *,
        mode: str = "global",
        eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute a normalized absolute-error map between prediction and target.

    Supported normalization modes:

        ``"global"``:
            ``abs(prediction - target) / max(abs(target))``

        ``"l2"``:
            ``abs(prediction - target) / RMS(target)``

        ``"pointwise"``:
            ``abs(prediction - target) / (abs(target) + eps)``

    Args:
        prediction: Predicted scalar field.
        target: Reference scalar field with the same shape as ``prediction``.
        mode: Error-map normalization strategy.
        eps: Small positive denominator stabilizer.

    Returns:
        Float64 normalized absolute-error map with the same shape as the input
        prediction.

    Raises:
        ValueError: If ``mode`` is not ``"global"``, ``"l2"``, or
            ``"pointwise"``.

    Notes:
        Pointwise normalization can strongly emphasize errors in regions where
        the true wavefront is close to zero. Global and L2 normalization are
        usually more suitable for visually comparing complete wavefronts.
    """
    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )
    target = np.asarray(
        target,
        dtype=np.float64,
    )

    absolute_error = np.abs(prediction - target)

    if mode == "pointwise":
        return absolute_error / (np.abs(target) + eps)

    if mode == "l2":
        denominator = np.sqrt(
            np.mean(target ** 2)
        ) + eps
        return absolute_error / denominator

    if mode == "global":
        denominator = np.max(
            np.abs(target)
        ) + eps
        return absolute_error / denominator

    raise ValueError(
        f"Unknown error-map mode {mode!r}. "
        "Expected 'global', 'l2', or 'pointwise'."
    )


def _finite_values(
        arrays,
) -> np.ndarray:
    """
    Collect all finite scalar values from one or more arrays.

    Args:
        arrays: Iterable of arrays that may contain NaN or infinite values.

    Returns:
        One-dimensional float64 array containing every finite value from the
        supplied arrays. Returns an empty array when no finite values exist.
    """
    values = []

    for array in arrays:
        array = np.asarray(
            array,
            dtype=np.float64,
        )
        array = array[np.isfinite(array)]

        if array.size:
            values.append(array.reshape(-1))

    if not values:
        return np.asarray(
            [],
            dtype=np.float64,
        )

    return np.concatenate(values)


def _percentile_limits(
        arrays,
        *,
        low: float,
        high: float,
) -> tuple[float, float]:
    """
    Compute robust lower and upper visualization limits from finite values.

    Args:
        arrays: Iterable of arrays used to derive common color limits.
        low: Lower percentile.
        high: Upper percentile.

    Returns:
        Tuple ``(vmin, vmax)`` suitable for a shared scalar-field color scale.

    Notes:
        If percentile bounds collapse to one value, the function falls back to
        the data extrema and then adds a small margin when necessary.
    """
    values = _finite_values(arrays)

    if values.size == 0:
        return 0.0, 1.0

    vmin = float(
        np.percentile(values, low)
    )
    vmax = float(
        np.percentile(values, high)
    )

    # Fall back to finite extrema if percentile limits are unusable.
    if (
            not np.isfinite(vmin)
            or not np.isfinite(vmax)
            or vmin == vmax
    ):
        vmin = float(np.min(values))
        vmax = float(np.max(values))

    # Ensure a non-zero range for constant-valued arrays.
    if vmin == vmax:
        margin = max(
            abs(vmin) * 0.05,
            1.0,
        )
        vmin -= margin
        vmax += margin

    return vmin, vmax


def _symmetric_limit(
        arrays,
        *,
        percentile: float,
        fallback: float = 1.0,
) -> float:
    """
    Compute a symmetric plotting limit for signed fields or differences.

    Args:
        arrays: Iterable of signed arrays.
        percentile: Percentile of absolute values used as the limit. Values of
            100 or greater use the maximum absolute value.
        fallback: Positive limit returned when no usable finite values exist.

    Returns:
        Positive scalar ``limit`` intended for ``vmin=-limit`` and
        ``vmax=limit``.
    """
    values = _finite_values(arrays)

    if values.size == 0:
        return fallback

    absolute_values = np.abs(values)

    if percentile >= 100.0:
        limit = float(np.max(absolute_values))
    else:
        limit = float(
            np.percentile(
                absolute_values,
                percentile,
            )
        )

    if not np.isfinite(limit) or limit <= 0.0:
        return fallback

    return limit


def _positive_limit(
        arrays,
        *,
        percentile: float,
        fallback: float = 1.0,
) -> float:
    """
    Compute an upper visualization limit for non-negative error maps.

    Args:
        arrays: Iterable of non-negative error arrays.
        percentile: Percentile used as the upper display limit. Values of 100
            or greater use the maximum value.
        fallback: Positive limit returned when no usable finite values exist.

    Returns:
        Positive scalar upper limit suitable for ``vmin=0``.
    """
    values = _finite_values(arrays)

    if values.size == 0:
        return fallback

    if percentile >= 100.0:
        limit = float(np.max(values))
    else:
        limit = float(
            np.percentile(
                values,
                percentile,
            )
        )

    if not np.isfinite(limit) or limit <= 0.0:
        return fallback

    return limit


def _is_poisson(
        name: str,
) -> bool:
    """
    Return whether a method label identifies the Poisson baseline.

    Args:
        name: Display label for a reconstruction method.

    Returns:
        True when the label contains either the English or Russian Poisson
        baseline name.
    """
    return (
            "poisson" in name.lower()
            or "пуассон" in name.lower()
    )


def save_benchmark_comparison(
        *,
        sample_index: int,
        wavefront_true,
        predictions: Mapping[str, np.ndarray],
        output_path: str | Path,
        grid_size: int,
        error_mode: str = "global",
        solution_percentiles: tuple[float, float] = (0.0, 100.0),
        error_percentile: float = 100.0,
        separate_poisson_scale: bool = False,
        cmap_solution: str = "magma",
        cmap_difference: str = "seismic",
        cmap_error: str = "viridis",
        dpi: int = 180,
) -> dict:
    """
    Save a three-row visual comparison of wavefront reconstructions.

    Figure layout:

        Row 1:
            Ground-truth wavefront followed by all predictions, using one
            common solution color scale.

        Row 2:
            Signed differences defined as ``prediction - ground_truth``.

        Row 3:
            Normalized absolute-error maps using ``error_mode``.

    Args:
        sample_index: Sample identifier displayed in the figure title.
        wavefront_true: Ground-truth wavefront for one sample.
        predictions: Mapping from method label to predicted wavefront.
        output_path: Destination image path.
        grid_size: Number of spatial points along each wavefront axis.
        error_mode: Normalization used for absolute-error maps. Supported
            values are ``"global"``, ``"l2"``, and ``"pointwise"``.
        solution_percentiles: Lower and upper percentiles used to derive the
            common solution color scale.
        error_percentile: Percentile used to derive robust signed-difference
            and normalized-error limits.
        separate_poisson_scale: Whether to use separate error scales for
            predictions whose labels identify the Poisson baseline.
        cmap_solution: Matplotlib colormap for true and predicted wavefronts.
        cmap_difference: Matplotlib diverging colormap for signed differences.
        cmap_error: Matplotlib sequential colormap for absolute-error maps.
        dpi: Saved figure resolution.

    Returns:
        Dictionary containing:

            ``path``:
                Saved figure path.

            ``relative_l2``:
                Per-method relative L2 reconstruction errors.

            ``solution_vmin`` and ``solution_vmax``:
                Shared color-scale limits used for Row 1.

    Raises:
        ValueError: If ``predictions`` is empty or an input wavefront cannot
            be converted to the requested grid shape.

    Notes:
        Separate Poisson scaling can improve readability when Poisson errors
        are substantially larger than neural-operator errors. For strict
        visual comparability across all methods, leave it disabled.
    """
    if not predictions:
        raise ValueError(
            "At least one prediction is required."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Normalize target and prediction layouts before deriving errors or scales.
    wavefront_true = _as_solution_grid(
        wavefront_true,
        grid_size=grid_size,
    )

    prediction_grids = {
        name: _as_solution_grid(
            prediction,
            grid_size=grid_size,
        )
        for name, prediction in predictions.items()
    }

    # Signed reconstruction residuals: predicted wavefront minus truth.
    differences = {
        name: prediction - wavefront_true
        for name, prediction in prediction_grids.items()
    }

    # Normalized non-negative absolute-error maps.
    error_maps = {
        name: relative_error_map(
            prediction,
            wavefront_true,
            mode=error_mode,
        )
        for name, prediction in prediction_grids.items()
    }

    # Use a common solution scale across the true field and all predictions.
    solution_vmin, solution_vmax = _percentile_limits(
        [wavefront_true, *prediction_grids.values()],
        low=float(solution_percentiles[0]),
        high=float(solution_percentiles[1]),
    )

    # Optionally separate Poisson error scales from neural-operator scales.
    if separate_poisson_scale:
        neural_names = [
            name
            for name in prediction_grids
            if not _is_poisson(name)
        ]

        poisson_names = [
            name
            for name in prediction_grids
            if _is_poisson(name)
        ]

        neural_difference_limit = _symmetric_limit(
            [
                differences[name]
                for name in neural_names
            ],
            percentile=error_percentile,
        )

        poisson_difference_limit = _symmetric_limit(
            [
                differences[name]
                for name in poisson_names
            ],
            percentile=error_percentile,
            fallback=neural_difference_limit,
        )

        neural_error_limit = _positive_limit(
            [
                error_maps[name]
                for name in neural_names
            ],
            percentile=error_percentile,
        )

        poisson_error_limit = _positive_limit(
            [
                error_maps[name]
                for name in poisson_names
            ],
            percentile=error_percentile,
            fallback=neural_error_limit,
        )
    else:
        common_difference_limit = _symmetric_limit(
            differences.values(),
            percentile=error_percentile,
        )

        common_error_limit = _positive_limit(
            error_maps.values(),
            percentile=error_percentile,
        )

    method_names = list(prediction_grids)
    ncols = len(method_names) + 1

    figure, axes = plt.subplots(
        3,
        ncols,
        figsize=(3.4 * ncols, 9.0),
        squeeze=False,
    )

    figure.suptitle(
        f"Benchmark sample #{sample_index}",
        fontsize=14,
    )

    # The first column contains only the ground-truth wavefront.
    image = axes[0, 0].imshow(
        wavefront_true,
        cmap=cmap_solution,
        vmin=solution_vmin,
        vmax=solution_vmax,
        extent=[-1, 1, 1, -1],
        origin="upper",
    )

    axes[0, 0].set_title("True wavefront")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    figure.colorbar(
        image,
        ax=axes[0, 0],
        fraction=0.046,
    )

    # Difference and error rows have no ground-truth counterpart.
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")

    relative_l2 = {}

    for column, name in enumerate(method_names, start=1):
        prediction = prediction_grids[name]
        difference = differences[name]
        error_map = error_maps[name]

        relative_l2[name] = float(
            np.linalg.norm(difference)
            / (
                    np.linalg.norm(wavefront_true)
                    + 1e-8
            )
        )

        # Assign the appropriate shared or group-specific error limits.
        if separate_poisson_scale and _is_poisson(name):
            difference_limit = poisson_difference_limit
            error_limit = poisson_error_limit
        elif separate_poisson_scale:
            difference_limit = neural_difference_limit
            error_limit = neural_error_limit
        else:
            difference_limit = common_difference_limit
            error_limit = common_error_limit

        # Row 1: predicted wavefront on the common solution scale.
        image = axes[0, column].imshow(
            prediction,
            cmap=cmap_solution,
            vmin=solution_vmin,
            vmax=solution_vmax,
            extent=[-1, 1, 1, -1],
            origin="upper",
        )

        axes[0, column].set_title(
            f"{name}\nrelative L2 = {relative_l2[name]:.3f}"
        )
        axes[0, column].set_xticks([])
        axes[0, column].set_yticks([])

        figure.colorbar(
            image,
            ax=axes[0, column],
            fraction=0.046,
        )

        # Row 2: signed prediction-minus-truth difference.
        image = axes[1, column].imshow(
            difference,
            cmap=cmap_difference,
            vmin=-difference_limit,
            vmax=difference_limit,
            extent=[-1, 1, 1, -1],
            origin="upper",
        )

        axes[1, column].set_title(
            f"{name} − true"
        )
        axes[1, column].set_xticks([])
        axes[1, column].set_yticks([])

        figure.colorbar(
            image,
            ax=axes[1, column],
            fraction=0.046,
        )

        # Row 3: normalized absolute reconstruction error.
        image = axes[2, column].imshow(
            error_map,
            cmap=cmap_error,
            vmin=0.0,
            vmax=error_limit,
            extent=[-1, 1, 1, -1],
            origin="upper",
        )

        axes[2, column].set_title(
            f"Normalized absolute error: {name}"
        )
        axes[2, column].set_xticks([])
        axes[2, column].set_yticks([])

        figure.colorbar(
            image,
            ax=axes[2, column],
            fraction=0.046,
        )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)

    return {
        "path": str(output_path),
        "relative_l2": relative_l2,
        "solution_vmin": solution_vmin,
        "solution_vmax": solution_vmax,
    }


def save_stage1_gradient_comparison(
    *,
    sample_index: int,
    reference_gradient,
    interpolated_gradient,
    deeponet_gradient,
    output_path: str | Path,
    grid_size: int,
    reference_label: str = "Reference gradient",
    interpolated_label: str = "Sensor interpolation",
    dpi: int = 180,
) -> str:
    """
    Save a Stage-1 gradient-reconstruction diagnostic figure.

    The figure contains two rows, one for each derivative component, and three
    columns:

        1. Reference regular-grid gradient.
        2. Gradient produced by sensor-to-grid interpolation.
        3. Stage-1 DeepONet gradient-map prediction.

    Args:
        sample_index: Sample identifier displayed in the figure title.
        reference_gradient: Clean reference regular-grid gradients.
        interpolated_gradient: Gradient field reconstructed through direct
            sensor-to-grid interpolation.
        deeponet_gradient: Gradient field predicted by the Stage-1 DeepONet.
        output_path: Destination image path.
        grid_size: Number of spatial points along each gradient-grid axis.
        reference_label: Title used for the reference-gradient column.
        dpi: Saved figure resolution.

    Returns:
        String representation of the saved figure path.

    Raises:
        ValueError: If any gradient cannot be converted to a regular grid with
            shape ``(grid_size, grid_size, 2)``.

    Notes:
        Each derivative-component row uses one symmetric color scale shared
        across its three displayed fields.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Normalize all inputs to channels-last regular-grid form.
    reference_gradient = _as_gradient_grid(
        reference_gradient,
        grid_size=grid_size,
    )

    interpolated_gradient = _as_gradient_grid(
        interpolated_gradient,
        grid_size=grid_size,
    )

    deeponet_gradient = _as_gradient_grid(
        deeponet_gradient,
        grid_size=grid_size,
    )

    fields = [
        reference_gradient,
        interpolated_gradient,
        deeponet_gradient,
    ]

    titles = [
        reference_label,
        interpolated_label,
        "Stage-1 DeepONet",
    ]

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 6.5),
        squeeze=False,
    )

    figure.suptitle(
        f"Gradient reconstruction for sample #{sample_index}",
        fontsize=14,
    )

    # Plot g_x and g_y in separate rows using a shared symmetric scale per
    # component across reference, interpolation, and DeepONet prediction.
    for component, component_name in enumerate(("gₓ", "gᵧ")):
        component_images = [
            field[..., component]
            for field in fields
        ]

        limit = _symmetric_limit(
            component_images,
            percentile=100.0,
        )

        for column, (image_data, title) in enumerate(
                zip(component_images, titles)
        ):
            image = axes[component, column].imshow(
                image_data,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                extent=[-1, 1, 1, -1],
                origin="upper",
            )

            axes[component, column].set_title(title)
            axes[component, column].set_xticks([])
            axes[component, column].set_yticks([])

            axes[component, column].set_ylabel(
                component_name if column == 0 else ""
            )

            figure.colorbar(
                image,
                ax=axes[component, column],
                fraction=0.046,
            )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)

    return str(output_path)
