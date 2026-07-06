from __future__ import annotations

import gc
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Literal

import jax
import numpy as np
import orbax.checkpoint as ocp

from wavefront.inference.predictor import build_eval_model_from_config
from wavefront.metamodel.gradmap import (
    DeepONetGradMapConfig,
    setup_deeponet_gradmap,
)
from wavefront.models.factory import setup_fno

CheckpointSelector = Literal["best", "last"]


def _read_json(
        path: str | Path,
) -> dict[str, Any]:
    """
    Load a JSON file and verify that its top-level value is an object.

    Args:
        path: Path to the required JSON file.

    Returns:
        Parsed JSON mapping.

    Raises:
        FileNotFoundError: If the requested JSON file does not exist.
        json.JSONDecodeError: If the file does not contain valid JSON.
        TypeError: If the top-level JSON value is not an object.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON file was not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a JSON object in {path}, "
            f"but got {type(payload).__name__}."
        )

    return payload


def _require_directory(
        path: str | Path,
) -> Path:
    """
    Resolve and validate a required directory path.

    Args:
        path: Directory path, optionally using a home-directory shortcut.

    Returns:
        Resolved absolute directory path.

    Raises:
        FileNotFoundError: If the path does not exist or is not a directory.
    """
    path = Path(path).expanduser().resolve()

    if not path.is_dir():
        raise FileNotFoundError(
            f"Required directory was not found: {path}"
        )

    return path


def _resolve_checkpoint_dir(
        root: Path,
        checkpoint: CheckpointSelector,
        *,
        prefix: str = "params",
) -> Path:
    """
    Resolve the directory containing a requested Orbax parameter checkpoint.

    For ``checkpoint="best"``, the function searches in this order:

        1. ``<prefix>_best``
        2. ``<prefix>``

    For ``checkpoint="last"``, the function searches in this order:

        1. ``<prefix>_last``
        2. ``<prefix>``

    Args:
        root: Parent directory containing checkpoint artifacts.
        checkpoint: Requested checkpoint selector, either ``"best"`` or
            ``"last"``.
        prefix: Base checkpoint prefix, such as ``"params"``,
            ``"deeponet_params"``, or ``"fno_params"``.

    Returns:
        Resolved directory containing the selected Orbax checkpoint.

    Raises:
        ValueError: If checkpoint is not ``"best"`` or ``"last"``.
        FileNotFoundError: If none of the expected checkpoint directories
            exists.
    """
    if checkpoint not in {"best", "last"}:
        raise ValueError(
            f"Unknown checkpoint selector {checkpoint!r}. "
            "Expected 'best' or 'last'."
        )

    candidates = (
        [f"{prefix}_best", prefix]
        if checkpoint == "best"
        else [f"{prefix}_last", prefix]
    )

    for name in candidates:
        candidate = root / name

        if candidate.is_dir():
            return candidate

    names = ", ".join(candidates)

    raise FileNotFoundError(
        f"Could not find parameter artifacts in {root}. "
        f"Expected one of: {names}."
    )


def _restore_params(
        root: Path,
        checkpoint: CheckpointSelector,
        *,
        prefix: str = "params",
):
    """
    Restore one JAX/Flax parameter pytree from an Orbax checkpoint.

    Args:
        root: Parent artifact directory.
        checkpoint: Requested checkpoint selector.
        prefix: Base checkpoint directory prefix.

    Returns:
        Restored parameter pytree.
    """
    params_dir = _resolve_checkpoint_dir(
        root=root,
        checkpoint=checkpoint,
        prefix=prefix,
    )

    return ocp.StandardCheckpointer().restore(
        str(params_dir)
    )


def _build_fno_state(
        args_payload: dict[str, Any],
        params,
) -> dict[str, Any]:
    """
    Rebuild an FNO architecture and attach a restored parameter tree.

    Args:
        args_payload: Saved FNO configuration dictionary, usually read from
            ``args_inference.json`` or a stage manifest.
        params: Restored FNO parameter pytree.

    Returns:
        FNO state dictionary containing the normalized configuration, model
        module, model function, and restored parameters.

    Notes:
        The model is initialized only to reconstruct the expected Flax
        parameter structure. The initialized parameter tree returned by
        ``setup_fno`` is discarded and replaced with ``params``.
    """
    args = Namespace(**args_payload)

    # Normalize fields required by the current FNO factory, including support
    # for saved configurations that may omit some values.
    args.operator_type = "fno"

    args.grid_size = int(
        getattr(
            args,
            "grid_size",
            getattr(args, "nx", 24),
        )
    )

    args.nx = int(
        getattr(args, "nx", args.grid_size)
    )

    args.ny = int(
        getattr(args, "ny", args.grid_size)
    )

    args.in_channels = int(
        getattr(args, "in_channels", 2)
    )

    args.num_outputs = int(
        getattr(args, "num_outputs", 1)
    )

    seed = int(
        getattr(args, "seed", 0)
    )

    key = jax.random.PRNGKey(seed)

    # Recreate the FNO module. Its freshly initialized parameters are ignored.
    args, model, model_fn, _ = setup_fno(
        args,
        key,
    )

    return {
        "operator_type": "fno",
        "args": args,
        "model": model,
        "model_fn": model_fn,
        "params": params,
    }


def _load_gradmap_state(
        stage1_dir: Path,
        params_root: Path,
        checkpoint: CheckpointSelector,
        *,
        params_prefix: str = "params",
) -> dict[str, Any]:
    """
    Rebuild a Stage-1 gradient-map DeepONet and restore its parameters.

    The model configuration is read from the Stage-1 metrics artifact, while
    the branch sensor-coordinate file determines the branch input dimension.

    Args:
        stage1_dir: Directory containing Stage-1 metrics and sensor-layout
            artifacts.
        params_root: Directory containing the target DeepONet checkpoint.
            This may be the Stage-1 directory itself or a later joint-training
            stage directory.
        checkpoint: Requested checkpoint selector.
        params_prefix: Prefix used by the DeepONet parameter artifacts.

    Returns:
        State dictionary for the gradient-map DeepONet.

    Raises:
        KeyError: If ``metrics.json`` does not contain the saved ``config``.
        FileNotFoundError: If required metrics, coordinates, or checkpoint
            artifacts are missing.
    """
    # The Stage-1 metrics file preserves all architecture settings required to
    # reconstruct the DeepONet gradient mapper.
    metrics = _read_json(
        stage1_dir / "metrics.json"
    )

    if "config" not in metrics:
        raise KeyError(
            f"Missing 'config' in Stage-1 metrics file: "
            f"{stage1_dir / 'metrics.json'}"
        )

    config = DeepONetGradMapConfig(
        **metrics["config"]
    )

    # The coordinate array establishes the number and ordering of branch
    # sensors expected by the pretrained model.
    coords_path = stage1_dir / "branch_sensor_coords.npy"

    if not coords_path.is_file():
        raise FileNotFoundError(
            "Stage-1 sensor coordinates are required to rebuild the model: "
            f"{coords_path}"
        )

    branch_sensor_coords = np.load(coords_path)

    n_sensors = int(
        branch_sensor_coords.shape[0]
    )

    # Recreate the Flax module using the stored Stage-1 configuration.
    model_fn, _ = setup_deeponet_gradmap(
        cfg=config,
        n_sensors=n_sensors,
        key=jax.random.PRNGKey(int(config.seed)),
    )

    # Restore parameters from the selected source checkpoint.
    params = _restore_params(
        root=params_root,
        checkpoint=checkpoint,
        prefix=params_prefix,
    )

    return {
        "operator_type": "deeponet_gradmap",
        "cfg": config,
        "model": model_fn,
        "model_fn": model_fn,
        "params": params,
        "branch_sensor_coords": branch_sensor_coords,
    }


def load_standalone_deeponet_state(
        run_dir: str | Path,
        checkpoint: CheckpointSelector = "best",
) -> dict[str, Any]:
    """
    Load a standalone DeepONet run created by ``main_routine_deeponet``.

    Expected artifact layout:

        <run_dir>/
            artifacts/
                args_inference.json
                params_best/
                params_last/
                params/

    Args:
        run_dir: Directory of the standalone DeepONet training run.
        checkpoint: Parameter checkpoint to restore, either ``"best"`` or
            ``"last"``.

    Returns:
        State dictionary containing the rebuilt model module, restored
        parameters, configuration, result directory, and optional branch
        sensor coordinates.
    """
    run_dir = _require_directory(run_dir)

    artifacts_dir = _require_directory(
        run_dir / "artifacts"
    )

    # Rebuild the model architecture from exported inference settings.
    config = _read_json(
        artifacts_dir / "args_inference.json"
    )

    args, model_fn = build_eval_model_from_config(config)

    # Restore selected trained parameters.
    params = _restore_params(
        root=artifacts_dir,
        checkpoint=checkpoint,
        prefix="params",
    )

    # Sensor coordinates are optional because some saved configurations may use
    # regular-grid branch inputs instead of an irregular sensor layout.
    coords_path = (
            artifacts_dir
            / "branch_sensor_coords.npy"
    )

    branch_sensor_coords = (
        np.load(coords_path)
        if coords_path.is_file()
        else None
    )

    return {
        "operator_type": "deeponet",
        "args": args,
        "model": model_fn,
        "model_fn": model_fn,
        "model_eval_fn": model_fn,
        "params": params,
        "result_dir": str(run_dir),
        "branch_sensor_coords": branch_sensor_coords,
    }


def load_standalone_fno_state(
        run_dir: str | Path,
        checkpoint: CheckpointSelector = "best",
) -> dict[str, Any]:
    """
    Load a standalone FNO run created by ``main_routine_fno``.

    Expected artifact layout:

        <run_dir>/
            artifacts/
                args_inference.json
                params_best/
                params_last/
                params/

    Args:
        run_dir: Directory of the standalone FNO training run.
        checkpoint: Parameter checkpoint to restore, either ``"best"`` or
            ``"last"``.

    Returns:
        State dictionary containing the rebuilt FNO module, restored
        parameters, configuration, and result directory.
    """
    run_dir = _require_directory(run_dir)

    artifacts_dir = _require_directory(
        run_dir / "artifacts"
    )

    args_payload = _read_json(
        artifacts_dir / "args_inference.json"
    )

    params = _restore_params(
        root=artifacts_dir,
        checkpoint=checkpoint,
        prefix="params",
    )

    state = _build_fno_state(
        args_payload=args_payload,
        params=params,
    )

    state["result_dir"] = str(run_dir)

    return state


def load_two_stage_metamodel_state(
        run_dir: str | Path,
        checkpoint: CheckpointSelector = "best",
) -> dict[str, Any]:
    """
    Load the jointly trained DeepONet–FNO pair from a two-stage run.

    Expected artifact layout:

        <run_dir>/
            stage1_deeponet_gradmap/
                metrics.json
                branch_sensor_coords.npy

            stage2_joint_random_fno/
                initialization.json
                deeponet_params_best/
                deeponet_params_last/
                fno_params_best/
                fno_params_last/

    Args:
        run_dir: Two-stage pipeline output directory.
        checkpoint: Parameter checkpoint selector for both component models.

    Returns:
        Nested metamodel state dictionary with ``"deeponet"`` and ``"fno"``
        entries ready for evaluation or inference.
    """
    run_dir = _require_directory(run_dir)

    stage1_dir = _require_directory(
        run_dir / "stage1_deeponet_gradmap"
    )

    stage2_dir = _require_directory(
        run_dir / "stage2_joint_random_fno"
    )

    # Rebuild the Stage-1 DeepONet architecture but restore its selected joint
    # fine-tuning parameters from the Stage-2 directory.
    deeponet_state = _load_gradmap_state(
        stage1_dir=stage1_dir,
        params_root=stage2_dir,
        checkpoint=checkpoint,
        params_prefix="deeponet_params",
    )

    # The initialization artifact stores the FNO architecture configuration.
    initialization = _read_json(
        stage2_dir / "initialization.json"
    )

    if "fno_args" not in initialization:
        raise KeyError(
            "Missing 'fno_args' in two-stage initialization.json."
        )

    fno_params = _restore_params(
        root=stage2_dir,
        checkpoint=checkpoint,
        prefix="fno_params",
    )

    fno_state = _build_fno_state(
        args_payload=initialization["fno_args"],
        params=fno_params,
    )

    return {
        "operator_type": "two_stage_metamodel",
        "run_dir": str(run_dir),
        "deeponet": deeponet_state,
        "fno": fno_state,
    }


def load_three_stage_metamodel_state(
        run_dir: str | Path,
        checkpoint: CheckpointSelector = "best",
) -> dict[str, Any]:
    """
    Load the jointly fine-tuned DeepONet–FNO pair from a three-stage run.

    Expected artifact layout:

        <run_dir>/
            stage1_deeponet_gradmap/
                metrics.json
                branch_sensor_coords.npy

            stage2_fno/
                stage2_manifest.json

            stage3_joint_finetune/
                deeponet_params_best/
                deeponet_params_last/
                fno_params_best/
                fno_params_last/

    Args:
        run_dir: Three-stage pipeline output directory.
        checkpoint: Parameter checkpoint selector for both component models.

    Returns:
        Nested metamodel state dictionary with ``"deeponet"`` and ``"fno"``
        entries ready for evaluation or inference.
    """
    run_dir = _require_directory(run_dir)

    stage1_dir = _require_directory(
        run_dir / "stage1_deeponet_gradmap"
    )

    stage2_dir = _require_directory(
        run_dir / "stage2_fno"
    )

    stage3_dir = _require_directory(
        run_dir / "stage3_joint_finetune"
    )

    # The DeepONet architecture comes from Stage 1, while parameters are
    # restored from the final joint fine-tuning stage.
    deeponet_state = _load_gradmap_state(
        stage1_dir=stage1_dir,
        params_root=stage3_dir,
        checkpoint=checkpoint,
        params_prefix="deeponet_params",
    )

    # The Stage-2 manifest stores the architecture settings used to construct
    # the FNO before Stage-3 joint fine-tuning.
    stage2_manifest = _read_json(
        stage2_dir / "stage2_manifest.json"
    )

    if "fno_args" not in stage2_manifest:
        raise KeyError(
            "Missing 'fno_args' in three-stage stage2_manifest.json."
        )

    fno_params = _restore_params(
        root=stage3_dir,
        checkpoint=checkpoint,
        prefix="fno_params",
    )

    fno_state = _build_fno_state(
        args_payload=stage2_manifest["fno_args"],
        params=fno_params,
    )

    return {
        "operator_type": "three_stage_metamodel",
        "run_dir": str(run_dir),
        "deeponet": deeponet_state,
        "fno": fno_state,
    }


def release_jax_memory(
        state: dict[str, Any] | None = None,
        *,
        clear_compilation_cache: bool = True,
) -> None:
    """
    Release Python references and request cleanup of JAX compilation caches.

    This utility is useful when independently loading and evaluating several
    large operators in one long-running notebook or accelerator session.

    Args:
        state: Optional mutable state dictionary to clear before releasing
            Python references.
        clear_compilation_cache: Whether to call ``jax.clear_caches()`` when
            that helper is available in the installed JAX version.

    Side Effects:
        Clears the supplied state dictionary in place, runs Python garbage
        collection, and optionally clears JAX compilation caches.

    Notes:
        Cache clearing is requested on a best-effort basis. Actual device
        memory release timing remains controlled by JAX and the accelerator
        runtime.
    """
    # Remove references held by the caller's state dictionary.
    if state is not None:
        state.clear()

    gc.collect()

    # JAX versions may or may not expose clear_caches, so use a guarded lookup.
    if clear_compilation_cache:
        clear_caches = getattr(
            jax,
            "clear_caches",
            None,
        )

        if callable(clear_caches):
            clear_caches()

    gc.collect()
