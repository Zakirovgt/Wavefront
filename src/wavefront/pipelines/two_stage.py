from __future__ import annotations

import copy
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orbax.checkpoint as ocp

from wavefront.pipelines.dataset_source import (
    load_pipeline_dataset_artifact,
    validate_stage1_dimensions,
)
from wavefront.config import load_config
from wavefront.data.generators import generate_mixed_span_dataset
from wavefront.metamodel.gradmap import (
    DeepONetGradMapConfig,
    train_deeponet_gradmap,
)
from wavefront.metamodel.joint import finetune_joint_deeponet_fno
from wavefront.metamodel.random_fno import setup_random_fno_state
from wavefront.training.precision import set_mixed_precision


@dataclass
class TwoStageDataConfig:
    """
    Synthetic-data settings shared by the two-stage baseline pipeline.

    Separate datasets are generated for Stage 1 and Stage 2 using identical
    distribution settings but different random seeds.

    Attributes:
        grid_size: Number of regular-grid points along each spatial axis.
        n_train: Number of synthetic training functions.
        n_test: Number of synthetic held-out functions.
        frac_zernike: Fraction of samples generated from Zernike wavefronts.
        frac_spiral: Fraction of samples generated from spiral wavefronts.
        frac_distortion: Fraction of samples generated from distortion fields.
        with_noise: Whether noise is added to sensor-gradient measurements.
        noise_percentage: Relative amplitude of measurement noise.
        noise_lambda: Noise correlation or smoothing parameter.
        apply_blur: Whether optional image-space blur is applied.
        sigma_pix: Blur standard deviation in pixel units.
        stage1_seed: Random seed used to create the Stage-1 dataset.
        joint_data_seed: Random seed used to create the joint-training dataset.
    """

    grid_size: int = 24
    n_train: int = 10_000
    n_test: int = 1_000

    frac_zernike: float = 1.0
    frac_spiral: float = 0.0
    frac_distortion: float = 0.0

    with_noise: bool = True
    noise_percentage: float = 1.0
    noise_lambda: float = 0.03

    apply_blur: bool = False
    sigma_pix: float = 1.0

    stage1_seed: int = 42
    joint_data_seed: int = 43


def _to_jsonable(value: Any) -> Any:
    """
    Convert common Python, NumPy, and filesystem values into JSON-safe objects.

    Args:
        value: Arbitrary value that may contain NumPy scalars, arrays, paths,
            mappings, sequences, or non-finite floating-point values.

    Returns:
        A JSON-serializable Python object.

    Notes:
        NaN and infinite floating-point values are converted to None to keep
        JSON output portable and standards-compliant.
    """
    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    if isinstance(value, np.generic):
        return _to_jsonable(value.item())

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]

    return str(value)


def _write_json(
        path: Path,
        payload: dict[str, Any],
) -> None:
    """
    Save a dictionary as a UTF-8 JSON artifact.

    Args:
        path: Destination JSON file path.
        payload: Dictionary to serialize.

    Side Effects:
        Creates parent directories as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            _to_jsonable(payload),
            file,
            ensure_ascii=False,
            indent=2,
        )


def _save_params(
        path: Path,
        params,
) -> None:
    """
    Save a JAX/Flax parameter pytree using Orbax.

    Args:
        path: Destination checkpoint directory.
        params: Parameter pytree to persist.

    Side Effects:
        Creates parent directories as needed and overwrites an existing
        checkpoint at the requested destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    ocp.StandardCheckpointer().save(
        str(path),
        params,
        force=True,
    )


def _generate_dataset(
        data_cfg: TwoStageDataConfig,
        seed: int,
        branch_grid_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one synthetic train/test dataset for the two-stage pipeline.

    Args:
        data_cfg: Shared synthetic-data configuration.
        seed: Random seed used for generation.
        branch_grid_path: CSV path defining the sensor layout.

    Returns:
        A tuple containing:

            wavefronts:
                Wavefront targets with shape ``(N, grid_size, grid_size)``,
                or an equivalent flattened representation.

            sensor_gradients:
                Noisy sensor-gradient measurements with shape
                ``(N, P_sensor, 2)``.

            clean_grid_gradients:
                Clean gradient targets with shape
                ``(N, grid_size ** 2, 2)``.

    Notes:
        The underlying dataset generator may return additional arrays. This
        helper keeps only the data required by the two-stage baseline.
    """
    (
        wavefronts,
        sensor_gradients,
        clean_grid_gradients,
        *_,
    ) = generate_mixed_span_dataset(
        coarse_size=int(data_cfg.grid_size),
        num_train=int(data_cfg.n_train),
        num_test=int(data_cfg.n_test),
        frac_zernike=float(data_cfg.frac_zernike),
        frac_spiral=float(data_cfg.frac_spiral),
        frac_distortion=float(data_cfg.frac_distortion),
        with_noise=bool(data_cfg.with_noise),
        noise_percentage=float(data_cfg.noise_percentage),
        noise_lambda=float(data_cfg.noise_lambda),
        apply_blur=bool(data_cfg.apply_blur),
        sigma_pix=float(data_cfg.sigma_pix),
        seed=int(seed),
        save_dir=None,
        sensor_csv_path=branch_grid_path,
    )

    return (
        np.asarray(wavefronts, dtype=np.float32),
        np.asarray(sensor_gradients, dtype=np.float32),
        np.asarray(clean_grid_gradients, dtype=np.float32),
    )


def _validate_stage1_config(
        data_cfg: TwoStageDataConfig,
        stage1_cfg: DeepONetGradMapConfig,
) -> None:
    """
    Ensure Stage-1 settings are consistent with the shared dataset definition.

    Args:
        data_cfg: Shared synthetic-data configuration.
        stage1_cfg: Stage-1 DeepONet gradient-map configuration.

    Raises:
        ValueError: If the grid size, training count, or test count differs
            between Stage 1 and the dataset configuration.
    """
    if int(stage1_cfg.grid_size) != int(data_cfg.grid_size):
        raise ValueError(
            "Stage-1 grid_size must match data_cfg.grid_size: "
            f"{stage1_cfg.grid_size} != {data_cfg.grid_size}."
        )

    if int(stage1_cfg.n_train) != int(data_cfg.n_train):
        raise ValueError(
            "Stage-1 n_train must match data_cfg.n_train: "
            f"{stage1_cfg.n_train} != {data_cfg.n_train}."
        )

    if int(stage1_cfg.n_test) != int(data_cfg.n_test):
        raise ValueError(
            "Stage-1 n_test must match data_cfg.n_test: "
            f"{stage1_cfg.n_test} != {data_cfg.n_test}."
        )


def run_two_stage_pipeline(
        branch_grid_path: str = "1.csv",
        *,
        data_cfg: TwoStageDataConfig | None = None,
        stage1_cfg: DeepONetGradMapConfig | None = None,
        fno_args=None,
        config_dir: str | Path = "configs",
        results_root: str | Path = "results/two_stage",
        run_name: str | None = None,
        mixed_precision: bool = False,
        mp_dtype: str = "bfloat16",
        fno_seed: int = 123,
        joint_steps: int = 50_000,
        joint_batch_size: int = 32,
        joint_lr: float = 1e-3,
        joint_weight_decay: float = 1e-6,
        joint_seed: int = 0,
        dataset_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Train and persist the two-stage baseline reconstruction pipeline.

    Pipeline:

        Stage 1:
            Noisy sensor gradients
                -> DeepONet
                -> clean regular-grid gradient fields

        Stage 2:
            Pretrained DeepONet + randomly initialized FNO
                -> end-to-end joint optimization
                -> reconstructed wavefronts

    Unlike the three-stage pipeline, this baseline does not perform standalone
    FNO pretraining. The FNO begins from random initialization and is trained
    jointly with the pretrained DeepONet against wavefront targets.

    Args:
        branch_grid_path: CSV path defining the sensor positions used by the
            DeepONet branch network.
        data_cfg: Shared synthetic-data settings. Default settings are used
            when None.
        stage1_cfg: Stage-1 DeepONet gradient-map settings. When None, a
            compatible configuration is created from data_cfg.
        fno_args: Optional FNO configuration namespace. When None, settings
            are loaded with ``load_config``.
        config_dir: Directory containing FNO configuration files.
        results_root: Root directory for pipeline-run result folders.
        run_name: Optional explicit run-folder name. A timestamp is used when
            None.
        mixed_precision: Whether global mixed precision should be enabled.
        mp_dtype: Mixed-precision dtype, such as ``"bfloat16"``.
        fno_seed: Random seed for FNO parameter initialization.
        joint_steps: Number of end-to-end Stage-2 optimization iterations.
        joint_batch_size: Number of functions sampled per joint-training step.
        joint_lr: FNO learning rate supplied to joint fine-tuning. DeepONet
            uses one tenth of this value inside the joint trainer.
        joint_weight_decay: AdamW weight decay for joint optimization.
        joint_seed: JAX random seed for joint-training batches and dropout.
        dataset_dir:
                Optional portable dataset artifact directory. When provided, Stage 1
                and joint fine-tuning reuse its fixed train/test split instead of
                generating synthetic datasets internally.

    Returns:
        A manifest dictionary containing artifact directories and scalar
        validation metrics.

    Memory Behavior:
        Intermediate Stage-1 targets and completed model states are explicitly
        released when they are no longer required. The returned manifest keeps
        only paths and summary metrics, rather than live JAX model objects.
    """
    # Build default shared and Stage-1 configurations when necessary.
    if data_cfg is None:
        data_cfg = TwoStageDataConfig()

    if stage1_cfg is None:
        stage1_cfg = DeepONetGradMapConfig(
            grid_size=int(data_cfg.grid_size),
            n_train=int(data_cfg.n_train),
            n_test=int(data_cfg.n_test),
        )

    _validate_stage1_config(data_cfg, stage1_cfg)

    # Configure global mixed-precision behavior before model construction.
    set_mixed_precision(
        enabled=bool(mixed_precision),
        dtype=str(mp_dtype),
    )
    validate_stage1_dimensions(
        data_cfg=data_cfg,
        stage1_cfg=stage1_cfg,
    )

    artifact_dataset = None

    if dataset_dir is not None:
        artifact_dataset = load_pipeline_dataset_artifact(
            dataset_dir,
            grid_size=int(data_cfg.grid_size),
            n_train=int(data_cfg.n_train),
            n_test=int(data_cfg.n_test),
        )

        # The local sensor-layout copy inside the artifact is authoritative.
        branch_grid_path = artifact_dataset.branch_grid_path

        stage1_wavefronts = artifact_dataset.wavefronts
        stage1_sensor_gradients = artifact_dataset.sensor_gradients
        stage1_grid_gradients = artifact_dataset.grid_gradients

        joint_wavefronts = artifact_dataset.wavefronts
        joint_sensor_gradients = artifact_dataset.sensor_gradients

        dataset_provenance = artifact_dataset.provenance

    else:
        # Keep your current two independent synthetic-data generation blocks.
        # They should still use data_cfg.stage1_seed and data_cfg.joint_data_seed.
        (
            stage1_wavefronts,
            stage1_sensor_gradients,
            stage1_grid_gradients,
        ) = _generate_dataset(
            data_cfg=data_cfg,
            seed=int(data_cfg.stage1_seed),
            branch_grid_path=branch_grid_path,
        )

        (
            joint_wavefronts,
            joint_sensor_gradients,
            _,
        ) = _generate_dataset(
            data_cfg=data_cfg,
            seed=int(data_cfg.joint_data_seed),
            branch_grid_path=branch_grid_path,
        )

        dataset_provenance = {
            "source": "generated_in_process",
            "stage1_seed": int(data_cfg.stage1_seed),
            "joint_data_seed": int(data_cfg.joint_data_seed),
            "branch_grid_path": str(branch_grid_path),
        }
    # Create the run directory and stable subdirectories for both stages.
    run_id = run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(results_root) / run_id

    stage1_dir = run_dir / "stage1_deeponet_gradmap"
    stage2_dir = run_dir / "stage2_joint_random_fno"

    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)

    # Persist all top-level pipeline settings before training begins.
    _write_json(
        run_dir / "pipeline_config.json",
        {
            "data": asdict(data_cfg),
            "dataset": dataset_provenance,
            "stage1": asdict(stage1_cfg),
            "branch_grid_path": branch_grid_path,
            "mixed_precision": bool(mixed_precision),
            "mp_dtype": str(mp_dtype),
            "fno_seed": int(fno_seed),
            "joint_steps": int(joint_steps),
            "joint_batch_size": int(joint_batch_size),
            "joint_lr": float(joint_lr),
            "joint_weight_decay": float(joint_weight_decay),
            "joint_seed": int(joint_seed),
        },
    )
    _write_json(
        run_dir / "dataset_source.json",
        dataset_provenance,
    )
    # --------------------------------------------------------------
    # Stage 1: noisy sensor gradients -> clean regular-grid gradients.
    # --------------------------------------------------------------
    deeponet_state = train_deeponet_gradmap(
        cfg=stage1_cfg,
        grad_sensor_noisy=stage1_sensor_gradients,
        grad_grid_clean=stage1_grid_gradients,
        branch_grid_path=branch_grid_path,
    )

    # Save both the selected validation checkpoint and final optimization state.
    _save_params(
        stage1_dir / "params_best",
        deeponet_state["params"],
    )

    _save_params(
        stage1_dir / "params_last",
        deeponet_state["last_params"],
    )

    np.save(
        stage1_dir / "branch_sensor_coords.npy",
        np.asarray(deeponet_state["branch_sensor_coords"]),
    )

    _write_json(
        stage1_dir / "metrics.json",
        {
            "best_relative_l2": deeponet_state["best_e"],
            "last_relative_l2": deeponet_state["last_e"],
            "config": asdict(stage1_cfg),
        },
    )

    # Stage-1 wavefronts, sensor data, and clean-grid targets are no longer
    # needed after DeepONet gradient-map training has completed.
    del stage1_wavefronts
    del stage1_sensor_gradients
    del stage1_grid_gradients
    gc.collect()

    # --------------------------------------------------------------
    # Stage 2: random FNO + pretrained DeepONet -> joint training.
    # --------------------------------------------------------------

    # Load an FNO configuration or work from a safe independent copy of a
    # caller-provided configuration namespace.
    if fno_args is None:
        fno_args = load_config(
            operator_type="fno",
            config_dir=config_dir,
            data_mode="regular_grid",
            grid_size=int(data_cfg.grid_size),
            n_train=int(data_cfg.n_train),
            n_test=int(data_cfg.n_test),
            steps=int(joint_steps),
            batch_size=int(joint_batch_size),
            lr=float(joint_lr),
        )
    else:
        fno_args = copy.deepcopy(fno_args)

    # Match the FNO architecture and data settings to the generated grid.
    fno_args.data_mode = "regular_grid"
    fno_args.grid_size = int(data_cfg.grid_size)
    fno_args.nx = int(data_cfg.grid_size)
    fno_args.ny = int(data_cfg.grid_size)
    fno_args.n_train = int(data_cfg.n_train)
    fno_args.n_test = int(data_cfg.n_test)

    # Initialize an FNO without standalone pretraining.
    fno_state = setup_random_fno_state(
        fno_args=fno_args,
        seed=int(fno_seed),
    )

    _write_json(
        stage2_dir / "initialization.json",
        {
            "fno_seed": int(fno_seed),
            "fno_args": vars(fno_state["args"]),
            "description": (
                "FNO was randomly initialized and received no standalone "
                "pretraining before joint optimization."
            ),
        },
    )

    n_train = int(data_cfg.n_train)
    n_test = int(data_cfg.n_test)

    # Fine-tune both models end to end against final wavefront targets.
    train_slice = slice(0, int(data_cfg.n_train))
    test_slice = slice(
        int(data_cfg.n_train),
        int(data_cfg.n_train) + int(data_cfg.n_test),
    )
    joint = finetune_joint_deeponet_fno(
        fno_state=fno_state,
        deeponet_state=deeponet_state,
        grad_sensor_noisy=joint_sensor_gradients[train_slice],
        wavefront_true=joint_wavefronts[train_slice],
        grad_sensor_test=joint_sensor_gradients[test_slice],
        wavefront_test=joint_wavefronts[test_slice],
        grid_size=int(data_cfg.grid_size),
        batch_size=int(joint_batch_size),
        steps=int(joint_steps),
        lr=float(joint_lr),
        weight_decay=float(joint_weight_decay),
        seed=int(joint_seed),
    )

    # Save selected and final checkpoints for both jointly trained models.
    _save_params(
        stage2_dir / "deeponet_params_best",
        joint["deeponet"]["params"],
    )

    _save_params(
        stage2_dir / "deeponet_params_last",
        joint["deeponet"]["last_params"],
    )

    _save_params(
        stage2_dir / "fno_params_best",
        joint["fno"]["params"],
    )

    _save_params(
        stage2_dir / "fno_params_last",
        joint["fno"]["last_params"],
    )

    _write_json(
        stage2_dir / "metrics.json",
        {
            "best_relative_l2": joint["best_e"],
            "last_relative_l2": joint["last_e"],
            "fno_lr": joint["fno_lr"],
            "deeponet_lr": joint["deeponet_lr"],
            "joint_steps": int(joint_steps),
            "joint_batch_size": int(joint_batch_size),
            "joint_weight_decay": float(joint_weight_decay),
        },
    )

    # Save compact output locations and headline validation metrics.
    manifest = {
        "run_dir": str(run_dir),
        "stage1_dir": str(stage1_dir),
        "stage2_dir": str(stage2_dir),
        "dataset": dataset_provenance,
        "stage1_best_relative_l2": deeponet_state["best_e"],
        "stage2_best_relative_l2": joint["best_e"],
    }

    _write_json(
        run_dir / "manifest.json",
        manifest,
    )

    # Release potentially large JAX/NumPy model states and generated arrays.
    del joint
    del fno_state
    del deeponet_state
    del joint_wavefronts
    del joint_sensor_gradients
    gc.collect()

    return manifest
