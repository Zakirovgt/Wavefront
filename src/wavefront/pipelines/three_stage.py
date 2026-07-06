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

from wavefront.config import load_config
from wavefront.data.generators import generate_mixed_span_dataset
from wavefront.evaluation.timing import benchmark_inference_deeponet
from wavefront.metamodel.gradmap import (
    DeepONetGradMapConfig,
    train_deeponet_gradmap,
)
from wavefront.pipelines.dataset_source import (
    load_pipeline_dataset_artifact,
    validate_stage1_dimensions,
)
from wavefront.metamodel.joint import finetune_joint_deeponet_fno
from wavefront.metamodel.stage2_fno import train_fno_on_deeponet_outputs
from wavefront.training.precision import set_mixed_precision


@dataclass
class ThreeStageDataConfig:
    """
    Synthetic dataset configuration shared by all stages of the pipeline.

    The generated data is split into a training partition and a test partition.
    Stage 1 and Stage 2 use independently generated datasets with the same
    composition settings but different random seeds.

    Attributes:
        grid_size: Number of spatial grid points along each dimension.
        n_train: Number of synthetic training samples.
        n_test: Number of synthetic test samples.
        frac_zernike: Fraction of generated samples based on Zernike modes.
        frac_spiral: Fraction of generated samples based on spiral wavefronts.
        frac_distortion: Fraction of generated samples based on distorted
            wavefronts.
        with_noise: Whether sensor-gradient noise is applied.
        noise_percentage: Relative magnitude of the added measurement noise.
        noise_lambda: Spatial correlation or filtering parameter for noise.
        apply_blur: Whether optional smoothing is applied to generated fields.
        sigma_pix: Blur standard deviation in pixels.
        stage1_seed: Random seed used for the Stage-1 synthetic dataset.
        stage2_seed: Random seed used for the Stage-2 synthetic dataset.
    """

    grid_size: int = 24
    n_train: int = 10_000
    n_test: int = 1_000

    frac_zernike: float = 0.0
    frac_spiral: float = 0.0
    frac_distortion: float = 1.0

    with_noise: bool = True
    noise_percentage: float = 1.0
    noise_lambda: float = 0.03

    apply_blur: bool = False
    sigma_pix: float = 1.0

    stage1_seed: int = 42
    stage2_seed: int = 43


def _to_jsonable(value: Any) -> Any:
    """
    Convert common Python, NumPy, and filesystem objects into JSON-safe values.

    Args:
        value: Arbitrary object that may contain NumPy scalars, arrays, paths,
            dictionaries, sequences, or non-finite floating-point values.

    Returns:
        A value suitable for serialization with json.dump.

    Notes:
        Non-finite floating-point values, including NaN and infinity, are
        converted to None because they are not portable JSON values.
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
    Write a dictionary as a UTF-8 JSON artifact.

    Args:
        path: Destination JSON file path.
        payload: Dictionary to serialize.

    Side Effects:
        Creates the parent directory if it does not already exist.
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
    Save a JAX/Flax parameter pytree using an Orbax standard checkpoint.

    Args:
        path: Destination checkpoint directory.
        params: Parameter pytree to save.

    Side Effects:
        Creates the checkpoint parent directory if needed and overwrites any
        existing checkpoint at the requested path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpointer = ocp.StandardCheckpointer()

    checkpointer.save(
        str(path),
        params,
        force=True,
    )


def _generate_synthetic_dataset(
        data_cfg: ThreeStageDataConfig,
        seed: int,
        branch_grid_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate one synthetic dataset split for the three-stage pipeline.

    Args:
        data_cfg: Shared synthetic-data configuration.
        seed: Random seed used for generation.
        branch_grid_path: CSV path describing the branch sensor layout.

    Returns:
        A tuple containing:

            wavefronts:
                Wavefront targets with shape ``(N, grid_size, grid_size)``,
                or an equivalent flattened representation.

            noisy_sensor_gradients:
                Sensor-gradient measurements with shape
                ``(N, P_sensor, 2)``.

            clean_grid_gradients:
                Clean regular-grid derivatives with shape
                ``(N, grid_size * grid_size, 2)``.

    Notes:
        The lower-level generator may return additional metadata. This helper
        intentionally retains only the arrays required by the three-stage
        training pipeline.
    """
    wavefronts, noisy_sensor_gradients, clean_grid_gradients, *_ = (
        generate_mixed_span_dataset(
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
    )

    return (
        np.asarray(wavefronts, dtype=np.float32),
        np.asarray(noisy_sensor_gradients, dtype=np.float32),
        np.asarray(clean_grid_gradients, dtype=np.float32),
    )


def _validate_stage1_config(
        data_cfg: ThreeStageDataConfig,
        stage1_cfg: DeepONetGradMapConfig,
) -> None:
    """
    Verify that synthetic-data and Stage-1 configurations describe one dataset.

    Args:
        data_cfg: Shared dataset configuration.
        stage1_cfg: DeepONet gradient-map training configuration.

    Raises:
        ValueError: If grid size, training-sample count, or test-sample count
            differs between the shared data configuration and Stage 1.
    """
    if int(stage1_cfg.grid_size) != int(data_cfg.grid_size):
        raise ValueError(
            "Stage-1 grid_size must match the synthetic dataset grid_size: "
            f"{stage1_cfg.grid_size} != {data_cfg.grid_size}."
        )

    if int(stage1_cfg.n_train) != int(data_cfg.n_train):
        raise ValueError(
            "Stage-1 n_train must match the synthetic dataset n_train: "
            f"{stage1_cfg.n_train} != {data_cfg.n_train}."
        )

    if int(stage1_cfg.n_test) != int(data_cfg.n_test):
        raise ValueError(
            "Stage-1 n_test must match the synthetic dataset n_test: "
            f"{stage1_cfg.n_test} != {data_cfg.n_test}."
        )


def run_three_stage_pipeline(
        branch_grid_path: str = "1.csv",
        *,
        data_cfg: ThreeStageDataConfig | None = None,
        stage1_cfg: DeepONetGradMapConfig | None = None,
        fno_args=None,
        config_dir: str | Path = "configs",
        results_root: str | Path = "results/three_stage",
        run_name: str | None = None,
        mixed_precision: bool = False,
        mp_dtype: str = "bfloat16",
        stage2_steps: int = 20_000,
        stage2_batch_size: int = 64,
        stage2_lr: float = 1e-3,
        stage2_infer_batch: int = 64,
        joint_steps: int = 10_000,
        joint_batch_size: int = 32,
        joint_weight_decay: float = 1e-6,
        joint_fno_lr_factor: float = 0.1,
        joint_seed: int = 0,
        dataset_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Train, fine-tune, and persist a three-stage wavefront reconstruction model.

    The complete pipeline consists of:

        Stage 1:
            Noisy sensor gradients
                -> DeepONet
                -> clean regular-grid gradients

        Stage 2:
            DeepONet-predicted gradient grids
                -> FNO
                -> wavefront reconstruction

        Stage 3:
            End-to-end joint fine-tuning of DeepONet and FNO using only the
            final wavefront reconstruction loss.

    Args:
        branch_grid_path: CSV path defining the sensor positions used by the
            DeepONet branch network.
        data_cfg: Shared synthetic-data configuration. When None, default
            ThreeStageDataConfig settings are used.
        stage1_cfg: Stage-1 DeepONet gradient-map configuration. When None,
            a matching configuration is created from data_cfg.
        fno_args: Optional FNO training configuration namespace. When None,
            configuration is loaded through ``load_config``.
        config_dir: Directory containing FNO configuration files.
        results_root: Root directory where pipeline-run folders are created.
        run_name: Optional explicit output folder name. When None, a timestamp
            is used.
        mixed_precision: Whether global mixed precision is enabled.
        mp_dtype: Mixed-precision dtype, such as ``"bfloat16"`` or
            ``"float16"``.
        stage2_steps: Number of FNO training steps during Stage 2.
        stage2_batch_size: FNO mini-batch size during Stage 2.
        stage2_lr: FNO learning rate during Stage 2.
        stage2_infer_batch: DeepONet inference batch size used while producing
            Stage-2 FNO inputs.
        joint_steps: Number of end-to-end fine-tuning steps during Stage 3.
        joint_batch_size: Joint-training mini-batch size.
        joint_weight_decay: AdamW weight decay used during Stage 3.
        joint_fno_lr_factor: Factor applied to the Stage-2 FNO learning rate
            to determine the Stage-3 FNO learning rate. It must be in
            ``(0, 1]``.
        joint_seed: Random seed for Stage-3 mini-batch sampling and dropout.
        dataset_dir:
            Optional portable dataset artifact directory. When provided, all three
            stages reuse its fixed train/test split instead of generating synthetic
            datasets internally.

    Returns:
        A manifest dictionary containing output paths and selected scalar
        metrics.

    Raises:
        ValueError: If Stage-1 and dataset configurations are inconsistent, or
            joint_fno_lr_factor is outside the interval ``(0, 1]``.

    Memory Behavior:
        - Stage 1 keeps only the DeepONet model.
        - Stage 2 keeps both the DeepONet and FNO models.
        - Stage 3 jointly fine-tunes the same DeepONet and FNO models.
        - No separate third model is constructed.
        - Intermediate arrays and model states are explicitly released when
          they are no longer needed.

    Notes:
        The function saves all required checkpoints and metadata but returns
        only paths and scalar metadata. Future inference or evaluation should
        reload model artifacts one model at a time.
    """
    # Create default configurations when the caller does not provide them.
    if data_cfg is None:
        data_cfg = ThreeStageDataConfig()

    if stage1_cfg is None:
        stage1_cfg = DeepONetGradMapConfig(
            grid_size=int(data_cfg.grid_size),
            n_train=int(data_cfg.n_train),
            n_test=int(data_cfg.n_test),
        )

    validate_stage1_dimensions(
        data_cfg=data_cfg,
        stage1_cfg=stage1_cfg,
    )

    if not 0.0 < float(joint_fno_lr_factor) <= 1.0:
        raise ValueError(
            "joint_fno_lr_factor must be in the interval (0, 1]."
        )

    set_mixed_precision(
        enabled=bool(mixed_precision),
        dtype=str(mp_dtype),
    )
    artifact_dataset = None

    if dataset_dir is not None:
        artifact_dataset = load_pipeline_dataset_artifact(
            dataset_dir,
            grid_size=int(data_cfg.grid_size),
            n_train=int(data_cfg.n_train),
            n_test=int(data_cfg.n_test),
        )

        branch_grid_path = artifact_dataset.branch_grid_path

        stage1_wavefronts = artifact_dataset.wavefronts
        stage1_sensor_gradients = artifact_dataset.sensor_gradients
        stage1_grid_gradients = artifact_dataset.grid_gradients

        stage2_wavefronts = artifact_dataset.wavefronts
        stage2_sensor_gradients = artifact_dataset.sensor_gradients

        stage3_wavefronts = artifact_dataset.wavefronts
        stage3_sensor_gradients = artifact_dataset.sensor_gradients

        dataset_provenance = artifact_dataset.provenance

    else:
        (
            stage1_wavefronts,
            stage1_sensor_gradients,
            stage1_grid_gradients,
        ) = _generate_synthetic_dataset(
            data_cfg=data_cfg,
            seed=int(data_cfg.stage1_seed),
            branch_grid_path=branch_grid_path,
        )

        (
            stage2_wavefronts,
            stage2_sensor_gradients,
            _,
        ) = _generate_synthetic_dataset(
            data_cfg=data_cfg,
            seed=int(data_cfg.stage2_seed),
            branch_grid_path=branch_grid_path,
        )

        (
            stage3_wavefronts,
            stage3_sensor_gradients,
            _,
        ) = _generate_synthetic_dataset(
            data_cfg=data_cfg,
            seed=int(data_cfg.stage3_seed),
            branch_grid_path=branch_grid_path,
        )

        dataset_provenance = {
            "source": "generated_in_process",
            "stage1_seed": int(data_cfg.stage1_seed),
            "stage2_seed": int(data_cfg.stage2_seed),
            "stage3_seed": int(data_cfg.stage3_seed),
            "branch_grid_path": str(branch_grid_path),
        }
    # Use a timestamped directory unless a specific run name was provided.
    run_id = run_name or time.strftime("%Y%m%d-%H%M%S")

    run_dir = Path(results_root) / run_id

    stage1_dir = run_dir / "stage1_deeponet_gradmap"
    stage2_dir = run_dir / "stage2_fno"
    stage3_dir = run_dir / "stage3_joint_finetune"

    for directory in (stage1_dir, stage2_dir, stage3_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Save the top-level run configuration before training begins.
    _write_json(
        run_dir / "pipeline_config.json",
        {
            "data": asdict(data_cfg),
            "dataset": dataset_provenance,
            "stage1": asdict(stage1_cfg),
            "mixed_precision": bool(mixed_precision),
            "mp_dtype": str(mp_dtype),
            "stage2_steps": int(stage2_steps),
            "stage2_batch_size": int(stage2_batch_size),
            "stage2_lr": float(stage2_lr),
            "stage2_infer_batch": int(stage2_infer_batch),
            "joint_steps": int(joint_steps),
            "joint_batch_size": int(joint_batch_size),
            "joint_weight_decay": float(joint_weight_decay),
            "joint_fno_lr_factor": float(joint_fno_lr_factor),
            "joint_seed": int(joint_seed),
            "branch_grid_path": str(branch_grid_path),
        },
    )

    _write_json(
        run_dir / "dataset_source.json",
        dataset_provenance,
    )
    # ------------------------------------------------------------------
    # Stage 1: noisy sensor gradients -> clean regular-grid gradients.
    # ------------------------------------------------------------------

    deeponet_state = train_deeponet_gradmap(
        cfg=stage1_cfg,
        grad_sensor_noisy=stage1_sensor_gradients,
        grad_grid_clean=stage1_grid_gradients,
        branch_grid_path=branch_grid_path,
    )

    # Persist both the best validation checkpoint and the final checkpoint.
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

    # Attempt to record DeepONet inference timing. Benchmark failure does not
    # interrupt the training pipeline; its error is written separately.
    try:
        stage1_test_start = int(stage1_cfg.n_train)

        stage1_test_stop = (
                stage1_test_start
                + int(stage1_cfg.n_test)
        )

        benchmark = benchmark_inference_deeponet(
            model_fn=deeponet_state["model_fn"],
            params=deeponet_state["params"],
            grad_sensor_test=np.asarray(
                sensor_gradients_stage1[
                stage1_test_start:stage1_test_stop
                ]
            ),
            p_test=int(data_cfg.grid_size) ** 2,
            n=min(1_000, int(stage1_cfg.n_test)),
            batch_size=64,
        )

        _write_json(
            stage1_dir / "inference_benchmark.json",
            benchmark,
        )

    except Exception as error:
        (stage1_dir / "inference_benchmark_error.txt").write_text(
            str(error),
            encoding="utf-8",
        )

    # Stage-1 wavefront targets and clean gradient labels are not used after
    # gradient-map training is complete.
    del _wavefronts_stage1
    del clean_grid_gradients_stage1
    del sensor_gradients_stage1
    gc.collect()

    # ------------------------------------------------------------------
    # Stage 2: Stage-1 predicted gradient grids -> FNO -> wavefront.
    # ------------------------------------------------------------------
    (
        wavefronts_stage2,
        sensor_gradients_stage2,
        _clean_grid_gradients_stage2,
    ) = _generate_synthetic_dataset(
        data_cfg=data_cfg,
        seed=int(data_cfg.stage2_seed),
        branch_grid_path=branch_grid_path,
    )

    # Build an FNO configuration from defaults when no explicit configuration
    # namespace was supplied.
    if fno_args is None:
        fno_args = load_config(
            operator_type="fno",
            config_dir=config_dir,
            data_mode="regular_grid",
            grid_size=int(data_cfg.grid_size),
            n_train=int(data_cfg.n_train),
            n_test=int(data_cfg.n_test),
            steps=int(stage2_steps),
            batch_size=int(stage2_batch_size),
            lr=float(stage2_lr),
        )
    else:
        # Avoid mutating a configuration object owned by the caller.
        fno_args = copy.deepcopy(fno_args)

    # The Stage-2 FNO trainer creates its own timestamped output folder inside
    # this stage parent directory.
    fno_args.result_dir = str(stage2_dir)
    fno_args.data_mode = "regular_grid"
    fno_args.grid_size = int(data_cfg.grid_size)
    fno_args.nx = int(data_cfg.grid_size)
    fno_args.ny = int(data_cfg.grid_size)
    fno_args.n_train = int(data_cfg.n_train)
    fno_args.n_test = int(data_cfg.n_test)

    fno_state = train_fno_on_deeponet_outputs(
        fno_args=fno_args,
        deeponet_state=deeponet_state,
        grad_sensor_noisy=stage2_sensor_gradients,
        wavefront_true=stage2_wavefronts,
        batch_infer=int(stage2_infer_batch),
    )

    _write_json(
        stage2_dir / "stage2_manifest.json",
        {
            "fno_trainer_result_dir": fno_state["result_dir"],
            "fno_args": vars(fno_state["args"]),
            "input_source": (
                "Stage-1 DeepONet predictions on the Stage-2 dataset."
            ),
        },
    )

    # Clean Stage-2 gradients are not required by FNO training or joint
    # end-to-end fine-tuning.
    del _clean_grid_gradients_stage2
    gc.collect()

    # ------------------------------------------------------------------
    # Stage 3: end-to-end DeepONet + FNO fine-tuning.
    # ------------------------------------------------------------------
    train_slice = slice(0, int(data_cfg.n_train))
    test_slice = slice(
        int(data_cfg.n_train),
        int(data_cfg.n_train) + int(data_cfg.n_test),
    )
    joint_fno_lr = (
            float(fno_args.lr)
            * float(joint_fno_lr_factor)
    )
    joint_state = finetune_joint_deeponet_fno(
        fno_state=fno_state,
        deeponet_state=deeponet_state,
        grad_sensor_noisy=stage3_sensor_gradients[train_slice],
        wavefront_true=stage3_wavefronts[train_slice],
        grad_sensor_test=stage3_sensor_gradients[test_slice],
        wavefront_test=stage3_wavefronts[test_slice],
        grid_size=int(data_cfg.grid_size),
        batch_size=int(joint_batch_size),
        steps=int(joint_steps),
        lr=joint_fno_lr,
        weight_decay=float(joint_weight_decay),
        seed=int(joint_seed),
    )

    # Save selected and final checkpoints for both jointly fine-tuned models.
    _save_params(
        stage3_dir / "deeponet_params_best",
        joint_state["deeponet"]["params"],
    )

    _save_params(
        stage3_dir / "deeponet_params_last",
        joint_state["deeponet"]["last_params"],
    )

    _save_params(
        stage3_dir / "fno_params_best",
        joint_state["fno"]["params"],
    )

    _save_params(
        stage3_dir / "fno_params_last",
        joint_state["fno"]["last_params"],
    )

    _write_json(
        stage3_dir / "metrics.json",
        {
            "best_relative_l2": joint_state["best_e"],
            "last_relative_l2": joint_state["last_e"],
            "fno_lr": joint_state["fno_lr"],
            "deeponet_lr": joint_state["deeponet_lr"],
            "joint_steps": int(joint_steps),
            "joint_batch_size": int(joint_batch_size),
            "joint_weight_decay": float(joint_weight_decay),
        },
    )

    # Save a compact manifest that links all stage artifacts and headline
    # validation metrics.
    manifest = {
        "run_dir": str(run_dir),
        "stage1_dir": str(stage1_dir),
        "stage2_dir": str(stage2_dir),
        "stage2_fno_trainer_result_dir": fno_state["result_dir"],
        "stage3_dir": str(stage3_dir),
        "dataset": dataset_provenance,
        "stage1_best_relative_l2": deeponet_state["best_e"],
        "stage3_best_relative_l2": joint_state["best_e"],
    }

    _write_json(
        run_dir / "manifest.json",
        manifest,
    )

    # Return only paths and scalar metadata. Release model states and arrays so
    # long-running notebook sessions do not retain unnecessary memory.
    # Release model states and array references.
    del joint_state
    del fno_state
    del deeponet_state

    del stage2_wavefronts
    del stage2_sensor_gradients

    del stage3_wavefronts
    del stage3_sensor_gradients

    del artifact_dataset
    gc.collect()

    return manifest
