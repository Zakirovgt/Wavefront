from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from wavefront.metamodel.gradmap import DeepONetGradMapConfig
from wavefront.pipelines.three_stage import ThreeStageDataConfig


def _read_yaml_mapping(
        path: str | Path,
) -> dict[str, Any]:
    """
    Load a YAML file and verify that its top-level value is a mapping.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed YAML content as a dictionary. An empty YAML file returns an
        empty dictionary.

    Raises:
        TypeError: If the YAML document exists but its top-level value is not
            a mapping.
        yaml.YAMLError: If the YAML content cannot be parsed.
        FileNotFoundError: If the requested configuration file does not exist.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    # An empty YAML document is treated as an empty configuration.
    if payload is None:
        return {}

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a YAML mapping in {path}, "
            f"but got {type(payload).__name__}."
        )

    return payload


def _allowed_field_names(
        dataclass_type: type,
) -> set[str]:
    """
    Return the valid constructor argument names for a dataclass type.

    Args:
        dataclass_type: Dataclass class whose declared field names should be
            retrieved.

    Returns:
        Set of field names accepted by the dataclass constructor.
    """
    return {
        field.name
        for field in fields(dataclass_type)
    }


def _validate_keys(
        section_name: str,
        values: dict[str, Any],
        allowed_keys: set[str],
) -> None:
    """
    Validate that a YAML configuration section contains no unknown keys.

    Args:
        section_name: Human-readable YAML section name.
        values: Parsed key-value mapping for the section.
        allowed_keys: Set of accepted keys for the section.

    Raises:
        KeyError: If one or more unsupported keys are present.
    """
    unknown_keys = set(values) - allowed_keys

    if unknown_keys:
        unknown_text = ", ".join(sorted(unknown_keys))

        raise KeyError(
            f"Unknown key(s) in '{section_name}': {unknown_text}."
        )


def load_three_stage_run_kwargs(
        config_path: str | Path,
) -> dict[str, Any]:
    """
    Load a three-stage YAML configuration into pipeline-runner keyword args.

    The returned dictionary can be passed directly to:

        run_three_stage_pipeline(**kwargs)

    Expected YAML structure:

        runtime:
            branch_grid_path: ...
            results_root: ...
            run_name: ...
            mixed_precision: ...
            mp_dtype: ...

        data:
            # Fields of ThreeStageDataConfig.

        stage1:
            # Optional DeepONetGradMapConfig overrides.

        stage2:
            config_dir: ...
            steps: ...
            batch_size: ...
            lr: ...
            infer_batch: ...

        stage3:
            steps: ...
            batch_size: ...
            weight_decay: ...
            fno_lr_factor: ...
            seed: ...

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing keyword arguments compatible with
        ``run_three_stage_pipeline``.

    Raises:
        TypeError: If any expected YAML section is not a mapping.
        KeyError: If a section contains unsupported configuration keys.
        ValueError: If dataclass construction fails because supplied values are
            invalid for the corresponding configuration objects.

    Notes:
        ``grid_size``, ``n_train``, and ``n_test`` in the Stage-1
        configuration are always inherited from the shared ``data`` section.
        This guarantees that Stage 1 uses the same generated dataset layout
        as the rest of the three-stage pipeline.
    """
    config = _read_yaml_mapping(config_path)

    # Missing sections are represented by empty mappings and therefore use
    # pipeline defaults.
    runtime = config.get("runtime", {})
    data = config.get("data", {})
    stage1 = config.get("stage1", {})
    stage2 = config.get("stage2", {})
    stage3 = config.get("stage3", {})

    # Every named section must be a YAML mapping.
    for section_name, section in {
        "runtime": runtime,
        "data": data,
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
    }.items():
        if not isinstance(section, dict):
            raise TypeError(
                f"Section '{section_name}' must be a YAML mapping."
            )

    # Validate keys early so configuration typos do not silently fall back to
    # defaults or pass unnoticed into a later training stage.
    _validate_keys(
        "runtime",
        runtime,
        {
            "branch_grid_path",
            "results_root",
            "run_name",
            "mixed_precision",
            "mp_dtype",
        },
    )

    _validate_keys(
        "data",
        data,
        _allowed_field_names(ThreeStageDataConfig),
    )

    _validate_keys(
        "stage1",
        stage1,
        _allowed_field_names(DeepONetGradMapConfig),
    )

    _validate_keys(
        "stage2",
        stage2,
        {
            "config_dir",
            "steps",
            "batch_size",
            "lr",
            "infer_batch",
        },
    )

    _validate_keys(
        "stage3",
        stage3,
        {
            "steps",
            "batch_size",
            "weight_decay",
            "fno_lr_factor",
            "seed",
        },
    )

    # Build the shared synthetic-data configuration first.
    data_cfg = ThreeStageDataConfig(**data)

    # Stage-1 dimensions and split sizes must always match the common
    # synthetic-data configuration. Explicit stage1 values for these fields
    # are intentionally overridden.
    stage1_values = {
        "grid_size": data_cfg.grid_size,
        "n_train": data_cfg.n_train,
        "n_test": data_cfg.n_test,
        **stage1,
    }

    stage1_cfg = DeepONetGradMapConfig(**stage1_values)

    # Translate compact YAML section names into the argument names expected by
    # run_three_stage_pipeline.
    return {
        "branch_grid_path": runtime.get(
            "branch_grid_path",
            "1.csv",
        ),
        "data_cfg": data_cfg,
        "stage1_cfg": stage1_cfg,
        "config_dir": stage2.get(
            "config_dir",
            "configs",
        ),
        "results_root": runtime.get(
            "results_root",
            "results/three_stage",
        ),
        "run_name": runtime.get("run_name"),
        "mixed_precision": bool(
            runtime.get("mixed_precision", False)
        ),
        "mp_dtype": str(
            runtime.get("mp_dtype", "bfloat16")
        ),
        "stage2_steps": int(
            stage2.get("steps", 20_000)
        ),
        "stage2_batch_size": int(
            stage2.get("batch_size", 64)
        ),
        "stage2_lr": float(
            stage2.get("lr", 1e-3)
        ),
        "stage2_infer_batch": int(
            stage2.get("infer_batch", 64)
        ),
        "joint_steps": int(
            stage3.get("steps", 10_000)
        ),
        "joint_batch_size": int(
            stage3.get("batch_size", 32)
        ),
        "joint_weight_decay": float(
            stage3.get("weight_decay", 1e-6)
        ),
        "joint_fno_lr_factor": float(
            stage3.get("fno_lr_factor", 0.1)
        ),
        "joint_seed": int(
            stage3.get("seed", 0)
        ),
    }
