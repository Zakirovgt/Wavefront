from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from wavefront.metamodel.gradmap import DeepONetGradMapConfig
from wavefront.pipelines.two_stage import TwoStageDataConfig


def _read_yaml_mapping(
        path: str | Path,
) -> dict[str, Any]:
    """
    Load a YAML configuration file and verify that its top-level value is a
    mapping.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed YAML content as a dictionary. An empty YAML document produces
        an empty dictionary.

    Raises:
        FileNotFoundError: If the requested configuration file does not exist.
        yaml.YAMLError: If the YAML content cannot be parsed.
        TypeError: If the top-level YAML value is not a mapping.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    # Treat an empty YAML document as an empty configuration.
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
    Return valid constructor argument names for a dataclass.

    Args:
        dataclass_type: Dataclass type whose declared fields should be read.

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
    Validate that a YAML section contains only supported configuration keys.

    Args:
        section_name: Human-readable name of the YAML section.
        values: Parsed key-value mapping from the YAML section.
        allowed_keys: Set of supported keys for that section.

    Raises:
        KeyError: If one or more unsupported keys are found.
    """
    unknown_keys = set(values) - allowed_keys

    if unknown_keys:
        unknown_text = ", ".join(sorted(unknown_keys))

        raise KeyError(
            f"Unknown key(s) in '{section_name}': {unknown_text}."
        )


def load_two_stage_run_kwargs(
        config_path: str | Path,
) -> dict[str, Any]:
    """
    Load a two-stage YAML configuration into pipeline-runner keyword arguments.

    The returned dictionary can be passed directly to:

        run_two_stage_pipeline(**kwargs)

    Expected YAML structure:

        runtime:
            branch_grid_path: ...
            results_root: ...
            run_name: ...
            mixed_precision: ...
            mp_dtype: ...

        data:
            # Fields of TwoStageDataConfig.

        stage1:
            # Optional DeepONetGradMapConfig overrides.

        fno:
            config_dir: ...
            seed: ...

        joint:
            steps: ...
            batch_size: ...
            lr: ...
            weight_decay: ...
            seed: ...

    Args:
        config_path: Path to the two-stage YAML configuration file.

    Returns:
        Dictionary of keyword arguments compatible with
        ``run_two_stage_pipeline``.

    Raises:
        FileNotFoundError: If config_path does not exist.
        yaml.YAMLError: If the YAML document cannot be parsed.
        TypeError: If a named configuration section is not a YAML mapping.
        KeyError: If a section contains unsupported keys.

    Notes:
        The shared ``data`` section is the source of truth for ``grid_size``,
        ``n_train``, and ``n_test``. Those values are always imposed on the
        Stage-1 configuration so generated data and DeepONet training use the
        same grid resolution and train/test split.
    """
    config = _read_yaml_mapping(config_path)

    # Missing sections use empty mappings so their downstream defaults apply.
    runtime = config.get("runtime", {})
    data = config.get("data", {})
    stage1 = config.get("stage1", {})
    fno = config.get("fno", {})
    joint = config.get("joint", {})

    # All named sections must be mappings rather than scalar YAML values or
    # lists. This keeps later configuration construction predictable.
    for section_name, section in {
        "runtime": runtime,
        "data": data,
        "stage1": stage1,
        "fno": fno,
        "joint": joint,
    }.items():
        if not isinstance(section, dict):
            raise TypeError(
                f"Section '{section_name}' must be a YAML mapping."
            )

    # Reject misspelled or unsupported settings before model construction.
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
        _allowed_field_names(TwoStageDataConfig),
    )

    _validate_keys(
        "stage1",
        stage1,
        _allowed_field_names(DeepONetGradMapConfig),
    )

    _validate_keys(
        "fno",
        fno,
        {
            "config_dir",
            "seed",
        },
    )

    _validate_keys(
        "joint",
        joint,
        {
            "steps",
            "batch_size",
            "lr",
            "weight_decay",
            "seed",
        },
    )

    # Build the shared synthetic-data configuration first.
    data_cfg = TwoStageDataConfig(**data)

    # Ensure Stage 1 uses the same regular-grid resolution and data split as
    # the dataset generator. The values from data_cfg intentionally override
    # any duplicate fields supplied in the stage1 YAML section.
    stage1_values = {
        **stage1,
        "grid_size": data_cfg.grid_size,
        "n_train": data_cfg.n_train,
        "n_test": data_cfg.n_test,
    }

    stage1_cfg = DeepONetGradMapConfig(**stage1_values)

    # Convert compact YAML section names into the parameter names expected by
    # run_two_stage_pipeline.
    return {
        "branch_grid_path": runtime.get(
            "branch_grid_path",
            "1.csv",
        ),
        "data_cfg": data_cfg,
        "stage1_cfg": stage1_cfg,
        "config_dir": fno.get(
            "config_dir",
            "configs",
        ),
        "results_root": runtime.get(
            "results_root",
            "results/two_stage",
        ),
        "run_name": runtime.get("run_name"),
        "mixed_precision": bool(
            runtime.get("mixed_precision", False)
        ),
        "mp_dtype": str(
            runtime.get("mp_dtype", "bfloat16")
        ),
        "fno_seed": int(
            fno.get("seed", 123)
        ),
        "joint_steps": int(
            joint.get("steps", 50_000)
        ),
        "joint_batch_size": int(
            joint.get("batch_size", 32)
        ),
        "joint_lr": float(
            joint.get("lr", 1e-3)
        ),
        "joint_weight_decay": float(
            joint.get("weight_decay", 1e-6)
        ),
        "joint_seed": int(
            joint.get("seed", 0)
        ),
    }
