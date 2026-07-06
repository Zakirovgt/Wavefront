from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp


@dataclass(frozen=True)
class ModelArtifacts:
    """
    Container for inference-time model artifacts.

    Attributes:
        model_config:
            Dictionary loaded from ``args_inference.json``. It typically
            contains the model architecture, preprocessing settings, and
            other configuration values needed to rebuild the model.

        params:
            Restored model parameter pytree loaded from the Orbax checkpoint.

        branch_sensor_coords:
            Optional normalized sensor-coordinate array with shape
            ``(P_sensor, 2)``. This is present for sensor-based models such
            as DeepONet and may be absent for purely regular-grid models.
    """

    model_config: dict
    params: object  # JAX/Flax parameter pytree.
    branch_sensor_coords: np.ndarray | None = None


def load_artifacts(
        artifacts_dir: Path,
) -> ModelArtifacts:
    """
    Load model configuration, trained parameters, and optional sensor metadata.

    Expected artifact-directory contents:

        args_inference.json
            JSON configuration exported during training.

        params/
            Orbax checkpoint directory containing the selected parameter tree.

        branch_sensor_coords.npy
            Optional normalized sensor-coordinate array.

    Args:
        artifacts_dir: Path to the artifact directory created by a training
            routine. The path may be relative, absolute, or use ``~``.

    Returns:
        A ModelArtifacts instance containing:

            model_config:
                Parsed inference configuration dictionary.

            params:
                Restored JAX/Flax parameter pytree.

            branch_sensor_coords:
                Sensor-coordinate array when available; otherwise None.

    Raises:
        FileNotFoundError: If required files such as ``args_inference.json``
            or the ``params`` checkpoint directory do not exist.
        json.JSONDecodeError: If the configuration file is not valid JSON.
        Exception: Any checkpoint restoration error raised by Orbax.

    Notes:
        ``Path.resolve()`` converts the supplied directory into an absolute
        canonical path before files are loaded. This helps avoid ambiguity when
        inference is launched from a different working directory.
    """
    # Normalize the artifact directory path and expand a leading home shortcut.
    artifacts_dir = Path(artifacts_dir).expanduser().resolve()

    # Load the model and inference configuration saved during training.
    cfg_path = artifacts_dir / "args_inference.json"

    cfg = json.loads(
        cfg_path.read_text(encoding="utf-8")
    )

    # Restore the selected parameter tree from the standard Orbax checkpoint.
    params_path = (artifacts_dir / "params").resolve()

    params = ocp.StandardCheckpointer().restore(
        str(params_path)
    )

    # Sensor coordinates are optional because grid-only models, such as some
    # FNO configurations, may not use an irregular sensor layout.
    branch_path = artifacts_dir / "branch_sensor_coords.npy"

    branch_sensor_coords = (
        np.load(branch_path)
        if branch_path.exists()
        else None
    )

    return ModelArtifacts(
        model_config=cfg,
        params=params,
        branch_sensor_coords=branch_sensor_coords,
    )
