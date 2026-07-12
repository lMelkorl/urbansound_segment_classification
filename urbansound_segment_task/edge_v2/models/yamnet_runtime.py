"""Lazy, local-only YAMNet SavedModel loading and output contracts."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .yamnet_artifact import ArtifactVerificationError, verify_yamnet_artifact


@dataclass(frozen=True)
class LoadedYamnet:
    model: Any
    tensorflow: Any
    loader_method: str
    tensorflow_version: Optional[str]
    visible_devices: tuple[str, ...]
    artifact_identity: dict


def _shape_list(tensor: Any) -> list[Optional[int]]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise ValueError("tensor shape is unavailable")
    values = shape.as_list() if hasattr(shape, "as_list") else list(shape)
    return [int(value) if value is not None else None for value in values]


def validate_yamnet_outputs(outputs: Any) -> dict:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise ValueError("YAMNet output must contain scores, embeddings, and spectrogram")
    scores, embeddings, spectrogram = outputs
    scores_shape = _shape_list(scores)
    embeddings_shape = _shape_list(embeddings)
    spectrogram_shape = _shape_list(spectrogram)
    if not scores_shape or scores_shape[-1] != 521:
        raise ValueError("YAMNet scores dimension must be 521")
    if not embeddings_shape or embeddings_shape[-1] != 1024:
        raise ValueError("YAMNet embedding dimension must be 1024")
    if not spectrogram_shape:
        raise ValueError("YAMNet spectrogram shape is unavailable")
    if scores_shape[0] != embeddings_shape[0]:
        raise ValueError("YAMNet score and embedding frame counts differ")
    return {
        "scores_shape": scores_shape,
        "embeddings_shape": embeddings_shape,
        "spectrogram_shape": spectrogram_shape,
        "frame_count": scores_shape[0],
    }


def _safe_visible_devices(tensorflow: Any) -> tuple[str, ...]:
    try:
        devices: Sequence[Any] = tensorflow.config.get_visible_devices()
    except Exception:
        return ()
    safe_types = {
        str(getattr(device, "device_type", "unknown")).upper() for device in devices
    }
    return tuple(sorted(safe_types))


def load_local_yamnet(
    artifact_directory: Path,
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> LoadedYamnet:
    """Verify an artifact, then lazily import TensorFlow and load only a local path."""

    identity = verify_yamnet_artifact(artifact_directory)
    model_path = Path(artifact_directory) / identity["model_relative_path"]
    tensorflow = import_module("tensorflow")
    loader_method = "tf.saved_model.load"
    try:
        model = tensorflow.saved_model.load(str(model_path))
        if not callable(model):
            raise TypeError("loaded SavedModel is not callable")
    except Exception:
        tensorflow_hub = import_module("tensorflow_hub")
        model = tensorflow_hub.load(str(model_path))
        if not callable(model):
            raise TypeError("loaded TF Hub model is not callable")
        loader_method = "tensorflow_hub.load(local_path)"
    return LoadedYamnet(
        model=model,
        tensorflow=tensorflow,
        loader_method=loader_method,
        tensorflow_version=getattr(tensorflow, "__version__", None),
        visible_devices=_safe_visible_devices(tensorflow),
        artifact_identity=identity,
    )
