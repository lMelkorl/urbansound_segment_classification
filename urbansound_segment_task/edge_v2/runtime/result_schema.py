"""Schema constants and safe serialization for offline audio inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from urbansound_segment_task.edge_v2.evaluation.result_schema import write_result


OFFLINE_AUDIO_INFERENCE_SCHEMA_VERSION = "edge-v2.offline-audio-inference.v1"
SOFTMAX_LIMITATION = (
    "Softmax confidence is not calibrated and must not be interpreted as a guaranteed "
    "probability of correctness."
)
CLASS_NAMES = (
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
)


def write_inference_result(path: Path, document: Mapping[str, Any], *, pretty: bool) -> None:
    write_result(Path(path), document, pretty=pretty)


__all__ = [
    "CLASS_NAMES", "OFFLINE_AUDIO_INFERENCE_SCHEMA_VERSION", "SOFTMAX_LIMITATION",
    "write_inference_result",
]

