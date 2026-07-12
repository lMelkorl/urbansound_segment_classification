"""Injectable stage contracts and a standard-library synthetic pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union


PIPELINE_STAGE_ORDER = ("decode", "preprocess", "inference", "aggregate", "end_to_end")
MEASURED_STAGE_ORDER = PIPELINE_STAGE_ORDER[:-1]


@dataclass(frozen=True)
class PipelineStages:
    decode: Callable[[Any], Any]
    preprocess: Callable[[Any], Any]
    inference: Callable[[Any], Any]
    aggregate: Callable[[Any], Any]


class PipelineExecutionTracker:
    """Per-run state used only to identify the active stage after a failure."""

    def __init__(self) -> None:
        self.current_stage: Optional[str] = None


def execute_pipeline(
    stages: PipelineStages,
    source: Any,
    *,
    tracker: Optional[PipelineExecutionTracker] = None,
) -> Any:
    """Execute every stage in order, passing each output to the next stage."""

    if tracker is not None:
        tracker.current_stage = "decode"
    decoded = stages.decode(source)
    if tracker is not None:
        tracker.current_stage = "preprocess"
    processed = stages.preprocess(decoded)
    if tracker is not None:
        tracker.current_stage = "inference"
    prediction = stages.inference(processed)
    if tracker is not None:
        tracker.current_stage = "aggregate"
    result = stages.aggregate(prediction)
    if tracker is not None:
        tracker.current_stage = "complete"
    return result


def build_synthetic_pipeline(input_size: int) -> tuple[PipelineStages, int]:
    """Build a deterministic, local pipeline with no file or framework access."""

    if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size < 1:
        raise ValueError("input_size must be a positive integer")
    if input_size > 1_000_000:
        raise ValueError("input_size must not exceed 1000000")

    def decode(source_size: int) -> list[int]:
        state = 0x00C0FFEE
        decoded: list[int] = []
        for index in range(source_size):
            state = (1_664_525 * state + 1_013_904_223 + index) & 0xFFFFFFFF
            decoded.append((state >> 16) & 0xFF)
        return decoded

    def preprocess(decoded: list[int]) -> list[float]:
        if not decoded:
            return []
        center = sum(decoded) / len(decoded)
        scale = max(1.0, max(abs(value - center) for value in decoded))
        return [(value - center) / scale for value in decoded]

    def inference(processed: list[float]) -> list[float]:
        scores: list[float] = []
        denominator = max(1, len(processed))
        for class_index in range(8):
            accumulator = 0.0
            for position, value in enumerate(processed):
                weight = ((position + 3) * (class_index + 5) % 29 - 14) / 14.0
                accumulator += value * weight
            scores.append(accumulator / denominator)
        return scores

    def aggregate(scores: list[float]) -> dict[str, Union[float, int]]:
        if not scores:
            return {"class_index": -1, "score": 0.0}
        class_index = max(range(len(scores)), key=scores.__getitem__)
        magnitude = sum(abs(score) for score in scores) or 1.0
        return {
            "class_index": class_index,
            "score": scores[class_index] / magnitude,
        }

    return PipelineStages(decode, preprocess, inference, aggregate), input_size
