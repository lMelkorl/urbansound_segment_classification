"""Exact legacy Goal 1 full-window segmentation policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SEGMENTATION_POLICY_ID = "yamnet-legacy-full-window-v1"


@dataclass(frozen=True)
class SegmentationPolicy:
    policy_id: str = SEGMENTATION_POLICY_ID
    target_sample_rate: int = 16_000
    channels: str = "mono"
    window_samples: int = 15_360
    window_duration_seconds: float = 0.960
    hop_samples: int = 7_680
    overlap_fraction: float = 0.5
    tail_policy: str = "drop"
    short_clip_policy: str = "emit_zero_segment"
    model_input_dtype: str = "float32"
    normalization: str = "pcm_waveform_to_minus_one_plus_one"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


LEGACY_SEGMENTATION_POLICY = SegmentationPolicy()


def segment_plan(sample_count: int, policy: SegmentationPolicy = LEGACY_SEGMENTATION_POLICY) -> dict[str, Any]:
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError("sample_count must be a non-negative integer")
    if sample_count < policy.window_samples:
        starts: list[int] = []
        dropped_tail = sample_count
    else:
        starts = list(range(0, sample_count - policy.window_samples + 1, policy.hop_samples))
        dropped_tail = sample_count - (starts[-1] + policy.window_samples)
    return {
        "segment_count": len(starts),
        "segment_start_samples": starts,
        "dropped_tail_samples": dropped_tail,
        "short_clip_zero_segment": len(starts) == 0,
    }


def iter_segments(waveform: Any, policy: SegmentationPolicy = LEGACY_SEGMENTATION_POLICY):
    plan = segment_plan(int(waveform.shape[0]), policy)
    for start in plan["segment_start_samples"]:
        yield waveform[start : start + policy.window_samples]


__all__ = [
    "LEGACY_SEGMENTATION_POLICY", "SEGMENTATION_POLICY_ID", "SegmentationPolicy",
    "iter_segments", "segment_plan",
]
