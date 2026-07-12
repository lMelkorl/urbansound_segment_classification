"""Pure presentation transforms for the local Gradio demo."""

from __future__ import annotations

from typing import Any, Mapping


CONFIDENCE_NOTICE = (
    "Confidence is not calibrated and is not a guaranteed probability of correctness."
)
DEPLOYMENT_NOTICE = (
    "The all-data deployment artifact is deployment-only and has no independent held-out test metric."
)
OFFICIAL_METRICS_CONTEXT = (
    "Official 10-fold Linear baseline:\n"
    "Clip macro-F1 0.789331 ± 0.033921\n"
    "Clip accuracy 0.780485 ± 0.036271\n\n"
    "These cross-fold metrics belong to the evaluated Linear architecture.\n"
    "The all-data deployment artifact itself has no independent held-out test metric."
)


def _milliseconds(value: Any) -> float:
    return round(float(value) / 1_000_000.0, 3)


def empty_presentation(message: str = "") -> dict[str, Any]:
    return {
        "message": message,
        "prediction": "No analysis yet.",
        "top3": [],
        "timeline": [],
        "audio_info": {},
        "request_timing": {},
        "technical": "",
    }


def present_result(result: Mapping[str, Any], startup: Mapping[str, Any]) -> dict[str, Any]:
    status = result.get("status", {})
    segmentation = result.get("segmentation", {})
    if status.get("outcome") == "no_prediction":
        prediction = "No prediction — audio is shorter than one 0.96 s segment."
        top3: list[list[Any]] = []
    elif status.get("outcome") == "success":
        clip = result["clip_prediction"]
        label = str(clip["class_name"]).replace("_", " ").upper()
        prediction = (
            f"# {label}\n## {float(clip['confidence']) * 100:.1f}%\n"
            f"Class ID: `{int(clip['class_id'])}` · Segments: `{int(clip['segment_count'])}`\n\n"
            f"⚠️ {CONFIDENCE_NOTICE}"
        )
        top3 = [
            [rank, str(row["class_name"]).replace("_", " "), float(row["confidence"])]
            for rank, row in enumerate(clip["top3"], start=1)
        ]
    else:
        return empty_presentation("Analysis could not be completed safely.")
    sample_rate = int(segmentation.get("target_sample_rate", 16_000))
    window_samples = int(segmentation.get("window_samples", 15_360))
    timeline = [
        [
            round(int(row["start_sample"]) / sample_rate, 3),
            round((int(row["start_sample"]) + window_samples) / sample_rate, 3),
            str(row["class_name"]).replace("_", " "),
            float(row["confidence"]),
        ]
        for row in result.get("segment_predictions", [])
    ]
    metadata = result["audio_metadata"]
    preprocessing = result["preprocessing"]
    dropped_seconds = int(segmentation.get("dropped_tail_samples", 0)) / sample_rate
    audio_info = {
        "original_sample_rate_hz": int(metadata["original_sample_rate"]),
        "channels": int(metadata["original_channel_count"]),
        "duration_seconds": float(metadata["original_duration_seconds"]),
        "resampled_sample_count": int(preprocessing["resampled_sample_count"]),
        "segment_count": int(segmentation["segment_count"]),
        "dropped_tail_seconds": round(dropped_seconds, 6),
    }
    timing = result["timing"]
    request_timing = {
        "decode_ms": _milliseconds(timing["audio_decode_ns"]),
        "resampling_preprocessing_ms": _milliseconds(timing["resampling_and_preprocessing_ns"]),
        "segment_inference_ms": _milliseconds(timing["segment_inference_total_ns"]),
        "aggregation_ms": _milliseconds(timing["aggregation_ns"]),
        "request_end_to_end_ms": _milliseconds(timing["end_to_end_ns"]),
        "startup_one_time_ms": {
            key.removesuffix("_ns"): _milliseconds(value) for key, value in startup.items()
        },
    }
    classifier = result["classifier_artifact"]
    yamnet = result["yamnet_artifact"]
    runtime = result["runtime"]
    technical = (
        f"**YAMNet artifact:** `{yamnet['tree_sha256']}`  \n"
        f"**FP32 ONNX artifact:** `{classifier['onnx_sha256']}`  \n"
        f"**Provider:** `{', '.join(result['providers'])}`  \n"
        f"**Segmentation:** `{segmentation['policy_id']}`  \n"
        f"**TensorFlow:** `{runtime.get('tensorflow_version', 'unknown')}` · "
        f"**ONNX Runtime:** `{runtime.get('onnxruntime_version', 'unknown')}`  \n"
        f"**Model loads:** YAMNet `{runtime['model_load_counts']['yamnet']}`, "
        f"ONNX `{runtime['model_load_counts']['onnx_classifier']}`  \n\n"
        f"⚠️ {DEPLOYMENT_NOTICE}\n\n{OFFICIAL_METRICS_CONTEXT}"
    )
    return {
        "message": "",
        "prediction": prediction,
        "top3": top3,
        "timeline": timeline,
        "audio_info": audio_info,
        "request_timing": request_timing,
        "technical": technical,
    }


def presentation_outputs(presentation: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        presentation["message"], presentation["prediction"], presentation["top3"],
        presentation["timeline"], presentation["audio_info"],
        presentation["request_timing"], presentation["technical"],
    )


__all__ = [
    "CONFIDENCE_NOTICE", "DEPLOYMENT_NOTICE", "OFFICIAL_METRICS_CONTEXT",
    "empty_presentation", "present_result", "presentation_outputs",
]

