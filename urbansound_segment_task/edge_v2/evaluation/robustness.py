"""Deterministic, framework-light contracts for audio robustness evaluation."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from urbansound_segment_task.edge_v2.data.manifest import document_sha256
from urbansound_segment_task.edge_v2.evaluation.aggregation import CANONICAL_CLASSES
from urbansound_segment_task.edge_v2.evaluation.metrics import classification_metrics
from urbansound_segment_task.edge_v2.evaluation.segmentation import segment_plan


SCHEMA_VERSION = "edge-v2.audio-robustness-cross-fold.v1"
PANEL_SCHEMA_VERSION = "edge-v2.audio-robustness-panel.v1"
UNIT_SCHEMA_VERSION = "edge-v2.audio-robustness-fold-condition.v1"
CONDITIONS = (
    "clean",
    "white_noise_snr_20db",
    "white_noise_snr_10db",
    "white_noise_snr_0db",
    "gain_minus_12db",
    "bandlimit_8khz_roundtrip",
)
CLASS_NAMES = (
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
)
PERTURBATION_CONTRACTS: dict[str, dict[str, Any]] = {
    "clean": {"operation": "identity"},
    "white_noise_snr_20db": {"operation": "gaussian_white_noise", "snr_db": 20.0},
    "white_noise_snr_10db": {"operation": "gaussian_white_noise", "snr_db": 10.0},
    "white_noise_snr_0db": {"operation": "gaussian_white_noise", "snr_db": 0.0},
    "gain_minus_12db": {"operation": "scalar_gain", "gain_db": -12.0},
    "bandlimit_8khz_roundtrip": {
        "operation": "resample_roundtrip", "source_rate_hz": 16_000,
        "intermediate_rate_hz": 8_000, "target_rate_hz": 16_000,
        "resample_type": "soxr_hq", "length_policy": "trim_or_zero_extend_to_input_length",
    },
}


def deterministic_seed(clip_key: str, condition: str) -> int:
    if condition not in CONDITIONS:
        raise ValueError("unsupported robustness condition")
    digest = hashlib.sha256(f"{clip_key}\0{condition}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array), dtype=np.float64))) if array.size else 0.0


def measured_snr_db(signal: np.ndarray, perturbed: np.ndarray) -> float | None:
    signal_rms = rms(signal)
    noise_rms = rms(np.asarray(perturbed, dtype=np.float64) - np.asarray(signal, dtype=np.float64))
    if signal_rms == 0.0 or noise_rms == 0.0:
        return None
    return float(20.0 * math.log10(signal_rms / noise_rms))


def _fix_length(values: np.ndarray, length: int) -> np.ndarray:
    current = int(values.shape[0])
    if current >= length:
        return np.asarray(values[:length], dtype=np.float32)
    return np.pad(values, (0, length - current), mode="constant").astype(np.float32, copy=False)


def apply_perturbation(
    waveform: np.ndarray,
    *,
    clip_key: str,
    condition: str,
    resample: Callable[..., np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(waveform, dtype=np.float32)
    if source.ndim != 1 or not np.all(np.isfinite(source)):
        raise ValueError("waveform must be finite mono float32")
    if condition not in CONDITIONS:
        raise ValueError("unsupported robustness condition")
    metadata: dict[str, Any] = {
        "condition": condition, "input_sample_count": int(source.shape[0]),
        "hard_clipping_applied": False,
    }
    if condition == "clean":
        output = source.copy()
    elif condition.startswith("white_noise_snr_"):
        target = float(PERTURBATION_CONTRACTS[condition]["snr_db"])
        signal_rms = rms(source)
        metadata.update({"target_snr_db": target, "silent_input": signal_rms == 0.0})
        if signal_rms == 0.0 or source.size == 0:
            output = source.copy()
            metadata["measured_snr_db"] = None
        else:
            generator = np.random.default_rng(deterministic_seed(clip_key, condition))
            noise = generator.standard_normal(source.shape[0]).astype(np.float64)
            noise -= noise.mean(dtype=np.float64)
            noise_rms = rms(noise)
            if noise_rms == 0.0:
                raise ValueError("generated noise has zero RMS")
            target_noise_rms = signal_rms / (10.0 ** (target / 20.0))
            scaled = noise * (target_noise_rms / noise_rms)
            output = (source.astype(np.float64) + scaled).astype(np.float32)
            metadata["measured_snr_db"] = measured_snr_db(source, output)
    elif condition == "gain_minus_12db":
        factor = float(10.0 ** (-12.0 / 20.0))
        output = (source * np.float32(factor)).astype(np.float32, copy=False)
        metadata.update({"gain_db": -12.0, "gain_factor": factor})
    else:
        if resample is None:
            import librosa
            resample = librosa.resample
        down = np.asarray(
            resample(source, orig_sr=16_000, target_sr=8_000, res_type="soxr_hq"),
            dtype=np.float32,
        )
        up = np.asarray(
            resample(down, orig_sr=8_000, target_sr=16_000, res_type="soxr_hq"),
            dtype=np.float32,
        )
        output = _fix_length(up, int(source.shape[0]))
        metadata["roundtrip_sample_counts"] = [int(source.shape[0]), int(down.shape[0]), int(up.shape[0])]
    if output.dtype != np.float32 or output.shape != source.shape or not np.all(np.isfinite(output)):
        raise ValueError("perturbation output contract mismatch")
    metadata["output_sample_count"] = int(output.shape[0])
    metadata["segment_start_samples_unchanged"] = (
        segment_plan(int(source.shape[0]))["segment_start_samples"]
        == segment_plan(int(output.shape[0]))["segment_start_samples"]
    )
    return output, metadata


def select_panel(
    metadata_rows: Iterable[Mapping[str, Any]],
    *,
    evaluable_clip_keys: set[str],
    clips_per_class_per_fold: int,
    dataset_manifest_sha256: str,
    cache_identity: str,
) -> dict[str, Any]:
    if clips_per_class_per_fold < 1:
        raise ValueError("clips per class per fold must be positive")
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for metadata_index, row in enumerate(metadata_rows):
        fold, class_id = int(row["fold"]), int(row["class_id"])
        clip_key = str(row["clip_key"])
        if clip_key in seen:
            raise ValueError("duplicate clip key in metadata")
        seen.add(clip_key)
        if clip_key in evaluable_clip_keys:
            groups[(fold, class_id)].append(
                {"metadata_index": metadata_index, "clip_key": clip_key, "fold": fold,
                 "class_id": class_id, "class_name": str(row["class_name"])}
            )
    selected: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for fold in range(1, 11):
        for class_id in CANONICAL_CLASSES:
            ordered = sorted(groups[(fold, class_id)], key=lambda item: (item["metadata_index"], item["clip_key"]))
            chosen = ordered[:clips_per_class_per_fold]
            selected.extend({key: item[key] for key in ("clip_key", "fold", "class_id", "class_name")} for item in chosen)
            coverage.append({
                "fold": fold, "class_id": class_id, "requested": clips_per_class_per_fold,
                "available": len(ordered), "selected": len(chosen),
                "shortfall": max(0, clips_per_class_per_fold - len(chosen)),
            })
    keys = [item["clip_key"] for item in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("panel contains duplicate clips")
    document: dict[str, Any] = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "selection_policy": {
            "order": "metadata_row_order_then_stable_clip_key",
            "eligibility": "at_least_one_legacy_segment",
            "clips_per_class_per_fold": clips_per_class_per_fold,
            "prediction_dependent_selection": False,
        },
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "cache_identity": cache_identity,
        "clip_count": len(selected), "coverage": coverage, "clips": selected,
    }
    document["panel_identity_sha256"] = document_sha256(document, excluded_fields=("panel_identity_sha256",))
    return document


def clip_probability_mean(segment_probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(segment_probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != 10:
        raise ValueError("segment probabilities must have shape [N,10] with N >= 1")
    result = values.mean(axis=0)
    if not np.all(np.isfinite(result)):
        raise ValueError("clip probabilities are not finite")
    return result


def fold_condition_metrics(
    clips: Sequence[Mapping[str, Any]], *, clean_clips: Sequence[Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    true = [int(item["class_id"]) for item in clips]
    predicted = [int(item["predicted_class_id"]) for item in clips]
    result = classification_metrics(true, predicted, labels=CANONICAL_CLASSES)
    result["segment_count"] = sum(int(item["segment_count"]) for item in clips)
    if clean_clips is None:
        result.update({"prediction_flip_rate": 0.0, "confidence_change_mean": 0.0})
        return result
    clean_by_key = {str(item["clip_key"]): item for item in clean_clips}
    if set(clean_by_key) != {str(item["clip_key"]) for item in clips}:
        raise ValueError("clean and perturbed clip identities differ")
    flips: list[bool] = []
    confidence_changes: list[float] = []
    for item in clips:
        clean = clean_by_key[str(item["clip_key"])]
        flips.append(int(item["predicted_class_id"]) != int(clean["predicted_class_id"]))
        confidence_changes.append(float(item["confidence"]) - float(clean["confidence"]))
    result["prediction_flip_rate"] = float(np.mean(flips))
    result["confidence_change_mean"] = float(np.mean(confidence_changes))
    return result


def degradation_classification(macro_f1_drop: float) -> str:
    if macro_f1_drop <= 0.025:
        return "minor"
    if macro_f1_drop <= 0.075:
        return "moderate"
    return "major"


def aggregate_conditions(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in fold_results:
        grouped[str(row["condition"])].append(row)
    aggregates: dict[str, Any] = {}
    for condition, rows in grouped.items():
        accuracy = [float(row["metrics"]["accuracy"]) for row in rows]
        f1 = [float(row["metrics"]["macro_f1"]) for row in rows]
        flips = [float(row["metrics"]["prediction_flip_rate"]) for row in rows]
        aggregates[condition] = {
            "fold_count": len(rows),
            "clip_accuracy_mean": statistics.fmean(accuracy),
            "clip_accuracy_population_standard_deviation": statistics.pstdev(accuracy),
            "clip_macro_f1_mean": statistics.fmean(f1),
            "clip_macro_f1_population_standard_deviation": statistics.pstdev(f1),
            "prediction_flip_rate_mean": statistics.fmean(flips),
        }
    clean = aggregates.get("clean")
    if clean:
        for condition, aggregate in aggregates.items():
            accuracy_drop = clean["clip_accuracy_mean"] - aggregate["clip_accuracy_mean"]
            f1_drop = clean["clip_macro_f1_mean"] - aggregate["clip_macro_f1_mean"]
            clean_f1 = clean["clip_macro_f1_mean"]
            aggregate.update({
                "absolute_accuracy_drop": accuracy_drop,
                "absolute_macro_f1_drop": f1_drop,
                "relative_macro_f1_drop_percent": 0.0 if clean_f1 == 0.0 else 100.0 * f1_drop / clean_f1,
                "degradation_classification": degradation_classification(f1_drop),
            })
    return aggregates


def require_clean_agreement(runtime_predictions: Sequence[int], cached_predictions: Sequence[int]) -> dict[str, Any]:
    if len(runtime_predictions) != len(cached_predictions) or not runtime_predictions:
        raise ValueError("clean control prediction arrays must be equal and non-empty")
    matches = [int(left) == int(right) for left, right in zip(runtime_predictions, cached_predictions)]
    agreement = float(np.mean(matches))
    if agreement != 1.0:
        raise ValueError("clean runtime/cache top-1 agreement must be 100%")
    return {"clip_count": len(matches), "matching_top1_count": sum(matches), "top1_agreement": agreement}


__all__ = [
    "CLASS_NAMES", "CONDITIONS", "PANEL_SCHEMA_VERSION", "PERTURBATION_CONTRACTS",
    "SCHEMA_VERSION", "UNIT_SCHEMA_VERSION", "aggregate_conditions", "apply_perturbation",
    "clip_probability_mean", "degradation_classification", "deterministic_seed",
    "fold_condition_metrics", "measured_snr_db", "require_clean_agreement", "rms",
    "select_panel",
]
