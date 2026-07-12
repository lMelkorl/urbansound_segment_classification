"""Deterministic cache and metric parity validation for the Linear FP32 ONNX export."""

from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from urbansound_segment_task.edge_v2.data.manifest import document_sha256
from urbansound_segment_task.edge_v2.evaluation.aggregation import aggregate_clip_probabilities
from urbansound_segment_task.edge_v2.evaluation.result_schema import utc_timestamp, write_result
from urbansound_segment_task.edge_v2.features.cache_dataset import (
    VerifiedCacheRecords, build_cache_split, load_verified_cache_records,
)
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256
from .onnx_linear import (
    EXPORT_SCHEMA_VERSION, build_linear_keras_model, inspect_onnx_model, resolve_source_model,
)


PARITY_SCHEMA_VERSION = "edge-v2.onnx-parity.v1"
FIXTURE_SCHEMA_VERSION = "edge-v2.onnx-fixture.v1"
MAX_ABSOLUTE_ERROR = 1e-5
MAX_MEAN_ABSOLUTE_ERROR = 1e-6
MIN_TOP1_AGREEMENT = 1.0
PROBABILITY_SUM_TOLERANCE = 1e-5
REQUIRED_BATCH_SIZES = (1, 7, 32, 128)


def select_deterministic_fixture(
    verified: VerifiedCacheRecords, sample_count: int
) -> tuple[dict[str, Any], np.ndarray]:
    if sample_count < 100:
        raise ValueError("fixture sample count must cover all 100 fold-class groups")
    groups: dict[tuple[int, int], list[tuple[dict[str, Any], np.ndarray]]] = {
        (fold, class_id): [] for fold in range(1, 11) for class_id in range(10)
    }
    for record in verified.records:
        fold = int(record["fold"])
        class_id = int(record["class_id"])
        clip_key = str(record["clip_key"])
        embeddings = np.asarray(record["embeddings"], dtype=np.float32)
        for segment_index in range(embeddings.shape[0]):
            identity = {
                "clip_key": clip_key,
                "segment_index": segment_index,
                "segment_start_sample": segment_index * 7680,
                "fold": fold,
                "class_id": class_id,
            }
            groups[(fold, class_id)].append((identity, embeddings[segment_index]))
    for values in groups.values():
        values.sort(key=lambda item: (item[0]["clip_key"], item[0]["segment_start_sample"]))
        if not values:
            raise ValueError("fixture fold-class coverage group is empty")
    selected: list[tuple[dict[str, Any], np.ndarray]] = []
    depth = 0
    ordered_groups = sorted(groups)
    while len(selected) < sample_count:
        added = False
        for key in ordered_groups:
            values = groups[key]
            if depth < len(values):
                selected.append(values[depth])
                added = True
                if len(selected) == sample_count:
                    break
        if not added:
            raise ValueError("fixture sample count exceeds available unique segments")
        depth += 1
    identities = [item[0] for item in selected]
    unique = {
        (item["clip_key"], item["segment_start_sample"]) for item in identities
    }
    if len(unique) != sample_count:
        raise ValueError("fixture contains duplicate segment identities")
    folds = sorted({int(item["fold"]) for item in identities})
    classes = sorted({int(item["class_id"]) for item in identities})
    if folds != list(range(1, 11)) or classes != list(range(10)):
        raise ValueError("fixture does not cover all folds and classes")
    document: dict[str, Any] = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "cache_identity": verified.cache_identity,
        "cache_index_sha256": verified.index_sha256,
        "dataset_manifest_sha256": verified.dataset_manifest_sha256,
        "selection": {
            "method": "round_robin_over_sorted_fold_class_groups",
            "sort_key": ["clip_key", "segment_start_sample"],
            "sample_count": sample_count,
            "hop_samples": 7680,
            "duplicate_count": 0,
            "fold_coverage": folds,
            "class_coverage": classes,
        },
        "selected_records": identities,
    }
    document["fixture_identity"] = document_sha256(
        document, excluded_fields=("created_at_utc", "fixture_identity")
    )
    features = np.stack([item[1] for item in selected]).astype(np.float32, copy=False)
    return document, features


def compare_probability_outputs(
    keras_output: np.ndarray, onnx_output: np.ndarray, identities: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    reference = np.asarray(keras_output, dtype=np.float32)
    candidate = np.asarray(onnx_output, dtype=np.float32)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 10:
        raise ValueError("parity output shape mismatch")
    if len(identities) != reference.shape[0]:
        raise ValueError("parity identities and output rows differ")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("parity outputs contain NaN or Inf")
    absolute = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    flat_index = int(np.argmax(absolute))
    row_index, class_index = np.unravel_index(flat_index, absolute.shape)
    top1_reference = reference.argmax(axis=1)
    top1_candidate = candidate.argmax(axis=1)
    return {
        "sample_count": int(reference.shape[0]),
        "shape": list(reference.shape),
        "maximum_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "rmse": float(math.sqrt(float(np.mean(np.square(absolute))))),
        "top1_agreement": float(np.mean(top1_reference == top1_candidate)),
        "maximum_error_location": {
            "fixture_record": dict(identities[row_index]),
            "class_index": int(class_index),
        },
        "probability_sum": {
            "keras_maximum_deviation_from_one": float(np.max(np.abs(reference.sum(axis=1) - 1.0))),
            "onnx_maximum_deviation_from_one": float(np.max(np.abs(candidate.sum(axis=1) - 1.0))),
        },
    }


def parity_passes(result: Mapping[str, Any]) -> bool:
    probability = result["probability_sum"]
    return (
        float(result["maximum_absolute_error"]) <= MAX_ABSOLUTE_ERROR
        and float(result["mean_absolute_error"]) <= MAX_MEAN_ABSOLUTE_ERROR
        and float(result["top1_agreement"]) >= MIN_TOP1_AGREEMENT
        and float(probability["keras_maximum_deviation_from_one"]) <= PROBABILITY_SUM_TOLERANCE
        and float(probability["onnx_maximum_deviation_from_one"]) <= PROBABILITY_SUM_TOLERANCE
    )


def _fixed_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(true, dtype=np.int64)
    y_pred = np.asarray(predicted, dtype=np.int64)
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    precision = np.divide(diagonal, predicted_count, out=np.zeros(10), where=predicted_count != 0)
    recall = np.divide(diagonal, support, out=np.zeros(10), where=support != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(10),
        where=(precision + recall) != 0,
    )
    return {
        "accuracy": float(diagonal.sum() / matrix.sum()),
        "macro_f1": float(f1.mean()),
        "prediction_count": int(matrix.sum()),
    }


def _predict_in_chunks(function: Any, features: np.ndarray, chunk_size: int = 128) -> np.ndarray:
    outputs = [np.asarray(function(features[start:start + chunk_size]), dtype=np.float32)
               for start in range(0, features.shape[0], chunk_size)]
    return np.concatenate(outputs, axis=0)


def _metric_parity(
    verified: VerifiedCacheRecords, keras_predict: Any, onnx_predict: Any
) -> dict[str, Any]:
    split = build_cache_split("test", (1,), verified.records)
    keras_probabilities = _predict_in_chunks(keras_predict, split.X)
    onnx_probabilities = _predict_in_chunks(onnx_predict, split.X)
    keras_segment = keras_probabilities.argmax(axis=1)
    onnx_segment = onnx_probabilities.argmax(axis=1)
    keras_clip_true, keras_clip_predicted, keras_keys = aggregate_clip_probabilities(
        split.clip_keys, split.y, keras_probabilities
    )
    onnx_clip_true, onnx_clip_predicted, onnx_keys = aggregate_clip_probabilities(
        split.clip_keys, split.y, onnx_probabilities
    )
    if keras_keys != onnx_keys or not np.array_equal(keras_clip_true, onnx_clip_true):
        raise ValueError("Fold 1 clip aggregation identity mismatch")
    keras_clip = _fixed_metrics(keras_clip_true, keras_clip_predicted)
    onnx_clip = _fixed_metrics(onnx_clip_true, onnx_clip_predicted)
    keras_segment_metrics = _fixed_metrics(split.y, keras_segment)
    onnx_segment_metrics = _fixed_metrics(split.y, onnx_segment)
    return {
        "fold": 1,
        "segment_count": split.segment_count,
        "metadata_clip_count": split.metadata_clip_count,
        "evaluable_clip_count": split.evaluable_clip_count,
        "excluded_zero_segment_clip_count": split.zero_segment_clip_count,
        "keras": {"segment": keras_segment_metrics, "clip": keras_clip},
        "onnx": {"segment": onnx_segment_metrics, "clip": onnx_clip},
        "segment_top1_agreement": float(np.mean(keras_segment == onnx_segment)),
        "clip_top1_agreement": float(np.mean(keras_clip_predicted == onnx_clip_predicted)),
        "clip_accuracy_delta": abs(keras_clip["accuracy"] - onnx_clip["accuracy"]),
        "clip_macro_f1_delta": abs(keras_clip["macro_f1"] - onnx_clip["macro_f1"]),
    }


def _synthetic_cases() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(42)
    return [
        ("synthetic_zero", np.zeros((1, 1024), dtype=np.float32)),
        ("synthetic_small_positive", np.full((7, 1024), 1e-3, dtype=np.float32)),
        ("synthetic_small_negative", np.full((32, 1024), -1e-3, dtype=np.float32)),
        ("synthetic_seeded_random", rng.normal(0.0, 0.1, size=(128, 1024)).astype(np.float32)),
    ]


def validate_linear_onnx(
    *, run_manifest_path: Path, cache_root: Path, onnx_artifact_directory: Path,
    sample_count: int, output_directory: Path, pretty: bool,
    tf_module: Optional[Any] = None, onnx_module: Optional[Any] = None,
    ort_module: Optional[Any] = None,
) -> dict[str, Any]:
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("parity output directory already exists")
    artifact_root = Path(onnx_artifact_directory)
    manifest_path = artifact_root / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXPORT_SCHEMA_VERSION or manifest.get("status", {}).get("outcome") != "success":
        raise ValueError("ONNX artifact manifest is not successful or compatible")
    if manifest.get("manifest_sha256") != document_sha256(
        manifest, excluded_fields=("created_at_utc", "manifest_sha256")
    ):
        raise ValueError("ONNX artifact manifest SHA-256 mismatch")
    source = resolve_source_model(run_manifest_path, model="linear", fold=1)
    if manifest.get("source_model_sha256") != source.source_model_sha256:
        raise ValueError("ONNX source model SHA-256 mismatch")
    relative = Path(str(manifest["onnx_artifact"]["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("ONNX artifact relative path is unsafe")
    onnx_path = artifact_root / relative
    if onnx_path.stat().st_size != int(manifest["onnx_artifact"]["size_bytes"]):
        raise ValueError("ONNX artifact size mismatch")
    onnx_sha256 = streaming_file_sha256(onnx_path)
    if onnx_sha256 != manifest["onnx_artifact"]["sha256"]:
        raise ValueError("ONNX artifact SHA-256 mismatch")
    if tf_module is None:
        import tensorflow as tf_module
    if onnx_module is None:
        import onnx as onnx_module
    if ort_module is None:
        import onnxruntime as ort_module
    model_proto = onnx_module.load(str(onnx_path))
    graph = inspect_onnx_model(model_proto, checker=onnx_module.checker)
    keras_model = build_linear_keras_model(tf_module)
    keras_model.load_weights(str(source.weights_path))
    session = ort_module.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    active_providers = list(session.get_providers())
    if active_providers != ["CPUExecutionProvider"]:
        raise ValueError("ONNX Runtime session activated a non-CPU provider")

    def keras_predict(values: np.ndarray) -> np.ndarray:
        return np.asarray(keras_model(np.asarray(values, dtype=np.float32), training=False), dtype=np.float32)

    def onnx_predict(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            session.run(["probabilities"], {"embedding": np.asarray(values, dtype=np.float32)})[0],
            dtype=np.float32,
        )

    verified = load_verified_cache_records(cache_root)
    fixture, fixture_features = select_deterministic_fixture(verified, sample_count)
    batch_results = []
    all_keras = []
    all_onnx = []
    all_identities: list[dict[str, Any]] = []
    for case_name, values in _synthetic_cases():
        identities = [{"fixture_id": case_name, "sample_index": index} for index in range(values.shape[0])]
        keras_output = keras_predict(values)
        onnx_output = onnx_predict(values)
        comparison = compare_probability_outputs(keras_output, onnx_output, identities)
        comparison["fixture_id"] = case_name
        comparison["batch_size"] = values.shape[0]
        batch_results.append(comparison)
        all_keras.append(keras_output)
        all_onnx.append(onnx_output)
        all_identities.extend(identities)
    fixture_ids = fixture["selected_records"]
    cached_keras = _predict_in_chunks(keras_predict, fixture_features)
    cached_onnx = _predict_in_chunks(onnx_predict, fixture_features)
    cached_comparison = compare_probability_outputs(cached_keras, cached_onnx, fixture_ids)
    cached_comparison["fixture_id"] = fixture["fixture_identity"]
    cached_comparison["batch_size"] = 128
    batch_results.append(cached_comparison)
    all_keras.append(cached_keras)
    all_onnx.append(cached_onnx)
    all_identities.extend(dict(item) for item in fixture_ids)
    global_comparison = compare_probability_outputs(
        np.concatenate(all_keras, axis=0), np.concatenate(all_onnx, axis=0), all_identities
    )
    metric_parity = _metric_parity(verified, keras_predict, onnx_predict)
    numeric_success = parity_passes(global_comparison)
    metric_success = (
        metric_parity["segment_top1_agreement"] == 1.0
        and metric_parity["clip_top1_agreement"] == 1.0
        and metric_parity["clip_accuracy_delta"] <= 1e-12
        and metric_parity["clip_macro_f1_delta"] <= 1e-12
    )
    success = numeric_success and metric_success
    document: dict[str, Any] = {
        "schema_version": PARITY_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "onnx_sha256": onnx_sha256,
        "source_model_sha256": source.source_model_sha256,
        "source_model_identity": source.source_model_identity,
        "fixture_identity": fixture["fixture_identity"],
        "sample_count": sample_count,
        "synthetic_sample_count": sum(values.shape[0] for _, values in _synthetic_cases()),
        "total_numeric_sample_count": global_comparison["sample_count"],
        "batch_sizes": list(REQUIRED_BATCH_SIZES),
        "batch_results": batch_results,
        "numeric_error": global_comparison,
        "thresholds": {
            "maximum_absolute_error": MAX_ABSOLUTE_ERROR,
            "mean_absolute_error": MAX_MEAN_ABSOLUTE_ERROR,
            "top1_agreement": MIN_TOP1_AGREEMENT,
            "probability_sum_tolerance": PROBABILITY_SUM_TOLERANCE,
        },
        "top1_agreement": global_comparison["top1_agreement"],
        "fold1_metric_parity": metric_parity,
        "onnx_graph": graph,
        "runtime": {
            "onnxruntime_version": importlib.metadata.version("onnxruntime"),
            "configured_providers": ["CPUExecutionProvider"],
            "active_session_providers": active_providers,
            "available_runtime_providers": list(ort_module.get_available_providers()),
        },
        "scope": {
            "classifier_only": True,
            "input": "precomputed_yamnet_embedding_1024_float32",
            "yamnet_included": False,
            "raw_audio_decode_included": False,
            "full_system_latency_or_metric_experiment": False,
        },
        "status": {
            "outcome": "success" if success else "failure",
            "error": None if success else {"type": "ParityThresholdFailure"},
        },
    }
    document["parity_sha256"] = document_sha256(
        document, excluded_fields=("created_at_utc", "parity_sha256")
    )
    output.mkdir(parents=True, exist_ok=False)
    write_result(output / "export-result.json", manifest, pretty=pretty)
    write_result(output / "fixture-manifest.json", fixture, pretty=pretty)
    write_result(output / "parity-result.json", document, pretty=pretty)
    return document


__all__ = [
    "FIXTURE_SCHEMA_VERSION", "MAX_ABSOLUTE_ERROR", "MAX_MEAN_ABSOLUTE_ERROR",
    "MIN_TOP1_AGREEMENT", "PARITY_SCHEMA_VERSION", "PROBABILITY_SUM_TOLERANCE",
    "REQUIRED_BATCH_SIZES", "compare_probability_outputs", "parity_passes",
    "select_deterministic_fixture", "validate_linear_onnx",
]
