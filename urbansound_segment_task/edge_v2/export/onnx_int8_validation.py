"""Keras, FP32 ONNX, and dynamic-INT8 ONNX parity on the fixed cache fixture."""

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
from urbansound_segment_task.edge_v2.features.cache_dataset import build_cache_split, load_verified_cache_records
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256
from .onnx_linear import build_linear_keras_model, inspect_onnx_model, resolve_source_model
from .onnx_quantization import QUANTIZED_SCHEMA_VERSION, summarize_quantized_graph, validate_fp32_artifact
from .onnx_validation import (
    _fixed_metrics, _predict_in_chunks, _synthetic_cases, select_deterministic_fixture,
)


INT8_PARITY_SCHEMA_VERSION = "edge-v2.onnx-int8-parity.v1"
EXPECTED_FIXTURE_IDENTITY = "a3bed1830532864c4dbf61ed55a4bdaf82bd11174ac409a2029074e9ee180c4d"
MAX_ABSOLUTE_ERROR = 0.02
MAX_MEAN_ABSOLUTE_ERROR = 0.002
MIN_TOP1_AGREEMENT = 0.995
MAX_CLIP_ACCURACY_DELTA = 0.002
MAX_CLIP_MACRO_F1_DELTA = 0.002
MIN_CLIP_PREDICTION_AGREEMENT = 0.995


def validate_int8_artifact(
    fp32_artifact_directory: Path, int8_artifact_directory: Path,
    *, onnx_module: Optional[Any] = None,
) -> tuple[dict[str, Any], Path]:
    fp32_manifest, _ = validate_fp32_artifact(fp32_artifact_directory)
    root = Path(int8_artifact_directory)
    manifest = json.loads((root / "artifact-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != QUANTIZED_SCHEMA_VERSION or manifest.get("status", {}).get("outcome") != "success":
        raise ValueError("INT8 ONNX manifest is incompatible or unsuccessful")
    if manifest.get("manifest_sha256") != document_sha256(
        manifest, excluded_fields=("created_at_utc", "manifest_sha256")
    ):
        raise ValueError("INT8 ONNX manifest SHA-256 mismatch")
    if manifest.get("source_fp32_identity") != fp32_manifest["manifest_sha256"]:
        raise ValueError("INT8 source FP32 identity mismatch")
    if manifest.get("source_fp32_sha256") != fp32_manifest["onnx_artifact"]["sha256"]:
        raise ValueError("INT8 source FP32 SHA-256 mismatch")
    if manifest.get("source_model_sha256") != fp32_manifest["source_model_sha256"]:
        raise ValueError("INT8 source model SHA-256 mismatch")
    relative = Path(str(manifest["int8_artifact"]["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("INT8 artifact relative path is unsafe")
    model_path = root / relative
    if model_path.stat().st_size != int(manifest["int8_artifact"]["size_bytes"]):
        raise ValueError("INT8 ONNX artifact size mismatch")
    if streaming_file_sha256(model_path) != manifest["int8_artifact"]["sha256"]:
        raise ValueError("INT8 ONNX artifact SHA-256 mismatch")
    if onnx_module is not None:
        proto = onnx_module.load(str(model_path))
        summarize_quantized_graph(proto, onnx_module)
    return manifest, model_path


def compare_fp32_int8_outputs(
    fp32_output: np.ndarray, int8_output: np.ndarray,
    identities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fp32 = np.asarray(fp32_output, dtype=np.float32)
    int8 = np.asarray(int8_output, dtype=np.float32)
    if fp32.shape != int8.shape or fp32.ndim != 2 or fp32.shape[1] != 10:
        raise ValueError("FP32 and INT8 output shapes differ or violate [N,10]")
    if len(identities) != fp32.shape[0]:
        raise ValueError("INT8 parity identities and output rows differ")
    if not np.isfinite(fp32).all() or not np.isfinite(int8).all():
        raise ValueError("FP32 or INT8 output contains NaN or Inf")
    absolute = np.abs(fp32.astype(np.float64) - int8.astype(np.float64))
    flat_index = int(np.argmax(absolute))
    row_index, class_index = np.unravel_index(flat_index, absolute.shape)
    fp32_top1 = fp32.argmax(axis=1)
    int8_top1 = int8.argmax(axis=1)
    changed_indices = np.flatnonzero(fp32_top1 != int8_top1)
    changed = []
    for index in changed_indices.tolist():
        ordered = np.sort(fp32[index])
        changed.append(
            {
                "fixture_record": dict(identities[index]),
                "fp32_top1_class": int(fp32_top1[index]),
                "int8_top1_class": int(int8_top1[index]),
                "fp32_top1_margin": float(ordered[-1] - ordered[-2]),
            }
        )
    return {
        "sample_count": int(fp32.shape[0]),
        "shape": list(fp32.shape),
        "maximum_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "rmse": float(math.sqrt(float(np.mean(np.square(absolute))))),
        "top1_agreement": float(np.mean(fp32_top1 == int8_top1)),
        "changed_top1_count": len(changed),
        "changed_top1_samples": changed,
        "maximum_error_location": {
            "fixture_record": dict(identities[row_index]),
            "class_index": int(class_index),
        },
        "probability_sum": {
            "fp32_maximum_deviation_from_one": float(np.max(np.abs(fp32.sum(axis=1) - 1.0))),
            "int8_maximum_deviation_from_one": float(np.max(np.abs(int8.sum(axis=1) - 1.0))),
        },
    }


def numeric_thresholds_pass(result: Mapping[str, Any]) -> bool:
    return (
        float(result["maximum_absolute_error"]) <= MAX_ABSOLUTE_ERROR
        and float(result["mean_absolute_error"]) <= MAX_MEAN_ABSOLUTE_ERROR
        and float(result["top1_agreement"]) >= MIN_TOP1_AGREEMENT
        and float(result["probability_sum"]["fp32_maximum_deviation_from_one"]) <= 1e-5
        and float(result["probability_sum"]["int8_maximum_deviation_from_one"]) <= 1e-5
    )


def _fold1_metric_parity(verified: Any, keras_predict: Any, fp32_predict: Any, int8_predict: Any) -> dict[str, Any]:
    split = build_cache_split("test", (1,), verified.records)
    keras_probabilities = _predict_in_chunks(keras_predict, split.X)
    fp32_probabilities = _predict_in_chunks(fp32_predict, split.X)
    int8_probabilities = _predict_in_chunks(int8_predict, split.X)
    predictions = {
        "keras": keras_probabilities.argmax(axis=1),
        "fp32": fp32_probabilities.argmax(axis=1),
        "int8": int8_probabilities.argmax(axis=1),
    }
    metrics: dict[str, Any] = {}
    clip_predictions: dict[str, np.ndarray] = {}
    reference_keys: Optional[list[str]] = None
    reference_true: Optional[np.ndarray] = None
    for runtime, probabilities in (
        ("keras", keras_probabilities), ("fp32", fp32_probabilities), ("int8", int8_probabilities)
    ):
        clip_true, clip_predicted, clip_keys = aggregate_clip_probabilities(
            split.clip_keys, split.y, probabilities
        )
        if reference_keys is None:
            reference_keys, reference_true = clip_keys, clip_true
        elif clip_keys != reference_keys or not np.array_equal(clip_true, reference_true):
            raise ValueError("INT8 Fold 1 clip aggregation identity mismatch")
        clip_predictions[runtime] = clip_predicted
        metrics[runtime] = {
            "segment": _fixed_metrics(split.y, predictions[runtime]),
            "clip": _fixed_metrics(clip_true, clip_predicted),
        }
    fp32_clip = metrics["fp32"]["clip"]
    int8_clip = metrics["int8"]["clip"]
    return {
        "fold": 1,
        "segment_count": split.segment_count,
        "metadata_clip_count": split.metadata_clip_count,
        "evaluable_clip_count": split.evaluable_clip_count,
        "excluded_zero_segment_clip_count": split.zero_segment_clip_count,
        "metrics": metrics,
        "fp32_int8_segment_prediction_agreement": float(np.mean(predictions["fp32"] == predictions["int8"])),
        "fp32_int8_clip_prediction_agreement": float(np.mean(clip_predictions["fp32"] == clip_predictions["int8"])),
        "clip_accuracy_delta": abs(fp32_clip["accuracy"] - int8_clip["accuracy"]),
        "clip_macro_f1_delta": abs(fp32_clip["macro_f1"] - int8_clip["macro_f1"]),
    }


def metric_thresholds_pass(result: Mapping[str, Any]) -> bool:
    return (
        float(result["clip_accuracy_delta"]) <= MAX_CLIP_ACCURACY_DELTA
        and float(result["clip_macro_f1_delta"]) <= MAX_CLIP_MACRO_F1_DELTA
        and float(result["fp32_int8_clip_prediction_agreement"]) >= MIN_CLIP_PREDICTION_AGREEMENT
    )


def validate_linear_int8_onnx(
    *, run_manifest_path: Path, cache_root: Path, fp32_artifact_directory: Path,
    int8_artifact_directory: Path, sample_count: int, output_directory: Path,
    pretty: bool,
) -> dict[str, Any]:
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("INT8 parity output directory already exists")
    import onnx
    import onnxruntime as ort
    import tensorflow as tf

    fp32_manifest, fp32_path = validate_fp32_artifact(fp32_artifact_directory)
    int8_manifest, int8_path = validate_int8_artifact(
        fp32_artifact_directory, int8_artifact_directory, onnx_module=onnx
    )
    inspect_onnx_model(onnx.load(str(fp32_path)), checker=onnx.checker)
    source = resolve_source_model(run_manifest_path, model="linear", fold=1)
    if source.source_model_sha256 != fp32_manifest["source_model_sha256"]:
        raise ValueError("Keras source and FP32 manifest identity mismatch")
    keras_model = build_linear_keras_model(tf)
    keras_model.load_weights(str(source.weights_path))
    fp32_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    if fp32_session.get_providers() != ["CPUExecutionProvider"] or int8_session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("INT8 parity activated a non-CPU provider")

    def keras_predict(values: np.ndarray) -> np.ndarray:
        return np.asarray(keras_model(np.asarray(values, dtype=np.float32), training=False), dtype=np.float32)

    def fp32_predict(values: np.ndarray) -> np.ndarray:
        return np.asarray(fp32_session.run(["probabilities"], {"embedding": np.asarray(values, dtype=np.float32)})[0], dtype=np.float32)

    def int8_predict(values: np.ndarray) -> np.ndarray:
        return np.asarray(int8_session.run(["probabilities"], {"embedding": np.asarray(values, dtype=np.float32)})[0], dtype=np.float32)

    verified = load_verified_cache_records(cache_root)
    fixture, fixture_features = select_deterministic_fixture(verified, sample_count)
    if fixture["fixture_identity"] != EXPECTED_FIXTURE_IDENTITY:
        raise ValueError("INT8 parity fixture identity differs from FP32 parity fixture")
    batch_results = []
    all_fp32 = []
    all_int8 = []
    all_identities: list[dict[str, Any]] = []
    keras_fp32_agreements = []
    for case_name, values in _synthetic_cases():
        identities = [{"fixture_id": case_name, "sample_index": index} for index in range(values.shape[0])]
        keras_output = keras_predict(values)
        fp32_output = fp32_predict(values)
        int8_output = int8_predict(values)
        comparison = compare_fp32_int8_outputs(fp32_output, int8_output, identities)
        comparison.update({"fixture_id": case_name, "batch_size": values.shape[0]})
        batch_results.append(comparison)
        keras_fp32_agreements.append(float(np.mean(keras_output.argmax(axis=1) == fp32_output.argmax(axis=1))))
        all_fp32.append(fp32_output)
        all_int8.append(int8_output)
        all_identities.extend(identities)
    fixture_ids = fixture["selected_records"]
    keras_fixture = _predict_in_chunks(keras_predict, fixture_features)
    fp32_fixture = _predict_in_chunks(fp32_predict, fixture_features)
    int8_fixture = _predict_in_chunks(int8_predict, fixture_features)
    fixture_comparison = compare_fp32_int8_outputs(fp32_fixture, int8_fixture, fixture_ids)
    fixture_comparison.update({"fixture_id": fixture["fixture_identity"], "batch_size": 128})
    batch_results.append(fixture_comparison)
    keras_fp32_agreements.append(float(np.mean(keras_fixture.argmax(axis=1) == fp32_fixture.argmax(axis=1))))
    all_fp32.append(fp32_fixture)
    all_int8.append(int8_fixture)
    all_identities.extend(dict(row) for row in fixture_ids)
    global_comparison = compare_fp32_int8_outputs(
        np.concatenate(all_fp32, axis=0), np.concatenate(all_int8, axis=0), all_identities
    )
    metric_parity = _fold1_metric_parity(verified, keras_predict, fp32_predict, int8_predict)
    numeric_success = numeric_thresholds_pass(global_comparison)
    metric_success = metric_thresholds_pass(metric_parity)
    success = numeric_success and metric_success and min(keras_fp32_agreements) == 1.0
    document: dict[str, Any] = {
        "schema_version": INT8_PARITY_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "source_model_sha256": source.source_model_sha256,
        "fp32_onnx_sha256": fp32_manifest["onnx_artifact"]["sha256"],
        "int8_onnx_sha256": int8_manifest["int8_artifact"]["sha256"],
        "fixture_identity": fixture["fixture_identity"],
        "sample_count": sample_count,
        "synthetic_sample_count": sum(values.shape[0] for _, values in _synthetic_cases()),
        "total_numeric_sample_count": global_comparison["sample_count"],
        "batch_sizes": [1, 7, 32, 128],
        "batch_results": batch_results,
        "numeric_error": global_comparison,
        "keras_fp32_top1_agreement": min(keras_fp32_agreements),
        "numeric_thresholds": {
            "maximum_absolute_error": MAX_ABSOLUTE_ERROR,
            "mean_absolute_error": MAX_MEAN_ABSOLUTE_ERROR,
            "minimum_top1_agreement": MIN_TOP1_AGREEMENT,
        },
        "fold1_metric_parity": metric_parity,
        "metric_thresholds": {
            "maximum_clip_accuracy_delta": MAX_CLIP_ACCURACY_DELTA,
            "maximum_clip_macro_f1_delta": MAX_CLIP_MACRO_F1_DELTA,
            "minimum_clip_prediction_agreement": MIN_CLIP_PREDICTION_AGREEMENT,
        },
        "runtime": {
            "onnxruntime_version": importlib.metadata.version("onnxruntime"),
            "configured_providers": ["CPUExecutionProvider"],
            "fp32_active_providers": list(fp32_session.get_providers()),
            "int8_active_providers": list(int8_session.get_providers()),
        },
        "status": {
            "outcome": "success" if success else "failure",
            "error": None if success else {"type": "Int8ParityThresholdFailure"},
        },
    }
    document["parity_sha256"] = document_sha256(
        document, excluded_fields=("created_at_utc", "parity_sha256")
    )
    output.mkdir(parents=True, exist_ok=False)
    write_result(output / "fixture-manifest.json", fixture, pretty=pretty)
    write_result(output / "parity-result.json", document, pretty=pretty)
    return document


__all__ = [
    "EXPECTED_FIXTURE_IDENTITY", "INT8_PARITY_SCHEMA_VERSION", "MAX_ABSOLUTE_ERROR",
    "MAX_CLIP_ACCURACY_DELTA", "MAX_CLIP_MACRO_F1_DELTA", "MAX_MEAN_ABSOLUTE_ERROR",
    "MIN_CLIP_PREDICTION_AGREEMENT", "MIN_TOP1_AGREEMENT", "compare_fp32_int8_outputs",
    "metric_thresholds_pass", "numeric_thresholds_pass", "validate_int8_artifact",
    "validate_linear_int8_onnx",
]
