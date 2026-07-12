"""FP32 ONNX export and numeric parity for the all-cache deployment Linear model."""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from urbansound_segment_task.edge_v2.data.manifest import document_sha256
from urbansound_segment_task.edge_v2.evaluation.result_schema import utc_timestamp, write_result
from urbansound_segment_task.edge_v2.features.cache_dataset import load_verified_cache_records
from urbansound_segment_task.edge_v2.models.deployment_classifier import (
    DEPLOYMENT_ID,
    DEPLOYMENT_SCHEMA_VERSION,
    EXPECTED_PARAMETER_COUNT,
)
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256

from .onnx_linear import (
    EXPECTED_OPSET,
    EXPORT_SCHEMA_VERSION,
    _publish_binary_new,
    build_linear_keras_model,
    inspect_onnx_model,
)
from .onnx_validation import (
    MAX_ABSOLUTE_ERROR,
    MAX_MEAN_ABSOLUTE_ERROR,
    MIN_TOP1_AGREEMENT,
    PROBABILITY_SUM_TOLERANCE,
    REQUIRED_BATCH_SIZES,
    compare_probability_outputs,
    parity_passes,
    select_deterministic_fixture,
)


DEPLOYMENT_PARITY_SCHEMA_VERSION = "edge-v2.deployment-onnx-parity.v1"


@dataclass(frozen=True)
class DeploymentSource:
    weights_path: Path
    model_sha256: str
    model_size_bytes: int
    result_sha256: str
    config_sha256: str
    cache_identity: str
    cache_index_sha256: str
    dataset_manifest_sha256: str
    yamnet_artifact_tree_sha256: str
    training_segment_count: int


def load_deployment_source(source_run: Path) -> DeploymentSource:
    root = Path(source_run)
    result = json.loads((root / "deployment-result.json").read_text(encoding="utf-8"))
    if result.get("schema_version") != DEPLOYMENT_SCHEMA_VERSION or result.get("status", {}).get("outcome") != "success":
        raise ValueError("deployment source result is incompatible or unsuccessful")
    if result.get("deployment_id") != DEPLOYMENT_ID:
        raise ValueError("deployment source identity mismatch")
    if result.get("deployment_only") is not True or result.get("independent_test_metrics_available") is not False:
        raise ValueError("deployment-only scientific labels are missing")
    if result.get("result_sha256") != document_sha256(
        result, excluded_fields=("created_at_utc", "training", "result_sha256")
    ):
        raise ValueError("deployment source result SHA-256 mismatch")
    if int(result.get("architecture", {}).get("parameter_count", 0)) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("deployment source parameter count mismatch")
    artifact = result["artifacts"]["model"]
    relative = Path(str(artifact["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("deployment source model relative path is unsafe")
    weights = root / relative
    if weights.stat().st_size != int(artifact["size_bytes"]):
        raise ValueError("deployment source model size mismatch")
    digest = streaming_file_sha256(weights)
    if digest != artifact["sha256"]:
        raise ValueError("deployment source model SHA-256 mismatch")
    return DeploymentSource(
        weights_path=weights, model_sha256=digest, model_size_bytes=weights.stat().st_size,
        result_sha256=str(result["result_sha256"]), config_sha256=str(result["config_sha256"]),
        cache_identity=str(result["cache_identity"]), cache_index_sha256=str(result["cache_index_sha256"]),
        dataset_manifest_sha256=str(result["dataset_manifest_sha256"]),
        yamnet_artifact_tree_sha256=str(result["yamnet_artifact_tree_sha256"]),
        training_segment_count=int(result["training_data"]["segment_count"]),
    )


def export_deployment_linear_onnx(
    *, source_run: Path, output_path: Path, manifest_path: Path, opset: int, pretty: bool,
    tf_module: Optional[Any] = None, tf2onnx_module: Optional[Any] = None,
    onnx_module: Optional[Any] = None,
) -> dict[str, Any]:
    if opset != EXPECTED_OPSET:
        raise ValueError("only fixed ONNX opset 15 is supported")
    output = Path(output_path)
    manifest_destination = Path(manifest_path)
    if output.exists() or manifest_destination.exists():
        raise FileExistsError("deployment ONNX output or manifest already exists")
    source = load_deployment_source(source_run)
    if tf_module is None:
        import tensorflow as tf_module
    if tf2onnx_module is None:
        import tf2onnx as tf2onnx_module
    if onnx_module is None:
        import onnx as onnx_module
    model = build_linear_keras_model(tf_module)
    model.load_weights(str(source.weights_path))
    if int(model.count_params()) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("deployment export model parameter count mismatch")
    signature = (tf_module.TensorSpec([None, 1024], tf_module.float32, name="embedding"),)
    model_proto, _ = tf2onnx_module.convert.from_keras(
        model, input_signature=signature, opset=opset, output_path=None
    )
    graph = inspect_onnx_model(model_proto, checker=onnx_module.checker)
    payload = model_proto.SerializeToString()
    _publish_binary_new(output, payload)
    digest = streaming_file_sha256(output)
    document: dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "deployment_id": DEPLOYMENT_ID,
        "deployment_only": True,
        "independent_test_metrics_available": False,
        "scientific_scope": "deployment-only; not an independently evaluated test model",
        "source_model_identity": source.result_sha256,
        "source_model_sha256": source.model_sha256,
        "source_model": {
            "deployment_result_sha256": source.result_sha256,
            "config_sha256": source.config_sha256,
            "cache_identity": source.cache_identity,
            "cache_index_sha256": source.cache_index_sha256,
            "dataset_manifest_sha256": source.dataset_manifest_sha256,
            "yamnet_artifact_tree_sha256": source.yamnet_artifact_tree_sha256,
            "training_segment_count": source.training_segment_count,
        },
        "architecture_id": "linear",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "input_contract": {"name": "embedding", "dtype": "float32", "shape": ["batch", 1024]},
        "output_contract": {
            "name": "probabilities", "dtype": "float32", "shape": ["batch", 10],
            "semantic": "softmax_probabilities",
        },
        "converter": {
            "method": "tf2onnx.convert.from_keras",
            "tensorflow_version": importlib.metadata.version("tensorflow"),
            "tf_keras_version": importlib.metadata.version("tf-keras"),
            "tf2onnx_version": importlib.metadata.version("tf2onnx"),
            "onnx_version": importlib.metadata.version("onnx"),
        },
        "opset": opset,
        "onnx_graph": graph,
        "onnx_artifact": {
            "relative_path": output.name, "size_bytes": output.stat().st_size,
            "sha256": digest, "precision": "fp32",
        },
        "runtime": {"target": "onnxruntime_cpu"},
        "limitations": [
            "Deployment-only model trained on all evaluable segments from all ten folds.",
            "No independent test metrics are available for this artifact.",
            "Accepts only 1024-value YAMNet embeddings and excludes preprocessing/YAMNet.",
            "No INT8, FP16, or alternate provider conversion is included.",
        ],
        "status": {"outcome": "success", "error": None},
    }
    document["manifest_sha256"] = document_sha256(
        document, excluded_fields=("created_at_utc", "manifest_sha256")
    )
    write_result(manifest_destination, document, pretty=pretty)
    return document


def _synthetic_cases() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(42)
    return [
        ("synthetic_zero", np.zeros((1, 1024), dtype=np.float32)),
        ("synthetic_small_positive", np.full((7, 1024), 1e-3, dtype=np.float32)),
        ("synthetic_small_negative", np.full((32, 1024), -1e-3, dtype=np.float32)),
        ("synthetic_seeded_random", rng.normal(0.0, 0.1, size=(128, 1024)).astype(np.float32)),
    ]


def validate_deployment_onnx(
    *, source_run: Path, cache_root: Path, onnx_artifact_directory: Path,
    sample_count: int, output_directory: Path, pretty: bool,
    tf_module: Optional[Any] = None, onnx_module: Optional[Any] = None,
    ort_module: Optional[Any] = None,
) -> dict[str, Any]:
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("deployment parity output directory already exists")
    source = load_deployment_source(source_run)
    artifact_root = Path(onnx_artifact_directory)
    manifest = json.loads((artifact_root / "artifact-manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != EXPORT_SCHEMA_VERSION
        or manifest.get("deployment_only") is not True
        or manifest.get("independent_test_metrics_available") is not False
        or manifest.get("status", {}).get("outcome") != "success"
    ):
        raise ValueError("deployment ONNX manifest is incompatible")
    if manifest.get("manifest_sha256") != document_sha256(
        manifest, excluded_fields=("created_at_utc", "manifest_sha256")
    ):
        raise ValueError("deployment ONNX manifest SHA-256 mismatch")
    if manifest.get("source_model_sha256") != source.model_sha256:
        raise ValueError("deployment ONNX source model mismatch")
    onnx_path = artifact_root / str(manifest["onnx_artifact"]["relative_path"])
    if streaming_file_sha256(onnx_path) != manifest["onnx_artifact"]["sha256"]:
        raise ValueError("deployment ONNX artifact SHA-256 mismatch")
    if tf_module is None:
        import tensorflow as tf_module
    if onnx_module is None:
        import onnx as onnx_module
    if ort_module is None:
        import onnxruntime as ort_module
    graph = inspect_onnx_model(onnx_module.load(str(onnx_path)), checker=onnx_module.checker)
    keras_model = build_linear_keras_model(tf_module)
    keras_model.load_weights(str(source.weights_path))
    session = ort_module.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    providers = list(session.get_providers())
    if providers != ["CPUExecutionProvider"]:
        raise ValueError("deployment parity must use only CPUExecutionProvider")

    def keras_predict(values: np.ndarray) -> np.ndarray:
        return np.asarray(keras_model(np.asarray(values, dtype=np.float32), training=False), dtype=np.float32)

    def onnx_predict(values: np.ndarray) -> np.ndarray:
        return np.asarray(session.run(["probabilities"], {"embedding": np.asarray(values, dtype=np.float32)})[0], dtype=np.float32)

    verified = load_verified_cache_records(cache_root)
    fixture, fixture_features = select_deterministic_fixture(verified, sample_count)
    cases = _synthetic_cases() + [(fixture["fixture_identity"], fixture_features)]
    batch_results = []
    all_keras = []
    all_onnx = []
    all_identities = []
    for name, values in cases:
        identities = (
            fixture["selected_records"] if name == fixture["fixture_identity"]
            else [{"fixture_id": name, "sample_index": index} for index in range(values.shape[0])]
        )
        keras_output = keras_predict(values)
        onnx_output = onnx_predict(values)
        comparison = compare_probability_outputs(keras_output, onnx_output, identities)
        comparison.update({"fixture_id": name, "batch_size": int(values.shape[0])})
        batch_results.append(comparison)
        all_keras.append(keras_output)
        all_onnx.append(onnx_output)
        all_identities.extend(dict(item) for item in identities)
    numeric = compare_probability_outputs(
        np.concatenate(all_keras), np.concatenate(all_onnx), all_identities
    )
    success = parity_passes(numeric)
    document: dict[str, Any] = {
        "schema_version": DEPLOYMENT_PARITY_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "deployment_id": DEPLOYMENT_ID,
        "deployment_only": True,
        "independent_test_metrics_available": False,
        "source_model_sha256": source.model_sha256,
        "onnx_sha256": manifest["onnx_artifact"]["sha256"],
        "fixture_identity": fixture["fixture_identity"],
        "sample_count": sample_count,
        "synthetic_sample_count": sum(values.shape[0] for _, values in _synthetic_cases()),
        "total_numeric_sample_count": numeric["sample_count"],
        "batch_sizes": list(REQUIRED_BATCH_SIZES),
        "batch_results": batch_results,
        "numeric_error": numeric,
        "top1_agreement": numeric["top1_agreement"],
        "thresholds": {
            "maximum_absolute_error": MAX_ABSOLUTE_ERROR,
            "mean_absolute_error": MAX_MEAN_ABSOLUTE_ERROR,
            "top1_agreement": MIN_TOP1_AGREEMENT,
            "probability_sum_tolerance": PROBABILITY_SUM_TOLERANCE,
        },
        "onnx_graph": graph,
        "runtime": {
            "onnxruntime_version": importlib.metadata.version("onnxruntime"),
            "configured_providers": ["CPUExecutionProvider"],
            "active_session_providers": providers,
        },
        "scope": {
            "numeric_and_prediction_parity_only": True,
            "fold_metric_parity_performed": False,
            "accuracy_or_macro_f1_generated": False,
        },
        "status": {"outcome": "success" if success else "failure", "error": None if success else {"type": "ParityThresholdFailure"}},
    }
    document["parity_sha256"] = document_sha256(
        document, excluded_fields=("created_at_utc", "parity_sha256")
    )
    output.mkdir(parents=True, exist_ok=False)
    write_result(output / "fixture-manifest.json", fixture, pretty=pretty)
    write_result(output / "parity-result.json", document, pretty=pretty)
    return document


__all__ = [
    "DEPLOYMENT_PARITY_SCHEMA_VERSION", "DeploymentSource", "export_deployment_linear_onnx",
    "load_deployment_source", "validate_deployment_onnx",
]

