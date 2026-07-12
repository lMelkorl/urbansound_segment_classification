"""Provenance-bound FP32 ONNX export for the representative Linear Fold 1 model."""

from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from urbansound_segment_task.edge_v2.data.manifest import document_sha256
from urbansound_segment_task.edge_v2.evaluation.result_schema import utc_timestamp, write_result
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256


EXPORT_SCHEMA_VERSION = "edge-v2.onnx-export.v1"
EXPECTED_RUN_SCHEMA = "edge-v2.compact-classifier-run.v1"
EXPECTED_AGGREGATE_SCHEMA = "edge-v2.compact-classifier-cross-fold.v1"
EXPECTED_FOLD_SCHEMA = "edge-v2.compact-classifier-fold.v1"
EXPECTED_CACHE_IDENTITY = "6b0807688796f3f19ca7129867a518f893d18566ffd057cb31a5302bd6fd17ce"
EXPECTED_ARCHITECTURE = "linear"
EXPECTED_FOLD = 1
EXPECTED_VALIDATION_FOLD = 2
EXPECTED_PARAMETER_COUNT = 10_250
EXPECTED_OPSET = 15
ALLOWED_ONNX_DOMAINS = ("", "ai.onnx")


@dataclass(frozen=True)
class SourceLinearModel:
    weights_path: Path
    weights_relative_path: str
    source_model_sha256: str
    source_model_size_bytes: int
    source_model_identity: str
    run_identity_sha256: str
    fold_identity_sha256: str
    config_sha256: str
    cache_identity: str
    split_manifest_sha256: str
    architecture_id: str
    parameter_count: int
    test_fold: int
    validation_fold: int

    def safe_identity(self) -> dict[str, Any]:
        return {
            "source_model_identity": self.source_model_identity,
            "run_identity_sha256": self.run_identity_sha256,
            "fold_identity_sha256": self.fold_identity_sha256,
            "config_sha256": self.config_sha256,
            "cache_identity": self.cache_identity,
            "split_manifest_sha256": self.split_manifest_sha256,
            "architecture_id": self.architecture_id,
            "parameter_count": self.parameter_count,
            "test_fold": self.test_fold,
            "validation_fold": self.validation_fold,
            "weights_relative_path": self.weights_relative_path,
        }


def _safe_relative_path(value: Any) -> str:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("source artifact relative path is unsafe")
    return path.as_posix()


def resolve_source_model(run_manifest_path: Path, *, model: str, fold: int) -> SourceLinearModel:
    if model != EXPECTED_ARCHITECTURE or fold != EXPECTED_FOLD:
        raise ValueError("only deterministic Linear Fold 1 export is supported")
    run_path = Path(run_manifest_path)
    root = run_path.parent
    run = json.loads(run_path.read_text(encoding="utf-8"))
    aggregate_path = root / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if run.get("schema_version") != EXPECTED_RUN_SCHEMA:
        raise ValueError("compact run manifest schema mismatch")
    if aggregate.get("schema_version") != EXPECTED_AGGREGATE_SCHEMA or aggregate.get("status", {}).get("outcome") != "success":
        raise ValueError("compact aggregate is not a successful compatible result")
    if aggregate.get("run_identity_sha256") != run.get("run_identity_sha256"):
        raise ValueError("compact aggregate run identity mismatch")
    aggregate_hash = aggregate.get("aggregate_sha256")
    if aggregate_hash != document_sha256(
        aggregate, excluded_fields=("created_at_utc", "aggregate_sha256")
    ):
        raise ValueError("compact aggregate SHA-256 mismatch")
    rows = [
        row for row in aggregate["models"][model]["fold_results"]
        if int(row["test_fold"]) == fold
    ]
    if len(rows) != 1 or int(rows[0]["validation_fold"]) != EXPECTED_VALIDATION_FOLD:
        raise ValueError("deterministic Fold 1 aggregate selection mismatch")
    fold_directory = root / model / f"fold-{fold:02d}"
    fold_result_path = fold_directory / "fold-result.json"
    result = json.loads(fold_result_path.read_text(encoding="utf-8"))
    sidecar = (fold_directory / "fold-result.sha256").read_text(encoding="ascii").strip()
    if streaming_file_sha256(fold_result_path) != sidecar:
        raise ValueError("source fold result SHA-256 mismatch")
    if result.get("schema_version") != EXPECTED_FOLD_SCHEMA or result.get("status", {}).get("outcome") != "success":
        raise ValueError("source fold result is not successful or compatible")
    if (
        result.get("architecture_id") != model
        or int(result.get("test_fold", 0)) != fold
        or int(result.get("validation_fold", 0)) != EXPECTED_VALIDATION_FOLD
        or int(result.get("architecture", {}).get("parameter_count", 0)) != EXPECTED_PARAMETER_COUNT
    ):
        raise ValueError("source architecture fold or parameter count mismatch")
    if result.get("cache_identity") != EXPECTED_CACHE_IDENTITY or result.get("cache_identity") != run.get("cache_identity"):
        raise ValueError("source cache identity mismatch")
    if result.get("config_sha256") != run["model_config_sha256"][model]:
        raise ValueError("source config SHA-256 mismatch")
    if result.get("split_manifest_sha256") != run["split_manifest_sha256"][str(fold)]:
        raise ValueError("source split manifest identity mismatch")
    artifact = result["artifacts"]["model"]
    relative_path = _safe_relative_path(artifact["relative_path"])
    weights_path = fold_directory / relative_path
    if weights_path.stat().st_size != int(artifact["size_bytes"]):
        raise ValueError("source model size mismatch")
    actual_sha256 = streaming_file_sha256(weights_path)
    if actual_sha256 != artifact["sha256"]:
        raise ValueError("source model SHA-256 mismatch")
    if (
        artifact.get("architecture_id") != model
        or artifact.get("config_sha256") != result["config_sha256"]
        or artifact.get("fold_identity_sha256") != result["fold_identity_sha256"]
        or int(artifact.get("parameter_count", 0)) != EXPECTED_PARAMETER_COUNT
    ):
        raise ValueError("source model artifact identity mismatch")
    identity_document = {
        "run_identity_sha256": run["run_identity_sha256"],
        "fold_identity_sha256": result["fold_identity_sha256"],
        "source_model_sha256": actual_sha256,
        "config_sha256": result["config_sha256"],
        "cache_identity": result["cache_identity"],
        "split_manifest_sha256": result["split_manifest_sha256"],
        "architecture_id": model,
        "test_fold": fold,
        "validation_fold": EXPECTED_VALIDATION_FOLD,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
    }
    return SourceLinearModel(
        weights_path=weights_path,
        weights_relative_path=relative_path,
        source_model_sha256=actual_sha256,
        source_model_size_bytes=int(artifact["size_bytes"]),
        source_model_identity=document_sha256(identity_document),
        run_identity_sha256=run["run_identity_sha256"],
        fold_identity_sha256=result["fold_identity_sha256"],
        config_sha256=result["config_sha256"],
        cache_identity=result["cache_identity"],
        split_manifest_sha256=result["split_manifest_sha256"],
        architecture_id=model,
        parameter_count=EXPECTED_PARAMETER_COUNT,
        test_fold=fold,
        validation_fold=EXPECTED_VALIDATION_FOLD,
    )


def build_linear_keras_model(tf: Any) -> Any:
    embedding = tf.keras.layers.Input(shape=(1024,), dtype=tf.float32, name="embedding")
    probabilities = tf.keras.layers.Dense(
        10, activation="softmax", name="probabilities"
    )(embedding)
    model = tf.keras.Model(inputs=embedding, outputs=probabilities, name="linear_fold1_export")
    if int(model.count_params()) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("export model parameter count mismatch")
    return model


def _shape_contract(value_info: Any) -> list[Any]:
    dimensions = value_info.type.tensor_type.shape.dim
    result: list[Any] = []
    for dimension in dimensions:
        if dimension.dim_value:
            result.append(int(dimension.dim_value))
        elif dimension.dim_param:
            result.append(str(dimension.dim_param))
        else:
            result.append(None)
    return result


def inspect_onnx_model(model_proto: Any, *, checker: Optional[Any] = None) -> dict[str, Any]:
    if checker is not None:
        checker.check_model(model_proto)
    custom_domains = sorted(
        {str(node.domain) for node in model_proto.graph.node if str(node.domain) not in ALLOWED_ONNX_DOMAINS}
    )
    if custom_domains:
        raise ValueError("unexpected ONNX custom domain or operation")
    opsets = {str(item.domain): int(item.version) for item in model_proto.opset_import}
    default_opset = opsets.get("", opsets.get("ai.onnx"))
    if default_opset != EXPECTED_OPSET:
        raise ValueError("ONNX opset mismatch")
    if len(model_proto.graph.input) != 1 or len(model_proto.graph.output) != 1:
        raise ValueError("ONNX graph input or output count mismatch")
    graph_input = model_proto.graph.input[0]
    graph_output = model_proto.graph.output[0]
    input_shape = _shape_contract(graph_input)
    output_shape = _shape_contract(graph_output)
    if graph_input.name != "embedding" or input_shape[1:] != [1024] or len(input_shape) != 2:
        raise ValueError("ONNX input contract mismatch")
    if graph_output.name != "probabilities" or output_shape[1:] != [10] or len(output_shape) != 2:
        raise ValueError("ONNX output contract mismatch")
    if not input_shape[0] or isinstance(input_shape[0], int):
        raise ValueError("ONNX input batch dimension is not dynamic")
    if not output_shape[0] or isinstance(output_shape[0], int):
        raise ValueError("ONNX output batch dimension is not dynamic")
    node_types = [str(node.op_type) for node in model_proto.graph.node]
    if "Softmax" not in node_types:
        raise ValueError("ONNX graph does not contain softmax probabilities")
    return {
        "ir_version": int(model_proto.ir_version),
        "opset_imports": opsets,
        "opset": int(default_opset),
        "graph": {
            "input": {"name": graph_input.name, "dtype": "float32", "shape": input_shape},
            "output": {"name": graph_output.name, "dtype": "float32", "shape": output_shape},
            "node_count": len(model_proto.graph.node),
            "initializer_count": len(model_proto.graph.initializer),
            "node_types": node_types,
            "custom_domains": custom_domains,
        },
    }


def _publish_binary_new(path: Path, payload: bytes) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("ONNX output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=".onnx-", suffix=".tmp", delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        if os.name == "nt":
            os.rename(temporary_name, destination)
        else:
            os.link(temporary_name, destination)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def export_linear_onnx(
    *, run_manifest_path: Path, model: str, fold: int, opset: int,
    output_path: Path, manifest_path: Path, pretty: bool,
    tf_module: Optional[Any] = None, tf2onnx_module: Optional[Any] = None,
    onnx_module: Optional[Any] = None,
) -> dict[str, Any]:
    if opset != EXPECTED_OPSET:
        raise ValueError("only fixed ONNX opset 15 is supported")
    output = Path(output_path)
    manifest = Path(manifest_path)
    if output.exists() or manifest.exists():
        raise FileExistsError("ONNX output or manifest already exists")
    source = resolve_source_model(run_manifest_path, model=model, fold=fold)
    if tf_module is None:
        import tensorflow as tf_module
    if tf2onnx_module is None:
        import tf2onnx as tf2onnx_module
    if onnx_module is None:
        import onnx as onnx_module
    keras_model = build_linear_keras_model(tf_module)
    keras_model.load_weights(str(source.weights_path))
    smoke_input = np.zeros((1, 1024), dtype=np.float32)
    smoke_output = np.asarray(keras_model(smoke_input, training=False), dtype=np.float32)
    if smoke_output.shape != (1, 10) or not np.isfinite(smoke_output).all():
        raise ValueError("Keras source smoke inference failed")
    signature = (
        tf_module.TensorSpec([None, 1024], tf_module.float32, name="embedding"),
    )
    model_proto, _ = tf2onnx_module.convert.from_keras(
        keras_model, input_signature=signature, opset=opset, output_path=None
    )
    graph = inspect_onnx_model(model_proto, checker=onnx_module.checker)
    payload = model_proto.SerializeToString()
    _publish_binary_new(output, payload)
    onnx_sha256 = streaming_file_sha256(output)
    if onnx_sha256 != __import__("hashlib").sha256(payload).hexdigest():
        raise ValueError("published ONNX SHA-256 mismatch")
    output_relative = _safe_relative_path(output.name)
    document: dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "source_model_identity": source.source_model_identity,
        "source_model_sha256": source.source_model_sha256,
        "source_model": source.safe_identity(),
        "architecture_id": source.architecture_id,
        "parameter_count": source.parameter_count,
        "fold_identity": {
            "test_fold": source.test_fold,
            "validation_fold": source.validation_fold,
            "fold_identity_sha256": source.fold_identity_sha256,
            "selection_reason": "deterministic_first_fold_not_best_fold_selection",
        },
        "input_contract": {"name": "embedding", "dtype": "float32", "shape": ["batch", 1024]},
        "output_contract": {"name": "probabilities", "dtype": "float32", "shape": ["batch", 10], "semantic": "softmax_probabilities"},
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
            "relative_path": output_relative,
            "size_bytes": output.stat().st_size,
            "sha256": onnx_sha256,
            "precision": "fp32",
        },
        "runtime": {
            "python_version": __import__("platform").python_version(),
            "target": "onnxruntime_cpu",
        },
        "limitations": [
            "Accepts only precomputed 1024-value YAMNet embeddings.",
            "Does not include YAMNet, raw audio decoding, segmentation, preprocessing, or clip aggregation.",
            "Representative export smoke uses deterministic Fold 1 and is not a full deployment model.",
            "No FP16 or INT8 conversion is included.",
        ],
        "status": {"outcome": "success", "error": None},
    }
    document["manifest_sha256"] = document_sha256(
        document, excluded_fields=("created_at_utc", "manifest_sha256")
    )
    write_result(manifest, document, pretty=pretty)
    return document


__all__ = [
    "ALLOWED_ONNX_DOMAINS", "EXPECTED_ARCHITECTURE", "EXPECTED_FOLD", "EXPECTED_OPSET",
    "EXPECTED_PARAMETER_COUNT", "EXPORT_SCHEMA_VERSION", "SourceLinearModel",
    "build_linear_keras_model", "export_linear_onnx", "inspect_onnx_model",
    "resolve_source_model",
]
