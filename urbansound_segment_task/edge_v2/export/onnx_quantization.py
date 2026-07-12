"""Fixed dynamic-QInt8 quantization for the verified Linear Fold 1 FP32 ONNX graph."""

from __future__ import annotations

import collections
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

from urbansound_segment_task.edge_v2.data.manifest import document_sha256
from urbansound_segment_task.edge_v2.evaluation.result_schema import utc_timestamp, write_result
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256
from .onnx_linear import EXPORT_SCHEMA_VERSION, inspect_onnx_model


QUANTIZED_SCHEMA_VERSION = "edge-v2.onnx-quantized-artifact.v1"
EXPECTED_FP32_SHA256 = "641f41921612048c8ed51d7c7c9d8f8f534827d08a091f87dd5900929731a329"
EXPECTED_SOURCE_MODEL_SHA256 = "7538f47869e4ad5901074a1602e1217d3a9192f98fbd6d3d9f9ebac9bc708301"
QUANTIZATION_CONFIG = {
    "api": "onnxruntime.quantization.quantize_dynamic",
    "weight_type": "QInt8",
    "per_channel": False,
    "reduce_range": False,
}


def validate_fp32_artifact(artifact_directory: Path) -> tuple[dict[str, Any], Path]:
    root = Path(artifact_directory)
    manifest_path = root / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXPORT_SCHEMA_VERSION or manifest.get("status", {}).get("outcome") != "success":
        raise ValueError("FP32 ONNX manifest is incompatible or unsuccessful")
    if manifest.get("manifest_sha256") != document_sha256(
        manifest, excluded_fields=("created_at_utc", "manifest_sha256")
    ):
        raise ValueError("FP32 ONNX manifest SHA-256 mismatch")
    if manifest.get("source_model_sha256") != EXPECTED_SOURCE_MODEL_SHA256:
        raise ValueError("FP32 source model SHA-256 mismatch")
    relative = Path(str(manifest["onnx_artifact"]["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("FP32 ONNX relative path is unsafe")
    model_path = root / relative
    if model_path.stat().st_size != int(manifest["onnx_artifact"]["size_bytes"]):
        raise ValueError("FP32 ONNX artifact size mismatch")
    digest = streaming_file_sha256(model_path)
    if digest != EXPECTED_FP32_SHA256 or digest != manifest["onnx_artifact"]["sha256"]:
        raise ValueError("FP32 ONNX artifact SHA-256 mismatch")
    return manifest, model_path


def summarize_quantized_graph(model_proto: Any, onnx_module: Any) -> dict[str, Any]:
    contract = inspect_onnx_model(model_proto, checker=onnx_module.checker)
    node_types = [str(node.op_type) for node in model_proto.graph.node]
    node_type_counts = dict(sorted(collections.Counter(node_types).items()))
    initializer_dtypes: list[dict[str, Any]] = []
    dtype_counts: collections.Counter[str] = collections.Counter()
    for initializer in model_proto.graph.initializer:
        dtype_name = str(onnx_module.TensorProto.DataType.Name(int(initializer.data_type)))
        dtype_counts[dtype_name] += 1
        initializer_dtypes.append(
            {"name": str(initializer.name), "dtype": dtype_name, "shape": [int(value) for value in initializer.dims]}
        )
    quantized_node_types = sorted(
        {
            node_type for node_type in node_types
            if any(token in node_type for token in ("Quantize", "Integer", "QLinear", "Dequantize"))
        }
    )
    quantized_weight_initializers = [
        row for row in initializer_dtypes if row["dtype"] in ("INT8", "UINT8")
    ]
    if not quantized_weight_initializers:
        raise ValueError("dynamic quantization produced no INT8/UINT8 weight initializer")
    if "Softmax" not in node_types:
        raise ValueError("quantized graph lost float softmax output")
    return {
        "ir_version": contract["ir_version"],
        "opset": contract["opset"],
        "opset_imports": contract["opset_imports"],
        "input": contract["graph"]["input"],
        "output": contract["graph"]["output"],
        "node_count": len(model_proto.graph.node),
        "initializer_count": len(model_proto.graph.initializer),
        "node_type_counts": node_type_counts,
        "quantized_node_types": quantized_node_types,
        "initializer_dtype_counts": dict(sorted(dtype_counts.items())),
        "quantized_weight_initializers": quantized_weight_initializers,
        "softmax": {"present": True, "output_dtype": "float32"},
        "custom_domains": contract["graph"]["custom_domains"],
    }


def _publish_existing_new(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("INT8 ONNX output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open("rb") as handle:
        os.fsync(handle.fileno())
    if os.name == "nt":
        os.rename(source, destination)
    else:
        os.link(source, destination)


def quantize_linear_dynamic_int8(
    *, fp32_artifact_directory: Path, output_path: Path, manifest_path: Path,
    pretty: bool, onnx_module: Optional[Any] = None,
    quantize_dynamic_function: Optional[Any] = None, qint8_value: Optional[Any] = None,
) -> dict[str, Any]:
    output = Path(output_path)
    manifest_destination = Path(manifest_path)
    if output.exists() or manifest_destination.exists():
        raise FileExistsError("INT8 ONNX output or manifest already exists")
    fp32_manifest, fp32_path = validate_fp32_artifact(fp32_artifact_directory)
    if onnx_module is None:
        import onnx as onnx_module
    if quantize_dynamic_function is None or qint8_value is None:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic_function = quantize_dynamic
        qint8_value = QuantType.QInt8
    fp32_proto = onnx_module.load(str(fp32_path))
    fp32_graph = inspect_onnx_model(fp32_proto, checker=onnx_module.checker)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=".int8-", suffix=".onnx", delete=False
        ) as temporary:
            temporary_name = temporary.name
        quantize_dynamic_function(
            model_input=str(fp32_path),
            model_output=temporary_name,
            weight_type=qint8_value,
            per_channel=False,
            reduce_range=False,
        )
        temporary_path = Path(temporary_name)
        int8_proto = onnx_module.load(str(temporary_path))
        int8_graph = summarize_quantized_graph(int8_proto, onnx_module)
        _publish_existing_new(temporary_path, output)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    int8_sha256 = streaming_file_sha256(output)
    fp32_size = fp32_path.stat().st_size
    int8_size = output.stat().st_size
    document: dict[str, Any] = {
        "schema_version": QUANTIZED_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "source_fp32_identity": fp32_manifest["manifest_sha256"],
        "source_fp32_sha256": fp32_manifest["onnx_artifact"]["sha256"],
        "source_model_sha256": fp32_manifest["source_model_sha256"],
        "quantization_method": "dynamic_weight_only_int8",
        "quantization_config": dict(QUANTIZATION_CONFIG),
        "input_contract": dict(fp32_manifest["input_contract"]),
        "output_contract": dict(fp32_manifest["output_contract"]),
        "graph_summary": {
            "fp32": {
                "ir_version": fp32_graph["ir_version"],
                "opset": fp32_graph["opset"],
                "node_count": fp32_graph["graph"]["node_count"],
                "initializer_count": fp32_graph["graph"]["initializer_count"],
                "node_types": fp32_graph["graph"]["node_types"],
            },
            "int8": int8_graph,
            "changed_node_types": {
                "removed": sorted(set(fp32_graph["graph"]["node_types"]) - set(int8_graph["node_type_counts"])),
                "added": sorted(set(int8_graph["node_type_counts"]) - set(fp32_graph["graph"]["node_types"])),
            },
        },
        "size_comparison": {
            "fp32_size_bytes": fp32_size,
            "int8_size_bytes": int8_size,
            "int8_minus_fp32_bytes": int8_size - fp32_size,
            "size_reduction_percent": (fp32_size - int8_size) / fp32_size * 100.0,
        },
        "int8_artifact": {
            "relative_path": output.name,
            "size_bytes": int8_size,
            "sha256": int8_sha256,
            "precision": "dynamic_int8_weights_float32_io",
        },
        "tool_versions": {
            "onnx": importlib.metadata.version("onnx"),
            "onnxruntime": importlib.metadata.version("onnxruntime"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "limitations": [
            "Dynamic quantization targets eligible Linear MatMul weights only.",
            "Input and output remain float32; Softmax remains float.",
            "No static calibration, QAT, FP16, INT4, or YAMNet quantization is included.",
            "A larger or slower INT8 artifact is a valid outcome for this very small graph.",
        ],
        "status": {"outcome": "success", "error": None},
    }
    document["manifest_sha256"] = document_sha256(
        document, excluded_fields=("created_at_utc", "manifest_sha256")
    )
    write_result(manifest_destination, document, pretty=pretty)
    return document


__all__ = [
    "EXPECTED_FP32_SHA256", "EXPECTED_SOURCE_MODEL_SHA256", "QUANTIZATION_CONFIG",
    "QUANTIZED_SCHEMA_VERSION", "quantize_linear_dynamic_int8", "summarize_quantized_graph",
    "validate_fp32_artifact",
]
