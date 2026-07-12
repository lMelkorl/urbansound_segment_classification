"""Alternating fresh-process FP32 versus dynamic-INT8 ONNX Runtime benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from urbansound_segment_task.edge_v2.benchmarks.onnx_classifier import (
    BenchmarkConfig, run_fresh_process, select_benchmark_input, summarize_runtime,
)
from urbansound_segment_task.edge_v2.data.manifest import document_sha256
from urbansound_segment_task.edge_v2.evaluation.result_schema import safe_environment, utc_timestamp, write_result
from urbansound_segment_task.edge_v2.export.onnx_int8_validation import INT8_PARITY_SCHEMA_VERSION
from urbansound_segment_task.edge_v2.export.onnx_quantization import validate_fp32_artifact
from urbansound_segment_task.edge_v2.export.onnx_int8_validation import validate_int8_artifact
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256


INT8_BENCHMARK_SCHEMA_VERSION = "edge-v2.onnx-int8-benchmark.v1"
VARIANTS = ("fp32", "int8")


def alternating_runtime_order(repetition: int) -> tuple[str, str]:
    if repetition < 1:
        raise ValueError("repetition must be positive")
    return ("fp32", "int8") if repetition % 2 == 1 else ("int8", "fp32")


def deployment_decision(
    *, parity_passed: bool, clip_macro_f1_delta: float,
    fp32_size_bytes: int, int8_size_bytes: int,
    fp32_p50_ns: float, int8_p50_ns: float,
    fp32_incremental_rss_bytes: float, int8_incremental_rss_bytes: float,
) -> tuple[str, list[dict[str, Any]]]:
    f1_pass = clip_macro_f1_delta <= 0.002
    size_pass = int8_size_bytes < fp32_size_bytes
    latency_improvement = (
        (fp32_p50_ns - int8_p50_ns) / fp32_p50_ns * 100.0 if fp32_p50_ns else float("-inf")
    )
    memory_improvement = (
        (fp32_incremental_rss_bytes - int8_incremental_rss_bytes)
        / fp32_incremental_rss_bytes * 100.0
        if fp32_incremental_rss_bytes
        else float("-inf")
    )
    latency_pass = latency_improvement >= 5.0
    memory_pass = memory_improvement >= 10.0
    reasons = [
        {"criterion": "parity_thresholds_pass", "passed": parity_passed},
        {"criterion": "clip_macro_f1_delta_lte_0.002", "passed": f1_pass, "value": clip_macro_f1_delta},
        {"criterion": "int8_artifact_smaller_than_fp32", "passed": size_pass, "fp32_bytes": fp32_size_bytes, "int8_bytes": int8_size_bytes},
        {"criterion": "p50_latency_at_least_5_percent_lower", "passed": latency_pass, "improvement_percent": latency_improvement},
        {"criterion": "incremental_rss_at_least_10_percent_lower", "passed": memory_pass, "improvement_percent": memory_improvement},
        {"criterion": "latency_or_memory_improvement", "passed": latency_pass or memory_pass},
    ]
    selected = "dynamic_int8_onnx" if parity_passed and f1_pass and size_pass and (latency_pass or memory_pass) else "fp32_onnx"
    return selected, reasons


def _validate_parity_result(
    parity_path: Path, fp32_sha256: str, int8_sha256: str
) -> dict[str, Any]:
    document = json.loads(Path(parity_path).read_text(encoding="utf-8"))
    if document.get("schema_version") != INT8_PARITY_SCHEMA_VERSION:
        raise ValueError("INT8 parity result schema mismatch")
    if document.get("parity_sha256") != document_sha256(
        document, excluded_fields=("created_at_utc", "parity_sha256")
    ):
        raise ValueError("INT8 parity result SHA-256 mismatch")
    if document.get("fp32_onnx_sha256") != fp32_sha256 or document.get("int8_onnx_sha256") != int8_sha256:
        raise ValueError("INT8 parity artifact identity mismatch")
    return document


def build_int8_benchmark_summary(
    *, raw_rows: Sequence[Mapping[str, Any]], raw_references: Sequence[Mapping[str, Any]],
    fp32_artifact: Mapping[str, Any], int8_artifact: Mapping[str, Any],
    input_identity: Mapping[str, Any], parity_result: Mapping[str, Any],
    warmup: int, iterations: int, repetitions: int, threads: int,
) -> dict[str, Any]:
    failures = [row for row in raw_rows if row.get("status", {}).get("outcome") != "success"]
    if failures:
        return {
            "schema_version": INT8_BENCHMARK_SCHEMA_VERSION,
            "created_at_utc": utc_timestamp(),
            "fp32_artifact": dict(fp32_artifact), "int8_artifact": dict(int8_artifact),
            "input_identity": dict(input_identity), "raw_run_references": list(raw_references),
            "status": {"outcome": "failure", "error": {"type": "RawRepetitionFailure", "count": len(failures)}},
        }
    grouped = {variant: [row for row in raw_rows if row["runtime"] == variant] for variant in VARIANTS}
    if any(len(grouped[variant]) != repetitions for variant in VARIANTS):
        raise ValueError("FP32 or INT8 raw repetition count mismatch")
    input_hashes = {row["input_identity"]["input_identity_sha256"] for row in raw_rows}
    if input_hashes != {input_identity["input_identity_sha256"]}:
        raise ValueError("FP32 and INT8 benchmark inputs differ")
    fp32 = summarize_runtime(grouped["fp32"])
    int8 = summarize_runtime(grouped["int8"])
    fp32_classes = {int(row["output_validation"]["top1_class"]) for row in grouped["fp32"]}
    int8_classes = {int(row["output_validation"]["top1_class"]) for row in grouped["int8"]}
    top1_agreement = len(fp32_classes) == 1 and fp32_classes == int8_classes
    fp32_incremental = float(fp32["memory"]["median_approximate_incremental_peak_rss_bytes"])
    int8_incremental = float(int8["memory"]["median_approximate_incremental_peak_rss_bytes"])
    comparison = {
        "int8_latency_speedup_fp32_p50_over_int8_p50": fp32["median_steady_p50_ns"] / int8["median_steady_p50_ns"],
        "int8_throughput_ratio": int8["median_predictions_per_second"] / fp32["median_predictions_per_second"],
        "session_load_ratio_fp32_over_int8": fp32["median_model_load_ns"] / int8["median_model_load_ns"],
        "memory_reduction_percent": (
            None if fp32_incremental == 0 else (fp32_incremental - int8_incremental) / fp32_incremental * 100.0
        ),
        "artifact_size_reduction_percent": (
            (fp32["artifact_size_bytes"] - int8["artifact_size_bytes"])
            / fp32["artifact_size_bytes"] * 100.0
        ),
        "top1_prediction_agreement": top1_agreement,
    }
    parity_passed = parity_result.get("status", {}).get("outcome") == "success"
    clip_f1_delta = float(parity_result["fold1_metric_parity"]["clip_macro_f1_delta"])
    selected, reasons = deployment_decision(
        parity_passed=parity_passed,
        clip_macro_f1_delta=clip_f1_delta,
        fp32_size_bytes=int(fp32["artifact_size_bytes"]),
        int8_size_bytes=int(int8["artifact_size_bytes"]),
        fp32_p50_ns=float(fp32["median_steady_p50_ns"]),
        int8_p50_ns=float(int8["median_steady_p50_ns"]),
        fp32_incremental_rss_bytes=fp32_incremental,
        int8_incremental_rss_bytes=int8_incremental,
    )
    success = top1_agreement and parity_passed
    document: dict[str, Any] = {
        "schema_version": INT8_BENCHMARK_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "fp32_artifact": dict(fp32_artifact),
        "int8_artifact": dict(int8_artifact),
        "input_identity": dict(input_identity),
        "benchmark_config": {
            "warmup": warmup, "iterations": iterations, "repetitions": repetitions,
            "threads": threads, "batch_size": 1,
            "runtime_order_by_repetition": {
                str(index): list(alternating_runtime_order(index)) for index in range(1, repetitions + 1)
            },
            "provider": "CPUExecutionProvider",
            "execution_mode": "ORT_SEQUENTIAL",
            "graph_optimization": "ORT_ENABLE_ALL",
        },
        "fp32_runtime": fp32,
        "int8_runtime": int8,
        "comparison": comparison,
        "parity_result": {
            "relative_path": "../parity/parity-result.json",
            "sha256": parity_result["parity_sha256"],
            "status": parity_result["status"],
            "clip_macro_f1_delta": clip_f1_delta,
        },
        "deployment_decision": selected,
        "decision_reasons": reasons,
        "raw_run_references": list(raw_references),
        "environment": safe_environment(threads),
        "limitations": [
            "Classifier-only benchmark excludes YAMNet, audio decoding, segmentation, and clip aggregation.",
            "Dynamic quantization uses QInt8 weights with float32 input, output, and Softmax.",
            "macOS peak RSS is a process-lifetime high-water mark and incremental values are approximate.",
            "This very small graph may gain overhead from dynamic quantization; no speedup is assumed.",
            "Results apply to one cached embedding, one thread, this machine, and the locked runtime.",
        ],
        "status": {"outcome": "success" if success else "failure", "error": None if success else {"type": "BenchmarkOrParityFailure"}},
    }
    document["summary_sha256"] = document_sha256(
        document, excluded_fields=("created_at_utc", "summary_sha256")
    )
    return document


def run_int8_benchmark_suite(
    *, cache_root: Path, fp32_artifact_directory: Path, int8_artifact_directory: Path,
    warmup: int, iterations: int, repetitions: int, threads: int,
    output_directory: Path, pretty: bool, parity_result_path: Optional[Path] = None,
) -> dict[str, Any]:
    if warmup < 0 or iterations < 1 or repetitions < 1 or threads != 1:
        raise ValueError("INT8 benchmark config violates the fixed contract")
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("INT8 benchmark output directory already exists")
    fp32_manifest, fp32_path = validate_fp32_artifact(fp32_artifact_directory)
    int8_manifest, int8_path = validate_int8_artifact(fp32_artifact_directory, int8_artifact_directory)
    parity_path = Path(parity_result_path) if parity_result_path is not None else output.parent / "parity" / "parity-result.json"
    parity = _validate_parity_result(
        parity_path, fp32_manifest["onnx_artifact"]["sha256"], int8_manifest["int8_artifact"]["sha256"]
    )
    input_identity, embedding = select_benchmark_input(cache_root)
    output.mkdir(parents=True, exist_ok=False)
    raw_directory = output / "raw"
    raw_directory.mkdir()
    variants = {
        "fp32": (fp32_path, int(fp32_manifest["onnx_artifact"]["size_bytes"]), fp32_manifest["onnx_artifact"]["sha256"]),
        "int8": (int8_path, int(int8_manifest["int8_artifact"]["size_bytes"]), int8_manifest["int8_artifact"]["sha256"]),
    }
    raw_rows: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        for variant in alternating_runtime_order(repetition):
            path, size, digest = variants[variant]
            config = BenchmarkConfig(
                runtime="onnxruntime", repetition=repetition, warmup=warmup,
                iterations=iterations, threads=threads,
                input_bytes=embedding.tobytes(order="C"), input_identity=input_identity,
                keras_weights_path="unused", onnx_model_path=str(path),
                keras_artifact_size_bytes=size, onnx_artifact_size_bytes=size,
            )
            raw = run_fresh_process(config)
            raw["runtime"] = variant
            raw["model_variant"] = variant
            raw["artifact_sha256"] = digest
            name = f"{variant}-repetition-{repetition:02d}.json"
            raw_path = raw_directory / name
            write_result(raw_path, raw, pretty=pretty)
            raw_rows.append(raw)
            references.append(
                {
                    "runtime": variant, "repetition": repetition,
                    "relative_path": f"raw/{name}", "sha256": streaming_file_sha256(raw_path),
                }
            )
    fp32_identity = {
        "manifest_sha256": fp32_manifest["manifest_sha256"],
        "onnx_sha256": fp32_manifest["onnx_artifact"]["sha256"],
        "size_bytes": fp32_manifest["onnx_artifact"]["size_bytes"],
        "precision": "fp32",
    }
    int8_identity = {
        "manifest_sha256": int8_manifest["manifest_sha256"],
        "onnx_sha256": int8_manifest["int8_artifact"]["sha256"],
        "size_bytes": int8_manifest["int8_artifact"]["size_bytes"],
        "precision": "dynamic_int8_weights_float32_io",
    }
    summary = build_int8_benchmark_summary(
        raw_rows=raw_rows, raw_references=references,
        fp32_artifact=fp32_identity, int8_artifact=int8_identity,
        input_identity=input_identity, parity_result=parity,
        warmup=warmup, iterations=iterations, repetitions=repetitions, threads=threads,
    )
    write_result(output / "summary.json", summary, pretty=pretty)
    return summary


__all__ = [
    "INT8_BENCHMARK_SCHEMA_VERSION", "VARIANTS", "alternating_runtime_order",
    "build_int8_benchmark_summary", "deployment_decision", "run_int8_benchmark_suite",
]
