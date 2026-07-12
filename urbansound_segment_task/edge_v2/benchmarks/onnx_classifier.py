"""Fresh-process Keras versus ONNX Runtime classifier-only lifecycle benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from urbansound_segment_task.edge_v2.benchmarks.lifecycle import LifecycleRequest, run_lifecycle
from urbansound_segment_task.edge_v2.benchmarks.memory import read_peak_rss, summarize_peak_rss
from urbansound_segment_task.edge_v2.benchmarks.schema import timing_section
from urbansound_segment_task.edge_v2.data.manifest import document_sha256
from urbansound_segment_task.edge_v2.evaluation.result_schema import safe_environment, utc_timestamp, write_result
from urbansound_segment_task.edge_v2.features.cache_dataset import load_verified_cache_records
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256
from urbansound_segment_task.edge_v2.export.onnx_linear import (
    EXPORT_SCHEMA_VERSION, build_linear_keras_model, resolve_source_model,
)


SUMMARY_SCHEMA_VERSION = "edge-v2.onnx-classifier-benchmark.v1"
RAW_SCHEMA_VERSION = "edge-v2.onnx-classifier-benchmark-run.v1"
EXPECTED_ONNX_SHA256 = "641f41921612048c8ed51d7c7c9d8f8f534827d08a091f87dd5900929731a329"
EXPECTED_SOURCE_SHA256 = "7538f47869e4ad5901074a1602e1217d3a9192f98fbd6d3d9f9ebac9bc708301"
RUNTIMES = ("keras", "onnxruntime")
TEST_MODES = ("benchmark", "test_crash", "test_timeout")


@dataclass(frozen=True)
class BenchmarkConfig:
    runtime: str
    repetition: int
    warmup: int
    iterations: int
    threads: int
    input_bytes: bytes
    input_identity: dict[str, Any]
    keras_weights_path: str
    onnx_model_path: str
    keras_artifact_size_bytes: int
    onnx_artifact_size_bytes: int
    test_mode: str = "benchmark"

    def validate(self) -> None:
        if self.runtime not in RUNTIMES:
            raise ValueError("unsupported benchmark runtime")
        if self.test_mode not in TEST_MODES:
            raise ValueError("unsupported benchmark test mode")
        if self.repetition < 1 or self.warmup < 0 or self.iterations < 1:
            raise ValueError("invalid benchmark repetition warmup or iterations")
        if self.threads != 1:
            raise ValueError("classifier benchmark requires exactly one thread")
        if len(self.input_bytes) != 1024 * 4:
            raise ValueError("benchmark input byte size mismatch")


class MaterializingConsumer:
    def __init__(self) -> None:
        self.last: Optional[np.ndarray] = None
        self.consumed_count = 0
        self.checksum = 0.0

    def __call__(self, value: Any) -> None:
        output = np.asarray(value, dtype=np.float32)
        if output.shape != (1, 10):
            raise ValueError("runtime output shape mismatch")
        self.last = output
        self.checksum += float(output[0, self.consumed_count % 10])
        self.consumed_count += 1


def validate_runtime_output(output: np.ndarray) -> dict[str, Any]:
    probabilities = np.asarray(output, dtype=np.float32)
    if probabilities.shape != (1, 10):
        raise ValueError("runtime output must have shape [1,10]")
    if not np.isfinite(probabilities).all():
        raise ValueError("runtime output contains NaN or Inf")
    probability_sum = float(probabilities.sum())
    deviation = abs(probability_sum - 1.0)
    if deviation > 1e-5:
        raise ValueError("runtime output probability sum mismatch")
    return {
        "shape": [1, 10],
        "dtype": str(probabilities.dtype),
        "finite": True,
        "probability_sum": probability_sum,
        "probability_sum_deviation": deviation,
        "top1_class": int(probabilities.argmax(axis=1)[0]),
        "output_sha256": hashlib.sha256(probabilities.tobytes(order="C")).hexdigest(),
    }


def select_benchmark_input(cache_root: Path) -> tuple[dict[str, Any], np.ndarray]:
    verified = load_verified_cache_records(cache_root)
    candidates = [
        record for record in verified.records if int(record["embeddings"].shape[0]) > 0
    ]
    if not candidates:
        raise ValueError("verified cache has no evaluable segment")
    record = min(candidates, key=lambda item: str(item["clip_key"]))
    embedding = np.asarray(record["embeddings"][0], dtype=np.float32).reshape(1, 1024)
    if not np.isfinite(embedding).all():
        raise ValueError("benchmark input contains NaN or Inf")
    identity_document = {
        "cache_identity": verified.cache_identity,
        "cache_index_sha256": verified.index_sha256,
        "clip_key": str(record["clip_key"]),
        "fold": int(record["fold"]),
        "class_id": int(record["class_id"]),
        "segment_index": 0,
        "segment_start_sample": 0,
        "shape": [1, 1024],
        "dtype": "float32",
        "embedding_sha256": hashlib.sha256(embedding.tobytes(order="C")).hexdigest(),
    }
    identity_document["input_identity_sha256"] = document_sha256(
        identity_document, excluded_fields=("input_identity_sha256",)
    )
    return identity_document, embedding


def _duration(duration_ns: Optional[int]) -> dict[str, Any]:
    return {
        "unit": "nanoseconds",
        "timer_source": "time.perf_counter_ns",
        "duration": duration_ns,
    }


def _thread_environment() -> dict[str, str]:
    return {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
    }


def _child_result(connection: Connection, config_data: dict[str, Any]) -> None:
    connection.send({"kind": "started", "pid": os.getpid()})
    config = BenchmarkConfig(**config_data)
    try:
        config.validate()
        if config.test_mode == "test_crash":
            os._exit(70)
        if config.test_mode == "test_timeout":
            while True:
                time.sleep(1)
        for name, value in _thread_environment().items():
            os.environ[name] = value
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        baseline = read_peak_rss()
        import_start = time.perf_counter_ns()
        if config.runtime == "keras":
            import tensorflow as tf

            tf.config.set_visible_devices([], "GPU")
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            runtime_state: dict[str, Any] = {
                "runtime": "keras_tensorflow",
                "tensorflow_version": tf.__version__,
                "effective_thread_settings": {
                    "intra_op_num_threads": tf.config.threading.get_intra_op_parallelism_threads(),
                    "inter_op_num_threads": tf.config.threading.get_inter_op_parallelism_threads(),
                },
                "accelerator": "none_cpu_only",
            }
            loader = lambda: _load_keras(tf, Path(config.keras_weights_path))
            inference = lambda model, values: np.asarray(model(values, training=False), dtype=np.float32)
        else:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            runtime_state = {
                "runtime": "onnxruntime",
                "onnxruntime_version": ort.__version__,
                "requested_provider_list": ["CPUExecutionProvider"],
                "effective_session_options": {
                    "intra_op_num_threads": int(options.intra_op_num_threads),
                    "inter_op_num_threads": int(options.inter_op_num_threads),
                    "execution_mode": "ORT_SEQUENTIAL",
                    "graph_optimization_level": "ORT_ENABLE_ALL",
                },
            }
            loader = lambda: _load_onnx(ort, options, Path(config.onnx_model_path))
            inference = lambda session, values: np.asarray(
                session.run(["probabilities"], {"embedding": values})[0], dtype=np.float32
            )
        import_end = time.perf_counter_ns()
        input_data = np.frombuffer(config.input_bytes, dtype=np.float32).copy().reshape(1, 1024)
        consumer = MaterializingConsumer()
        lifecycle = run_lifecycle(
            loader,
            inference,
            input_data,
            LifecycleRequest(
                name=f"linear-{config.runtime}-classifier-only",
                description="Batch-1 Linear classifier inference on one cached embedding; excludes YAMNet.",
                warmup=config.warmup,
                iterations=config.iterations,
                items_per_call=1,
            ),
            consume_result=consumer,
        )
        if lifecycle["status"]["outcome"] != "success" or consumer.last is None:
            raise RuntimeError("runtime lifecycle did not complete")
        if consumer.consumed_count != 1 + config.warmup + config.iterations:
            raise ValueError("first-call warmup or steady-state consumption count mismatch")
        if config.runtime == "onnxruntime":
            # The loaded session is not returned by run_lifecycle, so create no second session;
            # effective providers are captured inside the loader wrapper.
            runtime_state["effective_provider_list"] = _LAST_ORT_PROVIDER_LIST.copy()
            if runtime_state["effective_provider_list"] != ["CPUExecutionProvider"]:
                raise ValueError("non-CPU ONNX Runtime provider became active")
        final = read_peak_rss()
        memory = summarize_peak_rss(baseline, final)
        payload = {
            "kind": "result",
            "pid": os.getpid(),
            "runtime_import_configuration": _duration(import_end - import_start),
            "runtime_state": runtime_state,
            "lifecycle": lifecycle,
            "memory": memory,
            "output_validation": validate_runtime_output(consumer.last),
            "consumer": {
                "consumed_count": consumer.consumed_count,
                "expected_count": 1 + config.warmup + config.iterations,
                "checksum": consumer.checksum,
                "output_materialized_each_call": True,
            },
            "status": {"outcome": "success", "error": None},
        }
        connection.send(payload)
    except Exception as exc:
        connection.send(
            {
                "kind": "child_failure",
                "pid": os.getpid(),
                "error": {
                    "stage": "runtime_benchmark",
                    "type": type(exc).__name__,
                    "message": "runtime benchmark child did not complete",
                },
            }
        )
    finally:
        connection.close()


def _load_keras(tf: Any, weights_path: Path) -> Any:
    model = build_linear_keras_model(tf)
    model.load_weights(str(weights_path))
    return model


_LAST_ORT_PROVIDER_LIST: list[str] = []


def _load_onnx(ort: Any, options: Any, model_path: Path) -> Any:
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    global _LAST_ORT_PROVIDER_LIST
    _LAST_ORT_PROVIDER_LIST = list(session.get_providers())
    return session


def _failure_raw(config: BenchmarkConfig, startup_ns: Optional[int], error: Mapping[str, Any], pid: Optional[int]) -> dict[str, Any]:
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "runtime": config.runtime,
        "repetition": config.repetition,
        "child_pid": pid,
        "process_startup": _duration(startup_ns),
        "input_identity": config.input_identity,
        "status": {"outcome": "failure", "error": dict(error)},
    }


def run_fresh_process(
    config: BenchmarkConfig, *, timeout_seconds: float = 120.0
) -> dict[str, Any]:
    config.validate()
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout must be finite and positive")
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_child_result, args=(send, asdict(config)))
    started_at = time.perf_counter_ns()
    deadline = time.monotonic() + timeout_seconds
    startup_ns: Optional[int] = None
    child_pid: Optional[int] = None
    message: Optional[Mapping[str, Any]] = None
    error: Optional[dict[str, Any]] = None
    try:
        process.start()
        send.close()
        if not receive.poll(max(0.0, deadline - time.monotonic())):
            error = {"stage": "process_startup", "type": "ChildProcessTimeout"}
        else:
            try:
                started = receive.recv()
            except EOFError:
                started = None
            if not isinstance(started, dict) or started.get("kind") != "started":
                error = {"stage": "process_startup", "type": "ChildProcessProtocolError"}
            else:
                startup_ns = time.perf_counter_ns() - started_at
                child_pid = int(started["pid"])
                if not receive.poll(max(0.0, deadline - time.monotonic())):
                    error = {"stage": "child_process", "type": "ChildProcessTimeout"}
                else:
                    try:
                        message = receive.recv()
                    except EOFError:
                        message = None
        process.join(timeout=max(0.0, deadline - time.monotonic()))
        if process.is_alive():
            error = {"stage": "child_process", "type": "ChildProcessTimeout"}
        elif process.exitcode not in (0, None):
            error = {
                "stage": "child_process",
                "type": "ChildProcessCrash" if process.exitcode < 0 or config.test_mode == "test_crash" else "ChildProcessNonZeroExit",
                "exit_code": process.exitcode,
            }
        elif isinstance(message, dict) and message.get("kind") == "child_failure":
            error = dict(message["error"])
        elif not isinstance(message, dict) or message.get("kind") != "result":
            error = error or {"stage": "child_process", "type": "ChildProcessProtocolError"}
    finally:
        receive.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=1.0)
    if error is not None or message is None:
        return _failure_raw(config, startup_ns, error or {"type": "ChildResultMissing"}, child_pid)
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "runtime": config.runtime,
        "repetition": config.repetition,
        "child_pid": int(message["pid"]),
        "process_startup": _duration(startup_ns),
        "runtime_import_configuration": dict(message["runtime_import_configuration"]),
        "runtime_state": dict(message["runtime_state"]),
        "lifecycle": dict(message["lifecycle"]),
        "memory": dict(message["memory"]),
        "input_identity": config.input_identity,
        "output_validation": dict(message["output_validation"]),
        "consumer": dict(message["consumer"]),
        "benchmark_config": {
            "warmup": config.warmup,
            "iterations": config.iterations,
            "threads": config.threads,
            "batch_size": 1,
            "process_isolation": "fresh_spawned_process",
        },
        "artifact_size_bytes": (
            config.keras_artifact_size_bytes if config.runtime == "keras" else config.onnx_artifact_size_bytes
        ),
        "status": {"outcome": "success", "error": None},
    }


def _median(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(statistics.median(values))


def summarize_runtime(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows or any(row.get("status", {}).get("outcome") != "success" for row in rows):
        raise ValueError("cannot summarize failed or empty runtime repetitions")
    pids = [int(row["child_pid"]) for row in rows]
    if len(set(pids)) != len(pids):
        raise ValueError("runtime repetitions did not use fresh child processes")
    return {
        "repetitions": len(rows),
        "fresh_child_pids_unique": True,
        "median_process_startup_ns": _median(rows, ("process_startup", "duration")),
        "median_runtime_import_configuration_ns": _median(rows, ("runtime_import_configuration", "duration")),
        "median_model_load_ns": _median(rows, ("lifecycle", "load_time", "duration")),
        "median_first_call_ns": _median(rows, ("lifecycle", "first_call_latency", "duration")),
        "median_steady_p50_ns": _median(rows, ("lifecycle", "steady_state", "timing", "p50")),
        "median_steady_p95_ns": _median(rows, ("lifecycle", "steady_state", "timing", "p95")),
        "median_steady_p99_ns": _median(rows, ("lifecycle", "steady_state", "timing", "p99")),
        "median_predictions_per_second": _median(rows, ("lifecycle", "steady_state", "throughput", "items_per_second")),
        "memory": {
            "median_baseline_peak_rss_bytes": _median(rows, ("memory", "baseline_peak_rss_bytes")),
            "median_final_peak_rss_bytes": _median(rows, ("memory", "final_peak_rss_bytes")),
            "median_approximate_incremental_peak_rss_bytes": _median(rows, ("memory", "approximate_incremental_peak_rss_bytes")),
            "source": rows[0]["memory"]["source"],
            "limitation": rows[0]["memory"]["platform_limitation"],
        },
        "artifact_size_bytes": int(rows[0]["artifact_size_bytes"]),
        "top1_class": int(rows[0]["output_validation"]["top1_class"]),
    }


def build_summary(
    *, raw_rows: Sequence[Mapping[str, Any]], raw_references: Sequence[Mapping[str, Any]],
    artifact_identity: Mapping[str, Any], input_identity: Mapping[str, Any],
    warmup: int, iterations: int, repetitions: int, threads: int,
) -> dict[str, Any]:
    failures = [row for row in raw_rows if row.get("status", {}).get("outcome") != "success"]
    if failures:
        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "created_at_utc": utc_timestamp(),
            "artifact_identity": dict(artifact_identity),
            "input_identity": dict(input_identity),
            "benchmark_config": {"warmup": warmup, "iterations": iterations, "repetitions": repetitions, "threads": threads, "batch_size": 1},
            "raw_run_references": list(raw_references),
            "status": {"outcome": "failure", "error": {"type": "RawRepetitionFailure", "count": len(failures)}},
        }
    grouped = {runtime: [row for row in raw_rows if row["runtime"] == runtime] for runtime in RUNTIMES}
    if any(len(grouped[runtime]) != repetitions for runtime in RUNTIMES):
        raise ValueError("raw runtime repetition count mismatch")
    keras = summarize_runtime(grouped["keras"])
    onnx = summarize_runtime(grouped["onnxruntime"])
    input_hashes = {row["input_identity"]["input_identity_sha256"] for row in raw_rows}
    if input_hashes != {input_identity["input_identity_sha256"]}:
        raise ValueError("runtime repetitions did not use the same input identity")
    keras_classes = {int(row["output_validation"]["top1_class"]) for row in grouped["keras"]}
    onnx_classes = {int(row["output_validation"]["top1_class"]) for row in grouped["onnxruntime"]}
    top1_agreement = len(keras_classes) == 1 and keras_classes == onnx_classes
    keras_incremental = float(keras["memory"]["median_approximate_incremental_peak_rss_bytes"])
    onnx_incremental = float(onnx["memory"]["median_approximate_incremental_peak_rss_bytes"])
    memory_reduction = None if keras_incremental == 0 else (keras_incremental - onnx_incremental) / keras_incremental * 100.0
    comparison = {
        "latency_speedup_keras_p50_over_onnx_p50": keras["median_steady_p50_ns"] / onnx["median_steady_p50_ns"],
        "throughput_ratio_onnx_over_keras": onnx["median_predictions_per_second"] / keras["median_predictions_per_second"],
        "load_time_ratio_keras_over_onnx": keras["median_model_load_ns"] / onnx["median_model_load_ns"],
        "memory_reduction_percent": memory_reduction,
        "artifact_size_reduction_percent": (keras["artifact_size_bytes"] - onnx["artifact_size_bytes"]) / keras["artifact_size_bytes"] * 100.0,
        "top1_prediction_agreement": top1_agreement,
        "top1_class": next(iter(keras_classes)) if top1_agreement else None,
    }
    success = bool(top1_agreement)
    document: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "artifact_identity": dict(artifact_identity),
        "input_identity": dict(input_identity),
        "benchmark_config": {
            "warmup": warmup,
            "iterations": iterations,
            "repetitions": repetitions,
            "batch_size": 1,
            "threads": threads,
            "runtime_order_per_repetition": ["keras", "onnxruntime"],
            "timer_source": "time.perf_counter_ns",
        },
        "keras_runtime": keras,
        "onnx_runtime": onnx,
        "comparison": comparison,
        "raw_run_references": list(raw_references),
        "environment": safe_environment(threads),
        "limitations": [
            "Classifier-only benchmark accepts one precomputed 1024-value YAMNet embedding.",
            "YAMNet, audio decoding, segmentation, embedding extraction, and clip aggregation are excluded.",
            "macOS peak RSS is a process-lifetime high-water mark; incremental RSS is approximate.",
            "Fresh process startup includes Python and benchmark module bootstrap before child entry.",
            "Results describe this machine, dependency lock, and one-thread configuration only.",
        ],
        "status": {"outcome": "success" if success else "failure", "error": None if success else {"type": "Top1Mismatch"}},
    }
    document["summary_sha256"] = document_sha256(
        document, excluded_fields=("created_at_utc", "summary_sha256")
    )
    return document


def _validate_artifacts(run_manifest_path: Path, artifact_directory: Path) -> tuple[dict[str, Any], Path, Path, int, int]:
    source = resolve_source_model(run_manifest_path, model="linear", fold=1)
    if source.source_model_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError("benchmark source model SHA-256 mismatch")
    root = Path(artifact_directory)
    manifest_path = root / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXPORT_SCHEMA_VERSION or manifest.get("status", {}).get("outcome") != "success":
        raise ValueError("benchmark ONNX manifest is incompatible or unsuccessful")
    if manifest.get("manifest_sha256") != document_sha256(manifest, excluded_fields=("created_at_utc", "manifest_sha256")):
        raise ValueError("benchmark ONNX manifest SHA-256 mismatch")
    if manifest.get("source_model_sha256") != source.source_model_sha256:
        raise ValueError("benchmark ONNX source model SHA-256 mismatch")
    relative = Path(str(manifest["onnx_artifact"]["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("benchmark ONNX path is unsafe")
    onnx_path = root / relative
    if onnx_path.stat().st_size != int(manifest["onnx_artifact"]["size_bytes"]):
        raise ValueError("benchmark ONNX artifact size mismatch")
    onnx_sha = streaming_file_sha256(onnx_path)
    if onnx_sha != EXPECTED_ONNX_SHA256 or onnx_sha != manifest["onnx_artifact"]["sha256"]:
        raise ValueError("benchmark ONNX artifact SHA-256 mismatch")
    artifact_identity = {
        "onnx_manifest_sha256": manifest["manifest_sha256"],
        "onnx_sha256": onnx_sha,
        "source_model_identity": source.source_model_identity,
        "source_model_sha256": source.source_model_sha256,
        "architecture_id": "linear",
        "test_fold": 1,
        "validation_fold": 2,
        "precision": "fp32",
    }
    return artifact_identity, source.weights_path, onnx_path, source.source_model_size_bytes, int(manifest["onnx_artifact"]["size_bytes"])


def run_benchmark_suite(
    *, run_manifest_path: Path, cache_root: Path, onnx_artifact_directory: Path,
    warmup: int, iterations: int, repetitions: int, threads: int,
    output_directory: Path, pretty: bool, timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    if warmup < 0 or iterations < 1 or repetitions < 1 or threads != 1:
        raise ValueError("benchmark config is outside the supported fixed contract")
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("benchmark output directory already exists")
    artifact_identity, weights_path, onnx_path, keras_size, onnx_size = _validate_artifacts(
        run_manifest_path, onnx_artifact_directory
    )
    input_identity, embedding = select_benchmark_input(cache_root)
    output.mkdir(parents=True, exist_ok=False)
    raw_directory = output / "raw"
    raw_directory.mkdir()
    raw_rows: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        for runtime in RUNTIMES:
            config = BenchmarkConfig(
                runtime=runtime,
                repetition=repetition,
                warmup=warmup,
                iterations=iterations,
                threads=threads,
                input_bytes=embedding.tobytes(order="C"),
                input_identity=input_identity,
                keras_weights_path=str(weights_path),
                onnx_model_path=str(onnx_path),
                keras_artifact_size_bytes=keras_size,
                onnx_artifact_size_bytes=onnx_size,
            )
            raw = run_fresh_process(config, timeout_seconds=timeout_seconds)
            name = f"{runtime}-repetition-{repetition:02d}.json"
            path = raw_directory / name
            write_result(path, raw, pretty=pretty)
            raw_rows.append(raw)
            references.append(
                {
                    "runtime": runtime,
                    "repetition": repetition,
                    "relative_path": f"raw/{name}",
                    "sha256": streaming_file_sha256(path),
                }
            )
    summary = build_summary(
        raw_rows=raw_rows,
        raw_references=references,
        artifact_identity=artifact_identity,
        input_identity=input_identity,
        warmup=warmup,
        iterations=iterations,
        repetitions=repetitions,
        threads=threads,
    )
    write_result(output / "summary.json", summary, pretty=pretty)
    return summary


__all__ = [
    "BenchmarkConfig", "EXPECTED_ONNX_SHA256", "EXPECTED_SOURCE_SHA256", "RAW_SCHEMA_VERSION",
    "RUNTIMES", "SUMMARY_SCHEMA_VERSION", "build_summary", "run_benchmark_suite",
    "run_fresh_process", "select_benchmark_input", "summarize_runtime", "validate_runtime_output",
]
