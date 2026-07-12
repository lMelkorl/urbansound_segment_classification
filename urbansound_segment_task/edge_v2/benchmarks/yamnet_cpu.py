"""Method-specific orchestration for offline YAMNet CPU measurements."""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from urbansound_segment_task.edge_v2.models.yamnet_artifact import verify_yamnet_artifact
from urbansound_segment_task.edge_v2.utils.environment import collect_system_info

from .lifecycle import LifecycleRequest, run_lifecycle
from .memory import read_peak_rss, summarize_peak_rss
from .pipeline import PIPELINE_SCHEMA_VERSION, PipelineBenchmarkRequest, run_pipeline_benchmark
from .schema import serialize_document, timing_section, utc_timestamp, write_document_atomic
from .thread_policy import temporary_thread_policy, validate_thread_count
from .timer import summarize_samples


YAMNET_CPU_SCHEMA_VERSION = "edge-v2.yamnet-cpu-benchmark.v1"
YAMNET_RAW_SCHEMA_VERSION = "edge-v2.yamnet-cpu-raw-run.v1"
EXPECTED_ARTIFACT_TREE_SHA256 = (
    "5d3bccc6549dcf864250dd52b9ffa35a1aec0f2b6f88a91229ff0336582e25c2"
)
SELECTION_RULE = (
    "highest_legacy_throughput_then_within_3_percent_lower_p95_then_lower_threads_"
    "then_lower_incremental_peak_rss"
)


@dataclass(frozen=True)
class YamnetRunConfig:
    artifact: str
    threads: int
    warmup: int
    iterations: int
    sample_count: int
    repetition: int = 1

    def validate(self) -> None:
        validate_thread_count(self.threads)
        if isinstance(self.warmup, bool) or not isinstance(self.warmup, int) or self.warmup < 0:
            raise ValueError("warmup must be a non-negative integer")
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int) or self.iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise ValueError("sample_count must be a positive integer")
        if isinstance(self.repetition, bool) or not isinstance(self.repetition, int) or self.repetition < 1:
            raise ValueError("repetition must be a positive integer")


def real_time_factor(segments_per_second: float, duration_seconds: float) -> float:
    if segments_per_second < 0 or duration_seconds <= 0:
        raise ValueError("throughput must be non-negative and duration must be positive")
    return segments_per_second * duration_seconds


def median_repetition_summary(repetitions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not repetitions:
        raise ValueError("at least one repetition is required")
    metrics = (
        "steady_state_model_p50_ns",
        "steady_state_model_p95_ns",
        "model_inference_segments_per_second",
        "legacy_pipeline_p50_ns",
        "legacy_pipeline_p95_ns",
        "legacy_feature_pipeline_segments_per_second",
        "real_time_factor",
        "incremental_peak_rss_bytes",
    )
    summary: dict[str, Any] = {"repetition_count": len(repetitions)}
    for metric in metrics:
        values = [item[metric] for item in repetitions if item.get(metric) is not None]
        summary["median_" + metric] = float(statistics.median(values)) if values else None
    return summary


def select_thread_configuration(configurations: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Apply the documented throughput/3%-p95/thread/RSS deterministic rule."""

    successful = [
        item
        for item in configurations
        if item.get("status") == "success"
        and item.get("median_legacy_feature_pipeline_segments_per_second") is not None
    ]
    if not successful:
        raise ValueError("no successful thread configuration is available")
    maximum = max(
        float(item["median_legacy_feature_pipeline_segments_per_second"])
        for item in successful
    )
    close = [
        item
        for item in successful
        if float(item["median_legacy_feature_pipeline_segments_per_second"]) >= maximum * 0.97
    ]
    return min(
        close,
        key=lambda item: (
            float(item.get("median_legacy_pipeline_p95_ns") or math.inf),
            int(item["thread_count"]),
            float(item.get("median_incremental_peak_rss_bytes") or math.inf),
        ),
    )


def _git_revision(repository_root: Path) -> Optional[str]:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return value if len(value) == 40 and all(char in "0123456789abcdef" for char in value.lower()) else None


def _duration_section(duration_ns: Optional[int]) -> dict[str, Any]:
    return {
        "unit": "nanoseconds",
        "timer_source": "time.perf_counter_ns",
        "duration": duration_ns,
        "measurement": "parent_observed_spawn_to_child_entry",
        "status": {"outcome": "success" if duration_ns is not None else "failure"},
    }


def _run_measured_child(config: YamnetRunConfig) -> dict[str, Any]:
    """Execute lifecycle and staged measurements with one shared warm model."""

    from urbansound_segment_task.edge_v2.models.yamnet_benchmark import (
        build_yamnet_pipeline,
        decode_pcm16,
        generate_pcm16_input,
        input_identity,
        load_benchmark_runtime,
        materialized_inference,
        preprocess_pcm,
        WarmupTimingRecorder,
    )

    pcm_bytes = generate_pcm16_input(config.sample_count)
    duration_seconds = config.sample_count / 16_000.0
    loader_metadata: dict[str, Any] = {}
    holder: dict[str, Any] = {}

    def loader() -> Any:
        runtime = load_benchmark_runtime(Path(config.artifact), config.threads)
        waveform = preprocess_pcm(
            decode_pcm16(pcm_bytes, runtime.numpy), config.sample_count, runtime.numpy
        )
        value = (runtime, waveform)
        holder["loaded"] = value
        loader_metadata.update(
            {
                "loader_method": runtime.loader_method,
                "artifact_identity": runtime.artifact_identity,
                "tensorflow_version": runtime.tensorflow_version,
                "visible_devices": list(runtime.visible_devices),
                "requested_intra_op_threads": runtime.requested_intra_op_threads,
                "effective_intra_op_threads": runtime.effective_intra_op_threads,
                "requested_inter_op_threads": runtime.requested_inter_op_threads,
                "effective_inter_op_threads": runtime.effective_inter_op_threads,
                **runtime.loader_components_ns,
            }
        )
        return value

    def inference(loaded: Any, unused_input: Any) -> dict[str, Any]:
        runtime, waveform = loaded
        result = materialized_inference(runtime.model, waveform)
        loader_metadata["output_contract"] = dict(result["contract"])
        return result

    recorded_inference = WarmupTimingRecorder(inference, config.warmup)
    baseline = read_peak_rss()
    lifecycle = run_lifecycle(
        loader,
        recorded_inference,
        None,
        LifecycleRequest(
            name="yamnet-model-inference",
            description="Local YAMNet model inference lifecycle on deterministic audio.",
            warmup=config.warmup,
            iterations=config.iterations,
            items_per_call=1,
            item_duration_seconds=duration_seconds,
        ),
    )
    pipeline = None
    if lifecycle["status"]["outcome"] == "success":
        runtime, _ = holder["loaded"]
        pipeline = run_pipeline_benchmark(
            build_yamnet_pipeline(runtime.model, config.sample_count, runtime.numpy),
            pcm_bytes,
            PipelineBenchmarkRequest(
                name="yamnet-legacy-feature-pipeline",
                description="In-memory PCM through local YAMNet embedding mean aggregation.",
                warmup=config.warmup,
                iterations=config.iterations,
                items_per_call=1,
                item_duration_seconds=duration_seconds,
                input_size=len(pcm_bytes),
            ),
        )
    final = read_peak_rss()
    memory = summarize_peak_rss(baseline, final)
    warmup_statistics = (
        summarize_samples(recorded_inference.raw_samples_ns)
        if recorded_inference.raw_samples_ns
        else None
    )
    success = lifecycle["status"]["outcome"] == "success" and (
        pipeline is not None and pipeline["status"]["outcome"] == "success"
    )
    if success:
        measurement_error = None
    elif lifecycle["status"]["outcome"] != "success":
        measurement_error = dict(lifecycle["status"].get("error") or {"type": "LifecycleFailure"})
    else:
        measurement_error = dict(
            (pipeline or {}).get("status", {}).get("error") or {"type": "PipelineFailure"}
        )
    return {
        "schema_version": YAMNET_RAW_SCHEMA_VERSION,
        "config": {
            "thread_count": config.threads,
            "warmup": config.warmup,
            "iterations": config.iterations,
            "sample_count": config.sample_count,
            "duration_seconds": duration_seconds,
            "repetition": config.repetition,
        },
        "input_identity": input_identity(pcm_bytes, config.sample_count),
        "loader_metadata": loader_metadata,
        "lifecycle": lifecycle,
        "warmup": {
            "timing": timing_section(warmup_statistics),
            "expected_calls": config.warmup,
            "recorded_calls": len(recorded_inference.raw_samples_ns),
            "status": {
                "outcome": (
                    "success"
                    if len(recorded_inference.raw_samples_ns) == config.warmup
                    else "failure"
                ),
                "error": (
                    None
                    if len(recorded_inference.raw_samples_ns) == config.warmup
                    else {"type": "WarmupSampleCountMismatch"}
                ),
            },
        },
        "pipeline": pipeline,
        "memory": memory,
        "status": {
            "outcome": "success" if success else "failure",
            "error": measurement_error,
        },
    }


def _child_entry(connection: Connection, config_data: dict[str, Any]) -> None:
    connection.send({"kind": "started"})
    config = YamnetRunConfig(**config_data)
    try:
        with temporary_thread_policy(config.threads) as policy:
            result = _run_measured_child(config)
            result["execution_policy"] = {
                **policy,
                "process_isolation": "fresh_child_process",
                "multiprocessing_start_method": "spawn",
                "tensorflow_inter_op_threads": 1,
            }
            connection.send({"kind": "result", "document": result})
    except Exception as exc:
        connection.send(
            {
                "kind": "failure",
                "error": {
                    "stage": "child_measurement",
                    "type": type(exc).__name__,
                    "message": "YAMNet child measurement did not complete",
                },
            }
        )
    finally:
        connection.close()


def run_yamnet_in_fresh_process(
    config: YamnetRunConfig, *, timeout_seconds: float
) -> dict[str, Any]:
    config.validate()
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_child_entry, args=(send, asdict(config)))
    start_ns = time.perf_counter_ns()
    deadline = time.monotonic() + timeout_seconds
    startup_ns: Optional[int] = None
    document: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
    try:
        process.start()
        send.close()
        remaining = max(0.0, deadline - time.monotonic())
        if not receive.poll(remaining):
            error = {"stage": "process_startup", "type": "ChildProcessTimeout"}
        else:
            started = receive.recv()
            if not isinstance(started, dict) or started.get("kind") != "started":
                error = {"stage": "process_startup", "type": "ChildProtocolError"}
            else:
                startup_ns = time.perf_counter_ns() - start_ns
                remaining = max(0.0, deadline - time.monotonic())
                if not receive.poll(remaining):
                    error = {"stage": "child_process", "type": "ChildProcessTimeout"}
                else:
                    message = receive.recv()
                    if message.get("kind") == "result":
                        document = message["document"]
                    else:
                        error = dict(message.get("error") or {"type": "ChildProtocolError"})
        process.join(timeout=max(0.0, deadline - time.monotonic()))
        if process.is_alive():
            error = {"stage": "child_process", "type": "ChildProcessTimeout"}
        elif process.exitcode not in (0, None) and document is None:
            error = {
                "stage": "child_process",
                "type": "ChildProcessCrash" if process.exitcode < 0 else "ChildProcessNonZeroExit",
                "exit_code": process.exitcode,
            }
    except (EOFError, OSError):
        error = {"stage": "child_process", "type": "ChildProcessCrash"}
    finally:
        receive.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=1.0)
    if document is None:
        return {
            "schema_version": YAMNET_RAW_SCHEMA_VERSION,
            "config": {
                "thread_count": config.threads,
                "sample_count": config.sample_count,
                "repetition": config.repetition,
            },
            "process_startup_time": _duration_section(startup_ns),
            "status": {"outcome": "failure", "error": error},
        }
    document["process_startup_time"] = _duration_section(startup_ns)
    return document


def _run_metrics(document: Mapping[str, Any]) -> dict[str, Any]:
    if document["status"]["outcome"] != "success":
        return {"status": "failure", "error": document["status"].get("error")}
    lifecycle = document["lifecycle"]
    pipeline = document["pipeline"]
    model = lifecycle["steady_state"]
    legacy = pipeline["stages"]["end_to_end"]
    # The output contract is recorded by the loaded model's first materialized call.
    first_contract = document["loader_metadata"].get("output_contract")
    return {
        "status": "success",
        "steady_state_model_p50_ns": model["timing"]["p50"],
        "steady_state_model_p95_ns": model["timing"]["p95"],
        "model_inference_segments_per_second": model["throughput"]["items_per_second"],
        "legacy_pipeline_p50_ns": legacy["timing"]["p50"],
        "legacy_pipeline_p95_ns": legacy["timing"]["p95"],
        "legacy_feature_pipeline_segments_per_second": legacy["throughput"]["items_per_second"],
        "real_time_factor": legacy["throughput"]["real_time_factor"],
        "incremental_peak_rss_bytes": document["memory"].get("approximate_incremental_peak_rss_bytes"),
        "frame_count": first_contract.get("frame_count") if first_contract else None,
    }


def _median_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return float(statistics.median(present)) if present else None


def _headline(
    documents: Sequence[Mapping[str, Any]], summary_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    process_startup_ns = _median_optional(
        [document["process_startup_time"]["duration"] for document in documents]
    )
    verification_ns = _median_optional(
        [document["loader_metadata"]["artifact_verification_ns"] for document in documents]
    )
    tensorflow_import_ns = _median_optional(
        [
            document["loader_metadata"]["tensorflow_import_and_configuration_ns"]
            for document in documents
        ]
    )
    model_load_ns = _median_optional(
        [document["loader_metadata"]["savedmodel_load_ns"] for document in documents]
    )
    first_call_ns = _median_optional(
        [document["lifecycle"]["first_call_latency"]["duration"] for document in documents]
    )
    baseline_rss = _median_optional(
        [document["memory"].get("baseline_peak_rss_bytes") for document in documents]
    )
    final_rss = _median_optional(
        [document["memory"].get("final_peak_rss_bytes") for document in documents]
    )
    incremental_rss = _median_optional(
        [document["memory"].get("approximate_incremental_peak_rss_bytes") for document in documents]
    )
    return {
        "sample_count": 15_360,
        "duration_seconds": 0.960,
        "aggregation_across_repetitions": "median",
        "process_startup_ms": process_startup_ns / 1e6 if process_startup_ns is not None else None,
        "artifact_verification_ms": verification_ns / 1e6 if verification_ns is not None else None,
        "tensorflow_import_ms": tensorflow_import_ns / 1e6 if tensorflow_import_ns is not None else None,
        "model_load_ms": model_load_ns / 1e6 if model_load_ns is not None else None,
        "first_call_ms": first_call_ns / 1e6 if first_call_ns is not None else None,
        "steady_state_model_p50_ms": summary_metrics["median_steady_state_model_p50_ns"] / 1e6,
        "steady_state_model_p95_ms": summary_metrics["median_steady_state_model_p95_ns"] / 1e6,
        "model_inference_segments_per_second": summary_metrics[
            "median_model_inference_segments_per_second"
        ],
        "legacy_pipeline_p50_ms": summary_metrics["median_legacy_pipeline_p50_ns"] / 1e6,
        "legacy_pipeline_p95_ms": summary_metrics["median_legacy_pipeline_p95_ns"] / 1e6,
        "legacy_feature_pipeline_segments_per_second": summary_metrics[
            "median_legacy_feature_pipeline_segments_per_second"
        ],
        "real_time_factor": summary_metrics["median_real_time_factor"],
        "baseline_peak_rss_bytes": baseline_rss,
        "final_peak_rss_bytes": final_rss,
        "incremental_peak_rss_bytes": incremental_rss,
    }


def run_yamnet_cpu_benchmark(
    *,
    artifact: Path,
    thread_counts: Sequence[int],
    warmup: int,
    iterations: int,
    repetitions: int,
    sample_count: int,
    input_sensitivity: Sequence[int],
    timeout_seconds: float,
    repository_root: Path,
    child_runner: Callable[..., dict[str, Any]] = run_yamnet_in_fresh_process,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    identity = verify_yamnet_artifact(artifact)
    if identity["tree_sha256"] != EXPECTED_ARTIFACT_TREE_SHA256:
        raise ValueError("artifact tree SHA-256 does not match the benchmark contract")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if sample_count != 15_360:
        raise ValueError("headline sample_count must be 15360 for the legacy segment contract")
    for count in thread_counts:
        validate_thread_count(count)

    raw_runs: list[dict[str, Any]] = []
    thread_sweep = []
    for threads in thread_counts:
        repetition_metrics = []
        for repetition in range(1, repetitions + 1):
            raw = child_runner(
                YamnetRunConfig(str(artifact), threads, warmup, iterations, sample_count, repetition),
                timeout_seconds=timeout_seconds,
            )
            raw_runs.append(raw)
            repetition_metrics.append(_run_metrics(raw))
        successful = [metric for metric in repetition_metrics if metric["status"] == "success"]
        item: dict[str, Any] = {
            "thread_count": threads,
            "status": "success" if len(successful) == repetitions else "failure",
            "repetitions": repetition_metrics,
        }
        if successful:
            item.update(median_repetition_summary(successful))
        thread_sweep.append(item)

    selected = select_thread_configuration(thread_sweep)
    selected_threads = int(selected["thread_count"])
    sensitivity = []
    for count in input_sensitivity:
        raw = child_runner(
            YamnetRunConfig(str(artifact), selected_threads, warmup, iterations, count, 1),
            timeout_seconds=timeout_seconds,
        )
        raw_runs.append(raw)
        metrics = _run_metrics(raw)
        sensitivity.append(
            {
                "sample_count": count,
                "duration_seconds": count / 16_000.0,
                **metrics,
                "two_frame_compute_warning": count == 16_000,
            }
        )

    selected_raw_runs = [
        raw
        for raw in raw_runs
        if raw.get("status", {}).get("outcome") == "success"
        and raw["config"]["thread_count"] == selected_threads
        and raw["config"]["sample_count"] == 15_360
    ][:repetitions]
    selected_raw = selected_raw_runs[0]
    system = collect_system_info()
    return {
        "schema_version": YAMNET_CPU_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(now),
        "git_revision": _git_revision(repository_root),
        "artifact_identity": identity["artifact_id"],
        "artifact_tree_sha256": identity["tree_sha256"],
        "tensorflow_version": selected_raw["loader_metadata"]["tensorflow_version"],
        "python_version": platform.python_version(),
        "hardware": system,
        "tensorflow_devices": selected_raw["loader_metadata"]["visible_devices"],
        "input_identity": selected_raw["input_identity"],
        "thread_sweep": thread_sweep,
        "selected_thread_configuration": {
            "thread_count": selected_threads,
            "selection_rule": SELECTION_RULE,
            "throughput_tie_threshold_percent": 3.0,
            "summary": {key: value for key, value in selected.items() if key != "repetitions"},
        },
        "input_length_sensitivity": sensitivity,
        "headline_result": _headline(selected_raw_runs, selected),
        "embedded_schema_versions": {
            "lifecycle": "edge-v2.lifecycle-benchmark.v1",
            "pipeline": PIPELINE_SCHEMA_VERSION,
            "raw_run": YAMNET_RAW_SCHEMA_VERSION,
        },
        "raw_runs": raw_runs,
        "limitations": [
            "Synthetic non-silent in-memory PCM; no file I/O or dataset quality metric is measured.",
            "Darwin ru_maxrss is a process-lifetime high-water mark; incremental RSS is approximate.",
            "Power state and concurrent system load are not machine-verifiable in this artifact.",
            "The historical A100 throughput is unverified context and is not used for a speed ratio.",
            "The 16000-sample case produces two YAMNet frames and is not equal compute to one-frame inputs.",
        ],
        "status": {"outcome": "success", "error": None},
    }


def externalize_raw_runs(summary: dict[str, Any], output: Path) -> dict[str, Any]:
    """Write immutable child artifacts and replace embedded data with relative references."""

    raw_directory = output.parent / (output.stem + ".raw")
    if output.exists() or raw_directory.exists():
        raise FileExistsError("benchmark output or raw directory already exists")
    references = []
    try:
        for index, document in enumerate(summary["raw_runs"], start=1):
            config = document["config"]
            filename = (
                f"run-{index:03d}-t{config['thread_count']}-n{config['sample_count']}"
                f"-r{config['repetition']}.json"
            )
            serialized = serialize_document(document, pretty=False)
            path = raw_directory / filename
            write_document_atomic(path, serialized)
            references.append(
                {
                    "relative_path": raw_directory.name + "/" + filename,
                    "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
                    "thread_count": config["thread_count"],
                    "sample_count": config["sample_count"],
                    "repetition": config["repetition"],
                }
            )
    except Exception:
        # Keep already-written immutable evidence; never silently overwrite or hide partial output.
        raise
    result = dict(summary)
    result["raw_runs"] = references
    return result


__all__ = [
    "EXPECTED_ARTIFACT_TREE_SHA256",
    "SELECTION_RULE",
    "YAMNET_CPU_SCHEMA_VERSION",
    "YamnetRunConfig",
    "externalize_raw_runs",
    "median_repetition_summary",
    "real_time_factor",
    "run_yamnet_cpu_benchmark",
    "run_yamnet_in_fresh_process",
    "select_thread_configuration",
]
