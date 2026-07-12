"""Independent stage and end-to-end benchmarking for an injected pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Optional

from .runner import BenchmarkRequest, run_benchmark
from .schema import (
    environment_section,
    safe_environment_summary,
    serialize_document,
    timing_section,
    utc_timestamp,
    validate_benchmark_metadata,
    write_document_atomic,
)
from .stages import (
    MEASURED_STAGE_ORDER,
    PIPELINE_STAGE_ORDER,
    PipelineExecutionTracker,
    PipelineStages,
    execute_pipeline,
)
from .timer import ResultConsumer


PIPELINE_SCHEMA_VERSION = "edge-v2.pipeline-benchmark.v1"


@dataclass(frozen=True)
class PipelineBenchmarkRequest:
    name: str
    description: str
    warmup: int = 3
    iterations: int = 20
    items_per_call: int = 1
    item_duration_seconds: Optional[float] = None
    input_size: Optional[int] = None

    def validate(self) -> None:
        BenchmarkRequest(
            name=self.name,
            description=self.description,
            warmup=self.warmup,
            iterations=self.iterations,
            items_per_call=self.items_per_call,
            item_duration_seconds=self.item_duration_seconds,
        ).validate()
        validate_benchmark_metadata(self.name, self.description)
        if self.input_size is not None and (
            isinstance(self.input_size, bool)
            or not isinstance(self.input_size, int)
            or self.input_size < 1
        ):
            raise ValueError("input_size must be a positive integer when provided")


def _stage_result(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "timing": dict(document["timing"]),
        "throughput": dict(document["throughput"]),
        "status": dict(document["status"]),
    }


def _failed_stage(
    stage_name: str,
    *,
    error_type: str,
    message: str,
    blocked_by: Optional[str] = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "type": error_type,
        "pipeline_stage": stage_name,
        "message": message,
    }
    if blocked_by is not None:
        error["blocked_by"] = blocked_by
    return {
        "timing": timing_section(None),
        "throughput": {
            "calls_per_second": None,
            "items_per_second": None,
            "real_time_factor": None,
        },
        "status": {"outcome": "failure", "error": error},
    }


def _stage_request(stage_name: str, request: PipelineBenchmarkRequest) -> BenchmarkRequest:
    return BenchmarkRequest(
        name="pipeline-stage-" + stage_name.replace("_", "-"),
        description="Synthetic pipeline stage " + stage_name.replace("_", " ") + ".",
        warmup=request.warmup,
        iterations=request.iterations,
        items_per_call=request.items_per_call,
        item_duration_seconds=request.item_duration_seconds,
    )


def _add_pipeline_stage_to_error(stage_result: dict[str, Any], stage_name: str) -> None:
    error = stage_result["status"].get("error")
    if isinstance(error, dict):
        error["pipeline_stage"] = stage_name


def _comparison(stages: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    means = [stages[name]["timing"]["mean"] for name in MEASURED_STAGE_ORDER]
    end_to_end_mean = stages["end_to_end"]["timing"]["mean"]
    if any(mean is None for mean in means) or end_to_end_mean is None:
        return (
            {"mean_nanoseconds": None},
            {
                "nanoseconds": None,
                "percent": None,
                "percentage_basis": "stage_sum_mean",
            },
        )
    stage_sum = float(sum(means))
    overhead = float(end_to_end_mean - stage_sum)
    overhead_percent = (overhead / stage_sum * 100.0) if stage_sum != 0 else None
    return (
        {"mean_nanoseconds": stage_sum},
        {
            "nanoseconds": overhead,
            "percent": overhead_percent,
            "percentage_basis": "stage_sum_mean",
        },
    )


def run_pipeline_benchmark(
    stages: PipelineStages,
    source: Any,
    request: PipelineBenchmarkRequest,
    *,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    environment_collector: Callable[[], Mapping[str, Any]] = safe_environment_summary,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Benchmark four stages separately and the full pipeline independently."""

    request.validate()
    environment = environment_collector()
    stage_results: dict[str, dict[str, Any]] = {}
    current_input = source
    blocked_by: Optional[str] = None

    for stage_name in MEASURED_STAGE_ORDER:
        if blocked_by is not None:
            stage_results[stage_name] = _failed_stage(
                stage_name,
                error_type="DependencyStageFailed",
                message="stage was not run because an upstream stage failed",
                blocked_by=blocked_by,
            )
            continue

        stage_function = getattr(stages, stage_name)
        consumer = ResultConsumer()
        stage_input = current_input
        document = run_benchmark(
            lambda function=stage_function, value=stage_input: function(value),
            _stage_request(stage_name, request),
            clock_ns=clock_ns,
            consume_result=consumer,
            environment_collector=lambda: environment,
            now=now,
        )
        stage_result = _stage_result(document)
        _add_pipeline_stage_to_error(stage_result, stage_name)
        stage_results[stage_name] = stage_result
        if stage_result["status"]["outcome"] == "success":
            current_input = consumer.last_result
        else:
            blocked_by = stage_name

    tracker = PipelineExecutionTracker()
    end_to_end_document = run_benchmark(
        lambda: execute_pipeline(stages, source, tracker=tracker),
        _stage_request("end_to_end", request),
        clock_ns=clock_ns,
        environment_collector=lambda: environment,
        now=now,
    )
    end_to_end_result = _stage_result(end_to_end_document)
    if end_to_end_result["status"]["outcome"] == "failure":
        failed_pipeline_stage = tracker.current_stage or "unknown"
        _add_pipeline_stage_to_error(end_to_end_result, failed_pipeline_stage)
    stage_results["end_to_end"] = end_to_end_result

    stage_sum, overhead = _comparison(stage_results)
    failed_stages = [
        name for name in PIPELINE_STAGE_ORDER if stage_results[name]["status"]["outcome"] != "success"
    ]
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(now),
        "pipeline": {
            "name": request.name,
            "description": request.description,
            "stage_order": list(PIPELINE_STAGE_ORDER),
            "warmup_runs": request.warmup,
            "measured_iterations": request.iterations,
            "items_per_call": request.items_per_call,
            "item_duration_seconds": request.item_duration_seconds,
            "input_size": request.input_size,
        },
        "stages": stage_results,
        "stage_sum": stage_sum,
        "end_to_end_overhead": overhead,
        "environment": environment_section(environment),
        "status": {
            "outcome": "failure" if failed_stages else "success",
            "error": (
                {"type": "PipelineStageFailure", "failed_stages": failed_stages}
                if failed_stages
                else None
            ),
        },
    }


__all__ = [
    "PIPELINE_SCHEMA_VERSION",
    "PipelineBenchmarkRequest",
    "run_pipeline_benchmark",
    "serialize_document",
    "write_document_atomic",
]
