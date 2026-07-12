"""Clock-injectable loader, first-call, warm-up, and steady-state timing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .runner import calculate_throughput
from .schema import timing_section, validate_benchmark_metadata
from .timer import (
    TIMER_SOURCE,
    BenchmarkExecutionError,
    ResultConsumer,
    measure_callable,
)


@dataclass(frozen=True)
class LifecycleRequest:
    name: str
    description: str
    warmup: int
    iterations: int
    items_per_call: int
    item_duration_seconds: Optional[float] = None

    def validate(self) -> None:
        from .runner import BenchmarkRequest

        BenchmarkRequest(
            name=self.name,
            description=self.description,
            warmup=self.warmup,
            iterations=self.iterations,
            items_per_call=self.items_per_call,
            item_duration_seconds=self.item_duration_seconds,
        ).validate()
        validate_benchmark_metadata(self.name, self.description)


def _duration_result(duration_ns: Optional[int], status: dict[str, Any]) -> dict[str, Any]:
    return {
        "unit": "nanoseconds",
        "timer_source": TIMER_SOURCE,
        "duration": duration_ns,
        "status": status,
    }


def _safe_failure(stage: str, error_type: str) -> dict[str, str]:
    return {
        "stage": stage,
        "type": error_type,
        "message": "lifecycle callable did not complete",
    }


def _empty_steady(error: dict[str, str]) -> dict[str, Any]:
    return {
        "timing": timing_section(None),
        "throughput": {
            "calls_per_second": None,
            "items_per_second": None,
            "real_time_factor": None,
        },
        "status": {"outcome": "failure", "error": error},
    }


def _failed_lifecycle(
    *,
    load_time: dict[str, Any],
    first_call: dict[str, Any],
    steady_state: dict[str, Any],
    error: dict[str, str],
) -> dict[str, Any]:
    return {
        "load_time": load_time,
        "first_call_latency": first_call,
        "steady_state": steady_state,
        "status": {"outcome": "failure", "error": error},
    }


def run_lifecycle(
    loader: Callable[[], Any],
    inference: Callable[[Any, Any], Any],
    input_data: Any,
    request: LifecycleRequest,
    *,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    consume_result: Optional[Callable[[Any], None]] = None,
) -> dict[str, Any]:
    """Measure lifecycle phases without placing first-call or warm-up in steady samples."""

    request.validate()
    consumer = consume_result or ResultConsumer()

    load_start = int(clock_ns())
    try:
        loaded_object = loader()
    except Exception as exc:
        error = _safe_failure("load", type(exc).__name__)
        return _failed_lifecycle(
            load_time=_duration_result(None, {"outcome": "failure", "error": error}),
            first_call=_duration_result(
                None,
                {"outcome": "failure", "error": _safe_failure("first_call", "NotRun")},
            ),
            steady_state=_empty_steady(_safe_failure("steady_state", "NotRun")),
            error=error,
        )
    load_end = int(clock_ns())
    load_duration = load_end - load_start
    if load_duration < 0:
        error = _safe_failure("load", "NonMonotonicClockError")
        return _failed_lifecycle(
            load_time=_duration_result(None, {"outcome": "failure", "error": error}),
            first_call=_duration_result(None, {"outcome": "failure", "error": error}),
            steady_state=_empty_steady(error),
            error=error,
        )
    load_result = _duration_result(load_duration, {"outcome": "success", "error": None})

    first_start = int(clock_ns())
    try:
        first_result = inference(loaded_object, input_data)
    except Exception as exc:
        error = _safe_failure("first_call", type(exc).__name__)
        return _failed_lifecycle(
            load_time=load_result,
            first_call=_duration_result(None, {"outcome": "failure", "error": error}),
            steady_state=_empty_steady(_safe_failure("steady_state", "NotRun")),
            error=error,
        )
    first_end = int(clock_ns())
    first_duration = first_end - first_start
    if first_duration < 0:
        error = _safe_failure("first_call", "NonMonotonicClockError")
        return _failed_lifecycle(
            load_time=load_result,
            first_call=_duration_result(None, {"outcome": "failure", "error": error}),
            steady_state=_empty_steady(error),
            error=error,
        )
    try:
        consumer(first_result)
    except Exception as exc:
        error = _safe_failure("first_call_result_consumption", type(exc).__name__)
        return _failed_lifecycle(
            load_time=load_result,
            first_call=_duration_result(None, {"outcome": "failure", "error": error}),
            steady_state=_empty_steady(_safe_failure("steady_state", "NotRun")),
            error=error,
        )
    first_result_section = _duration_result(
        first_duration, {"outcome": "success", "error": None}
    )

    try:
        statistics = measure_callable(
            lambda: inference(loaded_object, input_data),
            warmup=request.warmup,
            iterations=request.iterations,
            clock_ns=clock_ns,
            consume_result=consumer,
        )
    except BenchmarkExecutionError as exc:
        stage = "warmup" if exc.stage.startswith("warmup") else "steady_state"
        error = _safe_failure(stage, exc.error_type)
        return _failed_lifecycle(
            load_time=load_result,
            first_call=first_result_section,
            steady_state=_empty_steady(error),
            error=error,
        )

    steady_state = {
        "timing": timing_section(statistics),
        "throughput": calculate_throughput(
            statistics,
            items_per_call=request.items_per_call,
            item_duration_seconds=request.item_duration_seconds,
        ),
        "status": {"outcome": "success", "error": None},
    }
    return {
        "load_time": load_result,
        "first_call_latency": first_result_section,
        "steady_state": steady_state,
        "status": {"outcome": "success", "error": None},
    }

