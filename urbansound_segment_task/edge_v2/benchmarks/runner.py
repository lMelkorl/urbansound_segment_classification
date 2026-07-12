"""Orchestration for one callable and one aggregate timing result."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Optional

from .schema import (
    build_document,
    safe_environment_summary,
    safe_error,
    validate_benchmark_metadata,
)
from .timer import BenchmarkExecutionError, TimingStatistics, measure_callable


@dataclass(frozen=True)
class BenchmarkRequest:
    name: str
    description: str
    warmup: int = 3
    iterations: int = 20
    items_per_call: int = 1
    item_duration_seconds: Optional[float] = None

    def validate(self) -> None:
        if isinstance(self.warmup, bool) or not isinstance(self.warmup, int) or self.warmup < 0:
            raise ValueError("warmup must be a non-negative integer")
        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or self.iterations < 1
        ):
            raise ValueError("iterations must be an integer greater than or equal to 1")
        if (
            isinstance(self.items_per_call, bool)
            or not isinstance(self.items_per_call, int)
            or self.items_per_call < 1
        ):
            raise ValueError("items_per_call must be a positive integer")
        if self.item_duration_seconds is not None:
            if (
                isinstance(self.item_duration_seconds, bool)
                or not isinstance(self.item_duration_seconds, (int, float))
                or not math.isfinite(self.item_duration_seconds)
                or self.item_duration_seconds <= 0
            ):
                raise ValueError("item_duration_seconds must be finite and positive when provided")
        validate_benchmark_metadata(self.name, self.description)


def calculate_throughput(
    statistics: TimingStatistics,
    *,
    items_per_call: int,
    item_duration_seconds: Optional[float],
) -> dict[str, Optional[float]]:
    total_seconds = sum(statistics.raw_samples_ns) / 1_000_000_000.0
    if total_seconds <= 0:
        return {
            "calls_per_second": None,
            "items_per_second": None,
            "real_time_factor": None,
        }
    calls_per_second = len(statistics.raw_samples_ns) / total_seconds
    items_per_second = calls_per_second * items_per_call
    real_time_factor = (
        items_per_second * item_duration_seconds
        if item_duration_seconds is not None
        else None
    )
    return {
        "calls_per_second": calls_per_second,
        "items_per_second": items_per_second,
        "real_time_factor": real_time_factor,
    }


def run_benchmark(
    function: Callable[[], Any],
    request: BenchmarkRequest,
    *,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    consume_result: Optional[Callable[[Any], None]] = None,
    environment_collector: Callable[[], Mapping[str, Any]] = safe_environment_summary,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    request.validate()
    environment = environment_collector()
    try:
        statistics = measure_callable(
            function,
            warmup=request.warmup,
            iterations=request.iterations,
            clock_ns=clock_ns,
            consume_result=consume_result,
        )
    except BenchmarkExecutionError as exc:
        return build_document(
            name=request.name,
            description=request.description,
            warmup=request.warmup,
            iterations=request.iterations,
            items_per_call=request.items_per_call,
            item_duration_seconds=request.item_duration_seconds,
            statistics=None,
            throughput={
                "calls_per_second": None,
                "items_per_second": None,
                "real_time_factor": None,
            },
            status={
                "outcome": "failure",
                "error": safe_error(exc.error_type, exc.stage, exc.iteration_index),
            },
            environment=environment,
            now=now,
        )

    return build_document(
        name=request.name,
        description=request.description,
        warmup=request.warmup,
        iterations=request.iterations,
        items_per_call=request.items_per_call,
        item_duration_seconds=request.item_duration_seconds,
        statistics=statistics,
        throughput=calculate_throughput(
            statistics,
            items_per_call=request.items_per_call,
            item_duration_seconds=request.item_duration_seconds,
        ),
        status={"outcome": "success", "error": None},
        environment=environment,
        now=now,
    )
