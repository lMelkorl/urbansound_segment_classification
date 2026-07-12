"""Single-callable timing using an injectable monotonic nanosecond clock."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence


TIMER_SOURCE = "time.perf_counter_ns"
PERCENTILE_METHOD = "linear_interpolation_at_rank_(n-1)*q"


class BenchmarkExecutionError(RuntimeError):
    """A sanitized execution failure with no original exception message."""

    def __init__(self, stage: str, iteration_index: int, error_type: str) -> None:
        super().__init__("benchmark execution failed")
        self.stage = stage
        self.iteration_index = iteration_index
        self.error_type = error_type


class ResultConsumer:
    """Retain each result after timing so the return value is observably used."""

    def __init__(self) -> None:
        self.last_result: Any = None
        self.consumed_count = 0

    def __call__(self, result: Any) -> None:
        self.last_result = result
        self.consumed_count += 1


@dataclass(frozen=True)
class TimingStatistics:
    raw_samples_ns: tuple[int, ...]
    p50: float
    p95: float
    p99: float
    mean: float
    minimum: int
    maximum: int
    standard_deviation: float


def validate_timer_inputs(warmup: int, iterations: int) -> None:
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be an integer greater than or equal to 1")


def percentile(samples: Sequence[int], quantile: float) -> float:
    """Return a percentile using linear interpolation at rank ``(n-1)*q``.

    This is the common R-7/NumPy-default style definition. A single sample is
    returned unchanged for every quantile.
    """

    if not samples:
        raise ValueError("at least one sample is required")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(int(sample) for sample in samples)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    fraction = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def summarize_samples(samples: Sequence[int]) -> TimingStatistics:
    if not samples:
        raise ValueError("at least one timing sample is required")
    normalized = tuple(int(sample) for sample in samples)
    if any(sample < 0 for sample in normalized):
        raise ValueError("timing samples cannot be negative")
    return TimingStatistics(
        raw_samples_ns=normalized,
        p50=percentile(normalized, 0.50),
        p95=percentile(normalized, 0.95),
        p99=percentile(normalized, 0.99),
        mean=float(statistics.fmean(normalized)),
        minimum=min(normalized),
        maximum=max(normalized),
        standard_deviation=float(statistics.pstdev(normalized)),
    )


def measure_callable(
    function: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    consume_result: Optional[Callable[[Any], None]] = None,
) -> TimingStatistics:
    """Warm up and measure one callable once per recorded sample.

    Warm-up calls are consumed but never timed or recorded. Each measured call
    is surrounded by exactly two clock reads. Its result is consumed only after
    the end timestamp, keeping result consumption outside the timed region.
    """

    validate_timer_inputs(warmup, iterations)
    consumer = consume_result or ResultConsumer()

    for index in range(warmup):
        try:
            result = function()
        except Exception as exc:
            raise BenchmarkExecutionError("warmup", index, type(exc).__name__) from None
        try:
            consumer(result)
        except Exception as exc:
            raise BenchmarkExecutionError("warmup_result_consumption", index, type(exc).__name__) from None

    samples: list[int] = []
    for index in range(iterations):
        start_ns = int(clock_ns())
        try:
            result = function()
        except Exception as exc:
            raise BenchmarkExecutionError("measurement", index, type(exc).__name__) from None
        end_ns = int(clock_ns())
        duration_ns = end_ns - start_ns
        if duration_ns < 0:
            raise BenchmarkExecutionError("timer", index, "NonMonotonicClockError")
        samples.append(duration_ns)
        try:
            consumer(result)
        except Exception as exc:
            raise BenchmarkExecutionError(
                "measurement_result_consumption", index, type(exc).__name__
            ) from None

    return summarize_samples(samples)
