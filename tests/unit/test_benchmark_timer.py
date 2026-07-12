from __future__ import annotations

import math
import unittest

from urbansound_segment_task.edge_v2.benchmarks.timer import (
    BenchmarkExecutionError,
    ResultConsumer,
    measure_callable,
    percentile,
    summarize_samples,
)


class FakeClock:
    def __init__(self, values: list[int]) -> None:
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


class BenchmarkTimerTests(unittest.TestCase):
    def test_warmup_samples_are_not_recorded(self) -> None:
        calls: list[int] = []
        consumer = ResultConsumer()

        statistics = measure_callable(
            lambda: calls.append(len(calls)) or len(calls),
            warmup=2,
            iterations=2,
            clock_ns=FakeClock([100, 110, 200, 225]),
            consume_result=consumer,
        )

        self.assertEqual(len(calls), 4)
        self.assertEqual(consumer.consumed_count, 4)
        self.assertEqual(statistics.raw_samples_ns, (10, 25))

    def test_injected_clock_produces_exact_samples(self) -> None:
        statistics = measure_callable(
            lambda: 7,
            warmup=0,
            iterations=3,
            clock_ns=FakeClock([1, 11, 20, 50, 100, 145]),
        )

        self.assertEqual(statistics.raw_samples_ns, (10, 30, 45))

    def test_percentiles_use_fixed_linear_interpolation(self) -> None:
        samples = [10, 20, 30, 40]

        self.assertEqual(percentile(samples, 0.50), 25.0)
        self.assertAlmostEqual(percentile(samples, 0.95), 38.5)
        self.assertAlmostEqual(percentile(samples, 0.99), 39.7)

    def test_single_sample_statistics(self) -> None:
        statistics = summarize_samples([123])

        self.assertEqual(statistics.p50, 123.0)
        self.assertEqual(statistics.p95, 123.0)
        self.assertEqual(statistics.p99, 123.0)
        self.assertEqual(statistics.standard_deviation, 0.0)

    def test_mean_min_max_and_population_standard_deviation(self) -> None:
        statistics = summarize_samples([10, 20, 30, 40])

        self.assertEqual(statistics.mean, 25.0)
        self.assertEqual(statistics.minimum, 10)
        self.assertEqual(statistics.maximum, 40)
        self.assertTrue(math.isclose(statistics.standard_deviation, math.sqrt(125.0)))

    def test_invalid_timer_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            measure_callable(lambda: None, warmup=-1, iterations=1)
        with self.assertRaises(ValueError):
            measure_callable(lambda: None, warmup=0, iterations=0)

    def test_callable_exception_identifies_stage_and_iteration(self) -> None:
        def fail() -> None:
            raise RuntimeError("local path and secret must not be serialized")

        with self.assertRaises(BenchmarkExecutionError) as caught:
            measure_callable(fail, warmup=0, iterations=1, clock_ns=FakeClock([1]))

        self.assertEqual(caught.exception.stage, "measurement")
        self.assertEqual(caught.exception.iteration_index, 0)
        self.assertEqual(caught.exception.error_type, "RuntimeError")
        self.assertNotIn("secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
