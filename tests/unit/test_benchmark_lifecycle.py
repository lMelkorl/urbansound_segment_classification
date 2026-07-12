from __future__ import annotations

import unittest

from urbansound_segment_task.edge_v2.benchmarks.lifecycle import (
    LifecycleRequest,
    run_lifecycle,
)
from urbansound_segment_task.edge_v2.benchmarks.timer import ResultConsumer


class FakeClock:
    def __init__(self, values: list[int]) -> None:
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


def request(*, warmup: int = 1, iterations: int = 2) -> LifecycleRequest:
    return LifecycleRequest(
        name="synthetic-lifecycle-test",
        description="Safe lifecycle test.",
        warmup=warmup,
        iterations=iterations,
        items_per_call=2,
        item_duration_seconds=0.5,
    )


class BenchmarkLifecycleTests(unittest.TestCase):
    def test_load_first_call_warmup_and_steady_are_separate(self) -> None:
        loaded = [3, 4]
        calls: list[tuple[list[int], int]] = []
        consumer = ResultConsumer()

        def inference(model: list[int], value: int) -> int:
            calls.append((model, value))
            return sum(model) * value

        result = run_lifecycle(
            lambda: loaded,
            inference,
            2,
            request(),
            clock_ns=FakeClock([0, 10, 20, 50, 100, 110, 200, 220]),
            consume_result=consumer,
        )

        self.assertEqual(result["load_time"]["duration"], 10)
        self.assertEqual(result["first_call_latency"]["duration"], 30)
        self.assertEqual(result["steady_state"]["timing"]["raw_samples"], [10, 20])
        self.assertEqual(len(calls), 4)  # first call + one warm-up + two steady calls
        self.assertTrue(all(model is loaded for model, _ in calls))
        self.assertEqual(consumer.last_result, 14)

    def test_loader_failure_reports_load_stage(self) -> None:
        def fail_loader() -> None:
            raise RuntimeError("secret loader failure")

        result = run_lifecycle(
            fail_loader,
            lambda _model, _input: 1,
            None,
            request(warmup=0, iterations=1),
            clock_ns=FakeClock([0]),
        )

        self.assertEqual(result["status"]["error"]["stage"], "load")
        self.assertIsNone(result["load_time"]["duration"])
        self.assertEqual(result["steady_state"]["timing"]["raw_samples"], [])
        self.assertIsNone(result["steady_state"]["throughput"]["calls_per_second"])

    def test_first_call_failure_reports_first_call_stage(self) -> None:
        def fail_inference(_model: object, _input: object) -> None:
            raise ValueError("private first failure")

        result = run_lifecycle(
            lambda: object(),
            fail_inference,
            None,
            request(warmup=0, iterations=1),
            clock_ns=FakeClock([0, 10, 20]),
        )

        self.assertEqual(result["status"]["error"]["stage"], "first_call")
        self.assertIsNone(result["first_call_latency"]["duration"])
        self.assertEqual(result["steady_state"]["timing"]["raw_samples"], [])

    def test_steady_state_failure_reports_steady_stage(self) -> None:
        call_count = 0

        def inference(_model: object, _input: object) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise LookupError("private steady failure")
            return call_count

        result = run_lifecycle(
            lambda: object(),
            inference,
            None,
            request(warmup=0, iterations=1),
            clock_ns=FakeClock([0, 10, 20, 30, 40]),
        )

        self.assertEqual(result["status"]["error"]["stage"], "steady_state")
        self.assertEqual(result["status"]["error"]["type"], "LookupError")
        self.assertEqual(result["steady_state"]["timing"]["raw_samples"], [])
        self.assertIsNone(result["steady_state"]["throughput"]["items_per_second"])

    def test_warmup_failure_is_not_a_steady_sample(self) -> None:
        call_count = 0

        def inference(_model: object, _input: object) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ArithmeticError("warmup failed")
            return call_count

        result = run_lifecycle(
            lambda: object(),
            inference,
            None,
            request(warmup=1, iterations=1),
            clock_ns=FakeClock([0, 10, 20, 30]),
        )

        self.assertEqual(result["status"]["error"]["stage"], "warmup")
        self.assertEqual(result["steady_state"]["timing"]["raw_samples"], [])


if __name__ == "__main__":
    unittest.main()
