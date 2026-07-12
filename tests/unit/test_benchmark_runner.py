from __future__ import annotations

import json
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

from urbansound_segment_task.edge_v2.benchmarks.runner import (
    BenchmarkRequest,
    run_benchmark,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPOSITORY_ROOT / "scripts" / "run_synthetic_benchmark.py"
FIXED_TIME = datetime(2026, 7, 12, 12, 30, tzinfo=timezone.utc)
SAFE_ENVIRONMENT = {
    "os": "TestOS",
    "os_version": "1.0",
    "architecture": "test64",
    "python_version": "3.12.0",
}


class FakeClock:
    def __init__(self, values: list[int]) -> None:
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


def request(**changes: object) -> BenchmarkRequest:
    values = {
        "name": "synthetic-test",
        "description": "Safe synthetic test.",
        "warmup": 0,
        "iterations": 2,
        "items_per_call": 3,
        "item_duration_seconds": None,
    }
    values.update(changes)
    return BenchmarkRequest(**values)


class BenchmarkRunnerTests(unittest.TestCase):
    def test_calls_and_items_per_second(self) -> None:
        document = run_benchmark(
            lambda: 1,
            request(),
            clock_ns=FakeClock([0, 100_000_000, 200_000_000, 400_000_000]),
            environment_collector=lambda: SAFE_ENVIRONMENT,
            now=FIXED_TIME,
        )

        self.assertAlmostEqual(document["throughput"]["calls_per_second"], 2 / 0.3)
        self.assertAlmostEqual(document["throughput"]["items_per_second"], 20.0)

    def test_real_time_factor(self) -> None:
        document = run_benchmark(
            lambda: 1,
            request(item_duration_seconds=0.5),
            clock_ns=FakeClock([0, 100_000_000, 200_000_000, 400_000_000]),
            environment_collector=lambda: SAFE_ENVIRONMENT,
        )

        self.assertAlmostEqual(document["throughput"]["real_time_factor"], 10.0)

    def test_invalid_request_values_are_rejected(self) -> None:
        for invalid in (
            request(warmup=-1),
            request(iterations=0),
            request(items_per_call=0),
            request(item_duration_seconds=0.0),
            request(item_duration_seconds=float("inf")),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    run_benchmark(lambda: 1, invalid)

    def test_callable_failure_has_no_success_metrics(self) -> None:
        def fail() -> None:
            raise ValueError("secret local failure details")

        document = run_benchmark(
            fail,
            request(iterations=1),
            clock_ns=FakeClock([1]),
            environment_collector=lambda: SAFE_ENVIRONMENT,
        )

        self.assertEqual(document["status"]["outcome"], "failure")
        self.assertEqual(document["status"]["error"]["stage"], "measurement")
        self.assertEqual(document["status"]["error"]["type"], "ValueError")
        self.assertEqual(document["timing"]["raw_samples"], [])
        self.assertIsNone(document["timing"]["mean"])
        self.assertIsNone(document["throughput"]["calls_per_second"])
        self.assertNotIn("secret", json.dumps(document).lower())

    def test_cli_help_smoke(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "--help"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "--warmup",
            "--iterations",
            "--items-per-call",
            "--item-duration-seconds",
            "--output",
            "--pretty",
            "--workload",
        ):
            self.assertIn(option, result.stdout)

    def test_synthetic_cli_produces_parseable_result(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "--workload",
                "noop",
                "--warmup",
                "0",
                "--iterations",
                "2",
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = json.loads(result.stdout)
        self.assertEqual(parsed["schema_version"], "edge-v2.benchmark.v1")
        self.assertEqual(parsed["status"]["outcome"], "success")
        self.assertEqual(len(parsed["timing"]["raw_samples"]), 2)


if __name__ == "__main__":
    unittest.main()
