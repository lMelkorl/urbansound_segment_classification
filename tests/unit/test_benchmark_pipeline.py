from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from urbansound_segment_task.edge_v2.benchmarks.pipeline import (
    PIPELINE_SCHEMA_VERSION,
    PipelineBenchmarkRequest,
    run_pipeline_benchmark,
)
from urbansound_segment_task.edge_v2.benchmarks.schema import (
    serialize_document,
    write_document_atomic,
)
from urbansound_segment_task.edge_v2.benchmarks.stages import PipelineStages


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPOSITORY_ROOT / "scripts" / "run_synthetic_pipeline_benchmark.py"
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


def simple_stages(calls: Optional[list[str]] = None) -> PipelineStages:
    call_log = calls if calls is not None else []

    def decode(value: int) -> int:
        call_log.append("decode")
        return value + 1

    def preprocess(value: int) -> int:
        call_log.append("preprocess")
        return value * 2

    def inference(value: int) -> int:
        call_log.append("inference")
        return value + 3

    def aggregate(value: int) -> int:
        call_log.append("aggregate")
        return value // 2

    return PipelineStages(decode, preprocess, inference, aggregate)


def request(**changes: object) -> PipelineBenchmarkRequest:
    values = {
        "name": "synthetic-pipeline-test",
        "description": "Safe synthetic pipeline test.",
        "warmup": 0,
        "iterations": 1,
        "items_per_call": 1,
        "item_duration_seconds": None,
    }
    values.update(changes)
    return PipelineBenchmarkRequest(**values)


class BenchmarkPipelineTests(unittest.TestCase):
    def test_clock_injection_stage_samples_and_comparison(self) -> None:
        document = run_pipeline_benchmark(
            simple_stages(),
            1,
            request(),
            clock_ns=FakeClock([0, 10, 100, 120, 200, 230, 300, 340, 400, 520]),
            environment_collector=lambda: SAFE_ENVIRONMENT,
            now=FIXED_TIME,
        )

        expected = {
            "decode": [10],
            "preprocess": [20],
            "inference": [30],
            "aggregate": [40],
            "end_to_end": [120],
        }
        for stage_name, samples in expected.items():
            self.assertEqual(document["stages"][stage_name]["timing"]["raw_samples"], samples)
        self.assertEqual(document["stage_sum"]["mean_nanoseconds"], 100.0)
        self.assertEqual(document["end_to_end_overhead"]["nanoseconds"], 20.0)
        self.assertEqual(document["end_to_end_overhead"]["percent"], 20.0)

    def test_upstream_preparation_is_outside_downstream_timers(self) -> None:
        calls: list[str] = []
        document = run_pipeline_benchmark(
            simple_stages(calls),
            1,
            request(warmup=1, iterations=2),
            clock_ns=FakeClock(list(range(0, 200, 10))),
            environment_collector=lambda: SAFE_ENVIRONMENT,
        )

        for stage_name in ("decode", "preprocess", "inference", "aggregate"):
            self.assertEqual(calls.count(stage_name), 6)
            self.assertEqual(len(document["stages"][stage_name]["timing"]["raw_samples"]), 2)
        self.assertEqual(len(document["stages"]["end_to_end"]["timing"]["raw_samples"]), 2)

    def test_negative_overhead_is_reported_without_failure(self) -> None:
        document = run_pipeline_benchmark(
            simple_stages(),
            1,
            request(),
            clock_ns=FakeClock([0, 10, 20, 40, 50, 80, 90, 130, 140, 190]),
            environment_collector=lambda: SAFE_ENVIRONMENT,
        )

        self.assertEqual(document["stage_sum"]["mean_nanoseconds"], 100.0)
        self.assertEqual(document["end_to_end_overhead"]["nanoseconds"], -50.0)
        self.assertEqual(document["end_to_end_overhead"]["percent"], -50.0)
        self.assertEqual(document["status"]["outcome"], "success")

    def test_stage_exception_is_named_and_has_no_misleading_metrics(self) -> None:
        stages = simple_stages()

        def fail_preprocess(_value: int) -> int:
            raise RuntimeError("secret /Users/private failure")

        stages = PipelineStages(stages.decode, fail_preprocess, stages.inference, stages.aggregate)
        document = run_pipeline_benchmark(
            stages,
            1,
            request(),
            clock_ns=FakeClock([0, 10, 20, 30, 40, 50]),
            environment_collector=lambda: SAFE_ENVIRONMENT,
        )

        preprocess = document["stages"]["preprocess"]
        self.assertEqual(preprocess["status"]["error"]["pipeline_stage"], "preprocess")
        self.assertEqual(preprocess["timing"]["raw_samples"], [])
        self.assertIsNone(preprocess["throughput"]["calls_per_second"])
        self.assertEqual(
            document["stages"]["end_to_end"]["status"]["error"]["pipeline_stage"],
            "preprocess",
        )
        self.assertIsNone(document["stage_sum"]["mean_nanoseconds"])
        self.assertEqual(document["status"]["outcome"], "failure")
        self.assertNotIn("secret", json.dumps(document).lower())
        self.assertNotIn("/users/", json.dumps(document).lower())

    def test_json_round_trip_and_safe_environment_allowlist(self) -> None:
        unsafe_environment = dict(SAFE_ENVIRONMENT)
        unsafe_environment.update({"username": "private-user", "path": "/Users/private"})
        document = run_pipeline_benchmark(
            simple_stages(),
            1,
            request(),
            clock_ns=FakeClock([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            environment_collector=lambda: unsafe_environment,
            now=FIXED_TIME,
        )
        parsed = json.loads(serialize_document(document, pretty=True))

        self.assertEqual(parsed["schema_version"], PIPELINE_SCHEMA_VERSION)
        self.assertEqual(parsed["created_at_utc"], "2026-07-12T12:30:00Z")
        self.assertNotIn("private", json.dumps(parsed).lower())

    def test_atomic_output_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "pipeline.json"
            write_document_atomic(output, "first\n")
            with self.assertRaises(FileExistsError):
                write_document_atomic(output, "second\n")
            self.assertEqual(output.read_text(encoding="utf-8"), "first\n")

    def test_cli_help(self) -> None:
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
            "--input-size",
            "--output",
            "--pretty",
        ):
            self.assertIn(option, result.stdout)

    def test_cli_output_is_parseable(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "--warmup",
                "0",
                "--iterations",
                "1",
                "--input-size",
                "16",
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = json.loads(result.stdout)
        self.assertEqual(parsed["schema_version"], PIPELINE_SCHEMA_VERSION)
        self.assertEqual(parsed["pipeline"]["stage_order"][-1], "end_to_end")

    def test_cli_refuses_to_overwrite_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pipeline.json"
            command = [
                sys.executable,
                str(CLI_PATH),
                "--warmup",
                "0",
                "--iterations",
                "1",
                "--input-size",
                "8",
                "--output",
                str(output),
            ]
            first = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            original = output.read_text(encoding="utf-8")
            second = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 2)
            self.assertIn("refusing to overwrite", second.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
