from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from urbansound_segment_task.edge_v2.benchmarks.process_runner import (
    LIFECYCLE_SCHEMA_VERSION,
    LifecycleProcessConfig,
    run_lifecycle_in_fresh_process,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPOSITORY_ROOT / "scripts" / "run_synthetic_lifecycle_benchmark.py"


def small_config(**changes: object) -> LifecycleProcessConfig:
    values = {
        "threads": 2,
        "warmup": 1,
        "iterations": 2,
        "load_size": 1_000,
        "work_size": 100,
    }
    values.update(changes)
    return LifecycleProcessConfig(**values)


class BenchmarkProcessRunnerTests(unittest.TestCase):
    def test_fresh_spawn_process_runs_synthetic_benchmark(self) -> None:
        before = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
        result = run_lifecycle_in_fresh_process(small_config(), timeout_seconds=10)
        after = {name: os.environ.get(name) for name in before}

        self.assertEqual(result["schema_version"], LIFECYCLE_SCHEMA_VERSION)
        self.assertEqual(result["status"]["outcome"], "success")
        self.assertEqual(result["execution_policy"]["multiprocessing_start_method"], "spawn")
        self.assertTrue(
            all(
                value == "2"
                for value in result["execution_policy"]["effective_thread_environment"].values()
            )
        )
        self.assertEqual(
            len(result["lifecycle"]["steady_state"]["timing"]["raw_samples"]), 2
        )
        self.assertEqual(after, before)

    def test_child_timeout_is_structured(self) -> None:
        result = run_lifecycle_in_fresh_process(
            small_config(workload="test_busy", test_busy_seconds=1.0),
            timeout_seconds=0.1,
        )

        self.assertEqual(result["status"]["outcome"], "failure")
        self.assertEqual(result["status"]["error"]["type"], "ChildProcessTimeout")

    def test_child_crash_is_distinct(self) -> None:
        result = run_lifecycle_in_fresh_process(
            small_config(workload="test_crash"),
            timeout_seconds=5,
        )

        self.assertEqual(result["status"]["outcome"], "failure")
        self.assertEqual(result["status"]["error"]["type"], "ChildProcessCrash")
        self.assertEqual(result["status"]["error"]["exit_code"], 70)

    def test_child_nonzero_exit_is_distinct(self) -> None:
        result = run_lifecycle_in_fresh_process(
            small_config(workload="test_nonzero"),
            timeout_seconds=5,
        )

        self.assertEqual(result["status"]["outcome"], "failure")
        self.assertEqual(result["status"]["error"]["type"], "ChildProcessNonZeroExit")
        self.assertEqual(result["status"]["error"]["exit_code"], 7)

    def test_json_round_trip_has_no_sensitive_paths(self) -> None:
        result = run_lifecycle_in_fresh_process(small_config(), timeout_seconds=10)
        serialized = json.dumps(result, sort_keys=True)
        parsed = json.loads(serialized)

        self.assertEqual(parsed["schema_version"], LIFECYCLE_SCHEMA_VERSION)
        for forbidden in ("/Users/", "hostname", "username", "email", "secret", "token"):
            self.assertNotIn(forbidden.lower(), serialized.lower())

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
            "--threads",
            "--warmup",
            "--iterations",
            "--items-per-call",
            "--item-duration-seconds",
            "--load-size",
            "--work-size",
            "--timeout-seconds",
            "--output",
            "--pretty",
        ):
            self.assertIn(option, result.stdout)

    def test_cli_output_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lifecycle.json"
            command = [
                sys.executable,
                str(CLI_PATH),
                "--warmup",
                "0",
                "--iterations",
                "1",
                "--load-size",
                "100",
                "--work-size",
                "10",
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
            self.assertEqual(output.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
