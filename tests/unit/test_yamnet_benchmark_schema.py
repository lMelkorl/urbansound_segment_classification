from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from urbansound_segment_task.edge_v2.benchmarks.schema import serialize_document
from urbansound_segment_task.edge_v2.benchmarks.yamnet_cpu import (
    SELECTION_RULE,
    YAMNET_CPU_SCHEMA_VERSION,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI = REPOSITORY_ROOT / "scripts" / "run_yamnet_cpu_benchmark.py"


class YamnetBenchmarkSchemaTests(unittest.TestCase):
    def test_cli_help(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CLI), "--help"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "--artifact",
            "--threads",
            "--warmup",
            "--iterations",
            "--repetitions",
            "--sample-count",
            "--thread-sweep",
            "--input-sensitivity",
            "--timeout-seconds",
            "--output",
            "--pretty",
        ):
            self.assertIn(option, result.stdout)

    def test_minimal_schema_round_trip_and_no_sensitive_paths(self) -> None:
        document = {
            "schema_version": YAMNET_CPU_SCHEMA_VERSION,
            "created_at_utc": "2026-07-12T00:00:00Z",
            "git_revision": "a" * 40,
            "artifact_identity": "yamnet-tfhub-v1",
            "artifact_tree_sha256": "b" * 64,
            "tensorflow_version": "2.15.1",
            "python_version": "3.11.15",
            "hardware": {"cpu_name": "Test CPU"},
            "input_identity": {"sha256": "c" * 64},
            "thread_sweep": [],
            "selected_thread_configuration": {"selection_rule": SELECTION_RULE},
            "input_length_sensitivity": [
                {"sample_count": 16_000, "frame_count": 2, "two_frame_compute_warning": True}
            ],
            "headline_result": {"sample_count": 15_360},
            "limitations": [],
            "status": {"outcome": "success", "error": None},
        }
        serialized = serialize_document(document, pretty=False)
        parsed = json.loads(serialized)

        self.assertEqual(parsed, document)
        lowered = serialized.lower()
        for forbidden in ("/users/", "hostname", "username", "email", "token", "credential"):
            self.assertNotIn(forbidden, lowered)
        self.assertTrue(parsed["input_length_sensitivity"][0]["two_frame_compute_warning"])


if __name__ == "__main__":
    unittest.main()
