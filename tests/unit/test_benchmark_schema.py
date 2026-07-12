from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from urbansound_segment_task.edge_v2.benchmarks.schema import (
    SCHEMA_VERSION,
    build_document,
    serialize_document,
    write_document_atomic,
)
from urbansound_segment_task.edge_v2.benchmarks.timer import summarize_samples


FIXED_TIME = datetime(2026, 7, 12, 12, 30, tzinfo=timezone.utc)


def successful_document() -> dict[str, object]:
    return build_document(
        name="synthetic-test",
        description="Safe synthetic test.",
        warmup=1,
        iterations=2,
        items_per_call=1,
        item_duration_seconds=None,
        statistics=summarize_samples([10, 20]),
        throughput={
            "calls_per_second": 1.0,
            "items_per_second": 1.0,
            "real_time_factor": None,
        },
        status={"outcome": "success", "error": None},
        environment={
            "os": "TestOS",
            "os_version": "1.0",
            "architecture": "test64",
            "python_version": "3.12.0",
            "username": "private-user",
            "project_path": "/Users/private-user/private-project",
        },
        now=FIXED_TIME,
    )


class BenchmarkSchemaTests(unittest.TestCase):
    def test_json_round_trip(self) -> None:
        serialized = serialize_document(successful_document(), pretty=True)
        parsed = json.loads(serialized)

        self.assertEqual(parsed["schema_version"], SCHEMA_VERSION)
        self.assertEqual(parsed["created_at_utc"], "2026-07-12T12:30:00Z")
        self.assertEqual(parsed["timing"]["raw_samples"], [10, 20])

    def test_document_contains_no_sensitive_field_or_absolute_path(self) -> None:
        serialized = serialize_document(successful_document()).lower()

        for forbidden in (
            "/users/",
            "c:\\\\users\\",
            '"hostname"',
            '"username"',
            '"email"',
            '"token"',
            '"credential"',
            '"home"',
            "private-user",
            "private-project",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_atomic_writer_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "benchmark.json"
            write_document_atomic(output, "first\n")

            with self.assertRaises(FileExistsError):
                write_document_atomic(output, "second\n")

            self.assertEqual(output.read_text(encoding="utf-8"), "first\n")
            self.assertEqual(list(output.parent.glob(".benchmark-*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
