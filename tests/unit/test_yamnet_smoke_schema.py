from __future__ import annotations

import importlib
import json
import socket
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from urbansound_segment_task.edge_v2.models.yamnet_runtime import LoadedYamnet
from urbansound_segment_task.edge_v2.models.yamnet_smoke import (
    YAMNET_SMOKE_SCHEMA_VERSION,
    run_yamnet_smoke,
)
from tests.unit.yamnet_test_fixtures import FakeModel, FakeTensorFlow


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CLI = REPOSITORY_ROOT / "scripts" / "smoke_yamnet_local.py"


class FakeLightGBM:
    __version__ = "4.6.0"


def fake_loaded() -> LoadedYamnet:
    return LoadedYamnet(
        model=FakeModel(),
        tensorflow=FakeTensorFlow,
        loader_method="tf.saved_model.load",
        tensorflow_version="2.15.1",
        visible_devices=("CPU",),
        artifact_identity={
            "artifact_id": "yamnet-tfhub-v1",
            "tree_sha256": "a" * 64,
        },
    )


class YamnetSmokeSchemaTests(unittest.TestCase):
    def test_loader_failure_is_structured(self) -> None:
        def fail_loader(_path: Path) -> LoadedYamnet:
            raise RuntimeError("private local path")

        document = run_yamnet_smoke(
            Path("local-artifact"),
            runtime_loader=fail_loader,
            import_module=lambda _name: FakeLightGBM,
        )

        self.assertEqual(document["status"]["outcome"], "failure")
        self.assertEqual(document["status"]["error"], {"stage": "load", "type": "RuntimeError"})
        self.assertEqual(document["input_cases"], [])
        self.assertNotIn("private", json.dumps(document).lower())

    def test_smoke_schema_round_trip_and_input_cases(self) -> None:
        document = run_yamnet_smoke(
            Path("local-artifact"),
            runtime_loader=lambda _path: fake_loaded(),
            import_module=lambda name: FakeLightGBM if name == "lightgbm" else None,
            now=datetime(2026, 7, 12, 12, 30, tzinfo=timezone.utc),
        )
        parsed = json.loads(json.dumps(document, sort_keys=True))

        self.assertEqual(parsed["schema_version"], YAMNET_SMOKE_SCHEMA_VERSION)
        self.assertEqual([case["sample_count"] for case in parsed["input_cases"]], [15360, 15600, 16000])
        self.assertEqual(parsed["input_cases"][0]["frame_count"], 0)
        self.assertEqual(parsed["input_cases"][1]["scores_shape"], [1, 521])
        self.assertEqual(parsed["input_cases"][1]["embeddings_shape"], [1, 1024])
        self.assertTrue(parsed["lightgbm"]["importable"])

    def test_fake_offline_smoke_does_not_open_network(self) -> None:
        with mock.patch.object(socket, "socket", side_effect=AssertionError("network forbidden")):
            document = run_yamnet_smoke(
                Path("local-artifact"),
                runtime_loader=lambda _path: fake_loaded(),
                import_module=lambda _name: FakeLightGBM,
            )
        self.assertEqual(document["status"]["outcome"], "success")

    def test_smoke_document_contains_no_sensitive_fields(self) -> None:
        document = run_yamnet_smoke(
            Path("local-artifact"),
            runtime_loader=lambda _path: fake_loaded(),
            import_module=lambda _name: FakeLightGBM,
        )
        serialized = json.dumps(document).lower()
        for forbidden in ("/users/", "hostname", "username", "email", "credential", "token"):
            self.assertNotIn(forbidden, serialized)

    def test_smoke_cli_help_does_not_import_tensorflow(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SMOKE_CLI), "--help"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--artifact", result.stdout)


if __name__ == "__main__":
    unittest.main()
