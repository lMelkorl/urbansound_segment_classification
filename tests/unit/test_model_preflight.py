from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from urbansound_segment_task.edge_v2.benchmarks.schema import (
    serialize_document,
    write_document_atomic,
)
from urbansound_segment_task.edge_v2.models.base import ArtifactSpec, DependencySpec
from urbansound_segment_task.edge_v2.models.preflight import (
    MODEL_PREFLIGHT_SCHEMA_VERSION,
    PreflightInspector,
    run_model_preflight,
)

from tests.unit.model_preflight_fixtures import (
    git_stage,
    missing_packages,
    no_versions,
    write_fixture,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPOSITORY_ROOT / "scripts" / "run_model_preflight.py"


class ModelPreflightCoreTests(unittest.TestCase):
    def test_missing_optional_dependency_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inspector = PreflightInspector(
                Path(directory), find_spec=lambda _name: False, version_lookup=lambda _name: None
            )
            result = inspector.dependency(
                DependencySpec("optional-package", "optional_package", required=False)
            )

        self.assertFalse(result["installed"])
        self.assertIsNone(result["version"])
        self.assertFalse(result["required"])

    def test_installed_dependency_version_is_read_without_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(importlib, "import_module") as import_module:
                inspector = PreflightInspector(
                    Path(directory),
                    find_spec=lambda name: name == "example_import",
                    version_lookup=lambda name: "1.2.3" if name == "example-package" else None,
                )
                result = inspector.dependency(
                    DependencySpec("example-package", "example_import")
                )

        self.assertTrue(result["installed"])
        self.assertEqual(result["version"], "1.2.3")
        import_module.assert_not_called()

    def test_existing_artifact_has_streaming_sha256(self) -> None:
        content = b"small deterministic artifact"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "artifacts" / "model.bin"
            path.parent.mkdir(parents=True)
            path.write_bytes(content)
            inspector = PreflightInspector(root)
            result = inspector.artifact(
                ArtifactSpec("model", "artifacts/model.bin", "file")
            )

        self.assertEqual(result["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(result["size_bytes"], len(content))
        self.assertEqual(result["provenance_status"], "present_unverified")

    def test_missing_artifact_has_no_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = PreflightInspector(Path(directory)).artifact(
                ArtifactSpec("missing", "artifacts/missing.bin", "file")
            )

        self.assertFalse(result["exists"])
        self.assertIsNone(result["sha256"])
        self.assertEqual(result["provenance_status"], "missing")

    def test_summary_and_recommended_method_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            result = run_model_preflight(
                root,
                find_spec=missing_packages,
                version_lookup=no_versions,
                git_stage_lookup=git_stage,
            )

        self.assertEqual(result["schema_version"], MODEL_PREFLIGHT_SCHEMA_VERSION)
        self.assertEqual(result["summary"]["blocked_count"], 2)
        self.assertEqual(result["summary"]["unavailable_count"], 1)
        self.assertEqual(result["summary"]["recommended_first_method"], "yamnet_lgbm")

    def test_method_filter_returns_only_requested_method(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            result = run_model_preflight(
                root,
                method="audioclip",
                find_spec=missing_packages,
                version_lookup=no_versions,
            )

        self.assertEqual([method["method_id"] for method in result["methods"]], ["audioclip"])
        self.assertIsNone(result["summary"]["recommended_first_method"])

    def test_json_round_trip_does_not_expose_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            result = run_model_preflight(
                root,
                find_spec=missing_packages,
                version_lookup=no_versions,
                git_stage_lookup=git_stage,
            )
            serialized = serialize_document(result, pretty=True)
            parsed = json.loads(serialized)

        self.assertEqual(parsed["schema_version"], MODEL_PREFLIGHT_SCHEMA_VERSION)
        self.assertNotIn(str(root), serialized)
        for forbidden in ("hostname", "username", "email", "secret", "token"):
            self.assertNotIn(forbidden, serialized.lower())

    def test_atomic_output_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "preflight.json"
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
        for option in ("--method", "--repo-root", "--output", "--pretty"):
            self.assertIn(option, result.stdout)

    def test_cli_method_filter_output_and_overwrite_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            write_fixture(root)
            output = Path(directory) / "output" / "preflight.json"
            command = [
                sys.executable,
                str(CLI_PATH),
                "--method",
                "audioclip",
                "--repo-root",
                str(root),
                "--output",
                str(output),
                "--pretty",
            ]
            first = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            parsed = json.loads(output.read_text(encoding="utf-8"))
            original = output.read_bytes()
            second = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            after = output.read_bytes()

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual([item["method_id"] for item in parsed["methods"]], ["audioclip"])
        self.assertEqual(second.returncode, 2)
        self.assertIn("refusing to overwrite", second.stderr)
        self.assertEqual(after, original)


if __name__ == "__main__":
    unittest.main()
