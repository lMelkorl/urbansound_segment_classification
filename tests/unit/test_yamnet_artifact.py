from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from urbansound_segment_task.edge_v2.models.yamnet_artifact import (
    ArtifactVerificationError,
    NetworkPermissionRequired,
    acquire_yamnet_artifact,
    collect_model_files,
    deterministic_tree_sha256,
    streaming_file_sha256,
    verify_yamnet_artifact,
)
from tests.unit.yamnet_test_fixtures import FIXED_TIME, create_artifact


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ACQUIRE_CLI = REPOSITORY_ROOT / "scripts" / "acquire_yamnet_artifact.py"
VERIFY_CLI = REPOSITORY_ROOT / "scripts" / "verify_yamnet_artifact.py"


class YamnetArtifactTests(unittest.TestCase):
    def test_acquisition_requires_explicit_network_permission(self) -> None:
        resolver = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(NetworkPermissionRequired):
                acquire_yamnet_artifact(
                    Path(directory) / "artifact",
                    allow_network=False,
                    resolver=resolver,
                    package_versions={},
                )
        resolver.assert_not_called()

    def test_cli_refuses_without_network_flag_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ACQUIRE_CLI), "--output", str(Path(directory) / "artifact")],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("no network request was made", result.stderr)

    def test_streaming_sha256_and_tree_hash_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.bin").write_bytes(b"abc")
            (root / "b.bin").write_bytes(b"def")
            records = collect_model_files(root)

        self.assertEqual(streaming_file_sha256.__name__, "streaming_file_sha256")
        self.assertEqual(records[0]["sha256"], hashlib.sha256(b"abc").hexdigest())
        self.assertEqual(
            deterministic_tree_sha256(records),
            deterministic_tree_sha256(list(reversed(records))),
        )

    def test_manifest_contains_no_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            manifest = create_artifact(root)
            serialized = json.dumps(manifest, sort_keys=True)

        self.assertNotIn(str(root), serialized)
        self.assertTrue(all(item["relative_path"].startswith("model/") for item in manifest["files"]))

    def test_missing_file_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            create_artifact(root)
            (root / "model" / "saved_model.pb").unlink()
            with self.assertRaises(ArtifactVerificationError) as caught:
                verify_yamnet_artifact(root)
        self.assertEqual(caught.exception.code, "FILE_SET_MISMATCH")

    def test_changed_file_fails_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            create_artifact(root)
            (root / "model" / "saved_model.pb").write_bytes(b"changed-data")
            with self.assertRaises(ArtifactVerificationError) as caught:
                verify_yamnet_artifact(root)
        self.assertIn(caught.exception.code, ("FILE_SIZE_MISMATCH", "FILE_HASH_MISMATCH"))

    def test_existing_artifact_is_not_overwritten(self) -> None:
        resolver = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            output.mkdir()
            marker = output / "marker"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                acquire_yamnet_artifact(
                    output,
                    allow_network=True,
                    resolver=resolver,
                    package_versions={},
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        resolver.assert_not_called()

    def test_acquisition_publishes_verified_artifact_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resolved = root / "resolved"
            resolved.mkdir()
            (resolved / "saved_model.pb").write_bytes(b"model")
            output = root / "published"
            manifest = acquire_yamnet_artifact(
                output,
                allow_network=True,
                resolver=lambda: resolved,
                package_versions={"tensorflow": "2.15.1"},
                now=FIXED_TIME,
            )
            identity = verify_yamnet_artifact(output)

        self.assertEqual(identity["tree_sha256"], manifest["tree_sha256"])

    def test_artifact_cli_help(self) -> None:
        for script in (ACQUIRE_CLI, VERIFY_CLI):
            with self.subTest(script=script.name):
                result = subprocess.run(
                    [sys.executable, str(script), "--help"],
                    cwd=REPOSITORY_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
