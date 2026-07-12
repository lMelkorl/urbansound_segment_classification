from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests.unit.urbansound_test_fixtures import COLUMNS, create_dataset, rewrite_rows
from urbansound_segment_task.edge_v2.data.urbansound8k import (
    DATASET_SCHEMA_VERSION, DatasetLayoutError, DatasetValidationError,
    detect_dataset_layout, inspect_urbansound8k,
)


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class UrbanSoundManifestTests(unittest.TestCase):
    def test_layout_and_inventory_headers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = create_dataset(Path(directory))
            layout = detect_dataset_layout(root)
            document = inspect_urbansound8k(root, now=NOW)
        self.assertEqual(layout.metadata_csv.name, "UrbanSound8K.csv")
        self.assertEqual(document["schema_version"], DATASET_SCHEMA_VERSION)
        self.assertEqual(document["counts"]["metadata_clips"], 10)
        self.assertEqual(document["counts"]["existing_audio_files"], 10)
        self.assertEqual(document["clips"][0]["audio_header"]["sample_rate"], 16_000)

    def test_missing_required_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = create_dataset(Path(directory), folds=[1])
            rewrite_rows(root, [{key: "x" for key in COLUMNS[:-1]}], COLUMNS[:-1])
            with self.assertRaises(DatasetValidationError):
                inspect_urbansound8k(root)

    def test_invalid_fold_and_class_id_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = create_dataset(Path(directory), folds=[1])
            rows = [{
                "slice_file_name": "1-fixture.wav", "fsID": "1", "start": "0", "end": "1",
                "salience": "1", "fold": "11", "classID": "10", "class": "bad",
            }]
            rewrite_rows(root, rows)
            document = inspect_urbansound8k(root)
        types = {error["type"] for error in document["validation_errors"]}
        self.assertIn("InvalidFold", types)
        self.assertIn("InvalidClassID", types)

    def test_inconsistent_class_and_duplicate_clip_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = create_dataset(Path(directory), folds=[1])
            base = {
                "slice_file_name": "1-fixture.wav", "fsID": "1", "start": "0", "end": "1",
                "salience": "1", "fold": "1", "classID": "0", "class": "first",
            }
            rewrite_rows(root, [base, {**base, "class": "second"}])
            document = inspect_urbansound8k(root)
        types = {error["type"] for error in document["validation_errors"]}
        self.assertIn("InconsistentClassName", types)
        self.assertIn("DuplicateClipKey", types)

    def test_missing_and_extra_audio_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = create_dataset(Path(directory), folds=[1])
            (root / "audio" / "fold1" / "1-fixture.wav").unlink()
            (root / "audio" / "fold1" / "extra.wav").write_bytes(b"extra")
            document = inspect_urbansound8k(root)
        self.assertEqual(document["missing_audio"], ["fold1/1-fixture.wav"])
        self.assertEqual(document["extra_audio"], ["fold1/extra.wav"])

    def test_manifest_hash_is_deterministic_across_timestamps_and_has_no_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = create_dataset(Path(directory), folds=[1])
            first = inspect_urbansound8k(root, now=NOW)
            second = inspect_urbansound8k(root, now=datetime(2027, 1, 1, tzinfo=timezone.utc))
            serialized = json.dumps(first)
            self.assertNotIn(str(root), serialized)
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])

    def test_missing_layout_is_controlled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(DatasetLayoutError):
                detect_dataset_layout(Path(directory))


if __name__ == "__main__":
    unittest.main()
