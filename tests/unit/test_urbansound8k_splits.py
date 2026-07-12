from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests.unit.urbansound_test_fixtures import create_dataset
from urbansound_segment_task.edge_v2.data.splits import (
    SPLIT_POLICY_ID, build_all_split_manifests, build_split_manifest, split_folds,
)
from urbansound_segment_task.edge_v2.data.urbansound8k import inspect_urbansound8k


class UrbanSoundSplitTests(unittest.TestCase):
    def test_rotating_validation_rule_and_all_test_folds_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = inspect_urbansound8k(create_dataset(Path(directory)))
            manifests = build_all_split_manifests(dataset)
        self.assertEqual([item["test_fold"] for item in manifests], list(range(1, 11)))
        self.assertEqual([item["validation_fold"] for item in manifests], [2,3,4,5,6,7,8,9,10,1])
        self.assertTrue(all(item["policy_id"] == SPLIT_POLICY_ID for item in manifests))

    def test_each_split_is_disjoint_and_counts_are_8_1_1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = inspect_urbansound8k(create_dataset(Path(directory)))
            manifests = build_all_split_manifests(dataset)
        for manifest in manifests:
            self.assertEqual(manifest["clip_counts"], {"train": 8, "validation": 1, "test": 1})
            self.assertTrue(manifest["disjointness"]["valid"])
            self.assertEqual(len(manifest["training_folds"]), 8)

    def test_examples_and_deterministic_hash(self) -> None:
        self.assertEqual(split_folds(1), (list(range(3, 11)), 2, 1))
        self.assertEqual(split_folds(10), (list(range(2, 10)), 1, 10))
        with tempfile.TemporaryDirectory() as directory:
            dataset = inspect_urbansound8k(create_dataset(Path(directory)))
            first = build_split_manifest(dataset, 9, now=datetime(2026,1,1,tzinfo=timezone.utc))
            second = build_split_manifest(dataset, 9, now=datetime(2027,1,1,tzinfo=timezone.utc))
        self.assertEqual(first["split_manifest_sha256"], second["split_manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
