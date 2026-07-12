from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from urbansound_segment_task.edge_v2.features.cache_dataset import (
    EXPECTED_CACHE_IDENTITY, build_cache_split, load_legacy_cache_dataset,
)


class CacheDatasetTests(unittest.TestCase):
    def test_zero_segment_clip_is_excluded_from_matrix_but_counted_in_metadata(self) -> None:
        records = [
            {"fold": 1, "class_id": 2, "clip_key": "fold1/a.wav", "embeddings": np.ones((2,1024),dtype=np.float32)},
            {"fold": 1, "class_id": 3, "clip_key": "fold1/b.wav", "embeddings": np.empty((0,1024),dtype=np.float32)},
        ]
        split = build_cache_split("train", (1,), records)
        self.assertEqual(split.X.shape, (2, 1024))
        self.assertEqual(split.metadata_clip_count, 2)
        self.assertEqual(split.evaluable_clip_count, 1)
        self.assertEqual(split.zero_segment_clip_count, 1)
        self.assertEqual(split.clip_keys.tolist(), ["fold1/a.wav", "fold1/a.wav"])

    def test_wrong_cache_identity_is_rejected_before_index_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            namespace = Path(directory) / EXPECTED_CACHE_IDENTITY
            namespace.mkdir(parents=True)
            (namespace / "cache-summary.json").write_text(
                json.dumps({"cache_identity": "wrong"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "cache identity"):
                load_legacy_cache_dataset(Path(directory))


if __name__ == "__main__":
    unittest.main()
