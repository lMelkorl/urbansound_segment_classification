from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from urbansound_segment_task.edge_v2.evaluation.segmentation import LEGACY_SEGMENTATION_POLICY
from urbansound_segment_task.edge_v2.features.yamnet_cache import (
    CACHE_SCHEMA_VERSION, cache_namespace_identity, clip_cache_key, load_and_validate_cache,
    validate_embedding_arrays, verify_cache_artifact, write_cache_atomic,
)


MODEL_HASH = "a" * 64


def identity(model_hash=MODEL_HASH, window=15_360):
    config = LEGACY_SEGMENTATION_POLICY.as_dict()
    config["window_samples"] = window
    return cache_namespace_identity(
        yamnet_tree_sha256=model_hash, segmentation_config=config,
        resampling={"implementation": "test", "version": "1"},
    )


def metadata():
    return {
        "clip_key": "fold1/a.wav", "class_id": 0, "fold": 1,
        "source_audio_sha256": "b" * 64, "source_sample_rate": 44_100,
        "target_sample_rate": 16_000, "resampled_sample_count": 23_040,
        "segmentation_policy_id": LEGACY_SEGMENTATION_POLICY.policy_id,
        "yamnet_artifact_tree_sha256": MODEL_HASH,
    }


class YamnetEmbeddingCacheTests(unittest.TestCase):
    def test_cache_keys_change_with_audio_model_and_segmentation(self) -> None:
        key, fields = identity()
        self.assertNotEqual(
            clip_cache_key("1" * 64, fields, "fold1/a.wav"),
            clip_cache_key("2" * 64, fields, "fold1/a.wav"),
        )
        self.assertNotEqual(
            clip_cache_key("1" * 64, fields, "fold1/a.wav"),
            clip_cache_key("1" * 64, fields, "fold2/b.wav"),
        )
        self.assertNotEqual(key, identity(model_hash="c" * 64)[0])
        self.assertNotEqual(key, identity(window=16_000)[0])

    def test_round_trip_hash_and_atomic_no_overwrite(self) -> None:
        embeddings = np.ones((2, 1024), dtype=np.float32)
        starts = np.asarray([0, 7_680], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            digest = write_cache_atomic(path, embeddings=embeddings, segment_start_samples=starts, metadata=metadata())
            loaded = load_and_validate_cache(path)
            verified = verify_cache_artifact(path, digest)
            with self.assertRaises(FileExistsError):
                write_cache_atomic(path, embeddings=embeddings, segment_start_samples=starts, metadata=metadata())
        np.testing.assert_array_equal(loaded["embeddings"], embeddings)
        self.assertEqual(loaded["cache_schema_version"], CACHE_SCHEMA_VERSION)
        self.assertEqual(verified["segment_count"], 2)

    def test_empty_embedding_record_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.npz"
            write_cache_atomic(
                path, embeddings=np.empty((0, 1024), dtype=np.float32),
                segment_start_samples=np.empty((0,), dtype=np.int64), metadata=metadata(),
            )
            loaded = load_and_validate_cache(path)
        self.assertEqual(loaded["embeddings"].shape, (0, 1024))

    def test_invalid_shape_dtype_nan_and_starts_are_rejected(self) -> None:
        valid_starts = np.asarray([0], dtype=np.int64)
        cases = (
            (np.ones((1, 10), dtype=np.float32), valid_starts),
            (np.ones((1, 1024), dtype=np.float64), valid_starts),
            (np.full((1, 1024), np.nan, dtype=np.float32), valid_starts),
            (np.ones((2, 1024), dtype=np.float32), np.asarray([0, 1], dtype=np.int64)),
        )
        for embeddings, starts in cases:
            with self.subTest(shape=embeddings.shape, dtype=embeddings.dtype):
                with self.assertRaises(ValueError):
                    validate_embedding_arrays(embeddings, starts)

    def test_corruption_and_hash_mismatch_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            digest = write_cache_atomic(
                path, embeddings=np.ones((1,1024),dtype=np.float32),
                segment_start_samples=np.asarray([0],dtype=np.int64), metadata=metadata(),
            )
            path.write_bytes(path.read_bytes()[:50])
            with self.assertRaises(ValueError):
                load_and_validate_cache(path)
            with self.assertRaises(ValueError):
                verify_cache_artifact(path, digest)


if __name__ == "__main__":
    unittest.main()
