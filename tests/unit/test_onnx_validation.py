from __future__ import annotations

import json
import unittest

import numpy as np

from urbansound_segment_task.edge_v2.export.onnx_validation import (
    REQUIRED_BATCH_SIZES,
    compare_probability_outputs,
    parity_passes,
    select_deterministic_fixture,
)
from urbansound_segment_task.edge_v2.features.cache_dataset import VerifiedCacheRecords


def _fake_verified() -> VerifiedCacheRecords:
    records = []
    for fold in range(1, 11):
        for class_id in range(10):
            embeddings = np.stack(
                [
                    np.full(1024, fold + class_id / 10 + segment / 100, dtype=np.float32)
                    for segment in range(3)
                ]
            )
            records.append(
                {
                    "clip_key": f"fold{fold:02d}-class{class_id:02d}",
                    "class_id": class_id,
                    "fold": fold,
                    "embeddings": embeddings,
                }
            )
    return VerifiedCacheRecords(
        records=tuple(records), cache_identity="a" * 64, dataset_manifest_sha256="b" * 64,
        yamnet_artifact_tree_sha256="c" * 64, index_sha256="d" * 64,
        load_seconds=0.0, verified_artifact_count=len(records),
    )


class OnnxValidationTests(unittest.TestCase):
    def test_fixture_is_deterministic_covers_folds_classes_and_has_no_duplicates(self) -> None:
        first, first_features = select_deterministic_fixture(_fake_verified(), 200)
        second, second_features = select_deterministic_fixture(_fake_verified(), 200)
        self.assertEqual(first["fixture_identity"], second["fixture_identity"])
        self.assertEqual(first["selected_records"], second["selected_records"])
        np.testing.assert_array_equal(first_features, second_features)
        self.assertEqual(first["selection"]["fold_coverage"], list(range(1, 11)))
        self.assertEqual(first["selection"]["class_coverage"], list(range(10)))
        identities = {(row["clip_key"], row["segment_start_sample"]) for row in first["selected_records"]}
        self.assertEqual(len(identities), 200)

    def test_fixture_manifest_round_trip_has_no_embedding_or_absolute_path(self) -> None:
        fixture, _ = select_deterministic_fixture(_fake_verified(), 100)
        serialized = json.dumps(fixture, sort_keys=True)
        self.assertEqual(json.loads(serialized)["fixture_identity"], fixture["fixture_identity"])
        self.assertNotIn("embedding", serialized)
        self.assertNotIn("/Users/", serialized)

    def test_numeric_error_top1_and_probability_sums(self) -> None:
        keras = np.asarray([[0.7, 0.3] + [0] * 8, [0.1, 0.9] + [0] * 8], dtype=np.float32)
        onnx = keras.copy()
        onnx[0, 0] -= 1e-7
        onnx[0, 1] += 1e-7
        result = compare_probability_outputs(keras, onnx, [{"id": 0}, {"id": 1}])
        self.assertLessEqual(result["maximum_absolute_error"], 1e-5)
        self.assertLessEqual(result["mean_absolute_error"], 1e-6)
        self.assertEqual(result["top1_agreement"], 1.0)
        self.assertTrue(parity_passes(result))

    def test_parity_threshold_failure(self) -> None:
        keras = np.asarray([[0.9, 0.1] + [0] * 8], dtype=np.float32)
        onnx = np.asarray([[0.1, 0.9] + [0] * 8], dtype=np.float32)
        result = compare_probability_outputs(keras, onnx, [{"id": 0}])
        self.assertFalse(parity_passes(result))
        self.assertEqual(result["top1_agreement"], 0.0)

    def test_nonfinite_and_wrong_shapes_are_rejected(self) -> None:
        valid = np.full((1, 10), 0.1, dtype=np.float32)
        with self.assertRaises(ValueError):
            compare_probability_outputs(valid, np.full((1, 9), 1 / 9, dtype=np.float32), [{"id": 0}])
        invalid = valid.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            compare_probability_outputs(valid, invalid, [{"id": 0}])

    def test_required_dynamic_batch_sizes_are_fixed(self) -> None:
        self.assertEqual(REQUIRED_BATCH_SIZES, (1, 7, 32, 128))


if __name__ == "__main__":
    unittest.main()
