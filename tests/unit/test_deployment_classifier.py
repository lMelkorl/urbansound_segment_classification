from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from urbansound_segment_task.edge_v2.data.manifest import document_sha256
from urbansound_segment_task.edge_v2.export.onnx_linear import EXPORT_SCHEMA_VERSION, build_linear_keras_model
from urbansound_segment_task.edge_v2.export.onnx_validation import (
    MAX_ABSOLUTE_ERROR,
    MAX_MEAN_ABSOLUTE_ERROR,
    MIN_TOP1_AGREEMENT,
    compare_probability_outputs,
    parity_passes,
)
from urbansound_segment_task.edge_v2.features.cache_dataset import (
    EXPECTED_FOLD_SEGMENTS,
    VerifiedCacheRecords,
)
from urbansound_segment_task.edge_v2.models.deployment_classifier import (
    DEPLOYMENT_ID,
    EXPECTED_BEST_EPOCHS,
    EXPECTED_EPOCHS,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_SEGMENT_COUNT,
    balanced_class_weights_all_segments,
    build_deployment_training_set,
    deployment_training_config,
    select_deployment_epochs,
    validate_epoch_source,
)
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256
from urbansound_segment_task.edge_v2.runtime.pipeline import verify_classifier_artifact


ROOT = Path(__file__).resolve().parents[2]


def fake_verified() -> VerifiedCacheRecords:
    records = (
        {"clip_key": "fold1/a.wav", "class_id": 0, "fold": 1, "embeddings": np.ones((2, 1024), dtype=np.float32)},
        {"clip_key": "fold2/b.wav", "class_id": 1, "fold": 2, "embeddings": np.ones((1, 1024), dtype=np.float32)},
        {"clip_key": "fold3/empty.wav", "class_id": 2, "fold": 3, "embeddings": np.empty((0, 1024), dtype=np.float32)},
    )
    return VerifiedCacheRecords(
        records=records, cache_identity="c", dataset_manifest_sha256="d",
        yamnet_artifact_tree_sha256="y", index_sha256="i", load_seconds=0.0,
        verified_artifact_count=3,
    )


class DeploymentClassifierTests(unittest.TestCase):
    def test_all_expected_fold_segments_total_53918(self) -> None:
        self.assertEqual(sum(EXPECTED_FOLD_SEGMENTS.values()), EXPECTED_SEGMENT_COUNT)
        self.assertEqual(EXPECTED_SEGMENT_COUNT, 53_918)

    def test_zero_segment_clip_adds_no_artificial_feature(self) -> None:
        split = build_deployment_training_set(fake_verified(), enforce_expected_counts=False)
        self.assertEqual(split.metadata_clip_count, 3)
        self.assertEqual(split.zero_segment_clip_count, 1)
        self.assertEqual(split.segment_count, 3)
        self.assertEqual(split.X.shape, (3, 1024))
        self.assertNotIn(2, split.y.tolist())

    def test_epoch_selection_is_median_round_half_up_seven(self) -> None:
        result = select_deployment_epochs(EXPECTED_BEST_EPOCHS)
        self.assertEqual(result["median"], 6.5)
        self.assertEqual(result["rounding"], "round_half_up")
        self.assertEqual(result["selected_epochs"], EXPECTED_EPOCHS)

    def test_real_cross_fold_aggregate_is_epoch_source(self) -> None:
        selection, identity = validate_epoch_source(
            ROOT / "results/compact_classifier_cross_fold/fixed-baselines-v1/aggregate.json"
        )
        self.assertEqual(selection["best_epochs"], list(EXPECTED_BEST_EPOCHS))
        self.assertEqual(len(identity), 64)

    def test_config_is_fixed_deployment_only_without_metrics_or_splits(self) -> None:
        config = deployment_training_config(select_deployment_epochs(EXPECTED_BEST_EPOCHS), threads=1)
        serialized = json.dumps(config, sort_keys=True)
        self.assertTrue(config["deployment_only"])
        self.assertFalse(config["independent_test_metrics_available"])
        self.assertIsNone(config["validation_split"])
        self.assertIsNone(config["test_split"])
        self.assertEqual(config["epochs"], 7)
        self.assertNotIn("accuracy", serialized)
        self.assertNotIn("macro_f1", serialized)

    def test_linear_model_has_exact_parameter_count(self) -> None:
        import tensorflow as tf

        model = build_linear_keras_model(tf)
        self.assertEqual(model.count_params(), EXPECTED_PARAMETER_COUNT)

    def test_balanced_class_weights_use_every_supplied_segment(self) -> None:
        labels = np.asarray([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
        weights = balanced_class_weights_all_segments(labels)
        self.assertAlmostEqual(weights[0], 0.55)
        self.assertAlmostEqual(weights[1], 1.1)
        self.assertEqual(set(weights), set(range(10)))

    def test_deployment_numeric_thresholds_and_top1_are_fixed(self) -> None:
        reference = np.asarray([[0.1] * 10], dtype=np.float32)
        candidate = reference.copy()
        result = compare_probability_outputs(reference, candidate, [{"id": 1}])
        self.assertTrue(parity_passes(result))
        self.assertEqual(result["top1_agreement"], MIN_TOP1_AGREEMENT)
        self.assertEqual(MAX_ABSOLUTE_ERROR, 1e-5)
        self.assertEqual(MAX_MEAN_ABSOLUTE_ERROR, 1e-6)

    def test_runtime_accepts_provenance_bound_deployment_manifest_and_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.onnx"
            model.write_bytes(b"deployment-onnx")
            digest = streaming_file_sha256(model)
            manifest = {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "created_at_utc": "2026-01-01T00:00:00Z",
                "deployment_id": DEPLOYMENT_ID,
                "deployment_only": True,
                "independent_test_metrics_available": False,
                "source_model": {
                    "cache_identity": "6b0807688796f3f19ca7129867a518f893d18566ffd057cb31a5302bd6fd17ce",
                    "training_segment_count": 53_918,
                },
                "architecture_id": "linear",
                "input_contract": {"name": "embedding", "dtype": "float32", "shape": ["batch", 1024]},
                "output_contract": {"name": "probabilities", "dtype": "float32", "shape": ["batch", 10], "semantic": "softmax_probabilities"},
                "onnx_artifact": {"relative_path": "model.onnx", "size_bytes": model.stat().st_size, "sha256": digest},
                "status": {"outcome": "success", "error": None},
            }
            manifest["manifest_sha256"] = document_sha256(
                manifest, excluded_fields=("created_at_utc", "manifest_sha256")
            )
            (root / "artifact-manifest.json").write_text(json.dumps(manifest))
            verified = verify_classifier_artifact(root)
            self.assertIsNone(verified.test_fold)
            self.assertTrue(verified.deployment_only)
            model.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                verify_classifier_artifact(root)

    def test_identifiers_and_config_have_no_absolute_path(self) -> None:
        config = deployment_training_config(select_deployment_epochs(EXPECTED_BEST_EPOCHS), threads=1)
        serialized = json.dumps(config, sort_keys=True)
        self.assertEqual(config["deployment_id"], DEPLOYMENT_ID)
        self.assertNotIn("/Users/", serialized)


if __name__ == "__main__":
    unittest.main()
