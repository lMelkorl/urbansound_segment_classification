from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from urbansound_segment_task.edge_v2.models.compact_classifier import (
    LIGHTGBM_REFERENCE,
    SCHEMA_VERSION,
    ValidationClipF1Controller,
    _build_keras_model,
    _fold_result_valid,
    architecture_config,
    build_aggregate,
    classify_compact_result,
    pareto_front,
    parse_models,
    training_config,
    validate_resume_manifest,
)
from urbansound_segment_task.edge_v2.models.lightgbm_legacy import balanced_class_weights
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256


def _per_class(class_id: int, value: float) -> dict:
    return {"class_id": class_id, "precision": value, "recall": value, "f1": value, "support": 10}


def _metrics(value: float) -> dict:
    matrix = [[0] * 10 for _ in range(10)]
    for index in range(10):
        matrix[index][index] = 10
    return {
        "accuracy": value,
        "macro_f1": value,
        "prediction_count": 100,
        "class_order": list(range(10)),
        "per_class": [_per_class(index, value - index / 1000) for index in range(10)],
        "confusion_matrix": matrix,
    }


def _fold(architecture_id: str, fold: int, value: float) -> dict:
    metric = _metrics(value)
    return {
        "architecture_id": architecture_id,
        "test_fold": fold,
        "validation_fold": fold % 10 + 1,
        "selection": {"best_epoch": 3},
        "test_metrics": {
            "segment": dict(metric),
            "clip": {**metric, "excluded_zero_segment_clip_count": 2},
        },
        "training": {"duration_seconds": 1.5, "after_training_peak_rss_bytes": 1000},
        "artifacts": {"model": {"size_bytes": 100_000 if architecture_id == "linear" else 600_000}},
        "classifier_benchmark": {
            "load_time": {"duration": 1000},
            "steady_state": {
                "timing": {"p50": 2000, "p95": 3000},
                "throughput": {"items_per_second": 500_000},
            },
        },
    }


RUN_MANIFEST = {
    "run_identity_sha256": "a" * 64,
    "cache_identity": "b" * 64,
    "threads": 8,
    "class_order": [{"class_id": index, "class_name": f"class-{index}"} for index in range(10)],
}


class FakeModel:
    def __init__(self, predictions: list[np.ndarray]) -> None:
        self.predictions = predictions
        self.index = 0
        self.stop_training = False
        self.weight = np.asarray([0.0])
        self.restored = None

    def get_weights(self):
        return [self.weight]

    def set_weights(self, values):
        self.restored = [np.array(value, copy=True) for value in values]


class CompactClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import tensorflow as tf

        cls.tf = tf

    def test_real_model_parameter_counts(self) -> None:
        linear = _build_keras_model(self.tf, "linear", compile_model=False)
        mlp = _build_keras_model(self.tf, "mlp128", compile_model=False)
        self.assertEqual(linear.count_params(), 10_250)
        self.assertEqual(mlp.count_params(), 132_490)

    def test_architecture_and_training_config_are_fixed(self) -> None:
        self.assertEqual(architecture_config("linear")["layers"][0]["units"], 10)
        self.assertEqual(architecture_config("mlp128")["layers"][1]["rate"], 0.20)
        config = training_config("mlp128", threads=8)
        self.assertEqual(config["maximum_epochs"], 30)
        self.assertEqual(config["early_stopping"], {
            "metric": "validation_clip_macro_f1", "patience": 5,
            "min_delta": 0.001, "restore_best_weights": True,
        })
        self.assertEqual(config["validation_source"], "official_rotating_validation_fold")

    def test_train_only_class_weights(self) -> None:
        weights = balanced_class_weights(np.asarray([0, 0, 0, 1], dtype=np.int64))
        self.assertEqual(weights, {0: 2 / 3, 1: 2.0})

    def test_validation_clip_f1_and_best_weight_restore(self) -> None:
        keys = np.asarray(["a", "a", "b", "b"])
        labels = np.asarray([0, 0, 1, 1])
        perfect = np.asarray([[1, 0] + [0] * 8, [1, 0] + [0] * 8, [0, 1] + [0] * 8, [0, 1] + [0] * 8])
        wrong = np.asarray([[0, 1] + [0] * 8] * 4)
        model = FakeModel([perfect, wrong])

        def predictor(current, _features):
            value = current.predictions[current.index]
            current.index += 1
            return value

        controller = ValidationClipF1Controller(keys, labels, np.zeros((4, 1024)), predictor=predictor)
        model.weight = np.asarray([1.0])
        first = controller.observe(model, 0)
        model.weight = np.asarray([2.0])
        second = controller.observe(model, 1)
        controller.restore(model)
        self.assertGreater(first, second)
        self.assertEqual(controller.best_epoch, 1)
        self.assertEqual(model.restored[0].tolist(), [1.0])

    def test_early_stop_patience_and_no_test_input(self) -> None:
        keys = np.asarray(["a"])
        labels = np.asarray([0])
        prediction = np.asarray([[1] + [0] * 9])
        model = FakeModel([prediction] * 6)
        controller = ValidationClipF1Controller(
            keys, labels, np.zeros((1, 1024)), patience=5,
            predictor=lambda current, values: prediction,
        )
        for epoch in range(6):
            controller.observe(model, epoch)
        self.assertTrue(model.stop_training)
        self.assertNotIn("test", ValidationClipF1Controller.__dataclass_fields__)

    def test_success_thresholds_are_boundary_inclusive(self) -> None:
        self.assertEqual(classify_compact_result(0.010, 1_000_000), "excellent")
        self.assertEqual(classify_compact_result(0.025, 2_000_000), "strong")
        self.assertEqual(classify_compact_result(0.050, 5_000_000), "promising")
        self.assertEqual(classify_compact_result(0.051, 100), "insufficient")

    def test_pareto_uses_f1_size_and_latency(self) -> None:
        rows = [
            {"architecture_id": "a", "clip_macro_f1_mean": .8, "model_size_median_bytes": 2, "classifier_p50_median_ns": 2},
            {"architecture_id": "b", "clip_macro_f1_mean": .7, "model_size_median_bytes": 3, "classifier_p50_median_ns": 3},
            {"architecture_id": "c", "clip_macro_f1_mean": .9, "model_size_median_bytes": 4, "classifier_p50_median_ns": 1},
        ]
        self.assertEqual(pareto_front(rows), ["a", "c"])

    def test_aggregate_mean_std_delta_label_and_round_trip(self) -> None:
        folds = {
            name: [_fold(name, fold, 0.70 + fold / 100) for fold in range(1, 11)]
            for name in ("linear", "mlp128")
        }
        result = build_aggregate(folds, RUN_MANIFEST)
        self.assertEqual(result["schema_version"], SCHEMA_VERSION)
        self.assertEqual(result["status"]["outcome"], "success")
        self.assertAlmostEqual(result["models"]["linear"]["metrics"]["clip_accuracy"]["mean"], .755)
        self.assertEqual(result["models"]["linear"]["classifier_only_benchmark"]["label"], "classifier_only_cached_embedding_batch1")
        loss = result["models"]["linear"]["lightgbm_comparison"]["clip_macro_f1_loss"]
        self.assertAlmostEqual(loss, LIGHTGBM_REFERENCE["clip_macro_f1_mean"] - .755)
        serialized = json.dumps(result, sort_keys=True)
        self.assertEqual(json.loads(serialized)["schema_version"], SCHEMA_VERSION)
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("dataset_root", serialized)

    def test_missing_fold_is_not_success(self) -> None:
        folds = {
            "linear": [_fold("linear", fold, .8) for fold in range(1, 10)],
            "mlp128": [_fold("mlp128", fold, .8) for fold in range(1, 11)],
        }
        result = build_aggregate(folds, RUN_MANIFEST)
        self.assertEqual(result["status"]["outcome"], "incomplete")
        self.assertEqual(result["missing_folds"]["linear"], [10])

    def test_artifact_size_hash_and_resume_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {}
            for name, filename in (
                ("model", "model.weights.h5"), ("config", "training-config.json"),
                ("history", "epoch-history.csv"), ("per_class", "per-class-metrics.csv"),
                ("benchmark", "classifier-benchmark.json"),
            ):
                path = root / filename
                path.write_text(name, encoding="utf-8")
                artifacts[name] = {
                    "relative_path": filename, "size_bytes": path.stat().st_size,
                    "sha256": streaming_file_sha256(path),
                }
            result = {
                "run_identity_sha256": "a" * 64, "architecture_id": "linear", "test_fold": 1,
                "status": {"outcome": "success"}, "artifacts": artifacts,
            }
            result_path = root / "fold-result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            (root / "fold-result.sha256").write_text(streaming_file_sha256(result_path), encoding="ascii")
            self.assertTrue(_fold_result_valid(root, "a" * 64, "linear", 1))
            (root / "model.weights.h5").write_text("changed", encoding="utf-8")
            self.assertFalse(_fold_result_valid(root, "a" * 64, "linear", 1))

    def test_resume_mismatch_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "resume run identity"):
            validate_resume_manifest({"run_identity_sha256": "a"}, {"run_identity_sha256": "b"})

    def test_model_parser(self) -> None:
        self.assertEqual(parse_models("linear,mlp128"), ("linear", "mlp128"))
        with self.assertRaises(ValueError):
            parse_models("linear,linear")
        with self.assertRaises(ValueError):
            parse_models("other")


if __name__ == "__main__":
    unittest.main()
