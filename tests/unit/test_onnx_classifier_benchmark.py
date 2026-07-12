from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from urbansound_segment_task.edge_v2.benchmarks.onnx_classifier import (
    BenchmarkConfig,
    MaterializingConsumer,
    _load_onnx,
    _validate_artifacts,
    build_summary,
    run_fresh_process,
    select_benchmark_input,
    summarize_runtime,
    validate_runtime_output,
)
from urbansound_segment_task.edge_v2.features.cache_dataset import VerifiedCacheRecords


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/compact_classifier_cross_fold/fixed-baselines-v1/run-manifest.json"
ONNX_ROOT = ROOT / "artifacts/compact_classifier/linear-fold1-fp32-onnx-v1"


def _raw(runtime: str, repetition: int, *, input_hash: str = "i" * 64, top1: int = 3) -> dict:
    offset = repetition * 10
    return {
        "runtime": runtime,
        "repetition": repetition,
        "child_pid": 1000 + repetition + (100 if runtime == "onnxruntime" else 0),
        "process_startup": {"duration": 100 + offset},
        "runtime_import_configuration": {"duration": 200 + offset},
        "lifecycle": {
            "load_time": {"duration": 300 + offset},
            "first_call_latency": {"duration": 400 + offset},
            "steady_state": {
                "timing": {"p50": 10 + offset, "p95": 20 + offset, "p99": 30 + offset},
                "throughput": {"items_per_second": 1000 + offset},
            },
        },
        "memory": {
            "baseline_peak_rss_bytes": 1000,
            "final_peak_rss_bytes": 1100 if runtime == "keras" else 1120,
            "approximate_incremental_peak_rss_bytes": 100 if runtime == "keras" else 120,
            "source": "resource",
            "platform_limitation": "high-water",
        },
        "artifact_size_bytes": 47920 if runtime == "keras" else 41795,
        "input_identity": {"input_identity_sha256": input_hash},
        "output_validation": {"top1_class": top1},
        "status": {"outcome": "success", "error": None},
    }


def _config(mode: str) -> BenchmarkConfig:
    return BenchmarkConfig(
        runtime="onnxruntime", repetition=1, warmup=0, iterations=1, threads=1,
        input_bytes=np.zeros((1, 1024), dtype=np.float32).tobytes(),
        input_identity={"input_identity_sha256": "i" * 64},
        keras_weights_path="unused", onnx_model_path="unused",
        keras_artifact_size_bytes=47920, onnx_artifact_size_bytes=41795,
        test_mode=mode,
    )


class OnnxClassifierBenchmarkTests(unittest.TestCase):
    def test_artifact_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy2(ONNX_ROOT / "artifact-manifest.json", root / "artifact-manifest.json")
            (root / "model.onnx").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "size mismatch|SHA-256 mismatch"):
                _validate_artifacts(RUN, root)

    def test_onnx_loader_forces_cpu_provider(self) -> None:
        calls = {}

        class Session:
            def get_providers(self):
                return ["CPUExecutionProvider"]

        class Ort:
            @staticmethod
            def InferenceSession(path, sess_options, providers):
                calls["providers"] = providers
                return Session()

        _load_onnx(Ort(), object(), Path("model.onnx"))
        self.assertEqual(calls["providers"], ["CPUExecutionProvider"])

    def test_deterministic_input_identity_and_finite_float32_shape(self) -> None:
        records = (
            {"clip_key": "z", "fold": 2, "class_id": 1, "embeddings": np.ones((1, 1024), dtype=np.float32)},
            {"clip_key": "a", "fold": 1, "class_id": 0, "embeddings": np.full((2, 1024), 0.5, dtype=np.float32)},
        )
        verified = VerifiedCacheRecords(
            records=records, cache_identity="c" * 64, dataset_manifest_sha256="d" * 64,
            yamnet_artifact_tree_sha256="y" * 64, index_sha256="x" * 64,
            load_seconds=0.0, verified_artifact_count=2,
        )
        with mock.patch(
            "urbansound_segment_task.edge_v2.benchmarks.onnx_classifier.load_verified_cache_records",
            return_value=verified,
        ):
            first_identity, first = select_benchmark_input(Path("cache"))
            second_identity, second = select_benchmark_input(Path("cache"))
        self.assertEqual(first_identity, second_identity)
        self.assertEqual(first_identity["clip_key"], "a")
        self.assertEqual(first.shape, (1, 1024))
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, second)

    def test_consumer_materializes_and_counts_first_warmup_steady(self) -> None:
        consumer = MaterializingConsumer()
        for _ in range(1 + 20 + 1000):
            consumer(np.full((1, 10), 0.1, dtype=np.float32))
        self.assertEqual(consumer.consumed_count, 1021)
        self.assertEqual(consumer.last.shape, (1, 10))

    def test_output_shape_finite_probability_and_top1(self) -> None:
        output = np.asarray([[0.05] * 9 + [0.55]], dtype=np.float32)
        result = validate_runtime_output(output)
        self.assertEqual(result["shape"], [1, 10])
        self.assertEqual(result["top1_class"], 9)
        self.assertTrue(result["finite"])
        with self.assertRaises(ValueError):
            validate_runtime_output(np.full((1, 9), 1 / 9, dtype=np.float32))

    def test_five_repetition_medians_cover_all_lifecycle_fields(self) -> None:
        rows = [_raw("keras", index) for index in range(1, 6)]
        result = summarize_runtime(rows)
        self.assertEqual(result["repetitions"], 5)
        self.assertEqual(result["median_model_load_ns"], 330)
        self.assertEqual(result["median_first_call_ns"], 430)
        self.assertEqual(result["median_steady_p50_ns"], 40)
        self.assertEqual(result["median_steady_p95_ns"], 50)
        self.assertEqual(result["median_steady_p99_ns"], 60)
        self.assertEqual(result["median_predictions_per_second"], 1030)

    def test_summary_ratios_negative_memory_and_top1_agreement(self) -> None:
        rows = []
        for index in range(1, 6):
            rows.extend((_raw("keras", index), _raw("onnxruntime", index)))
        identity = {"input_identity_sha256": "i" * 64}
        result = build_summary(
            raw_rows=rows, raw_references=[], artifact_identity={"onnx_sha256": "o" * 64},
            input_identity=identity, warmup=20, iterations=1000, repetitions=5, threads=1,
        )
        self.assertEqual(result["status"]["outcome"], "success")
        self.assertTrue(result["comparison"]["top1_prediction_agreement"])
        self.assertEqual(result["comparison"]["memory_reduction_percent"], -20.0)
        self.assertGreater(result["comparison"]["artifact_size_reduction_percent"], 0)

    def test_summary_rejects_different_input_identity(self) -> None:
        rows = [_raw(runtime, index) for index in range(1, 6) for runtime in ("keras", "onnxruntime")]
        rows[-1]["input_identity"] = {"input_identity_sha256": "z" * 64}
        with self.assertRaisesRegex(ValueError, "same input"):
            build_summary(
                raw_rows=rows, raw_references=[], artifact_identity={},
                input_identity={"input_identity_sha256": "i" * 64},
                warmup=20, iterations=1000, repetitions=5, threads=1,
            )

    def test_summary_json_round_trip_has_no_sensitive_path(self) -> None:
        rows = [_raw(runtime, index) for index in range(1, 6) for runtime in ("keras", "onnxruntime")]
        result = build_summary(
            raw_rows=rows, raw_references=[], artifact_identity={"onnx_sha256": "o" * 64},
            input_identity={"input_identity_sha256": "i" * 64},
            warmup=20, iterations=1000, repetitions=5, threads=1,
        )
        serialized = json.dumps(result, sort_keys=True)
        self.assertEqual(json.loads(serialized)["schema_version"], "edge-v2.onnx-classifier-benchmark.v1")
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("melkor", serialized)

    def test_child_crash_is_structured(self) -> None:
        result = run_fresh_process(_config("test_crash"), timeout_seconds=5)
        self.assertEqual(result["status"]["outcome"], "failure")
        self.assertEqual(result["status"]["error"]["type"], "ChildProcessCrash")

    def test_child_timeout_is_structured(self) -> None:
        result = run_fresh_process(_config("test_timeout"), timeout_seconds=0.1)
        self.assertEqual(result["status"]["outcome"], "failure")
        self.assertEqual(result["status"]["error"]["type"], "ChildProcessTimeout")


if __name__ == "__main__":
    unittest.main()
