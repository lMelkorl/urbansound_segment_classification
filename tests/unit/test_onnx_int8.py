from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from urbansound_segment_task.edge_v2.benchmarks.onnx_int8_classifier import (
    alternating_runtime_order,
    build_int8_benchmark_summary,
    deployment_decision,
)
from urbansound_segment_task.edge_v2.export.onnx_int8_validation import (
    EXPECTED_FIXTURE_IDENTITY,
    compare_fp32_int8_outputs,
    metric_thresholds_pass,
    numeric_thresholds_pass,
)
from urbansound_segment_task.edge_v2.export.onnx_quantization import (
    QUANTIZATION_CONFIG,
    quantize_linear_dynamic_int8,
    summarize_quantized_graph,
    validate_fp32_artifact,
)
from urbansound_segment_task.edge_v2.data.manifest import document_sha256


ROOT = Path(__file__).resolve().parents[2]


def _value_info(name: str, second: int):
    dims = [SimpleNamespace(dim_value=0, dim_param="batch"), SimpleNamespace(dim_value=second, dim_param="")]
    return SimpleNamespace(name=name, type=SimpleNamespace(tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=dims))))


def _quantized_proto(domain: str = ""):
    nodes = [
        SimpleNamespace(domain=domain, op_type="DynamicQuantizeLinear"),
        SimpleNamespace(domain="", op_type="MatMulInteger"),
        SimpleNamespace(domain="", op_type="Cast"),
        SimpleNamespace(domain="", op_type="Mul"),
        SimpleNamespace(domain="", op_type="Add"),
        SimpleNamespace(domain="", op_type="Softmax"),
    ]
    initializers = [
        SimpleNamespace(name="weight_quantized", data_type=3, dims=[1024, 10]),
        SimpleNamespace(name="weight_scale", data_type=1, dims=[]),
        SimpleNamespace(name="bias", data_type=1, dims=[10]),
    ]
    graph = SimpleNamespace(
        input=[_value_info("embedding", 1024)], output=[_value_info("probabilities", 10)],
        node=nodes, initializer=initializers,
    )
    return SimpleNamespace(ir_version=8, graph=graph, opset_import=[SimpleNamespace(domain="", version=15)])


class FakeOnnx:
    checker = SimpleNamespace(check_model=lambda model: None)
    TensorProto = SimpleNamespace(DataType=SimpleNamespace(Name=lambda value: {1: "FLOAT", 3: "INT8"}[value]))


def _raw(variant: str, repetition: int) -> dict:
    base = 100 if variant == "fp32" else 80
    return {
        "runtime": variant,
        "child_pid": repetition + (0 if variant == "fp32" else 100),
        "process_startup": {"duration": base + repetition},
        "runtime_import_configuration": {"duration": base + repetition},
        "lifecycle": {
            "load_time": {"duration": base + repetition},
            "first_call_latency": {"duration": base + repetition},
            "steady_state": {
                "timing": {"p50": base + repetition, "p95": base + repetition + 10, "p99": base + repetition + 20},
                "throughput": {"items_per_second": 1000 if variant == "fp32" else 1200},
            },
        },
        "memory": {
            "baseline_peak_rss_bytes": 1000, "final_peak_rss_bytes": 1200,
            "approximate_incremental_peak_rss_bytes": 200 if variant == "fp32" else 150,
            "source": "resource", "platform_limitation": "high-water",
        },
        "artifact_size_bytes": 1000 if variant == "fp32" else 700,
        "input_identity": {"input_identity_sha256": "i" * 64},
        "output_validation": {"top1_class": 3},
        "status": {"outcome": "success", "error": None},
    }


class OnnxInt8Tests(unittest.TestCase):
    def test_dynamic_qint8_config_is_fixed(self) -> None:
        self.assertEqual(
            QUANTIZATION_CONFIG,
            {"api": "onnxruntime.quantization.quantize_dynamic", "weight_type": "QInt8", "per_channel": False, "reduce_range": False},
        )

    def test_fp32_source_hash_mismatch_is_rejected(self) -> None:
        source = ROOT / "artifacts/compact_classifier/linear-fold1-fp32-onnx-v1"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            shutil.copy2(source / "model.onnx", destination / "model.onnx")
            manifest = json.loads((source / "artifact-manifest.json").read_text())
            manifest["source_model_sha256"] = "0" * 64
            manifest["manifest_sha256"] = document_sha256(
                manifest, excluded_fields=("created_at_utc", "manifest_sha256")
            )
            (destination / "artifact-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "source model SHA-256 mismatch"):
                validate_fp32_artifact(destination)

    def test_quantized_graph_detects_nodes_initializers_and_float_softmax(self) -> None:
        result = summarize_quantized_graph(_quantized_proto(), FakeOnnx)
        self.assertEqual(result["input"]["shape"], ["batch", 1024])
        self.assertEqual(result["output"]["shape"], ["batch", 10])
        self.assertIn("MatMulInteger", result["quantized_node_types"])
        self.assertEqual(result["initializer_dtype_counts"]["INT8"], 1)
        self.assertEqual(result["softmax"], {"present": True, "output_dtype": "float32"})

    def test_quantized_custom_domain_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize_quantized_graph(_quantized_proto(domain="custom.int8"), FakeOnnx)

    def test_quantizer_refuses_existing_output_before_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.int8.onnx"
            output.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                quantize_linear_dynamic_int8(
                    fp32_artifact_directory=Path("unused"), output_path=output,
                    manifest_path=Path(directory) / "manifest.json", pretty=True,
                )

    def test_fixture_identity_is_reused_from_fp32_parity(self) -> None:
        fixture = json.loads(
            (ROOT / "results/onnx_export/linear-fold1-fp32-v1/fixture-manifest.json").read_text()
        )
        self.assertEqual(EXPECTED_FIXTURE_IDENTITY, fixture["fixture_identity"])

    def test_numeric_error_agreement_changed_samples_and_margin(self) -> None:
        fp32 = np.asarray([[0.6, 0.4] + [0] * 8, [0.51, 0.49] + [0] * 8], dtype=np.float32)
        int8 = np.asarray([[0.59, 0.41] + [0] * 8, [0.49, 0.51] + [0] * 8], dtype=np.float32)
        result = compare_fp32_int8_outputs(fp32, int8, [{"id": 0}, {"id": 1}])
        self.assertEqual(result["changed_top1_count"], 1)
        self.assertAlmostEqual(result["changed_top1_samples"][0]["fp32_top1_margin"], 0.02, places=6)
        self.assertEqual(result["maximum_error_location"]["class_index"], 0)

    def test_numeric_and_metric_threshold_failures(self) -> None:
        fp32 = np.asarray([[0.9, 0.1] + [0] * 8], dtype=np.float32)
        int8 = np.asarray([[0.1, 0.9] + [0] * 8], dtype=np.float32)
        self.assertFalse(numeric_thresholds_pass(compare_fp32_int8_outputs(fp32, int8, [{"id": 0}])))
        self.assertFalse(metric_thresholds_pass({
            "clip_accuracy_delta": 0.003, "clip_macro_f1_delta": 0.0,
            "fp32_int8_clip_prediction_agreement": 1.0,
        }))

    def test_alternating_runtime_order(self) -> None:
        self.assertEqual(alternating_runtime_order(1), ("fp32", "int8"))
        self.assertEqual(alternating_runtime_order(2), ("int8", "fp32"))
        self.assertEqual(alternating_runtime_order(3), ("fp32", "int8"))

    def test_deployment_rule_selects_int8_only_when_all_rules_pass(self) -> None:
        selected, reasons = deployment_decision(
            parity_passed=True, clip_macro_f1_delta=0.001,
            fp32_size_bytes=1000, int8_size_bytes=700,
            fp32_p50_ns=100, int8_p50_ns=90,
            fp32_incremental_rss_bytes=200, int8_incremental_rss_bytes=150,
        )
        self.assertEqual(selected, "dynamic_int8_onnx")
        self.assertTrue(all(row["passed"] for row in reasons))

    def test_deployment_rule_keeps_fp32_when_int8_is_larger_or_slower(self) -> None:
        selected, _ = deployment_decision(
            parity_passed=True, clip_macro_f1_delta=0.0,
            fp32_size_bytes=1000, int8_size_bytes=1200,
            fp32_p50_ns=100, int8_p50_ns=110,
            fp32_incremental_rss_bytes=200, int8_incremental_rss_bytes=210,
        )
        self.assertEqual(selected, "fp32_onnx")

    def test_benchmark_summary_medians_ratios_decision_and_json_privacy(self) -> None:
        rows = [_raw(variant, repetition) for repetition in range(1, 6) for variant in ("fp32", "int8")]
        parity = {
            "parity_sha256": "p" * 64, "status": {"outcome": "success", "error": None},
            "fold1_metric_parity": {"clip_macro_f1_delta": 0.001},
        }
        result = build_int8_benchmark_summary(
            raw_rows=rows, raw_references=[],
            fp32_artifact={"size_bytes": 1000}, int8_artifact={"size_bytes": 700},
            input_identity={"input_identity_sha256": "i" * 64}, parity_result=parity,
            warmup=20, iterations=1000, repetitions=5, threads=1,
        )
        self.assertEqual(result["status"]["outcome"], "success")
        self.assertEqual(result["deployment_decision"], "dynamic_int8_onnx")
        self.assertGreater(result["comparison"]["int8_latency_speedup_fp32_p50_over_int8_p50"], 1)
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("melkor", serialized)


if __name__ == "__main__":
    unittest.main()
