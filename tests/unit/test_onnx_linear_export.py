from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from urbansound_segment_task.edge_v2.export.onnx_linear import (
    _publish_binary_new,
    inspect_onnx_model,
    resolve_source_model,
)


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/compact_classifier_cross_fold/fixed-baselines-v1/run-manifest.json"


def _value_info(name: str, second: int):
    dimensions = [
        SimpleNamespace(dim_value=0, dim_param="batch"),
        SimpleNamespace(dim_value=second, dim_param=""),
    ]
    return SimpleNamespace(
        name=name,
        type=SimpleNamespace(tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=dimensions))),
    )


def _model_proto(*, domain: str = "", opset: int = 15, output: str = "probabilities"):
    graph = SimpleNamespace(
        input=[_value_info("embedding", 1024)],
        output=[_value_info(output, 10)],
        node=[SimpleNamespace(domain=domain, op_type="MatMul"), SimpleNamespace(domain="", op_type="Softmax")],
        initializer=[object(), object()],
    )
    return SimpleNamespace(
        ir_version=9,
        graph=graph,
        opset_import=[SimpleNamespace(domain="", version=opset)],
    )


class OnnxLinearExportTests(unittest.TestCase):
    def test_source_resolves_only_linear_fold1_with_full_identity(self) -> None:
        source = resolve_source_model(RUN, model="linear", fold=1)
        self.assertEqual(source.parameter_count, 10_250)
        self.assertEqual(source.validation_fold, 2)
        self.assertEqual(len(source.source_model_sha256), 64)
        self.assertEqual(source.weights_relative_path, "model.weights.h5")
        with self.assertRaises(ValueError):
            resolve_source_model(RUN, model="mlp128", fold=1)
        with self.assertRaises(ValueError):
            resolve_source_model(RUN, model="linear", fold=2)

    def test_source_model_hash_mismatch_is_rejected(self) -> None:
        source_root = RUN.parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "linear/fold-01").mkdir(parents=True)
            for name in ("run-manifest.json", "aggregate.json"):
                shutil.copy2(source_root / name, root / name)
            for name in ("fold-result.json", "fold-result.sha256", "model.weights.h5"):
                shutil.copy2(source_root / "linear/fold-01" / name, root / "linear/fold-01" / name)
            (root / "linear/fold-01/model.weights.h5").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "size mismatch|SHA-256 mismatch"):
                resolve_source_model(root / "run-manifest.json", model="linear", fold=1)

    def test_graph_contract_opset_shapes_softmax_and_counts(self) -> None:
        result = inspect_onnx_model(_model_proto(), checker=SimpleNamespace(check_model=lambda model: None))
        self.assertEqual(result["opset"], 15)
        self.assertEqual(result["graph"]["input"]["shape"], ["batch", 1024])
        self.assertEqual(result["graph"]["output"]["shape"], ["batch", 10])
        self.assertEqual(result["graph"]["node_count"], 2)
        self.assertEqual(result["graph"]["initializer_count"], 2)

    def test_wrong_opset_output_and_custom_domain_are_rejected(self) -> None:
        for model in (_model_proto(opset=16), _model_proto(output="logits"), _model_proto(domain="custom.edge")):
            with self.assertRaises(ValueError):
                inspect_onnx_model(model)

    def test_checker_failure_is_propagated(self) -> None:
        class Checker:
            @staticmethod
            def check_model(model):
                raise RuntimeError("invalid graph")

        with self.assertRaises(RuntimeError):
            inspect_onnx_model(_model_proto(), checker=Checker())

    def test_binary_publish_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            _publish_binary_new(path, b"first")
            with self.assertRaises(FileExistsError):
                _publish_binary_new(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_source_safe_identity_json_has_no_absolute_path(self) -> None:
        serialized = json.dumps(resolve_source_model(RUN, model="linear", fold=1).safe_identity())
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("melkor", serialized)


if __name__ == "__main__":
    unittest.main()
