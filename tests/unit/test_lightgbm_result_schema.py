from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from urbansound_segment_task.edge_v2.evaluation.metrics import classification_metrics
from urbansound_segment_task.edge_v2.evaluation.result_schema import (
    LIGHTGBM_REPRODUCTION_SCHEMA_VERSION, serialize_result, write_result,
)
from urbansound_segment_task.edge_v2.models.lightgbm_legacy import (
    delta_classification, historical_comparison,
)
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/run_lightgbm_legacy_reproduction.py"


class LightgbmResultSchemaTests(unittest.TestCase):
    def test_fixed_ten_class_metrics_and_confusion_order(self) -> None:
        metrics = classification_metrics([0, 0, 1], [0, 1, 1])
        self.assertEqual(metrics["class_order"], list(range(10)))
        self.assertEqual(len(metrics["per_class"]), 10)
        self.assertEqual(len(metrics["confusion_matrix"]), 10)
        self.assertLess(metrics["macro_f1"], 0.2)

    def test_delta_thresholds_and_historical_comparison(self) -> None:
        self.assertEqual(delta_classification(0.005), "exact_or_near_reproduction")
        self.assertEqual(delta_classification(0.015), "close_reproduction")
        self.assertEqual(delta_classification(0.016), "material_difference")
        historical = {
            "validation": {"segment_accuracy":.5,"segment_macroF1":.5,"clip_accuracy":.5,"clip_macroF1":.5},
            "test": {"segment_accuracy":.5,"segment_macroF1":.5,"clip_accuracy":.5,"clip_macroF1":.5},
        }
        reproduced = {"segment":{"accuracy":.5,"macro_f1":.5},"clip":{"accuracy":.52,"macro_f1":.5}}
        comparison = historical_comparison(historical, reproduced, reproduced)
        self.assertEqual(comparison["overall_classification"], "material_difference")

    def test_model_sha_json_round_trip_privacy_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.txt"
            model.write_text("safe model", encoding="utf-8")
            self.assertEqual(len(streaming_file_sha256(model)), 64)
            document = {"schema_version":LIGHTGBM_REPRODUCTION_SCHEMA_VERSION,"status":{"outcome":"success"}}
            path = root / "result.json"
            write_result(path, document, pretty=True)
            self.assertEqual(json.loads(path.read_text()), document)
            self.assertNotIn("/Users/", serialize_result(document, pretty=False))
            with self.assertRaises(FileExistsError):
                write_result(path, document, pretty=True)

    def test_cli_help(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CLI), "--help"], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--cache-root","--output-dir","--threads","--seed","--pretty"):
            self.assertIn(option, result.stdout)


if __name__ == "__main__":
    unittest.main()
