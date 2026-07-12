from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class OnnxCliTests(unittest.TestCase):
    def test_export_and_validation_help(self) -> None:
        cases = {
            "export_linear_onnx.py": ("--run-manifest", "--model", "--fold", "--opset", "--output", "--manifest"),
            "validate_linear_onnx.py": ("--run-manifest", "--cache-root", "--onnx-artifact", "--sample-count", "--output-dir"),
        }
        for script, options in cases.items():
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), "--help"],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0)
            for option in options:
                self.assertIn(option, completed.stdout)


if __name__ == "__main__":
    unittest.main()
