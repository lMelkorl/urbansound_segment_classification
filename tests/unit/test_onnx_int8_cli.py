from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class OnnxInt8CliTests(unittest.TestCase):
    def test_all_int8_cli_help_commands(self) -> None:
        cases = {
            "quantize_linear_onnx.py": ("--fp32-artifact", "--output", "--manifest"),
            "validate_linear_int8_onnx.py": ("--run-manifest", "--cache-root", "--fp32-artifact", "--int8-artifact", "--sample-count", "--output-dir"),
            "benchmark_linear_int8_onnx.py": ("--cache-root", "--fp32-artifact", "--int8-artifact", "--warmup", "--iterations", "--repetitions", "--threads", "--output-dir"),
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
