from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class OnnxClassifierBenchmarkCliTests(unittest.TestCase):
    def test_help_smoke(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/benchmark_linear_onnx.py"), "--help"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0)
        for option in (
            "--run-manifest", "--cache-root", "--onnx-artifact", "--warmup",
            "--iterations", "--repetitions", "--threads", "--output-dir", "--pretty",
        ):
            self.assertIn(option, completed.stdout)


if __name__ == "__main__":
    unittest.main()
