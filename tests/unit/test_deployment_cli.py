from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class DeploymentCliTests(unittest.TestCase):
    def test_all_deployment_cli_help_commands(self) -> None:
        scripts = {
            "train_deployment_linear.py": ("--cache-root", "--epochs", "--output-dir"),
            "export_deployment_linear_onnx.py": ("--source-run", "--output", "--manifest", "--opset"),
            "validate_deployment_linear_onnx.py": ("--source-run", "--cache-root", "--onnx-artifact", "--output-dir"),
        }
        for name, options in scripts.items():
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / name), "--help"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for option in options:
                self.assertIn(option, result.stdout)


if __name__ == "__main__":
    unittest.main()
