from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_offline_audio_inference.py"


class OfflineAudioCliTests(unittest.TestCase):
    def test_help_smoke(self) -> None:
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--audio", "--yamnet-artifact", "--classifier-artifact", "--output", "--pretty"):
            self.assertIn(option, result.stdout)

    def test_existing_output_is_rejected_before_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "exists.json"
            output.write_text("existing")
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT), "--audio", "missing.wav",
                    "--yamnet-artifact", "missing-yamnet", "--classifier-artifact", "missing-onnx",
                    "--output", str(output),
                ], capture_output=True, text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to overwrite", result.stderr)


if __name__ == "__main__":
    unittest.main()
