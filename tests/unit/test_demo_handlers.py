from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from urbansound_segment_task.edge_v2.demo.app import LOCAL_ENVIRONMENT, configure_local_environment


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_audio_demo.py"


class DemoHandlerTests(unittest.TestCase):
    def test_cli_help_and_localhost_default(self) -> None:
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "--yamnet-artifact", "--classifier-artifact", "--host", "--port",
            "--max-duration-seconds", "--no-browser",
        ):
            self.assertIn(option, result.stdout)
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('default="127.0.0.1"', source)
        self.assertIn('share=False', source)

    def test_local_privacy_environment_is_forced(self) -> None:
        for name in LOCAL_ENVIRONMENT:
            os.environ[name] = "unsafe"
        configured = configure_local_environment()
        self.assertEqual(configured, LOCAL_ENVIRONMENT)
        for name, value in LOCAL_ENVIRONMENT.items():
            self.assertEqual(os.environ[name], value)

    def test_non_localhost_binding_is_refused_before_runtime(self) -> None:
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT), "--yamnet-artifact", "missing",
                "--classifier-artifact", "missing", "--host", "0.0.0.0", "--no-browser",
            ], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("refusing non-localhost", result.stderr)

    def test_gradio_import_is_lazy_for_core_modules(self) -> None:
        code = (
            "import sys; import urbansound_segment_task.edge_v2.demo.presenter; "
            "import urbansound_segment_task.edge_v2.demo.state; "
            "raise SystemExit(1 if 'gradio' in sys.modules else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code])
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
