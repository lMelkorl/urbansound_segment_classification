from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class CompactClassifierCliTests(unittest.TestCase):
    def test_help_smoke_and_required_options(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/run_compact_classifier_cross_fold.py"), "--help"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0)
        for option in (
            "--cache-root", "--split-manifest-dir", "--output-dir", "--models",
            "--fold", "--threads", "--resume", "--pretty",
        ):
            self.assertIn(option, completed.stdout)


if __name__ == "__main__":
    unittest.main()
