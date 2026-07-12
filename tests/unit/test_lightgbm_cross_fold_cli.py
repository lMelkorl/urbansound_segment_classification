from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]
CLI=ROOT/'scripts/run_lightgbm_cross_fold.py'


class LightgbmCrossFoldCliTests(unittest.TestCase):
    def test_cli_help_and_required_options(self) -> None:
        result=subprocess.run([sys.executable,str(CLI),'--help'],cwd=ROOT,capture_output=True,text=True,check=False)
        self.assertEqual(result.returncode,0,result.stderr)
        for option in ('--cache-root','--split-manifest-dir','--output-dir','--threads','--fold','--resume','--pretty'):
            self.assertIn(option,result.stdout)


if __name__=='__main__': unittest.main()
