from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from urbansound_segment_task.edge_v2.models.preflight import run_model_preflight
from tests.unit.model_preflight_fixtures import missing_packages, no_versions, write_fixture


class ModelPreflightAudioclipTests(unittest.TestCase):
    def test_placeholder_and_missing_readme_artifacts_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            result = run_model_preflight(
                root,
                method="audioclip",
                find_spec=missing_packages,
                version_lookup=no_versions,
            )["methods"][0]

        self.assertEqual(result["implementation_status"], "placeholder")
        self.assertEqual(result["benchmark_readiness"], "unavailable")
        self.assertTrue(result["source_code"]["placeholder_detected"])
        self.assertEqual(len(result["source_code"]["missing_readme_artifacts"]), 3)
        self.assertFalse(result["cpu_support"]["code_path_detected"])
        self.assertFalse(result["cpu_support"]["runtime_verified"])


if __name__ == "__main__":
    unittest.main()
