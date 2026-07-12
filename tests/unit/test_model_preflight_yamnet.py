from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from urbansound_segment_task.edge_v2.models.preflight import run_model_preflight
from tests.unit.model_preflight_fixtures import missing_packages, no_versions, write_fixture


class ModelPreflightYamnetTests(unittest.TestCase):
    def test_runtime_url_and_local_artifact_blockers_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            result = run_model_preflight(
                root,
                method="yamnet_lgbm",
                find_spec=missing_packages,
                version_lookup=no_versions,
            )["methods"][0]

        issue_codes = {issue["code"] for issue in result["issues"]}
        self.assertTrue(result["source_code"]["runtime_url_detected"])
        self.assertEqual(result["source_code"]["runtime_urls"], ["https://tfhub.dev/google/yamnet/1"])
        self.assertIn("YAMNET_RUNTIME_DOWNLOAD", issue_codes)
        self.assertIn("MISSING_LOCAL_ARTIFACTS", issue_codes)
        self.assertEqual(result["benchmark_readiness"], "blocked")
        self.assertTrue(result["cpu_support"]["code_path_detected"])
        self.assertFalse(result["cpu_support"]["runtime_verified"])


if __name__ == "__main__":
    unittest.main()
