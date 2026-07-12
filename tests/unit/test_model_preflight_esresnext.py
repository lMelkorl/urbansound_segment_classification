from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from urbansound_segment_task.edge_v2.models.preflight import run_model_preflight
from tests.unit.model_preflight_fixtures import (
    git_stage,
    missing_packages,
    no_versions,
    write_fixture,
)


class ModelPreflightEsresnextTests(unittest.TestCase):
    def test_broken_gitlink_and_path_import_inconsistency_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            result = run_model_preflight(
                root,
                method="esresnext",
                find_spec=missing_packages,
                version_lookup=no_versions,
                git_stage_lookup=git_stage,
            )["methods"][0]

        issue_codes = {issue["code"] for issue in result["issues"]}
        source_artifact = next(
            artifact for artifact in result["artifacts"] if artifact["artifact_id"] == "esresnext_source"
        )
        self.assertTrue(result["source_code"]["gitlink_detected"])
        self.assertFalse(result["source_code"]["gitmodules_mapping_present"])
        self.assertEqual(source_artifact["provenance_status"], "broken_reference")
        self.assertTrue(result["source_code"]["path_import_inconsistent"])
        self.assertIn("BROKEN_GITLINK_MAPPING", issue_codes)
        self.assertIn("EXTERNAL_PATH_IMPORT_INCONSISTENCY", issue_codes)
        self.assertTrue(result["cpu_support"]["code_path_detected"])
        self.assertFalse(result["cpu_support"]["runtime_verified"])


if __name__ == "__main__":
    unittest.main()
