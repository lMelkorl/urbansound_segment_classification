from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from urbansound_segment_task.edge_v2.models.yamnet_artifact import ArtifactVerificationError
from urbansound_segment_task.edge_v2.models.yamnet_runtime import (
    load_local_yamnet,
    validate_yamnet_outputs,
)
from tests.unit.yamnet_test_fixtures import FakeTensor, FakeTensorFlow, create_artifact


class YamnetRuntimeContractTests(unittest.TestCase):
    def test_runtime_rejects_url_before_framework_import(self) -> None:
        importer = mock.Mock()
        with self.assertRaises(ArtifactVerificationError) as caught:
            load_local_yamnet("https://tfhub.dev/google/yamnet/1", import_module=importer)
        self.assertEqual(caught.exception.code, "LOCAL_PATH_REQUIRED")
        importer.assert_not_called()

    def test_tensorflow_import_is_lazy(self) -> None:
        sys.modules.pop("tensorflow", None)
        importlib.reload(sys.modules["urbansound_segment_task.edge_v2.models.yamnet_runtime"])
        self.assertNotIn("tensorflow", sys.modules)

    def test_verified_local_artifact_loads_with_direct_savedmodel(self) -> None:
        importer = mock.Mock(return_value=FakeTensorFlow)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            create_artifact(root)
            loaded = load_local_yamnet(root, import_module=importer)

        self.assertEqual(loaded.loader_method, "tf.saved_model.load")
        self.assertEqual(loaded.visible_devices, ("CPU",))
        importer.assert_called_once_with("tensorflow")

    def test_score_and_embedding_contracts(self) -> None:
        valid = validate_yamnet_outputs(
            (FakeTensor([1, 521]), FakeTensor([1, 1024]), FakeTensor([96, 64]))
        )
        self.assertEqual(valid["frame_count"], 1)
        with self.assertRaises(ValueError):
            validate_yamnet_outputs(
                (FakeTensor([1, 520]), FakeTensor([1, 1024]), FakeTensor([96, 64]))
            )
        with self.assertRaises(ValueError):
            validate_yamnet_outputs(
                (FakeTensor([1, 521]), FakeTensor([1, 1000]), FakeTensor([96, 64]))
            )


if __name__ == "__main__":
    unittest.main()
