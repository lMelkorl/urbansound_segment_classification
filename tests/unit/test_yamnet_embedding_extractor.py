from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tests.unit.urbansound_test_fixtures import create_dataset
from urbansound_segment_task.edge_v2.features.extractor import extract_yamnet_embeddings


ROOT = Path(__file__).resolve().parents[2]
TREE_HASH = "5d3bccc6549dcf864250dd52b9ffa35a1aec0f2b6f88a91229ff0336582e25c2"


class FakeBackend:
    loads = 0

    def __init__(self, artifact, threads):
        type(self).loads += 1

    def extract(self, waveform):
        return np.full((1024,), float(waveform.shape[0]), dtype=np.float32)


def audio_loader(path):
    return np.linspace(-0.1, 0.1, 23_040, dtype=np.float32), 16_000


class YamnetEmbeddingExtractorTests(unittest.TestCase):
    def setUp(self):
        FakeBackend.loads = 0

    def run_extract(self, root, cache, **changes):
        values = dict(
            dataset_root=root, artifact=Path("safe-artifact"), cache_root=cache, threads=8,
            limit_clips=2, confirm_full_run=False, force=False,
            backend_factory=FakeBackend, audio_loader=audio_loader,
        )
        values.update(changes)
        with patch(
            "urbansound_segment_task.edge_v2.features.extractor.verify_yamnet_artifact",
            return_value={"tree_sha256": TREE_HASH, "artifact_id": "yamnet-tfhub-v1"},
        ):
            return extract_yamnet_embeddings(**values)

    def test_smoke_extract_model_load_once_and_resume_skip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = create_dataset(Path(directory) / "dataset", folds=[1, 2])
            cache = Path(directory) / "cache"
            first = self.run_extract(root, cache)
            second = self.run_extract(root, cache)
        self.assertEqual(first["counts"]["successful"], 2)
        self.assertEqual(first["segmentation_summary"]["total_segments"], 4)
        self.assertEqual(first["repeatability_validation"]["allclose"], True)
        self.assertEqual(second["counts"]["skipped"], 2)
        self.assertEqual(FakeBackend.loads, 1)

    def test_corrupt_cache_is_regenerated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = create_dataset(Path(directory) / "dataset", folds=[1])
            cache = Path(directory) / "cache"
            first = self.run_extract(root, cache, limit_clips=1)
            namespace = cache / first["cache_identity"]
            artifact = next(namespace.glob("fold1/*.npz"))
            with artifact.open("ab") as handle:
                handle.write(b"hash-mismatch-with-valid-zip-body")
            second = self.run_extract(root, cache, limit_clips=1)
        self.assertEqual(second["counts"]["regenerated_corrupt"], 1)

    def test_full_run_requires_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.run_extract(Path(directory), Path(directory) / "cache", limit_clips=None)

    def test_summary_contains_no_absolute_dataset_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = create_dataset(Path(directory) / "dataset", folds=[1])
            summary = self.run_extract(root, Path(directory) / "cache", limit_clips=1)
            self.assertNotIn(str(root), json.dumps(summary))

    def test_all_three_cli_help_commands(self) -> None:
        for name in (
            "inspect_urbansound8k.py", "build_urbansound8k_manifests.py",
            "extract_yamnet_embeddings.py",
        ):
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / name), "--help"], cwd=ROOT,
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
