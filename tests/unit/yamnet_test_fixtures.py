from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from urbansound_segment_task.edge_v2.models.yamnet_artifact import build_manifest


FIXED_TIME = datetime(2026, 7, 12, 12, 30, tzinfo=timezone.utc)


def create_artifact(root: Path) -> dict:
    model = root / "model"
    variables = model / "variables"
    variables.mkdir(parents=True)
    (model / "saved_model.pb").write_bytes(b"saved-model")
    (variables / "variables.index").write_bytes(b"index")
    (variables / "variables.data-00000-of-00001").write_bytes(b"data")
    manifest = build_manifest(
        model,
        package_versions={"tensorflow": "2.15.1", "tensorflow-hub": "0.16.1"},
        now=FIXED_TIME,
    )
    (root / "artifact-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


class FakeShape:
    def __init__(self, values: list[int]) -> None:
        self._values = values

    def as_list(self) -> list[int]:
        return list(self._values)


class FakeTensor:
    def __init__(self, shape: list[int]) -> None:
        self.shape = FakeShape(shape)


class FakeModel:
    def __call__(self, waveform: object) -> tuple[FakeTensor, FakeTensor, FakeTensor]:
        sample_count = int(getattr(waveform, "sample_count", 16_000))
        frames = 0 if sample_count == 15_360 else 1
        return FakeTensor([frames, 521]), FakeTensor([frames, 1024]), FakeTensor([96, 64])


class FakeWaveform:
    def __init__(self, sample_count: int) -> None:
        self.sample_count = sample_count


class FakeTensorFlow:
    __version__ = "2.15.1"
    float32 = "float32"

    class saved_model:
        model = FakeModel()

        @classmethod
        def load(cls, _path: str) -> FakeModel:
            return cls.model

    class config:
        @staticmethod
        def get_visible_devices() -> list[object]:
            return [type("Device", (), {"device_type": "CPU"})()]

    @staticmethod
    def zeros(shape: list[int], dtype: object) -> FakeWaveform:
        del dtype
        return FakeWaveform(shape[0])
