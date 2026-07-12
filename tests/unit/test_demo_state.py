from __future__ import annotations

import unittest
from pathlib import Path

from urbansound_segment_task.edge_v2.demo.state import DemoRuntimeState
from urbansound_segment_task.edge_v2.runtime.pipeline import AudioDurationLimitError

from tests.unit.test_demo_presenter import result_fixture


class FakeRuntime:
    constructions = 0

    def __init__(self, **kwargs) -> None:
        self.__class__.constructions += 1
        self.calls = []
        self.startup_timing = {"startup_end_to_end_ns": 1_000_000}

    def safe_status(self):
        return {
            "active_onnx_providers": ["CPUExecutionProvider"],
            "model_load_counts": {"yamnet": 1, "onnx_classifier": 1},
        }

    def analyze(self, path: Path, *, max_duration_seconds: float):
        self.calls.append((path, max_duration_seconds))
        return result_fixture()


class DemoStateTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeRuntime.constructions = 0

    def state(self, factory=FakeRuntime) -> DemoRuntimeState:
        return DemoRuntimeState(
            yamnet_artifact=Path("yamnet"), classifier_artifact=Path("onnx"),
            runtime_factory=factory,
        )

    def test_artifact_failure_never_becomes_ready(self) -> None:
        def failing(**kwargs):
            raise ValueError("hash mismatch at /Users/private/model")

        state = self.state(failing)
        self.assertFalse(state.initialize())
        self.assertFalse(state.ready)
        self.assertNotIn("/Users/", state.initialization_error)

    def test_runtime_initializes_only_once(self) -> None:
        state = self.state()
        self.assertTrue(state.initialize())
        self.assertTrue(state.initialize())
        self.assertEqual(FakeRuntime.constructions, 1)
        self.assertIn("Runtime ready", state.status_markdown())

    def test_multiple_file_and_microphone_filepath_requests_do_not_reload(self) -> None:
        state = self.state()
        state.initialize()
        for path in ("upload.wav", "microphone.wav", "upload-again.wav"):
            presentation = state.analyze_audio(path)
            self.assertEqual(presentation["message"], "")
        self.assertEqual(FakeRuntime.constructions, 1)
        self.assertEqual(len(state.runtime.calls), 3)
        self.assertEqual(state.runtime.safe_status()["model_load_counts"], {"yamnet": 1, "onnx_classifier": 1})

    def test_missing_audio_does_not_call_runtime(self) -> None:
        state = self.state()
        state.initialize()
        presentation = state.analyze_audio(None)
        self.assertIn("Select or record", presentation["message"])
        self.assertEqual(state.runtime.calls, [])

    def test_url_audio_is_rejected(self) -> None:
        state = self.state()
        state.initialize()
        presentation = state.analyze_audio("https://example.test/audio.wav")
        self.assertIn("URLs are not supported", presentation["message"])
        self.assertEqual(state.runtime.calls, [])

    def test_duration_limit_is_structured(self) -> None:
        class LongRuntime(FakeRuntime):
            def analyze(self, path, *, max_duration_seconds):
                raise AudioDurationLimitError("too long")

        state = self.state(LongRuntime)
        state.initialize()
        presentation = state.analyze_audio("long.wav")
        self.assertIn("30-second", presentation["message"])

    def test_traceback_and_sensitive_exception_do_not_reach_user(self) -> None:
        class BrokenRuntime(FakeRuntime):
            def analyze(self, path, *, max_duration_seconds):
                raise RuntimeError("secret /Users/private token=abc")

        state = self.state(BrokenRuntime)
        state.initialize()
        serialized = str(state.analyze_audio("recording.wav"))
        self.assertNotIn("Traceback", serialized)
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("token", serialized)

    def test_runtime_busy_is_structured(self) -> None:
        state = self.state()
        state.initialize()
        state._inference_lock.acquire()
        try:
            presentation = state.analyze_audio("audio.wav")
        finally:
            state._inference_lock.release()
        self.assertIn("Runtime busy", presentation["message"])

    def test_clear_resets_presentation_without_reloading(self) -> None:
        state = self.state()
        state.initialize()
        cleared = state.clear()
        self.assertEqual(cleared["prediction"], "No analysis yet.")
        self.assertEqual(FakeRuntime.constructions, 1)


if __name__ == "__main__":
    unittest.main()

