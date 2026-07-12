from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from urbansound_segment_task.edge_v2.runtime.audio import (
    DecodedAudio,
    RawAudio,
    decode_wav,
    preprocess_audio,
)
from urbansound_segment_task.edge_v2.runtime.pipeline import (
    EXPECTED_YAMNET_TREE_SHA256,
    LocalOnnxClassifier,
    LocalYamnetAdapter,
    VerifiedClassifier,
    run_offline_audio_inference,
    verify_classifier_artifact,
    verify_runtime_artifacts,
)
from urbansound_segment_task.edge_v2.runtime.result_schema import (
    CLASS_NAMES,
    OFFLINE_AUDIO_INFERENCE_SCHEMA_VERSION,
    SOFTMAX_LIMITATION,
    write_inference_result,
)


ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def raw_audio(samples: np.ndarray, rate: int = 16_000) -> RawAudio:
    values = np.asarray(samples, dtype=np.float32)
    return RawAudio(
        samples=values, audio_sha256="a" * 64, safe_name="fixture.wav",
        original_sample_rate=rate, original_channel_count=values.shape[1],
        original_frame_count=values.shape[0], original_duration_seconds=values.shape[0] / rate,
    )


def decoded_audio(sample_count: int) -> DecodedAudio:
    return DecodedAudio(
        waveform=np.linspace(-0.5, 0.5, sample_count, dtype=np.float32),
        audio_sha256="a" * 64, safe_name="fixture.wav", original_sample_rate=16_000,
        original_channel_count=1, original_frame_count=sample_count,
        original_duration_seconds=sample_count / 16_000, resampled_sample_count=sample_count,
    )


def verified() -> tuple[dict, VerifiedClassifier]:
    return (
        {"artifact_id": "yamnet-tfhub-v1", "tree_sha256": EXPECTED_YAMNET_TREE_SHA256, "file_count": 4},
        VerifiedClassifier(
            model_path=Path("model.onnx"), artifact_identity="m" * 64,
            onnx_sha256="o" * 64, size_bytes=41_795, architecture_id="linear",
            test_fold=1, validation_fold=2,
        ),
    )


class FakeYamnet:
    loader_method = "fake-local"
    tensorflow_version = "2.15.1"
    visible_devices = ("CPU",)

    def __init__(self, artifact: Path, *, threads: int) -> None:
        self.calls = 0

    def embed(self, waveform: np.ndarray):
        self.calls += 1
        feature = np.zeros(1024, dtype=np.float32)
        feature[self.calls - 1] = 1.0
        return feature, {
            "scores_shape": [2, 521], "embeddings_shape": [2, 1024],
            "spectrogram_shape": [96, 64], "frame_count": 2,
        }


class FakeClassifier:
    active_providers = ("CPUExecutionProvider",)
    onnxruntime_version = "1.20.1"
    instances = []

    def __init__(self, model_path: Path, *, threads: int) -> None:
        self.inputs = []
        self.__class__.instances.append(self)

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.inputs.append(features.copy())
        rows = np.zeros((features.shape[0], 10), dtype=np.float32)
        rows[:, 2] = 0.4
        rows[:, 3] = 0.6
        if rows.shape[0] > 1:
            rows[1, 2], rows[1, 3] = 0.8, 0.2
        return rows


class OfflineAudioRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeClassifier.instances.clear()

    def test_schema_and_canonical_class_order(self) -> None:
        self.assertEqual(OFFLINE_AUDIO_INFERENCE_SCHEMA_VERSION, "edge-v2.offline-audio-inference.v1")
        self.assertEqual(CLASS_NAMES[0], "air_conditioner")
        self.assertEqual(CLASS_NAMES[9], "street_music")
        self.assertEqual(len(CLASS_NAMES), 10)

    def test_only_local_wav_is_accepted(self) -> None:
        for value in ("https://example.test/a.wav", "clip.mp3"):
            with self.assertRaises(ValueError):
                decode_wav(Path(value))

    def test_mono_decode_records_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mono.wav"
            path.write_bytes(b"wav")
            sf = SimpleNamespace(
                info=lambda _: SimpleNamespace(samplerate=16_000, channels=1, frames=3),
                read=lambda *args, **kwargs: (np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32), 16_000),
            )
            result = decode_wav(path, import_module=lambda name: sf)
        self.assertEqual(result.samples.shape, (3, 1))
        self.assertEqual(result.original_channel_count, 1)
        self.assertEqual(result.safe_name, "mono.wav")

    def test_programmatically_generated_wav_fixture_round_trip(self) -> None:
        import soundfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.wav"
            samples = np.column_stack(
                [np.linspace(-0.5, 0.5, 800, dtype=np.float32), np.zeros(800, dtype=np.float32)]
            )
            soundfile.write(path, samples, 8_000, subtype="PCM_16")
            decoded = decode_wav(path)
            preprocessed = preprocess_audio(decoded)
        self.assertEqual(decoded.original_channel_count, 2)
        self.assertEqual(decoded.original_sample_rate, 8_000)
        self.assertEqual(preprocessed.waveform.ndim, 1)
        self.assertEqual(preprocessed.resampled_sample_count, 1_600)

    def test_stereo_is_arithmetic_mono_mean(self) -> None:
        result = preprocess_audio(raw_audio(np.asarray([[1, -1], [0.5, 0.25]], dtype=np.float32)))
        np.testing.assert_allclose(result.waveform, [0.0, 0.375])
        self.assertEqual(result.waveform.dtype, np.float32)

    def test_resampling_uses_soxr_hq_contract(self) -> None:
        calls = []
        librosa = SimpleNamespace(
            resample=lambda values, **kwargs: calls.append(kwargs) or np.zeros(16_000, dtype=np.float32)
        )
        result = preprocess_audio(
            raw_audio(np.zeros((8_000, 1), dtype=np.float32), rate=8_000),
            import_module=lambda name: librosa,
        )
        self.assertEqual(calls, [{"orig_sr": 8_000, "target_sr": 16_000, "res_type": "soxr_hq"}])
        self.assertEqual(result.resampled_sample_count, 16_000)

    def test_nonfinite_audio_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            preprocess_audio(raw_audio(np.asarray([[np.nan]], dtype=np.float32)))

    def test_onnx_hash_mismatch_is_rejected(self) -> None:
        source = ROOT / "artifacts/compact_classifier/linear-fold1-fp32-onnx-v1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy2(source / "artifact-manifest.json", root / "artifact-manifest.json")
            (root / "model.onnx").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "artifact SHA-256 mismatch"):
                verify_classifier_artifact(root)

    def test_yamnet_hash_mismatch_is_rejected(self) -> None:
        with patch(
            "urbansound_segment_task.edge_v2.runtime.pipeline.verify_classifier_artifact",
            return_value=verified()[1],
        ), patch(
            "urbansound_segment_task.edge_v2.runtime.pipeline.verify_yamnet_artifact",
            return_value={"tree_sha256": "0" * 64},
        ):
            with self.assertRaisesRegex(ValueError, "tree SHA-256 mismatch"):
                verify_runtime_artifacts(Path("yamnet"), Path("classifier"))

    def test_artifacts_are_verified_before_audio_decode(self) -> None:
        order = []
        with patch(
            "urbansound_segment_task.edge_v2.runtime.pipeline.verify_runtime_artifacts",
            side_effect=lambda *args: order.append("verify") or (_ for _ in ()).throw(ValueError("bad hash")),
        ):
            with self.assertRaises(ValueError):
                run_offline_audio_inference(
                    audio_path=Path("x.wav"), yamnet_artifact=Path("y"), classifier_artifact=Path("c"),
                    audio_decoder=lambda path: order.append("decode"),
                    yamnet_factory=lambda *args, **kwargs: order.append("tensorflow_import"),
                    classifier_factory=lambda *args, **kwargs: order.append("onnxruntime_import"),
                )
        self.assertEqual(order, ["verify"])

    def test_short_clip_returns_no_prediction_without_loading_models(self) -> None:
        with patch(
            "urbansound_segment_task.edge_v2.runtime.pipeline.verify_runtime_artifacts", return_value=verified()
        ):
            result = run_offline_audio_inference(
                audio_path=Path("x.wav"), yamnet_artifact=Path("y"), classifier_artifact=Path("c"),
                audio_decoder=lambda path: raw_audio(np.zeros((15_359, 1), dtype=np.float32)),
                audio_preprocessor=lambda raw: decoded_audio(15_359),
                yamnet_factory=lambda *args, **kwargs: self.fail("YAMNet must not load"),
                classifier_factory=lambda *args, **kwargs: self.fail("classifier must not load"),
                now=NOW,
            )
        self.assertEqual(result["status"]["outcome"], "no_prediction")
        self.assertEqual(result["status"]["reason"], "zero_segments_under_legacy_policy")
        self.assertEqual(result["runtime"]["model_load_counts"], {"yamnet": 0, "onnx_classifier": 0})

    def test_pipeline_reuses_legacy_segments_and_loads_models_once(self) -> None:
        with patch(
            "urbansound_segment_task.edge_v2.runtime.pipeline.verify_runtime_artifacts", return_value=verified()
        ):
            result = run_offline_audio_inference(
                audio_path=Path("x.wav"), yamnet_artifact=Path("y"), classifier_artifact=Path("c"),
                stable_clip_key="fold1/example.wav", ground_truth_class_id=2,
                audio_decoder=lambda path: raw_audio(np.zeros((23_040, 1), dtype=np.float32)),
                audio_preprocessor=lambda raw: decoded_audio(23_040),
                yamnet_factory=FakeYamnet, classifier_factory=FakeClassifier, now=NOW,
            )
        self.assertEqual(result["segmentation"]["segment_start_samples"], [0, 7_680])
        self.assertEqual(result["segmentation"]["segment_count"], 2)
        self.assertEqual(result["runtime"]["model_load_counts"], {"yamnet": 1, "onnx_classifier": 1})
        self.assertEqual(len(FakeClassifier.instances), 1)
        self.assertEqual(FakeClassifier.instances[0].inputs[0].shape, (2, 1024))

    def test_arithmetic_mean_top3_segment_predictions_and_ground_truth(self) -> None:
        with patch(
            "urbansound_segment_task.edge_v2.runtime.pipeline.verify_runtime_artifacts", return_value=verified()
        ):
            result = run_offline_audio_inference(
                audio_path=Path("x.wav"), yamnet_artifact=Path("y"), classifier_artifact=Path("c"),
                ground_truth_class_id=2,
                audio_decoder=lambda path: raw_audio(np.zeros((23_040, 1), dtype=np.float32)),
                audio_preprocessor=lambda raw: decoded_audio(23_040),
                yamnet_factory=FakeYamnet, classifier_factory=FakeClassifier, now=NOW,
            )
        self.assertEqual([row["class_id"] for row in result["segment_predictions"]], [3, 2])
        self.assertEqual(result["clip_prediction"]["class_id"], 2)
        self.assertAlmostEqual(result["clip_prediction"]["confidence"], 0.6)
        self.assertEqual([row["class_id"] for row in result["clip_prediction"]["top3"]], [2, 3, 0])
        self.assertTrue(result["ground_truth_comparison"]["prediction_matches"])

    def test_softmax_limitation_and_safe_json_round_trip(self) -> None:
        with patch(
            "urbansound_segment_task.edge_v2.runtime.pipeline.verify_runtime_artifacts", return_value=verified()
        ):
            result = run_offline_audio_inference(
                audio_path=Path("/Users/private/secret.wav"), yamnet_artifact=Path("y"), classifier_artifact=Path("c"),
                audio_decoder=lambda path: raw_audio(np.zeros((15_359, 1), dtype=np.float32)),
                audio_preprocessor=lambda raw: decoded_audio(15_359), now=NOW,
            )
        serialized = json.dumps(result, sort_keys=True)
        self.assertIn(SOFTMAX_LIMITATION, result["limitations"])
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("private", serialized)
        self.assertEqual(json.loads(serialized)["schema_version"], OFFLINE_AUDIO_INFERENCE_SCHEMA_VERSION)

    def test_yamnet_embedding_mean_is_1024_and_finite(self) -> None:
        adapter = object.__new__(LocalYamnetAdapter)
        adapter._numpy = np
        adapter._model = lambda waveform: (
            np.zeros((2, 521), dtype=np.float32), np.ones((2, 1024), dtype=np.float32),
            np.zeros((96, 64), dtype=np.float32),
        )
        feature, shapes = adapter.embed(np.zeros(15_360, dtype=np.float32))
        self.assertEqual(feature.shape, (1024,))
        self.assertTrue(np.all(feature == 1))
        self.assertEqual(shapes["frame_count"], 2)

    def test_onnx_probability_shape_finite_and_row_sum_validation(self) -> None:
        classifier = object.__new__(LocalOnnxClassifier)
        classifier._session = SimpleNamespace(
            run=lambda outputs, inputs: [np.asarray([[0.1] * 10], dtype=np.float32)]
        )
        result = classifier.predict(np.zeros((1, 1024), dtype=np.float32))
        self.assertEqual(result.shape, (1, 10))
        classifier._session = SimpleNamespace(
            run=lambda outputs, inputs: [np.asarray([[np.nan] * 10], dtype=np.float32)]
        )
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            classifier.predict(np.zeros((1, 1024), dtype=np.float32))
        classifier._session = SimpleNamespace(
            run=lambda outputs, inputs: [np.asarray([[0.01] * 10], dtype=np.float32)]
        )
        with self.assertRaisesRegex(ValueError, "sum to one"):
            classifier.predict(np.zeros((1, 1024), dtype=np.float32))

    def test_atomic_result_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            write_inference_result(output, {"status": "first"}, pretty=True)
            with self.assertRaises(FileExistsError):
                write_inference_result(output, {"status": "second"}, pretty=True)
            self.assertEqual(json.loads(output.read_text())["status"], "first")


if __name__ == "__main__":
    unittest.main()
