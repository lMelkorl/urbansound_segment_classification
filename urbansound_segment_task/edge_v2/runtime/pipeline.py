"""Verified, network-free WAV -> YAMNet -> FP32 ONNX inference pipeline."""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from urbansound_segment_task.edge_v2.benchmarks.schema import utc_timestamp
from urbansound_segment_task.edge_v2.data.manifest import document_sha256
from urbansound_segment_task.edge_v2.evaluation.segmentation import (
    LEGACY_SEGMENTATION_POLICY,
    iter_segments,
    segment_plan,
)
from urbansound_segment_task.edge_v2.export.onnx_linear import EXPORT_SCHEMA_VERSION
from urbansound_segment_task.edge_v2.models.yamnet_artifact import (
    streaming_file_sha256,
    verify_yamnet_artifact,
)
from urbansound_segment_task.edge_v2.models.yamnet_runtime import validate_yamnet_outputs

from .audio import DecodedAudio, RESAMPLE_TYPE, RawAudio, decode_wav, preprocess_audio
from .result_schema import (
    CLASS_NAMES,
    OFFLINE_AUDIO_INFERENCE_SCHEMA_VERSION,
    SOFTMAX_LIMITATION,
)


EXPECTED_YAMNET_TREE_SHA256 = "5d3bccc6549dcf864250dd52b9ffa35a1aec0f2b6f88a91229ff0336582e25c2"
EXPECTED_ONNX_SHA256 = "641f41921612048c8ed51d7c7c9d8f8f534827d08a091f87dd5900929731a329"


@dataclass(frozen=True)
class VerifiedClassifier:
    model_path: Path
    artifact_identity: str
    onnx_sha256: str
    size_bytes: int
    architecture_id: str
    test_fold: Optional[int]
    validation_fold: Optional[int]
    deployment_only: bool = False


class LocalYamnetAdapter:
    def __init__(
        self,
        artifact_directory: Path,
        *,
        threads: int,
        import_module: Callable[[str], Any] = importlib.import_module,
    ) -> None:
        tensorflow = import_module("tensorflow")
        tensorflow.config.threading.set_intra_op_parallelism_threads(threads)
        tensorflow.config.threading.set_inter_op_parallelism_threads(1)
        self._numpy = import_module("numpy")
        self._model = tensorflow.saved_model.load(str(Path(artifact_directory) / "model"))
        if not callable(self._model):
            raise TypeError("local YAMNet SavedModel is not callable")
        self.loader_method = "tf.saved_model.load(local_path)"
        self.tensorflow_version = str(getattr(tensorflow, "__version__", "unknown"))
        self.visible_devices = tuple(
            sorted({str(getattr(item, "device_type", "unknown")).upper() for item in tensorflow.config.get_visible_devices()})
        )

    def embed(self, waveform: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        outputs = self._model(waveform)
        shapes = validate_yamnet_outputs(outputs)
        scores = self._numpy.asarray(outputs[0], dtype=self._numpy.float32)
        embeddings = self._numpy.asarray(outputs[1], dtype=self._numpy.float32)
        if not self._numpy.all(self._numpy.isfinite(scores)) or not self._numpy.all(self._numpy.isfinite(embeddings)):
            raise ValueError("YAMNet output contains NaN or Inf")
        feature = embeddings.mean(axis=0).astype(self._numpy.float32, copy=False)
        if feature.shape != (1024,) or not self._numpy.all(self._numpy.isfinite(feature)):
            raise ValueError("YAMNet mean embedding contract mismatch")
        return feature, shapes


class LocalOnnxClassifier:
    def __init__(
        self,
        model_path: Path,
        *,
        threads: int,
        import_module: Callable[[str], Any] = importlib.import_module,
    ) -> None:
        ort = import_module("onnxruntime")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.active_providers = tuple(self._session.get_providers())
        if self.active_providers != ("CPUExecutionProvider",):
            raise ValueError("ONNX Runtime must use only CPUExecutionProvider")
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "embedding" or inputs[0].type != "tensor(float)":
            raise ValueError("ONNX input contract mismatch")
        if len(outputs) != 1 or outputs[0].name != "probabilities" or outputs[0].type != "tensor(float)":
            raise ValueError("ONNX output contract mismatch")
        self.onnxruntime_version = str(getattr(ort, "__version__", "unknown"))

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 1024 or not np.all(np.isfinite(values)):
            raise ValueError("ONNX classifier input must be finite float32 [N,1024]")
        probabilities = np.asarray(
            self._session.run(["probabilities"], {"embedding": values})[0], dtype=np.float32
        )
        if probabilities.shape != (values.shape[0], len(CLASS_NAMES)):
            raise ValueError("ONNX probabilities shape mismatch")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("ONNX probabilities contain NaN or Inf")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5):
            raise ValueError("ONNX probability rows do not sum to one")
        return probabilities


def _reject_url(path: Path, label: str) -> Path:
    if str(path).lower().startswith(("http:/", "https:/")):
        raise ValueError(f"{label} must be a local path")
    return Path(path)


def verify_classifier_artifact(artifact_directory: Path) -> VerifiedClassifier:
    root = _reject_url(artifact_directory, "classifier artifact")
    manifest = json.loads((root / "artifact-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXPORT_SCHEMA_VERSION or manifest.get("status", {}).get("outcome") != "success":
        raise ValueError("FP32 ONNX manifest is incompatible or unsuccessful")
    if manifest.get("manifest_sha256") != document_sha256(
        manifest, excluded_fields=("created_at_utc", "manifest_sha256")
    ):
        raise ValueError("FP32 ONNX manifest SHA-256 mismatch")
    artifact = manifest.get("onnx_artifact", {})
    relative = Path(str(artifact.get("relative_path", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("FP32 ONNX relative path is unsafe")
    model_path = root / relative
    digest = streaming_file_sha256(model_path)
    deployment_only = manifest.get("deployment_only") is True
    if digest != artifact.get("sha256"):
        raise ValueError("FP32 ONNX artifact SHA-256 mismatch")
    if not deployment_only and digest != EXPECTED_ONNX_SHA256:
        raise ValueError("FP32 ONNX artifact SHA-256 mismatch")
    if deployment_only and (
        manifest.get("deployment_id") != "linear-all-data-deployment-v1"
        or manifest.get("independent_test_metrics_available") is not False
        or manifest.get("source_model", {}).get("cache_identity")
        != "6b0807688796f3f19ca7129867a518f893d18566ffd057cb31a5302bd6fd17ce"
        or int(manifest.get("source_model", {}).get("training_segment_count", 0)) != 53_918
    ):
        raise ValueError("deployment-only classifier provenance mismatch")
    size = model_path.stat().st_size
    if size != int(artifact.get("size_bytes", -1)):
        raise ValueError("FP32 ONNX artifact size mismatch")
    input_contract = manifest.get("input_contract")
    output_contract = manifest.get("output_contract")
    if input_contract != {"name": "embedding", "dtype": "float32", "shape": ["batch", 1024]}:
        raise ValueError("FP32 ONNX input manifest contract mismatch")
    if output_contract != {
        "name": "probabilities", "dtype": "float32", "shape": ["batch", 10],
        "semantic": "softmax_probabilities",
    }:
        raise ValueError("FP32 ONNX output manifest contract mismatch")
    fold = manifest.get("fold_identity", {})
    return VerifiedClassifier(
        model_path=model_path,
        artifact_identity=str(manifest["manifest_sha256"]),
        onnx_sha256=digest,
        size_bytes=size,
        architecture_id=str(manifest.get("architecture_id")),
        test_fold=None if deployment_only else int(fold.get("test_fold")),
        validation_fold=None if deployment_only else int(fold.get("validation_fold")),
        deployment_only=deployment_only,
    )


def verify_runtime_artifacts(yamnet_artifact: Path, classifier_artifact: Path) -> tuple[dict[str, Any], VerifiedClassifier]:
    yamnet_root = _reject_url(yamnet_artifact, "YAMNet artifact")
    classifier = verify_classifier_artifact(classifier_artifact)
    yamnet = verify_yamnet_artifact(yamnet_root)
    if yamnet["tree_sha256"] != EXPECTED_YAMNET_TREE_SHA256:
        raise ValueError("YAMNet artifact tree SHA-256 mismatch")
    return yamnet, classifier


def _top_predictions(probabilities: np.ndarray, count: int) -> list[dict[str, Any]]:
    order = np.argsort(-probabilities, kind="stable")[:count]
    return [
        {"class_id": int(index), "class_name": CLASS_NAMES[int(index)], "confidence": float(probabilities[index])}
        for index in order
    ]


def run_offline_audio_inference(
    *,
    audio_path: Path,
    yamnet_artifact: Path,
    classifier_artifact: Path,
    stable_clip_key: Optional[str] = None,
    ground_truth_class_id: Optional[int] = None,
    threads: int = 1,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    audio_decoder: Callable[[Path], RawAudio] = decode_wav,
    audio_preprocessor: Callable[[RawAudio], DecodedAudio] = preprocess_audio,
    yamnet_factory: Callable[..., Any] = LocalYamnetAdapter,
    classifier_factory: Callable[..., Any] = LocalOnnxClassifier,
    now: Any = None,
) -> dict[str, Any]:
    if threads < 1:
        raise ValueError("threads must be positive")
    if ground_truth_class_id is not None and ground_truth_class_id not in range(len(CLASS_NAMES)):
        raise ValueError("ground truth class ID must be canonical")
    start = int(clock_ns())
    stage_start = int(clock_ns())
    yamnet_identity, classifier_identity = verify_runtime_artifacts(yamnet_artifact, classifier_artifact)
    verification_ns = int(clock_ns()) - stage_start

    stage_start = int(clock_ns())
    raw_audio = audio_decoder(Path(audio_path))
    decode_ns = int(clock_ns()) - stage_start
    stage_start = int(clock_ns())
    audio = audio_preprocessor(raw_audio)
    preprocessing_ns = int(clock_ns()) - stage_start
    plan = segment_plan(audio.resampled_sample_count)
    base = {
        "schema_version": OFFLINE_AUDIO_INFERENCE_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(now),
        "runtime": {
            "mode": "offline_local_only", "threads": threads,
            "model_load_counts": {"yamnet": 0, "onnx_classifier": 0},
        },
        "audio_identity": {
            "safe_name": audio.safe_name, "stable_clip_key": stable_clip_key,
            "sha256": audio.audio_sha256,
        },
        "audio_metadata": {
            "original_sample_rate": audio.original_sample_rate,
            "original_channel_count": audio.original_channel_count,
            "original_frame_count": audio.original_frame_count,
            "original_duration_seconds": audio.original_duration_seconds,
        },
        "preprocessing": {
            "target_sample_rate": LEGACY_SEGMENTATION_POLICY.target_sample_rate,
            "resampled_sample_count": audio.resampled_sample_count,
            "output_channels": "mono", "output_dtype": "float32",
            "resampling": {"implementation": "librosa.resample", "res_type": RESAMPLE_TYPE},
        },
        "segmentation": {**LEGACY_SEGMENTATION_POLICY.as_dict(), **plan},
        "yamnet_artifact": {
            "artifact_id": yamnet_identity["artifact_id"],
            "tree_sha256": yamnet_identity["tree_sha256"],
            "file_count": yamnet_identity["file_count"],
        },
        "classifier_artifact": {
            "artifact_identity": classifier_identity.artifact_identity,
            "onnx_sha256": classifier_identity.onnx_sha256,
            "size_bytes": classifier_identity.size_bytes,
            "precision": "fp32", "architecture_id": classifier_identity.architecture_id,
            "test_fold": classifier_identity.test_fold,
            "validation_fold": classifier_identity.validation_fold,
            "deployment_only": classifier_identity.deployment_only,
        },
        "providers": [], "segment_predictions": [], "clip_prediction": None,
        "ground_truth_comparison": (
            None if ground_truth_class_id is None else {
                "class_id": ground_truth_class_id,
                "class_name": CLASS_NAMES[ground_truth_class_id],
                "prediction_matches": None,
            }
        ),
        "limitations": [
            SOFTMAX_LIMITATION,
            (
                "The classifier is deployment-only and trained on all evaluable cached folds; no independent test metrics are available."
                if classifier_identity.deployment_only
                else "The classifier is a deterministic Fold 1 export-smoke artifact, not a general production model."
            ),
            "This single-clip integration result is not an accuracy, macro-F1, or latency benchmark.",
            "YAMNet scores are validated but are not used as the final UrbanSound class prediction.",
        ],
    }
    if plan["segment_count"] == 0:
        base["timing"] = {
            "artifact_verification_ns": verification_ns,
            "audio_decode_ns": decode_ns,
            "resampling_and_preprocessing_ns": preprocessing_ns,
            "yamnet_load_ns": 0, "onnx_session_creation_ns": 0,
            "segment_inference_total_ns": 0, "aggregation_ns": 0,
            "end_to_end_ns": int(clock_ns()) - start,
            "classification": "descriptive_smoke_wall_time_not_a_benchmark",
        }
        base["status"] = {"outcome": "no_prediction", "reason": "zero_segments_under_legacy_policy", "error": None}
        return base

    stage_start = int(clock_ns())
    yamnet = yamnet_factory(Path(yamnet_artifact), threads=threads)
    yamnet_load_ns = int(clock_ns()) - stage_start
    base["runtime"]["model_load_counts"]["yamnet"] = 1
    stage_start = int(clock_ns())
    classifier = classifier_factory(classifier_identity.model_path, threads=threads)
    classifier_load_ns = int(clock_ns()) - stage_start
    base["runtime"]["model_load_counts"]["onnx_classifier"] = 1

    stage_start = int(clock_ns())
    features = []
    yamnet_shapes = []
    for segment in iter_segments(audio.waveform):
        feature, shapes = yamnet.embed(np.asarray(segment, dtype=np.float32))
        features.append(np.asarray(feature, dtype=np.float32))
        yamnet_shapes.append(shapes)
    matrix = np.stack(features).astype(np.float32, copy=False)
    probabilities = classifier.predict(matrix)
    inference_ns = int(clock_ns()) - stage_start

    stage_start = int(clock_ns())
    for index, row in enumerate(probabilities):
        top = _top_predictions(row, 1)[0]
        base["segment_predictions"].append(
            {
                "segment_index": index,
                "start_sample": plan["segment_start_samples"][index],
                "class_id": top["class_id"], "class_name": top["class_name"],
                "confidence": top["confidence"],
                "yamnet_frame_count": int(yamnet_shapes[index]["frame_count"]),
            }
        )
    mean_probability = probabilities.mean(axis=0, dtype=np.float64)
    top3 = _top_predictions(mean_probability, 3)
    final = top3[0]
    base["clip_prediction"] = {**final, "top3": top3, "segment_count": int(probabilities.shape[0])}
    if base["ground_truth_comparison"] is not None:
        base["ground_truth_comparison"]["prediction_matches"] = final["class_id"] == ground_truth_class_id
    aggregation_ns = int(clock_ns()) - stage_start
    base["providers"] = list(classifier.active_providers)
    base["runtime"].update(
        {
            "tensorflow_version": yamnet.tensorflow_version,
            "onnxruntime_version": classifier.onnxruntime_version,
            "yamnet_loader_method": yamnet.loader_method,
            "tensorflow_visible_device_types": list(yamnet.visible_devices),
        }
    )
    base["timing"] = {
        "artifact_verification_ns": verification_ns,
        "audio_decode_ns": decode_ns,
        "resampling_and_preprocessing_ns": preprocessing_ns,
        "yamnet_load_ns": yamnet_load_ns,
        "onnx_session_creation_ns": classifier_load_ns,
        "segment_inference_total_ns": inference_ns,
        "aggregation_ns": aggregation_ns,
        "end_to_end_ns": int(clock_ns()) - start,
        "classification": "descriptive_smoke_wall_time_not_a_benchmark",
    }
    base["status"] = {"outcome": "success", "reason": None, "error": None}
    return base


__all__ = [
    "EXPECTED_ONNX_SHA256", "EXPECTED_YAMNET_TREE_SHA256", "LocalOnnxClassifier",
    "LocalYamnetAdapter", "VerifiedClassifier", "run_offline_audio_inference",
    "verify_classifier_artifact", "verify_runtime_artifacts",
]
