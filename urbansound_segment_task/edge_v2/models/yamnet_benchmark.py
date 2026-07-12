"""Deterministic inputs and measured operations for the local YAMNet benchmark."""

from __future__ import annotations

import hashlib
import importlib
import math
import random
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from urbansound_segment_task.edge_v2.benchmarks.stages import PipelineStages

from .yamnet_artifact import verify_yamnet_artifact
from .yamnet_runtime import validate_yamnet_outputs


SAMPLE_RATE = 16_000
DEFAULT_SEED = 20_240_701
EXPECTED_FEATURE_SIZE = 1_024


def generate_pcm16_input(sample_count: int, *, seed: int = DEFAULT_SEED) -> bytes:
    """Create repeatable non-silent PCM without NumPy or platform-dependent math."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    rng = random.Random(seed)
    values = []
    for index in range(sample_count):
        t = index / SAMPLE_RATE
        signal = (
            0.34 * math.sin(2.0 * math.pi * 220.0 * t)
            + 0.17 * math.sin(2.0 * math.pi * 440.0 * t + 0.3)
            + 0.08 * math.sin(2.0 * math.pi * 880.0 * t + 0.7)
            + rng.uniform(-0.025, 0.025)
        )
        values.append(max(-32768, min(32767, round(signal * 32767.0))))
    return struct.pack("<" + "h" * sample_count, *values)


def input_identity(pcm_bytes: bytes, sample_count: int, *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    return {
        "generator": "seeded_multisine_with_uniform_noise_v1",
        "seed": seed,
        "sample_rate": SAMPLE_RATE,
        "sample_count": sample_count,
        "duration_seconds": sample_count / SAMPLE_RATE,
        "pcm_encoding": "signed_16_bit_little_endian_mono",
        "sha256": hashlib.sha256(pcm_bytes).hexdigest(),
    }


def decode_pcm16(pcm_bytes: bytes, numpy_module: Any) -> Any:
    return numpy_module.frombuffer(pcm_bytes, dtype="<i2")


def preprocess_pcm(decoded: Any, sample_count: int, numpy_module: Any) -> Any:
    waveform = decoded.astype(numpy_module.float32) / numpy_module.float32(32768.0)
    if waveform.ndim != 1:
        raise ValueError("waveform must be one-dimensional mono audio")
    if int(waveform.shape[0]) != sample_count:
        raise ValueError("waveform sample count does not match benchmark input")
    if not numpy_module.all(numpy_module.isfinite(waveform)):
        raise ValueError("waveform contains non-finite samples")
    if numpy_module.any(waveform < -1.0) or numpy_module.any(waveform > 1.0):
        raise ValueError("waveform normalization is outside [-1, 1]")
    return waveform


def _to_numpy(value: Any) -> Any:
    return value.numpy() if hasattr(value, "numpy") else value


def materialized_inference(model: Callable[[Any], Any], waveform: Any) -> dict[str, Any]:
    outputs = model(waveform)
    contract = validate_yamnet_outputs(outputs)
    scores, embeddings, spectrogram = (_to_numpy(value) for value in outputs)
    # Reading deterministic scalars makes synchronization and consumer use explicit.
    checksum = float(scores.reshape(-1)[0]) + float(spectrogram.reshape(-1)[0])
    return {
        "scores": scores,
        "embeddings": embeddings,
        "spectrogram": spectrogram,
        "contract": contract,
        "materialization_checksum": checksum,
    }


def aggregate_outputs(inference_result: dict[str, Any], numpy_module: Any) -> dict[str, Any]:
    embeddings = inference_result["embeddings"]
    feature_vector = numpy_module.asarray(embeddings).mean(axis=0)
    if feature_vector.ndim != 1 or int(feature_vector.shape[0]) != EXPECTED_FEATURE_SIZE:
        raise ValueError("YAMNet aggregate feature size must be 1024")
    score_mean = float(numpy_module.asarray(inference_result["scores"]).mean())
    checksum = float(feature_vector[:8].sum()) + score_mean
    return {
        "feature_vector": feature_vector,
        "feature_size": EXPECTED_FEATURE_SIZE,
        "score_mean": score_mean,
        "checksum": checksum,
        "contract": inference_result["contract"],
    }


def build_yamnet_pipeline(
    model: Callable[[Any], Any], sample_count: int, numpy_module: Any
) -> PipelineStages:
    return PipelineStages(
        decode=lambda source: decode_pcm16(source, numpy_module),
        preprocess=lambda decoded: preprocess_pcm(decoded, sample_count, numpy_module),
        inference=lambda waveform: materialized_inference(model, waveform),
        aggregate=lambda result: aggregate_outputs(result, numpy_module),
    )


@dataclass(frozen=True)
class LoadedBenchmarkRuntime:
    model: Any
    tensorflow: Any
    numpy: Any
    loader_method: str
    artifact_identity: dict[str, Any]
    tensorflow_version: Optional[str]
    visible_devices: tuple[str, ...]
    requested_intra_op_threads: int
    effective_intra_op_threads: int
    requested_inter_op_threads: int
    effective_inter_op_threads: int
    loader_components_ns: dict[str, int]


class WarmupTimingRecorder:
    """Record only lifecycle warm-up calls while preserving first/steady boundaries."""

    def __init__(
        self,
        inference: Callable[[Any, Any], Any],
        warmup_count: int,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if isinstance(warmup_count, bool) or not isinstance(warmup_count, int) or warmup_count < 0:
            raise ValueError("warmup_count must be a non-negative integer")
        self._inference = inference
        self._warmup_count = warmup_count
        self._clock_ns = clock_ns
        self.call_count = 0
        self.raw_samples_ns: list[int] = []

    def __call__(self, loaded: Any, input_data: Any) -> Any:
        self.call_count += 1
        is_warmup = 2 <= self.call_count <= self._warmup_count + 1
        if not is_warmup:
            return self._inference(loaded, input_data)
        start_ns = int(self._clock_ns())
        result = self._inference(loaded, input_data)
        duration_ns = int(self._clock_ns()) - start_ns
        if duration_ns < 0:
            raise RuntimeError("warm-up clock was not monotonic")
        self.raw_samples_ns.append(duration_ns)
        return result


def load_benchmark_runtime(
    artifact_directory: Path,
    thread_count: int,
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> LoadedBenchmarkRuntime:
    """Verify first, then import/configure TensorFlow before loading or executing ops."""

    verify_start = int(clock_ns())
    identity = verify_yamnet_artifact(Path(artifact_directory))
    verify_ns = int(clock_ns()) - verify_start

    import_start = int(clock_ns())
    tensorflow = import_module("tensorflow")
    tensorflow.config.threading.set_intra_op_parallelism_threads(thread_count)
    tensorflow.config.threading.set_inter_op_parallelism_threads(1)
    effective_intra = int(tensorflow.config.threading.get_intra_op_parallelism_threads())
    effective_inter = int(tensorflow.config.threading.get_inter_op_parallelism_threads())
    if effective_intra != thread_count or effective_inter != 1:
        raise RuntimeError("TensorFlow thread configuration did not take effect")
    numpy_module = import_module("numpy")
    import_and_configuration_ns = int(clock_ns()) - import_start

    model_path = Path(artifact_directory) / identity["model_relative_path"]
    load_start = int(clock_ns())
    model = tensorflow.saved_model.load(str(model_path))
    savedmodel_load_ns = int(clock_ns()) - load_start
    if not callable(model):
        raise TypeError("loaded SavedModel is not callable")
    devices = tuple(
        sorted(
            {
                str(getattr(device, "device_type", "unknown")).upper()
                for device in tensorflow.config.get_visible_devices()
            }
        )
    )
    return LoadedBenchmarkRuntime(
        model=model,
        tensorflow=tensorflow,
        numpy=numpy_module,
        loader_method="tf.saved_model.load",
        artifact_identity=identity,
        tensorflow_version=getattr(tensorflow, "__version__", None),
        visible_devices=devices,
        requested_intra_op_threads=thread_count,
        effective_intra_op_threads=effective_intra,
        requested_inter_op_threads=1,
        effective_inter_op_threads=effective_inter,
        loader_components_ns={
            "artifact_verification_ns": verify_ns,
            "tensorflow_import_and_configuration_ns": import_and_configuration_ns,
            "savedmodel_load_ns": savedmodel_load_ns,
        },
    )


__all__ = [
    "DEFAULT_SEED",
    "EXPECTED_FEATURE_SIZE",
    "SAMPLE_RATE",
    "LoadedBenchmarkRuntime",
    "WarmupTimingRecorder",
    "aggregate_outputs",
    "build_yamnet_pipeline",
    "decode_pcm16",
    "generate_pcm16_input",
    "input_identity",
    "load_benchmark_runtime",
    "materialized_inference",
    "preprocess_pcm",
]
