"""Local-WAV decoding and the legacy mono/16 kHz preprocessing contract."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from urbansound_segment_task.edge_v2.evaluation.segmentation import LEGACY_SEGMENTATION_POLICY
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256


RESAMPLE_TYPE = "soxr_hq"


@dataclass(frozen=True)
class RawAudio:
    samples: np.ndarray
    audio_sha256: str
    safe_name: str
    original_sample_rate: int
    original_channel_count: int
    original_frame_count: int
    original_duration_seconds: float


@dataclass(frozen=True)
class DecodedAudio:
    waveform: np.ndarray
    audio_sha256: str
    safe_name: str
    original_sample_rate: int
    original_channel_count: int
    original_frame_count: int
    original_duration_seconds: float
    resampled_sample_count: int


def _local_wav(path: Path) -> Path:
    value = str(path).lower()
    if value.startswith(("http:/", "https:/")):
        raise ValueError("audio must be a local WAV path")
    local = Path(path)
    if local.suffix.lower() != ".wav":
        raise ValueError("only local WAV input is supported")
    if not local.is_file():
        raise FileNotFoundError("audio WAV does not exist")
    return local


def decode_wav(
    path: Path,
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> RawAudio:
    """Decode a local WAV as float32 frames/channels without exposing its path."""

    local = _local_wav(path)
    soundfile = import_module("soundfile")
    info = soundfile.info(str(local))
    source_rate = int(info.samplerate)
    channels = int(info.channels)
    frames = int(info.frames)
    if source_rate <= 0 or channels <= 0 or frames < 0:
        raise ValueError("invalid WAV metadata")
    values, decoded_rate = soundfile.read(str(local), dtype="float32", always_2d=True)
    if int(decoded_rate) != source_rate or values.shape != (frames, channels):
        raise ValueError("decoded WAV shape or sample rate differs from metadata")
    samples = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(samples)):
        raise ValueError("decoded waveform contains NaN or Inf")
    return RawAudio(
        samples=samples,
        audio_sha256=streaming_file_sha256(local),
        safe_name=local.name,
        original_sample_rate=source_rate,
        original_channel_count=channels,
        original_frame_count=frames,
        original_duration_seconds=frames / source_rate,
    )


def preprocess_audio(
    raw: RawAudio,
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> DecodedAudio:
    """Convert channels by arithmetic mean and resample to 16 kHz with soxr_hq."""

    waveform = raw.samples.mean(axis=1, dtype=np.float32)
    source_rate = raw.original_sample_rate
    target_rate = LEGACY_SEGMENTATION_POLICY.target_sample_rate
    if source_rate != target_rate:
        librosa = import_module("librosa")
        waveform = np.asarray(
            librosa.resample(
                waveform, orig_sr=source_rate, target_sr=target_rate, res_type=RESAMPLE_TYPE
            ),
            dtype=np.float32,
        )
    if waveform.ndim != 1 or not np.all(np.isfinite(waveform)):
        raise ValueError("preprocessed waveform must be finite mono float32")
    return DecodedAudio(
        waveform=waveform,
        audio_sha256=raw.audio_sha256,
        safe_name=raw.safe_name,
        original_sample_rate=raw.original_sample_rate,
        original_channel_count=raw.original_channel_count,
        original_frame_count=raw.original_frame_count,
        original_duration_seconds=raw.original_duration_seconds,
        resampled_sample_count=int(waveform.shape[0]),
    )


def decode_local_wav(
    path: Path,
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> DecodedAudio:
    """Compatibility wrapper combining the separately timed decode and preprocessing stages."""

    return preprocess_audio(decode_wav(path, import_module=import_module), import_module=import_module)


__all__ = [
    "DecodedAudio", "RESAMPLE_TYPE", "RawAudio", "decode_local_wav", "decode_wav",
    "preprocess_audio",
]
