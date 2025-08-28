import librosa
import numpy as np
from typing import Iterable

# --- Goal 1 için (YAMNet, 16 kHz) ---
def load_mono_16k(path: str, target_sr: int = 16000):
    """Load audio as mono, resample to 16kHz (YAMNet)."""
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32), target_sr

# --- Goal 2 için (ESResNeXt, 44.1 kHz) ---
def load_mono_44k(path: str, target_sr: int = 44100):
    """Load audio as mono, resample to 44.1kHz (ESResNeXt)."""
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32), target_sr

# --- Generic resampler (herhangi bir sr için) ---
def load_mono_resample(path: str, target_sr: int):
    """Load audio as mono, resample to arbitrary sample rate."""
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32), target_sr

# --- Segmentleyici ---
def iter_segments(y: np.ndarray, sr: int,
                  win_sec: float = 0.96,
                  overlap: float = 0.5) -> Iterable[np.ndarray]:
    """
    Break audio into overlapping fixed-length segments.
    - win_sec: window length in seconds (default 0.96)
    - overlap: fraction overlap (default 0.5)
    """
    win = int(sr * win_sec)
    hop = int(win * (1 - overlap))
    if hop <= 0:
        hop = max(1, win // 2)
    n = len(y)
    start = 0
    while start + win <= n:
        yield y[start:start+win]
        start += hop
    # Kısa kuyruk segmenti atlanır
