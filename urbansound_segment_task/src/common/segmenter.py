import librosa
import numpy as np
from typing import Iterable

def load_mono_16k(path: str, target_sr: int = 16000):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32), target_sr

def iter_segments(y: np.ndarray, sr: int, win_sec: float = 0.96, overlap: float = 0.5) -> Iterable[np.ndarray]:
    win = int(sr * win_sec)
    hop = int(win * (1 - overlap))
    if hop <= 0:
        hop = max(1, win // 2)
    n = len(y)
    start = 0
    while start + win <= n:
        yield y[start:start+win]
        start += hop
    # kısa kuyruk segmenti atlanır
