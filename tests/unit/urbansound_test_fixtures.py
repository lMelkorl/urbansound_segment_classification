from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf


COLUMNS = ["slice_file_name", "fsID", "start", "end", "salience", "fold", "classID", "class"]


def create_dataset(root: Path, *, folds=range(1, 11), samples: int = 20_000) -> Path:
    metadata = root / "metadata"
    audio = root / "audio"
    metadata.mkdir(parents=True)
    rows = []
    for fold in folds:
        class_id = (fold - 1) % 10
        filename = f"{fold}-fixture.wav"
        fold_dir = audio / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        waveform = np.linspace(-0.25, 0.25, samples, dtype=np.float32)
        sf.write(fold_dir / filename, waveform, 16_000, subtype="PCM_16")
        rows.append(
            {
                "slice_file_name": filename, "fsID": str(fold), "start": "0", "end": "1.25",
                "salience": "1", "fold": str(fold), "classID": str(class_id),
                "class": f"class-{class_id}",
            }
        )
    with (metadata / "UrbanSound8K.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return root


def rewrite_rows(root: Path, rows: list[dict[str, str]], columns=COLUMNS) -> None:
    with (root / "metadata" / "UrbanSound8K.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
