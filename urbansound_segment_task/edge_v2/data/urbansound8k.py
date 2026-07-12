"""Offline UrbanSound8K layout detection and dataset inventory."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from urbansound_segment_task.edge_v2.benchmarks.schema import utc_timestamp
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256

from .manifest import document_sha256


DATASET_SCHEMA_VERSION = "edge-v2.urbansound8k-dataset-manifest.v1"
REQUIRED_COLUMNS = (
    "slice_file_name", "fsID", "start", "end", "salience", "fold", "classID", "class"
)


class DatasetLayoutError(RuntimeError):
    pass


class DatasetValidationError(RuntimeError):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        super().__init__("UrbanSound8K dataset validation failed")
        self.errors = errors


@dataclass(frozen=True)
class DatasetLayout:
    dataset_root: Path
    metadata_csv: Path
    audio_root: Path


def detect_dataset_layout(dataset_root: Path) -> DatasetLayout:
    root = Path(dataset_root)
    candidates = (
        (root / "UrbanSound8K.csv", root / "audio"),
        (root / "metadata" / "UrbanSound8K.csv", root / "audio"),
        (root / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv", root / "UrbanSound8K" / "audio"),
    )
    for metadata, audio in candidates:
        if metadata.is_file() and audio.is_dir():
            return DatasetLayout(root, metadata, audio)
    raise DatasetLayoutError(
        "expected UrbanSound8K.csv plus audio/fold1..fold10 under --dataset-root"
    )


def _read_audio_header(path: Path, soundfile_info: Optional[Callable[[str], Any]]) -> dict[str, Any]:
    if soundfile_info is None:
        try:
            import soundfile
            soundfile_info = soundfile.info
        except Exception:
            soundfile_info = None
    if soundfile_info is None:
        return {"status": "unavailable", "error_type": "SoundFileUnavailable"}
    try:
        info = soundfile_info(str(path))
        frames = int(info.frames)
        sample_rate = int(info.samplerate)
        return {
            "status": "success",
            "sample_rate": sample_rate,
            "frames": frames,
            "channels": int(info.channels),
            "duration_seconds": frames / sample_rate if sample_rate > 0 else None,
            "error_type": None,
        }
    except Exception as exc:
        return {"status": "failure", "error_type": type(exc).__name__}


def inspect_urbansound8k(
    dataset_root: Path,
    *,
    soundfile_info: Optional[Callable[[str], Any]] = None,
    now=None,
) -> dict[str, Any]:
    layout = detect_dataset_layout(dataset_root)
    errors: list[dict[str, Any]] = []
    clips: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, ...]] = set()
    seen_keys: set[str] = set()
    class_names: dict[int, str] = {}
    with layout.metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing_columns:
            raise DatasetValidationError(
                [{"type": "MissingMetadataColumns", "columns": missing_columns}]
            )
        for row_number, row in enumerate(reader, start=2):
            row_identity = tuple(str(row[column]) for column in REQUIRED_COLUMNS)
            if row_identity in seen_rows:
                errors.append({"type": "DuplicateMetadataRow", "row": row_number})
            seen_rows.add(row_identity)
            try:
                fold = int(row["fold"])
                class_id = int(row["classID"])
            except (TypeError, ValueError):
                errors.append({"type": "InvalidIntegerField", "row": row_number})
                continue
            if fold not in range(1, 11):
                errors.append({"type": "InvalidFold", "row": row_number, "value": fold})
            if class_id not in range(10):
                errors.append({"type": "InvalidClassID", "row": row_number, "value": class_id})
            class_name = str(row["class"])
            previous = class_names.setdefault(class_id, class_name)
            if previous != class_name:
                errors.append({"type": "InconsistentClassName", "row": row_number, "class_id": class_id})
            filename = Path(str(row["slice_file_name"])).name
            if filename != str(row["slice_file_name"]):
                errors.append({"type": "UnsafeSliceFileName", "row": row_number})
            clip_key = f"fold{fold}/{filename}"
            if clip_key in seen_keys:
                errors.append({"type": "DuplicateClipKey", "row": row_number, "clip_key": clip_key})
            seen_keys.add(clip_key)
            audio_path = layout.audio_root / f"fold{fold}" / filename
            exists = audio_path.is_file()
            if not exists:
                errors.append({"type": "MissingAudioFile", "clip_key": clip_key})
            header = _read_audio_header(audio_path, soundfile_info) if exists else {
                "status": "not_run", "error_type": "MissingAudioFile"
            }
            if exists and header["status"] == "failure":
                errors.append({"type": "AudioHeaderFailure", "clip_key": clip_key, "error_type": header["error_type"]})
            clips.append(
                {
                    "clip_key": clip_key,
                    "slice_file_name": filename,
                    "fold": fold,
                    "class_id": class_id,
                    "class_name": class_name,
                    "fs_id": str(row["fsID"]),
                    "start": str(row["start"]),
                    "end": str(row["end"]),
                    "salience": str(row["salience"]),
                    "audio_exists": exists,
                    "audio_header": header,
                }
            )
    clips.sort(key=lambda item: item["clip_key"])
    actual_audio = {
        path.relative_to(layout.audio_root).as_posix()
        for path in layout.audio_root.glob("fold*/*")
        if path.is_file()
    }
    expected_audio = {clip["clip_key"] for clip in clips}
    missing_audio = sorted(expected_audio - actual_audio)
    extra_audio = sorted(actual_audio - expected_audio)
    fold_counts = Counter(clip["fold"] for clip in clips)
    class_counts = Counter(clip["class_id"] for clip in clips)
    fold_class: dict[int, Counter[int]] = defaultdict(Counter)
    for clip in clips:
        fold_class[clip["fold"]][clip["class_id"]] += 1
    document: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(now),
        "dataset_identity": "UrbanSound8K",
        "layout": {
            "metadata_relative_path": layout.metadata_csv.relative_to(layout.dataset_root).as_posix(),
            "audio_relative_path": layout.audio_root.relative_to(layout.dataset_root).as_posix(),
        },
        "metadata_csv_sha256": streaming_file_sha256(layout.metadata_csv),
        "required_columns": list(REQUIRED_COLUMNS),
        "class_order": [
            {"class_id": class_id, "class_name": class_names.get(class_id)} for class_id in range(10)
        ],
        "counts": {
            "metadata_clips": len(clips),
            "existing_audio_files": sum(clip["audio_exists"] for clip in clips),
            "discovered_audio_files": len(actual_audio),
            "missing_audio_files": len(missing_audio),
            "extra_audio_files": len(extra_audio),
            "header_failures": sum(clip["audio_header"]["status"] == "failure" for clip in clips),
        },
        "fold_counts": {str(key): fold_counts.get(key, 0) for key in range(1, 11)},
        "class_counts": {str(key): class_counts.get(key, 0) for key in range(10)},
        "fold_class_counts": {
            str(fold): {str(class_id): fold_class[fold].get(class_id, 0) for class_id in range(10)}
            for fold in range(1, 11)
        },
        "missing_audio": missing_audio,
        "extra_audio": extra_audio,
        "clips": clips,
        "validation_errors": errors,
        "status": {"outcome": "success" if not errors else "failure", "error_count": len(errors)},
    }
    document["manifest_sha256"] = document_sha256(
        document, excluded_fields=("created_at_utc", "manifest_sha256")
    )
    return document


def audio_path_for_clip(layout: DatasetLayout, clip: dict[str, Any]) -> Path:
    return layout.audio_root / clip["clip_key"]


__all__ = [
    "DATASET_SCHEMA_VERSION", "DatasetLayout", "DatasetLayoutError", "DatasetValidationError",
    "REQUIRED_COLUMNS", "audio_path_for_clip", "detect_dataset_layout", "inspect_urbansound8k",
]
