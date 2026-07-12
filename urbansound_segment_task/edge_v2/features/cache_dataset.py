"""Verified in-memory datasets built only from the YAMNet embedding cache."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .yamnet_cache import load_and_validate_cache
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256


EXPECTED_CACHE_IDENTITY = "6b0807688796f3f19ca7129867a518f893d18566ffd057cb31a5302bd6fd17ce"
EXPECTED_DATASET_MANIFEST_SHA256 = "c77946581016eba9aacf00683ba073ccb537f0ca6bfd86ab50d9cea3734c7d2e"
EXPECTED_YAMNET_TREE_SHA256 = "5d3bccc6549dcf864250dd52b9ffa35a1aec0f2b6f88a91229ff0336582e25c2"
EXPECTED_FOLD_SEGMENTS = {1: 5437, 2: 5369, 3: 5844, 4: 6030, 5: 5672, 6: 5069, 7: 5265, 8: 4928, 9: 5102, 10: 5202}
EXPECTED_FOLD_CLIPS = {1: 873, 2: 888, 3: 925, 4: 990, 5: 936, 6: 823, 7: 838, 8: 806, 9: 816, 10: 837}


@dataclass(frozen=True)
class CacheSplit:
    name: str
    folds: tuple[int, ...]
    X: np.ndarray
    y: np.ndarray
    clip_keys: np.ndarray
    metadata_clip_count: int
    evaluable_clip_count: int
    zero_segment_clip_count: int
    segment_count: int


@dataclass(frozen=True)
class LegacyCacheDataset:
    train: CacheSplit
    validation: CacheSplit
    test: CacheSplit
    cache_identity: str
    dataset_manifest_sha256: str
    yamnet_artifact_tree_sha256: str
    index_sha256: str
    load_seconds: float
    verified_artifact_count: int


@dataclass(frozen=True)
class VerifiedCacheRecords:
    records: tuple[dict[str, Any], ...]
    cache_identity: str
    dataset_manifest_sha256: str
    yamnet_artifact_tree_sha256: str
    index_sha256: str
    load_seconds: float
    verified_artifact_count: int


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_cache_split(name: str, folds: Iterable[int], records: Iterable[dict[str, Any]]) -> CacheSplit:
    selected_folds = tuple(sorted(int(fold) for fold in folds))
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    keys: list[np.ndarray] = []
    metadata_clips = evaluable_clips = zero_clips = segments = 0
    for record in records:
        if int(record["fold"]) not in selected_folds:
            continue
        metadata_clips += 1
        embeddings = record["embeddings"]
        count = int(embeddings.shape[0])
        segments += count
        if count == 0:
            zero_clips += 1
            continue
        evaluable_clips += 1
        features.append(embeddings)
        labels.append(np.full(count, int(record["class_id"]), dtype=np.int64))
        keys.append(np.asarray([str(record["clip_key"])] * count, dtype=np.str_))
    X = np.concatenate(features, axis=0) if features else np.empty((0, 1024), dtype=np.float32)
    y = np.concatenate(labels) if labels else np.empty((0,), dtype=np.int64)
    clip_keys = np.concatenate(keys) if keys else np.empty((0,), dtype=np.str_)
    return CacheSplit(
        name=name, folds=selected_folds, X=X, y=y, clip_keys=clip_keys,
        metadata_clip_count=metadata_clips, evaluable_clip_count=evaluable_clips,
        zero_segment_clip_count=zero_clips, segment_count=segments,
    )


def load_verified_cache_records(cache_root: Path) -> VerifiedCacheRecords:
    start = time.perf_counter()
    namespace_root = Path(cache_root) / EXPECTED_CACHE_IDENTITY
    summary = json.loads((namespace_root / "cache-summary.json").read_text(encoding="utf-8"))
    if summary.get("cache_identity") != EXPECTED_CACHE_IDENTITY:
        raise ValueError("cache identity mismatch")
    if summary.get("dataset_manifest_sha256") != EXPECTED_DATASET_MANIFEST_SHA256:
        raise ValueError("dataset manifest SHA-256 mismatch")
    if summary.get("yamnet_artifact_tree_sha256") != EXPECTED_YAMNET_TREE_SHA256:
        raise ValueError("YAMNet artifact tree SHA-256 mismatch")
    index_text = (namespace_root / "cache-index.jsonl").read_text(encoding="utf-8")
    if _sha256_text(index_text) != summary["cache_index"]["sha256"]:
        raise ValueError("cache index SHA-256 mismatch")
    index = [json.loads(line) for line in index_text.splitlines()]
    if len(index) != 8732 or [item["clip_key"] for item in index] != sorted(item["clip_key"] for item in index):
        raise ValueError("cache index count or deterministic order mismatch")
    records: list[dict[str, Any]] = []
    fold_clips = {fold: 0 for fold in range(1, 11)}
    fold_segments = {fold: 0 for fold in range(1, 11)}
    for item in index:
        path = namespace_root / item["cache_relative_path"]
        if streaming_file_sha256(path) != item["cache_artifact_sha256"]:
            raise ValueError("cache artifact SHA-256 mismatch")
        source_hash = str(item.get("source_audio_sha256", ""))
        if len(source_hash) != 64:
            raise ValueError("source audio SHA-256 missing")
        expected = {
            "clip_key": item["clip_key"], "class_id": int(item["class_id"]),
            "fold": int(item["fold"]), "source_audio_sha256": source_hash,
            "yamnet_artifact_tree_sha256": EXPECTED_YAMNET_TREE_SHA256,
        }
        loaded = load_and_validate_cache(path, expected=expected)
        if loaded["segment_start_samples"].shape[0] != loaded["embeddings"].shape[0]:
            raise ValueError("segment starts and embeddings count mismatch")
        record = {
            "clip_key": str(loaded["clip_key"]), "class_id": int(loaded["class_id"]),
            "fold": int(loaded["fold"]), "embeddings": loaded["embeddings"],
        }
        records.append(record)
        fold_clips[record["fold"]] += 1
        fold_segments[record["fold"]] += int(record["embeddings"].shape[0])
    if fold_clips != EXPECTED_FOLD_CLIPS or fold_segments != EXPECTED_FOLD_SEGMENTS:
        raise ValueError("cache fold clip or segment counts mismatch")
    return VerifiedCacheRecords(
        records=tuple(records), cache_identity=EXPECTED_CACHE_IDENTITY,
        dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
        yamnet_artifact_tree_sha256=EXPECTED_YAMNET_TREE_SHA256,
        index_sha256=summary["cache_index"]["sha256"],
        load_seconds=time.perf_counter() - start,
        verified_artifact_count=len(records),
    )


def load_legacy_cache_dataset(cache_root: Path) -> LegacyCacheDataset:
    verified = load_verified_cache_records(cache_root)
    train = build_cache_split("train", range(1, 9), verified.records)
    validation = build_cache_split("validation", (9,), verified.records)
    test = build_cache_split("test", (10,), verified.records)
    if (train.metadata_clip_count, validation.metadata_clip_count, test.metadata_clip_count) != (7079, 816, 837):
        raise ValueError("legacy split metadata clip counts mismatch")
    if (train.segment_count, validation.segment_count, test.segment_count) != (43614, 5102, 5202):
        raise ValueError("legacy split segment counts mismatch")
    return LegacyCacheDataset(
        train=train, validation=validation, test=test,
        cache_identity=verified.cache_identity,
        dataset_manifest_sha256=verified.dataset_manifest_sha256,
        yamnet_artifact_tree_sha256=verified.yamnet_artifact_tree_sha256,
        index_sha256=verified.index_sha256,
        load_seconds=verified.load_seconds,
        verified_artifact_count=verified.verified_artifact_count,
    )


__all__ = [
    "CacheSplit", "LegacyCacheDataset", "VerifiedCacheRecords", "EXPECTED_CACHE_IDENTITY",
    "EXPECTED_DATASET_MANIFEST_SHA256", "EXPECTED_FOLD_CLIPS", "EXPECTED_FOLD_SEGMENTS",
    "EXPECTED_YAMNET_TREE_SHA256", "build_cache_split", "load_legacy_cache_dataset",
    "load_verified_cache_records",
]
