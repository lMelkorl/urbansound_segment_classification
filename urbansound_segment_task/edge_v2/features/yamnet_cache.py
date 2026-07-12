"""Content-addressed, atomically-written YAMNet embedding cache artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

from urbansound_segment_task.edge_v2.data.manifest import canonical_json_bytes
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256


CACHE_SCHEMA_VERSION = "edge-v2.yamnet-embedding-cache.v1"
OUTPUT_DTYPE = "float32"


def cache_namespace_identity(
    *, yamnet_tree_sha256: str, segmentation_config: Mapping[str, Any], resampling: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    identity = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "yamnet_artifact_tree_sha256": yamnet_tree_sha256,
        "segmentation": dict(segmentation_config),
        "resampling": dict(resampling),
        "target_sample_rate": int(segmentation_config["target_sample_rate"]),
        "output_dtype": OUTPUT_DTYPE,
    }
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest(), identity


def clip_cache_key(
    audio_sha256: str, namespace_identity: Mapping[str, Any], clip_key: str
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "audio_sha256": audio_sha256,
                "clip_key": clip_key,
                "namespace": dict(namespace_identity),
            }
        )
    ).hexdigest()


def cache_artifact_path(cache_root: Path, namespace_key: str, fold: int, cache_key: str) -> Path:
    return Path(cache_root) / namespace_key / f"fold{fold}" / f"{cache_key}.npz"


def validate_embedding_arrays(embeddings: Any, starts: Any) -> None:
    import numpy as np
    if embeddings.dtype != np.float32 or embeddings.ndim != 2 or embeddings.shape[1] != 1024:
        raise ValueError("embeddings must have dtype float32 and shape [segment_count, 1024]")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("embeddings contain NaN or Inf")
    if starts.dtype != np.int64 or starts.ndim != 1 or starts.shape[0] != embeddings.shape[0]:
        raise ValueError("segment starts must be int64 and align with embeddings")
    if starts.size and (np.any(starts < 0) or np.any(np.diff(starts) != 7_680)):
        raise ValueError("segment starts must be non-negative and spaced by 7680")


def _scalar(value: Any) -> Any:
    return value.item() if getattr(value, "shape", None) == () else value


def load_and_validate_cache(
    path: Path, *, expected: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    import numpy as np
    try:
        with Path(path).open("rb") as source:
            with np.load(source, allow_pickle=False) as archive:
                embeddings = archive["embeddings"]
                starts = archive["segment_start_samples"]
                metadata = {
                    name: _scalar(archive[name])
                    for name in (
                        "clip_key", "class_id", "fold", "source_audio_sha256", "source_sample_rate",
                        "target_sample_rate", "resampled_sample_count", "segmentation_policy_id",
                        "yamnet_artifact_tree_sha256", "cache_schema_version",
                    )
                }
    except Exception as exc:
        raise ValueError("cache artifact cannot be loaded") from exc
    validate_embedding_arrays(embeddings, starts)
    if metadata["cache_schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError("cache schema version mismatch")
    if expected:
        for key, value in expected.items():
            if key in metadata and metadata[key] != value:
                raise ValueError("cache metadata mismatch")
    return {"embeddings": embeddings, "segment_start_samples": starts, **metadata}


def write_cache_atomic(
    path: Path,
    *, embeddings: Any,
    segment_start_samples: Any,
    metadata: Mapping[str, Any],
    replace: bool = False,
) -> str:
    import numpy as np
    validate_embedding_arrays(embeddings, segment_start_samples)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace:
        raise FileExistsError("cache artifact already exists")
    fd, temporary_name = tempfile.mkstemp(prefix=".yamnet-cache-", suffix=".npz", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(
                handle,
                embeddings=embeddings,
                segment_start_samples=segment_start_samples,
                clip_key=np.asarray(str(metadata["clip_key"])),
                class_id=np.asarray(int(metadata["class_id"]), dtype=np.int64),
                fold=np.asarray(int(metadata["fold"]), dtype=np.int64),
                source_audio_sha256=np.asarray(str(metadata["source_audio_sha256"])),
                source_sample_rate=np.asarray(int(metadata["source_sample_rate"]), dtype=np.int64),
                target_sample_rate=np.asarray(int(metadata["target_sample_rate"]), dtype=np.int64),
                resampled_sample_count=np.asarray(int(metadata["resampled_sample_count"]), dtype=np.int64),
                segmentation_policy_id=np.asarray(str(metadata["segmentation_policy_id"])),
                yamnet_artifact_tree_sha256=np.asarray(str(metadata["yamnet_artifact_tree_sha256"])),
                cache_schema_version=np.asarray(CACHE_SCHEMA_VERSION),
            )
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary_name, destination)
        elif os.name == "nt":
            os.rename(temporary_name, destination)
        else:
            os.link(temporary_name, destination)
        load_and_validate_cache(destination)
        return streaming_file_sha256(destination)
    finally:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass


def verify_cache_artifact(path: Path, expected_sha256: str) -> dict[str, Any]:
    actual = streaming_file_sha256(Path(path))
    if actual != expected_sha256:
        raise ValueError("cache artifact SHA-256 mismatch")
    loaded = load_and_validate_cache(path)
    return {"sha256": actual, "segment_count": int(loaded["embeddings"].shape[0])}


def atomic_replace_text(path: Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".cache-index-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "CACHE_SCHEMA_VERSION", "OUTPUT_DTYPE", "atomic_replace_text", "cache_artifact_path",
    "cache_namespace_identity", "clip_cache_key", "load_and_validate_cache",
    "validate_embedding_arrays", "verify_cache_artifact", "write_cache_atomic",
]
