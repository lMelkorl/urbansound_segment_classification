"""Offline, resumable UrbanSound8K YAMNet embedding extraction."""

from __future__ import annotations

import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from urbansound_segment_task.edge_v2.benchmarks.schema import utc_timestamp
from urbansound_segment_task.edge_v2.data.manifest import document_sha256
from urbansound_segment_task.edge_v2.data.urbansound8k import (
    audio_path_for_clip, detect_dataset_layout, inspect_urbansound8k,
)
from urbansound_segment_task.edge_v2.evaluation.segmentation import (
    LEGACY_SEGMENTATION_POLICY, iter_segments, segment_plan,
)
from urbansound_segment_task.edge_v2.models.yamnet_artifact import (
    streaming_file_sha256, verify_yamnet_artifact,
)

from .yamnet_cache import (
    CACHE_SCHEMA_VERSION, atomic_replace_text, cache_artifact_path, cache_namespace_identity,
    clip_cache_key, load_and_validate_cache, verify_cache_artifact, write_cache_atomic,
)


EXPECTED_YAMNET_TREE_SHA256 = "5d3bccc6549dcf864250dd52b9ffa35a1aec0f2b6f88a91229ff0336582e25c2"
RESAMPLING_IMPLEMENTATION = "librosa.load-soxr_hq"


def resampling_identity() -> dict[str, Any]:
    return {
        "implementation": RESAMPLING_IMPLEMENTATION,
        "librosa_version": importlib.metadata.version("librosa"),
        "res_type": "soxr_hq",
        "mono": True,
    }


def load_audio_legacy(path: Path) -> tuple[Any, int]:
    import librosa
    import soundfile
    source_rate = int(soundfile.info(str(path)).samplerate)
    waveform, _ = librosa.load(
        str(path), sr=LEGACY_SEGMENTATION_POLICY.target_sample_rate, mono=True,
        dtype="float32", res_type="soxr_hq",
    )
    return waveform, source_rate


class LocalYamnetEmbeddingBackend:
    def __init__(self, artifact: Path, threads: int) -> None:
        from urbansound_segment_task.edge_v2.models.yamnet_benchmark import load_benchmark_runtime
        loaded = load_benchmark_runtime(Path(artifact), threads)
        self.model = loaded.model
        self.numpy = loaded.numpy
        self.tensorflow = loaded.tensorflow

    def extract(self, waveform: Any) -> Any:
        outputs = self.model(waveform)
        embeddings = outputs[1].numpy() if hasattr(outputs[1], "numpy") else outputs[1]
        result = self.numpy.asarray(embeddings, dtype=self.numpy.float32).mean(axis=0)
        if result.shape != (1024,) or not self.numpy.all(self.numpy.isfinite(result)):
            raise ValueError("invalid YAMNet segment embedding")
        return result


def _extract_clip(waveform: Any, backend: Any) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np
    plan = segment_plan(int(waveform.shape[0]))
    vectors = [backend.extract(segment) for segment in iter_segments(waveform)]
    embeddings = (
        np.stack(vectors).astype(np.float32, copy=False)
        if vectors else np.empty((0, 1024), dtype=np.float32)
    )
    starts = np.asarray(plan["segment_start_samples"], dtype=np.int64)
    return embeddings, starts, plan


def extract_yamnet_embeddings(
    *, dataset_root: Path, artifact: Path, cache_root: Path, threads: int,
    limit_clips: Optional[int], confirm_full_run: bool, force: bool = False,
    backend_factory: Callable[[Path, int], Any] = LocalYamnetEmbeddingBackend,
    audio_loader: Callable[[Path], tuple[Any, int]] = load_audio_legacy,
    now=None,
) -> dict[str, Any]:
    if limit_clips is None and not confirm_full_run:
        raise ValueError("--confirm-full-run is required when --limit-clips is not provided")
    artifact_identity = verify_yamnet_artifact(artifact)
    if artifact_identity["tree_sha256"] != EXPECTED_YAMNET_TREE_SHA256:
        raise ValueError("YAMNet artifact tree SHA-256 mismatch")
    dataset_manifest = inspect_urbansound8k(dataset_root, now=now)
    if dataset_manifest["status"]["outcome"] != "success":
        raise ValueError("dataset inventory must pass before extraction")
    layout = detect_dataset_layout(dataset_root)
    clips = list(dataset_manifest["clips"])
    if limit_clips is not None:
        if limit_clips < 1:
            raise ValueError("limit_clips must be positive")
        clips = clips[:limit_clips]
    namespace_key, namespace = cache_namespace_identity(
        yamnet_tree_sha256=artifact_identity["tree_sha256"],
        segmentation_config=LEGACY_SEGMENTATION_POLICY.as_dict(),
        resampling=resampling_identity(),
    )
    namespace_root = Path(cache_root) / namespace_key
    prior_index: dict[str, dict[str, Any]] = {}
    prior_index_path = namespace_root / "cache-index.jsonl"
    if prior_index_path.is_file():
        try:
            for line in prior_index_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if isinstance(item, dict) and isinstance(item.get("cache_key"), str):
                    prior_index[item["cache_key"]] = item
        except (OSError, json.JSONDecodeError):
            prior_index = {}
    backend = None
    index: list[dict[str, Any]] = []
    counts = {"successful": 0, "skipped": 0, "failed": 0, "regenerated_corrupt": 0}
    segmentation_totals = {
        "total_clips": len(clips), "clips_with_segments": 0, "short_clips_zero_segments": 0,
        "total_segments": 0, "total_dropped_tail_samples": 0,
    }
    extraction_start = time.perf_counter()
    newly_extracted_segments = 0
    repeatability: Optional[dict[str, Any]] = None
    for clip in clips:
        audio_path = audio_path_for_clip(layout, clip)
        audio_sha = streaming_file_sha256(audio_path)
        cache_key = clip_cache_key(audio_sha, namespace, clip["clip_key"])
        cache_path = cache_artifact_path(cache_root, namespace_key, int(clip["fold"]), cache_key)
        expected = {
            "clip_key": clip["clip_key"], "class_id": int(clip["class_id"]),
            "fold": int(clip["fold"]), "source_audio_sha256": audio_sha,
            "target_sample_rate": 16_000, "segmentation_policy_id": LEGACY_SEGMENTATION_POLICY.policy_id,
            "yamnet_artifact_tree_sha256": artifact_identity["tree_sha256"],
        }
        corrupt = False
        if cache_path.exists() and not force:
            try:
                prior_hash = prior_index.get(cache_key, {}).get("cache_artifact_sha256")
                current_hash = streaming_file_sha256(cache_path)
                if prior_hash is not None and current_hash != prior_hash:
                    raise ValueError("cache artifact SHA-256 differs from prior index")
                loaded = load_and_validate_cache(cache_path, expected=expected)
                artifact_hash = current_hash
                plan = segment_plan(int(loaded["resampled_sample_count"]))
                status = "skipped"
                counts["skipped"] += 1
            except ValueError:
                corrupt = True
        if not cache_path.exists() or force or corrupt:
            try:
                if backend is None:
                    backend = backend_factory(Path(artifact), threads)
                waveform, source_rate = audio_loader(audio_path)
                embeddings, starts, plan = _extract_clip(waveform, backend)
                metadata = {
                    **expected, "source_sample_rate": source_rate,
                    "resampled_sample_count": int(waveform.shape[0]),
                }
                artifact_hash = write_cache_atomic(
                    cache_path, embeddings=embeddings, segment_start_samples=starts,
                    metadata=metadata, replace=cache_path.exists(),
                )
                verify_cache_artifact(cache_path, artifact_hash)
                status = "regenerated_corrupt" if corrupt else "successful"
                counts["successful"] += 1
                newly_extracted_segments += int(embeddings.shape[0])
                if corrupt:
                    counts["regenerated_corrupt"] += 1
                if repeatability is None and embeddings.shape[0] > 0:
                    repeated, repeated_starts, _ = _extract_clip(waveform, backend)
                    import numpy as np
                    repeatability = {
                        "clip_key": clip["clip_key"],
                        "allclose": bool(np.allclose(embeddings, repeated, rtol=1e-5, atol=1e-6)),
                        "starts_equal": bool(np.array_equal(starts, repeated_starts)),
                        "rtol": 1e-5, "atol": 1e-6,
                    }
                    if not repeatability["allclose"] or not repeatability["starts_equal"]:
                        raise ValueError("repeat extraction numerical validation failed")
            except Exception as exc:
                counts["failed"] += 1
                index.append({
                    "clip_key": clip["clip_key"], "fold": clip["fold"], "class_id": clip["class_id"],
                    "cache_key": cache_key, "status": "failed", "error_type": type(exc).__name__,
                })
                continue
        segment_count = int(plan["segment_count"])
        segmentation_totals["clips_with_segments"] += int(segment_count > 0)
        segmentation_totals["short_clips_zero_segments"] += int(segment_count == 0)
        segmentation_totals["total_segments"] += segment_count
        segmentation_totals["total_dropped_tail_samples"] += int(plan["dropped_tail_samples"])
        index.append({
            "clip_key": clip["clip_key"], "fold": clip["fold"], "class_id": clip["class_id"],
            "cache_key": cache_key,
            "cache_relative_path": cache_path.relative_to(namespace_root).as_posix(),
            "cache_artifact_sha256": artifact_hash, "source_audio_sha256": audio_sha,
            "source_sample_rate": int(loaded["source_sample_rate"]) if status == "skipped" else source_rate,
            "target_sample_rate": 16_000,
            "resampled_sample_count": int(loaded["resampled_sample_count"]) if status == "skipped" else int(waveform.shape[0]),
            "segment_count": segment_count,
            "segment_start_samples": plan["segment_start_samples"],
            "dropped_tail_samples": plan["dropped_tail_samples"],
            "short_clip_zero_segment": plan["short_clip_zero_segment"],
            "status": status, "error_type": None,
        })
    elapsed = time.perf_counter() - extraction_start
    index.sort(key=lambda item: item["clip_key"])
    index_text = "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in index)
    atomic_replace_text(namespace_root / "cache-index.jsonl", index_text)
    cache_files = sorted(namespace_root.glob("fold*/*.npz"))
    cache_size_bytes = sum(path.stat().st_size for path in cache_files)
    summary: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(now),
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "artifact_identity": artifact_identity["artifact_id"],
        "yamnet_artifact_tree_sha256": artifact_identity["tree_sha256"],
        "cache_identity": namespace_key, "cache_identity_fields": namespace,
        "segmentation_policy": LEGACY_SEGMENTATION_POLICY.as_dict(),
        "operational": {"threads": threads, "limit_clips": limit_clips, "force": force},
        "counts": counts, "segmentation_summary": segmentation_totals,
        "cache_size_bytes": cache_size_bytes,
        "duration_seconds": elapsed,
        "operational_segments_per_second": (
            newly_extracted_segments / elapsed if elapsed > 0 else None
        ),
        "cache_validation": {
            "verified_artifacts": sum(item["status"] != "failed" for item in index),
            "failed_artifacts": counts["failed"],
            "all_valid": counts["failed"] == 0,
        },
        "repeatability_validation": repeatability,
        "cache_index": {"relative_path": "cache-index.jsonl", "sha256": hashlib_sha256(index_text)},
        "status": {
            "outcome": "success" if counts["failed"] == 0 else "failure",
            "error": None if counts["failed"] == 0 else {"type": "ClipExtractionFailure"},
        },
    }
    summary["summary_sha256"] = document_sha256(
        summary, excluded_fields=("created_at_utc", "duration_seconds", "operational_segments_per_second", "summary_sha256")
    )
    atomic_replace_text(namespace_root / "cache-summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def hashlib_sha256(content: str) -> str:
    import hashlib
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


__all__ = [
    "EXPECTED_YAMNET_TREE_SHA256", "LocalYamnetEmbeddingBackend", "extract_yamnet_embeddings",
    "load_audio_legacy", "resampling_identity",
]
