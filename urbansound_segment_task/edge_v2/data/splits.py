"""Official UrbanSound8K rotating-validation split manifests."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Optional

from urbansound_segment_task.edge_v2.benchmarks.schema import utc_timestamp

from .manifest import document_sha256


SPLIT_SCHEMA_VERSION = "edge-v2.urbansound8k-split-manifest.v1"
SPLIT_POLICY_ID = "official-fold-rotating-validation-v1"


def split_folds(test_fold: int) -> tuple[list[int], int, int]:
    if test_fold not in range(1, 11):
        raise ValueError("test_fold must be between 1 and 10")
    validation_fold = test_fold % 10 + 1
    training_folds = [fold for fold in range(1, 11) if fold not in (test_fold, validation_fold)]
    return training_folds, validation_fold, test_fold


def _distribution(clips: list[Mapping[str, Any]]) -> dict[str, Any]:
    folds = Counter(int(clip["fold"]) for clip in clips)
    classes = Counter(int(clip["class_id"]) for clip in clips)
    return {
        "fold_counts": {str(fold): folds[fold] for fold in sorted(folds)},
        "class_counts": {str(class_id): classes.get(class_id, 0) for class_id in range(10)},
    }


def build_split_manifest(
    dataset_manifest: Mapping[str, Any], test_fold: int, *, now: Optional[Any] = None
) -> dict[str, Any]:
    if dataset_manifest.get("status", {}).get("outcome") != "success":
        raise ValueError("dataset manifest must be successful before split generation")
    train_folds, validation_fold, test_fold = split_folds(test_fold)
    clips = list(dataset_manifest["clips"])
    partitions = {
        "train": sorted(
            [clip["clip_key"] for clip in clips if int(clip["fold"]) in train_folds]
        ),
        "validation": sorted(
            [clip["clip_key"] for clip in clips if int(clip["fold"]) == validation_fold]
        ),
        "test": sorted([clip["clip_key"] for clip in clips if int(clip["fold"]) == test_fold]),
    }
    sets = {name: set(values) for name, values in partitions.items()}
    intersections = {
        "train_validation": sorted(sets["train"] & sets["validation"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "validation_test": sorted(sets["validation"] & sets["test"]),
    }
    disjoint = not any(intersections.values())
    if not disjoint:
        raise ValueError("train validation and test clip keys must be disjoint")
    by_key = {clip["clip_key"]: clip for clip in clips}
    document: dict[str, Any] = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(now),
        "policy_id": SPLIT_POLICY_ID,
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "test_fold": test_fold,
        "validation_fold": validation_fold,
        "training_folds": train_folds,
        "class_order": list(dataset_manifest["class_order"]),
        "partitions": partitions,
        "clip_counts": {name: len(values) for name, values in partitions.items()},
        "distributions": {
            name: _distribution([by_key[key] for key in keys])
            for name, keys in partitions.items()
        },
        "disjointness": {"valid": True, "intersections": intersections},
        "status": {"outcome": "success", "error": None},
    }
    document["split_manifest_sha256"] = document_sha256(
        document, excluded_fields=("created_at_utc", "split_manifest_sha256")
    )
    return document


def build_all_split_manifests(
    dataset_manifest: Mapping[str, Any], *, now: Optional[Any] = None
) -> list[dict[str, Any]]:
    return [build_split_manifest(dataset_manifest, fold, now=now) for fold in range(1, 11)]


__all__ = [
    "SPLIT_POLICY_ID", "SPLIT_SCHEMA_VERSION", "build_all_split_manifests",
    "build_split_manifest", "split_folds",
]
