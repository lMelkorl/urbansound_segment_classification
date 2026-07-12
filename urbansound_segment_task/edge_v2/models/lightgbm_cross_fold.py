"""Resumable official-fold LightGBM evaluation over verified cached embeddings."""

from __future__ import annotations

import csv
import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from urbansound_segment_task.edge_v2.benchmarks.memory import read_peak_rss
from urbansound_segment_task.edge_v2.data.manifest import document_sha256
from urbansound_segment_task.edge_v2.data.splits import SPLIT_POLICY_ID
from urbansound_segment_task.edge_v2.evaluation.result_schema import safe_environment, utc_timestamp, write_result
from urbansound_segment_task.edge_v2.features.cache_dataset import (
    EXPECTED_DATASET_MANIFEST_SHA256, VerifiedCacheRecords, build_cache_split,
    load_verified_cache_records,
)
from urbansound_segment_task.edge_v2.features.yamnet_cache import atomic_replace_text
from urbansound_segment_task.edge_v2.models.lightgbm_legacy import (
    _evaluate_split, _rss_bytes, _split_counts, _write_per_class_csv,
    balanced_class_weights, legacy_training_config,
)
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256


CROSS_FOLD_SCHEMA_VERSION = "edge-v2.lightgbm-cross-fold.v1"
FOLD_RESULT_SCHEMA_VERSION = "edge-v2.lightgbm-cross-fold-fold.v1"
RUN_MANIFEST_SCHEMA_VERSION = "edge-v2.lightgbm-cross-fold-run.v1"
LEGACY_FOLD10_CONTEXT = {"clip_accuracy": 0.8124223602484472, "clip_macro_f1": 0.830150142616613}


def load_split_manifests(directory: Path) -> dict[int, dict[str, Any]]:
    root = Path(directory)
    manifests: dict[int, dict[str, Any]] = {}
    class_order = None
    for fold in range(1, 11):
        path = root / f"test-fold-{fold}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("policy_id") != SPLIT_POLICY_ID or int(document.get("test_fold", 0)) != fold:
            raise ValueError("split manifest policy or test fold mismatch")
        if document.get("dataset_manifest_sha256") != EXPECTED_DATASET_MANIFEST_SHA256:
            raise ValueError("split dataset manifest identity mismatch")
        if int(document["validation_fold"]) != fold % 10 + 1:
            raise ValueError("rotating validation fold mismatch")
        expected_train = [value for value in range(1, 11) if value not in (fold, fold % 10 + 1)]
        if document["training_folds"] != expected_train or not document["disjointness"]["valid"]:
            raise ValueError("split training folds or disjointness mismatch")
        actual_hash = document_sha256(
            document, excluded_fields=("created_at_utc", "split_manifest_sha256")
        )
        if actual_hash != document["split_manifest_sha256"]:
            raise ValueError("split manifest SHA-256 mismatch")
        current_order = json.dumps(document["class_order"], sort_keys=True)
        class_order = current_order if class_order is None else class_order
        if current_order != class_order:
            raise ValueError("split class order mismatch")
        manifests[fold] = document
    return manifests


def build_run_manifest(
    manifests: Mapping[int, Mapping[str, Any]], verified: VerifiedCacheRecords, *, threads: int
) -> dict[str, Any]:
    config = legacy_training_config(seed=42, threads=threads)
    document: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "policy_id": SPLIT_POLICY_ID,
        "cache_identity": verified.cache_identity,
        "cache_index_sha256": verified.index_sha256,
        "dataset_manifest_sha256": verified.dataset_manifest_sha256,
        "yamnet_artifact_tree_sha256": verified.yamnet_artifact_tree_sha256,
        "config_sha256": config["config_sha256"],
        "threads": threads,
        "seed": 42,
        "class_order": manifests[1]["class_order"],
        "split_manifest_sha256": {
            str(fold): manifests[fold]["split_manifest_sha256"] for fold in range(1, 11)
        },
    }
    document["run_identity_sha256"] = document_sha256(
        document, excluded_fields=("created_at_utc", "run_identity_sha256")
    )
    return document


def validate_resume_manifest(existing: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if existing.get("run_identity_sha256") != expected.get("run_identity_sha256"):
        raise ValueError("resume run identity does not match cache config or split manifests")


def _fold_result_valid(fold_directory: Path, run_identity: str) -> bool:
    try:
        result_path = fold_directory / "fold-result.json"
        expected_result_hash = (fold_directory / "fold-result.sha256").read_text(encoding="ascii").strip()
        if streaming_file_sha256(result_path) != expected_result_hash:
            return False
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["run_identity_sha256"] != run_identity or result["status"]["outcome"] != "success":
            return False
        for artifact_name in ("model", "config", "per_class"):
            artifact = result["artifacts"][artifact_name]
            path = fold_directory / artifact["relative_path"]
            if path.stat().st_size != artifact["size_bytes"]:
                return False
            if streaming_file_sha256(path) != artifact["sha256"]:
                return False
        return True
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _publish_fold_result(fold_directory: Path, result: Mapping[str, Any], *, pretty: bool) -> None:
    result_path = fold_directory / "fold-result.json"
    write_result(result_path, result, pretty=pretty)
    digest = streaming_file_sha256(result_path)
    atomic_replace_text(fold_directory / "fold-result.sha256", digest + "\n")


def _train_fold(
    *, fold: int, split_manifest: Mapping[str, Any], verified: VerifiedCacheRecords,
    fold_directory: Path, run_identity: str, threads: int, pretty: bool,
) -> dict[str, Any]:
    import lightgbm
    from lightgbm import LGBMClassifier

    train = build_cache_split("train", split_manifest["training_folds"], verified.records)
    validation = build_cache_split("validation", (int(split_manifest["validation_fold"]),), verified.records)
    test = build_cache_split("test", (fold,), verified.records)
    class_weights = balanced_class_weights(train.y)
    fold_directory.mkdir(parents=True, exist_ok=False)
    model = LGBMClassifier(
        n_estimators=700, num_leaves=64, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, n_jobs=threads,
        random_state=42, class_weight=class_weights,
    )
    before_rss = read_peak_rss()
    start = time.perf_counter()
    model.fit(train.X, train.y)
    training_seconds = time.perf_counter() - start
    after_rss = read_peak_rss()
    validation_metrics = _evaluate_split(model, validation)
    test_metrics = _evaluate_split(model, test)
    model_path = fold_directory / "model.txt"
    model.booster_.save_model(str(model_path))
    config = legacy_training_config(seed=42, threads=threads)
    config_path = fold_directory / "training-config.json"
    write_result(config_path, config, pretty=True)
    per_class_path = fold_directory / "per-class-metrics.csv"
    _write_per_class_csv(
        per_class_path, validation_metrics, test_metrics
    )
    before_bytes = _rss_bytes(before_rss)
    after_bytes = _rss_bytes(after_rss)
    result: dict[str, Any] = {
        "schema_version": FOLD_RESULT_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "run_identity_sha256": run_identity,
        "policy_id": SPLIT_POLICY_ID,
        "test_fold": fold,
        "validation_fold": int(split_manifest["validation_fold"]),
        "training_folds": list(split_manifest["training_folds"]),
        "split_manifest_sha256": split_manifest["split_manifest_sha256"],
        "cache_identity": verified.cache_identity,
        "config_sha256": config["config_sha256"],
        "class_weights": {str(key): value for key, value in sorted(class_weights.items())},
        "data_counts": {
            "train": _split_counts(train), "validation": _split_counts(validation),
            "test": _split_counts(test),
        },
        "training": {
            "duration_seconds": training_seconds,
            "before_training_peak_rss_bytes": before_bytes,
            "after_training_peak_rss_bytes": after_bytes,
            "approximate_incremental_peak_rss_bytes": (
                after_bytes - before_bytes if before_bytes is not None and after_bytes is not None else None
            ),
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "artifacts": {
            "model": {
                "relative_path": "model.txt", "size_bytes": model_path.stat().st_size,
                "sha256": streaming_file_sha256(model_path), "lightgbm_version": lightgbm.__version__,
            },
            "config": {
                "relative_path": "training-config.json", "size_bytes": config_path.stat().st_size,
                "sha256": streaming_file_sha256(config_path),
            },
            "per_class": {
                "relative_path": "per-class-metrics.csv", "size_bytes": per_class_path.stat().st_size,
                "sha256": streaming_file_sha256(per_class_path),
            },
        },
        "status": {"outcome": "success", "error": None},
    }
    _publish_fold_result(fold_directory, result, pretty=pretty)
    del model, train, validation, test
    gc.collect()
    return result


def _metric_statistics(rows: Sequence[Mapping[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append((int(row["test_fold"]), float(value)))
    numeric = [value for _, value in values]
    minimum = min(values, key=lambda item: item[1])
    maximum = max(values, key=lambda item: item[1])
    return {
        "mean": float(statistics.fmean(numeric)),
        "population_standard_deviation": float(statistics.pstdev(numeric)),
        "minimum": minimum[1], "minimum_fold": minimum[0],
        "maximum": maximum[1], "maximum_fold": maximum[0],
        "median": float(statistics.median(numeric)),
    }


def _pooled_from_confusions(rows: Sequence[Mapping[str, Any]], level: str) -> dict[str, Any]:
    matrix = np.zeros((10, 10), dtype=np.int64)
    for row in rows:
        matrix += np.asarray(row["test_metrics"][level]["confusion_matrix"], dtype=np.int64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    precision = np.divide(diagonal, predicted, out=np.zeros(10), where=predicted != 0)
    recall = np.divide(diagonal, support, out=np.zeros(10), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(10), where=(precision + recall) != 0)
    return {
        "label": "pooled_secondary",
        "accuracy": float(diagonal.sum() / matrix.sum()),
        "macro_f1": float(f1.mean()),
        "prediction_count": int(matrix.sum()),
        "class_order": list(range(10)),
        "confusion_matrix": matrix.tolist(),
        "per_class": [
            {"class_id": index, "precision": float(precision[index]), "recall": float(recall[index]),
             "f1": float(f1[index]), "support": int(support[index])}
            for index in range(10)
        ],
    }


def _per_class_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classes = []
    for class_id in range(10):
        clip_rows = [row["test_metrics"]["clip"]["per_class"][class_id] for row in rows]
        segment_rows = [row["test_metrics"]["segment"]["per_class"][class_id] for row in rows]
        def stats(values: Sequence[float]) -> dict[str, float]:
            return {"mean": float(statistics.fmean(values)), "population_standard_deviation": float(statistics.pstdev(values))}
        classes.append(
            {
                "class_id": class_id,
                "clip_precision": stats([float(item["precision"]) for item in clip_rows]),
                "clip_recall": stats([float(item["recall"]) for item in clip_rows]),
                "clip_f1": stats([float(item["f1"]) for item in clip_rows]),
                "segment_f1": stats([float(item["f1"]) for item in segment_rows]),
                "total_clip_support": sum(int(item["support"]) for item in clip_rows),
                "total_segment_support": sum(int(item["support"]) for item in segment_rows),
            }
        )
    strongest = sorted(classes, key=lambda item: item["clip_f1"]["mean"], reverse=True)[:3]
    weakest = sorted(classes, key=lambda item: item["clip_f1"]["mean"])[:3]
    variable = sorted(classes, key=lambda item: item["clip_f1"]["population_standard_deviation"], reverse=True)[:3]
    return {
        "classes": classes,
        "strongest_three_by_mean_clip_f1": [item["class_id"] for item in strongest],
        "weakest_three_by_mean_clip_f1": [item["class_id"] for item in weakest],
        "most_variable_three_by_clip_f1_std": [item["class_id"] for item in variable],
    }


def build_aggregate(fold_results: Sequence[Mapping[str, Any]], run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = sorted(fold_results, key=lambda item: int(item["test_fold"]))
    present = [int(item["test_fold"]) for item in rows]
    missing = sorted(set(range(1, 11)) - set(present))
    if missing:
        return {
            "schema_version": CROSS_FOLD_SCHEMA_VERSION,
            "created_at_utc": utc_timestamp(),
            "run_identity_sha256": run_manifest["run_identity_sha256"],
            "policy_id": SPLIT_POLICY_ID,
            "completed_folds": present, "missing_folds": missing,
            "status": {"outcome": "incomplete", "error": {"type": "MissingFoldResults"}},
        }
    fold_summary = [
        {
            "test_fold": row["test_fold"], "validation_fold": row["validation_fold"],
            "segment_accuracy": row["test_metrics"]["segment"]["accuracy"],
            "segment_macro_f1": row["test_metrics"]["segment"]["macro_f1"],
            "clip_accuracy": row["test_metrics"]["clip"]["accuracy"],
            "clip_macro_f1": row["test_metrics"]["clip"]["macro_f1"],
            "evaluable_clip_count": row["test_metrics"]["clip"]["evaluable_clip_count"],
            "zero_segment_clip_count": row["test_metrics"]["clip"]["excluded_zero_segment_clip_count"],
            "training_seconds": row["training"]["duration_seconds"],
            "model_size_bytes": row["artifacts"]["model"]["size_bytes"],
            "model_sha256": row["artifacts"]["model"]["sha256"],
        }
        for row in rows
    ]
    aggregate = {
        "segment_accuracy": _metric_statistics(rows, ("test_metrics", "segment", "accuracy")),
        "segment_macro_f1": _metric_statistics(rows, ("test_metrics", "segment", "macro_f1")),
        "clip_accuracy": _metric_statistics(rows, ("test_metrics", "clip", "accuracy")),
        "clip_macro_f1": _metric_statistics(rows, ("test_metrics", "clip", "macro_f1")),
    }
    document: dict[str, Any] = {
        "schema_version": CROSS_FOLD_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "run_identity_sha256": run_manifest["run_identity_sha256"],
        "policy_id": SPLIT_POLICY_ID,
        "class_order": run_manifest["class_order"],
        "cache_identity": run_manifest["cache_identity"],
        "config_sha256": run_manifest["config_sha256"],
        "fold_results": fold_summary,
        "aggregate": aggregate,
        "headline": {
            "mean_test_clip_accuracy": aggregate["clip_accuracy"]["mean"],
            "std_test_clip_accuracy": aggregate["clip_accuracy"]["population_standard_deviation"],
            "mean_test_clip_macro_f1": aggregate["clip_macro_f1"]["mean"],
            "std_test_clip_macro_f1": aggregate["clip_macro_f1"]["population_standard_deviation"],
            "mean_test_segment_accuracy": aggregate["segment_accuracy"]["mean"],
            "std_test_segment_accuracy": aggregate["segment_accuracy"]["population_standard_deviation"],
            "mean_test_segment_macro_f1": aggregate["segment_macro_f1"]["mean"],
            "std_test_segment_macro_f1": aggregate["segment_macro_f1"]["population_standard_deviation"],
        },
        "per_class_aggregate": _per_class_aggregate(rows),
        "pooled_secondary": {
            "segment": _pooled_from_confusions(rows, "segment"),
            "clip": _pooled_from_confusions(rows, "clip"),
        },
        "training": {
            "total_seconds": sum(float(row["training"]["duration_seconds"]) for row in rows),
            "peak_rss_bytes": max(int(row["training"]["after_training_peak_rss_bytes"] or 0) for row in rows),
            "model_size_bytes": _metric_statistics(rows, ("artifacts", "model", "size_bytes")),
        },
        "zero_segment_clips": {
            "total": sum(int(row["test_metrics"]["clip"]["excluded_zero_segment_clip_count"]) for row in rows),
            "by_test_fold": {str(row["test_fold"]): row["test_metrics"]["clip"]["excluded_zero_segment_clip_count"] for row in rows},
        },
        "legacy_single_split_context": {
            **LEGACY_FOLD10_CONTEXT,
            "note": "Legacy context is Fold 10 only; cross-fold headline is fold metric mean plus population standard deviation.",
        },
        "environment": safe_environment(int(run_manifest["threads"])),
        "status": {"outcome": "success", "error": None},
    }
    document["aggregate_sha256"] = document_sha256(
        document, excluded_fields=("created_at_utc", "aggregate_sha256")
    )
    return document


def run_cross_fold(
    *, cache_root: Path, split_manifest_dir: Path, output_dir: Path,
    threads: int, fold: Optional[int], resume: bool, pretty: bool,
) -> dict[str, Any]:
    if threads < 1 or (fold is not None and fold not in range(1, 11)):
        raise ValueError("threads or fold is outside the supported range")
    manifests = load_split_manifests(split_manifest_dir)
    verified = load_verified_cache_records(cache_root)
    expected_run = build_run_manifest(manifests, verified, threads=threads)
    root = Path(output_dir)
    if root.exists():
        if not resume:
            raise FileExistsError("output directory exists; --resume is required")
        existing = json.loads((root / "run-manifest.json").read_text(encoding="utf-8"))
        validate_resume_manifest(existing, expected_run)
        run_manifest = existing
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_result(root / "run-manifest.json", expected_run, pretty=True)
        run_manifest = expected_run
    targets = [fold] if fold is not None else list(range(1, 11))
    trained: list[int] = []
    skipped: list[int] = []
    for test_fold in targets:
        fold_directory = root / f"fold-{test_fold:02d}"
        if fold_directory.exists() and _fold_result_valid(
            fold_directory, run_manifest["run_identity_sha256"]
        ):
            skipped.append(test_fold)
            continue
        if fold_directory.exists():
            archived = root / f".invalid-fold-{test_fold:02d}-{time.time_ns()}"
            os.rename(fold_directory, archived)
        _train_fold(
            fold=test_fold, split_manifest=manifests[test_fold], verified=verified,
            fold_directory=fold_directory, run_identity=run_manifest["run_identity_sha256"],
            threads=threads, pretty=pretty,
        )
        trained.append(test_fold)
    fold_results = []
    for test_fold in range(1, 11):
        fold_directory = root / f"fold-{test_fold:02d}"
        if _fold_result_valid(fold_directory, run_manifest["run_identity_sha256"]):
            fold_results.append(json.loads((fold_directory / "fold-result.json").read_text(encoding="utf-8")))
    aggregate = build_aggregate(fold_results, run_manifest)
    aggregate["execution"] = {"trained_folds": trained, "skipped_folds": skipped, "requested_fold": fold, "resume": resume}
    if aggregate["status"]["outcome"] == "success":
        aggregate["aggregate_sha256"] = document_sha256(
            aggregate, excluded_fields=("created_at_utc", "aggregate_sha256")
        )
    atomic_replace_text(root / "aggregate.json", json.dumps(aggregate, indent=2 if pretty else None, sort_keys=True) + "\n")
    return aggregate


__all__ = [
    "CROSS_FOLD_SCHEMA_VERSION", "FOLD_RESULT_SCHEMA_VERSION", "RUN_MANIFEST_SCHEMA_VERSION",
    "build_aggregate", "build_run_manifest", "load_split_manifests", "run_cross_fold",
    "validate_resume_manifest",
]
