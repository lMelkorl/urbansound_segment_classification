"""Legacy Goal 1 LightGBM configuration and reproduction helpers."""

from __future__ import annotations

import json
import csv
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from urbansound_segment_task.edge_v2.data.manifest import document_sha256
from urbansound_segment_task.edge_v2.evaluation.aggregation import (
    aggregate_clip_probabilities, align_probability_columns,
)
from urbansound_segment_task.edge_v2.evaluation.metrics import classification_metrics
from urbansound_segment_task.edge_v2.evaluation.result_schema import (
    LIGHTGBM_REPRODUCTION_SCHEMA_VERSION, git_revision, safe_environment, utc_timestamp,
    write_result,
)
from urbansound_segment_task.edge_v2.features.cache_dataset import load_legacy_cache_dataset
from urbansound_segment_task.edge_v2.benchmarks.memory import read_peak_rss
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256


LEGACY_PROTOCOL_ID = "legacy-single-split-folds-1-8-train-9-validation-10-test"
LEGACY_SOURCE_PATH = "urbansound_segment_task/goals/goal1_yamnet_lgbm/run.py"
LEGACY_DEFAULT_SEED = 42


def validate_legacy_source(repository_root: Path) -> str:
    path = Path(repository_root) / LEGACY_SOURCE_PATH
    source = path.read_text(encoding="utf-8")
    required = (
        "n_estimators=700", "num_leaves=64", "learning_rate=0.05",
        "subsample=0.9", "colsample_bytree=0.9", "n_jobs=-1",
        "random_state=args.seed", 'compute_class_weight("balanced"',
        "clf.fit(X_tr, y_tr)",
    )
    if any(token not in source for token in required):
        raise ValueError("legacy source no longer matches the extracted configuration")
    import hashlib
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def legacy_training_config(*, seed: int = LEGACY_DEFAULT_SEED, threads: int = 8) -> dict[str, Any]:
    if threads < 1:
        raise ValueError("threads must be positive")
    config = {
        "source_path": LEGACY_SOURCE_PATH,
        "classifier": "lightgbm.LGBMClassifier",
        "explicit_legacy_parameters": {
            "n_estimators": 700, "num_leaves": 64, "learning_rate": 0.05,
            "subsample": 0.9, "colsample_bytree": 0.9,
            "n_jobs": -1, "random_state": LEGACY_DEFAULT_SEED,
        },
        "effective_reproduction_parameters": {
            "n_estimators": 700, "num_leaves": 64, "learning_rate": 0.05,
            "subsample": 0.9, "colsample_bytree": 0.9,
            "n_jobs": threads, "random_state": seed,
        },
        "library_defaults": {
            "boosting_type": "gbdt", "max_depth": -1, "min_child_samples": 20,
            "reg_alpha": 0.0, "reg_lambda": 0.0, "objective": "library_default_inferred_multiclass",
            "subsample_freq": 0,
        },
        "class_weight": {
            "method": "sklearn.utils.class_weight.compute_class_weight",
            "mode": "balanced", "source": "training_segments_only",
        },
        "early_stopping": False,
        "validation_used_during_fit": False,
        "subsample_effective_note": (
            "subsample=0.9 is explicit legacy code, but library-default subsample_freq=0 "
            "means row subsampling is not activated"
        ),
        "thread_deviation_from_legacy": threads != -1,
        "seed_deviation_from_legacy": seed != LEGACY_DEFAULT_SEED,
    }
    config["config_sha256"] = document_sha256(config, excluded_fields=("config_sha256",))
    return config


def balanced_class_weights(y_train: np.ndarray) -> dict[int, float]:
    labels = np.asarray(y_train, dtype=np.int64)
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}


def load_historical_metrics(repository_root: Path) -> dict[str, Any]:
    base = Path(repository_root) / "urbansound_segment_task/goals/goal1_yamnet_lgbm/results"
    validation = json.loads((base / "metrics_val.json").read_text(encoding="utf-8"))
    test = json.loads((base / "metrics_test.json").read_text(encoding="utf-8"))
    return {"validation": validation, "test": test}


def delta_classification(absolute_delta: float) -> str:
    if absolute_delta <= 0.005:
        return "exact_or_near_reproduction"
    if absolute_delta <= 0.015:
        return "close_reproduction"
    return "material_difference"


def historical_comparison(
    historical: Mapping[str, Any], validation_metrics: Mapping[str, Any], test_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    rows = []
    for split_name, reproduced in (("validation", validation_metrics), ("test", test_metrics)):
        source = historical[split_name]
        for level, historical_prefix in (("segment", "segment"), ("clip", "clip")):
            for metric, old_suffix in (("accuracy", "accuracy"), ("macro_f1", "macroF1")):
                historical_key = f"{historical_prefix}_{old_suffix}"
                historical_value = float(source[historical_key])
                reproduced_value = float(reproduced[level][metric])
                delta = abs(reproduced_value - historical_value)
                rows.append(
                    {
                        "split": split_name, "level": level, "metric": metric,
                        "historical_value": historical_value,
                        "reproduced_value": reproduced_value,
                        "absolute_delta": delta,
                        "classification": delta_classification(delta),
                    }
                )
    ranking = {"exact_or_near_reproduction": 0, "close_reproduction": 1, "material_difference": 2}
    overall = max(rows, key=lambda row: ranking[row["classification"]])["classification"]
    return {
        "thresholds": {"exact_or_near_max_delta": 0.005, "close_max_delta": 0.015},
        "metrics": rows, "overall_classification": overall,
        "possible_difference_factors": [
            "LightGBM version differs from an unpinned legacy environment.",
            "Legacy dependency versions and original embedding artifact provenance are incomplete.",
            "The verified local SavedModel and modern resampling stack may differ from the historical run.",
            "Probability columns are explicitly aligned in the reproduction; legacy code assumed order.",
        ],
    }


def _rss_bytes(reading: Mapping[str, Any]) -> Any:
    return reading.get("peak_rss_bytes") if reading.get("supported") else None


def _split_counts(split: Any) -> dict[str, Any]:
    return {
        "folds": list(split.folds), "metadata_clip_count": split.metadata_clip_count,
        "evaluable_clip_count": split.evaluable_clip_count,
        "excluded_zero_segment_clip_count": split.zero_segment_clip_count,
        "segment_count": split.segment_count,
        "feature_shape": list(split.X.shape), "feature_dtype": str(split.X.dtype),
    }


def _evaluate_split(model: Any, split: Any) -> dict[str, Any]:
    prediction_start = time.perf_counter()
    raw_probabilities = model.predict_proba(split.X)
    prediction_seconds = time.perf_counter() - prediction_start
    probabilities, alignment = align_probability_columns(raw_probabilities, model.classes_)
    segment_predictions = probabilities.argmax(axis=1)
    segment_metrics = classification_metrics(split.y, segment_predictions)
    clip_true, clip_predicted, clip_keys = aggregate_clip_probabilities(
        split.clip_keys, split.y, probabilities
    )
    clip_metrics = classification_metrics(clip_true, clip_predicted)
    if len(clip_keys) != split.evaluable_clip_count:
        raise ValueError("evaluable clip count differs from aggregation output")
    return {
        "segment": segment_metrics,
        "clip": {
            **clip_metrics,
            "metadata_clip_count": split.metadata_clip_count,
            "evaluable_clip_count": split.evaluable_clip_count,
            "excluded_zero_segment_clip_count": split.zero_segment_clip_count,
            "aggregation": "arithmetic_mean_of_canonical_class_probabilities",
        },
        "probability_alignment": alignment,
        "prediction_seconds": prediction_seconds,
    }


def _write_per_class_csv(path: Path, validation: Mapping[str, Any], test: Mapping[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "level", "class_id", "precision", "recall", "f1", "support"),
        )
        writer.writeheader()
        for split_name, metrics in (("validation", validation), ("test", test)):
            for level in ("segment", "clip"):
                for row in metrics[level]["per_class"]:
                    writer.writerow({"split": split_name, "level": level, **row})


def run_legacy_reproduction(
    *, cache_root: Path, output_dir: Path, threads: int, seed: int,
    repository_root: Path, pretty: bool,
) -> dict[str, Any]:
    import lightgbm
    from lightgbm import LGBMClassifier

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError("output directory already exists")
    source_sha256 = validate_legacy_source(repository_root)
    baseline_rss = read_peak_rss()
    dataset = load_legacy_cache_dataset(cache_root)
    post_load_rss = read_peak_rss()
    config = legacy_training_config(seed=seed, threads=threads)
    config["legacy_source_sha256"] = source_sha256
    config["config_sha256"] = document_sha256(config, excluded_fields=("config_sha256",))
    class_weights = balanced_class_weights(dataset.train.y)
    destination.mkdir(parents=True, exist_ok=False)
    model = LGBMClassifier(
        n_estimators=700, num_leaves=64, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, n_jobs=threads,
        random_state=seed, class_weight=class_weights,
    )
    training_start = time.perf_counter()
    model.fit(dataset.train.X, dataset.train.y)
    training_seconds = time.perf_counter() - training_start
    post_training_rss = read_peak_rss()
    validation_metrics = _evaluate_split(model, dataset.validation)
    test_metrics = _evaluate_split(model, dataset.test)
    model_path = destination / "model.txt"
    model.booster_.save_model(str(model_path))
    model_sha256 = streaming_file_sha256(model_path)
    config_path = destination / "training-config.json"
    write_result(config_path, config, pretty=True)
    per_class_path = destination / "per-class-metrics.csv"
    _write_per_class_csv(per_class_path, validation_metrics, test_metrics)
    historical = historical_comparison(
        load_historical_metrics(repository_root), validation_metrics, test_metrics
    )
    data_counts = {
        "train": _split_counts(dataset.train),
        "validation": _split_counts(dataset.validation),
        "test": _split_counts(dataset.test),
        "verified_cache_artifacts": dataset.verified_artifact_count,
        "cache_load_seconds": dataset.load_seconds,
    }
    baseline_bytes = _rss_bytes(baseline_rss)
    final_bytes = _rss_bytes(post_training_rss)
    result = {
        "schema_version": LIGHTGBM_REPRODUCTION_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "git_revision": git_revision(repository_root),
        "protocol": LEGACY_PROTOCOL_ID,
        "cache_identity": dataset.cache_identity,
        "dataset_manifest_sha256": dataset.dataset_manifest_sha256,
        "yamnet_artifact_tree_sha256": dataset.yamnet_artifact_tree_sha256,
        "lightgbm_version": lightgbm.__version__,
        "training_config": config,
        "class_weights": {str(key): value for key, value in sorted(class_weights.items())},
        "data_counts": data_counts,
        "training": {
            "duration_seconds": training_seconds,
            "baseline_peak_rss_bytes": baseline_bytes,
            "post_cache_load_peak_rss_bytes": _rss_bytes(post_load_rss),
            "post_training_peak_rss_bytes": final_bytes,
            "approximate_incremental_peak_rss_bytes": (
                final_bytes - baseline_bytes if final_bytes is not None and baseline_bytes is not None else None
            ),
            "peak_rss_source": post_training_rss.get("source"),
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "historical_comparison": historical,
        "artifacts": {
            "model": {
                "relative_path": "model.txt", "size_bytes": model_path.stat().st_size,
                "sha256": model_sha256, "lightgbm_version": lightgbm.__version__,
                "config_sha256": config["config_sha256"], "cache_identity": dataset.cache_identity,
            },
            "training_config": {"relative_path": "training-config.json"},
            "per_class_metrics": {"relative_path": "per-class-metrics.csv"},
        },
        "environment": safe_environment(threads),
        "limitations": [
            "This is the legacy single split, not official 10-fold evaluation.",
            "Zero-segment clips are excluded from clip metrics and denominators are explicit.",
            "The historical environment and original embedding artifact were not fully pinned.",
            "Effective n_jobs is explicitly bounded by --threads instead of legacy n_jobs=-1.",
        ],
        "status": {"outcome": "success", "error": None},
    }
    write_result(destination / "result.json", result, pretty=pretty)
    return result


__all__ = [
    "LEGACY_DEFAULT_SEED", "LEGACY_PROTOCOL_ID", "LEGACY_SOURCE_PATH",
    "balanced_class_weights", "delta_classification", "historical_comparison",
    "legacy_training_config", "load_historical_metrics", "run_legacy_reproduction",
    "validate_legacy_source",
]
