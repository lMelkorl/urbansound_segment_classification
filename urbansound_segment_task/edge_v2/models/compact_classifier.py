"""Fixed compact Keras classifiers over verified cached YAMNet embeddings."""

from __future__ import annotations

import csv
import gc
import importlib.metadata
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from urbansound_segment_task.edge_v2.benchmarks.lifecycle import LifecycleRequest, run_lifecycle
from urbansound_segment_task.edge_v2.benchmarks.memory import read_peak_rss
from urbansound_segment_task.edge_v2.data.manifest import document_sha256
from urbansound_segment_task.edge_v2.data.splits import SPLIT_POLICY_ID
from urbansound_segment_task.edge_v2.evaluation.aggregation import aggregate_clip_probabilities
from urbansound_segment_task.edge_v2.evaluation.metrics import classification_metrics
from urbansound_segment_task.edge_v2.evaluation.result_schema import safe_environment, utc_timestamp, write_result
from urbansound_segment_task.edge_v2.features.cache_dataset import (
    CacheSplit, VerifiedCacheRecords, build_cache_split, load_verified_cache_records,
)
from urbansound_segment_task.edge_v2.features.yamnet_cache import atomic_replace_text
from urbansound_segment_task.edge_v2.models.lightgbm_cross_fold import (
    _metric_statistics, _per_class_aggregate, _pooled_from_confusions, load_split_manifests,
)
from urbansound_segment_task.edge_v2.models.lightgbm_legacy import (
    _rss_bytes, _split_counts, _write_per_class_csv, balanced_class_weights,
)
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256


SCHEMA_VERSION = "edge-v2.compact-classifier-cross-fold.v1"
FOLD_SCHEMA_VERSION = "edge-v2.compact-classifier-fold.v1"
RUN_SCHEMA_VERSION = "edge-v2.compact-classifier-run.v1"
ARCHITECTURE_IDS = ("linear", "mlp128")
LIGHTGBM_REFERENCE = {
    "clip_accuracy_mean": 0.7785585802056633,
    "clip_accuracy_population_standard_deviation": 0.03427676472841665,
    "clip_macro_f1_mean": 0.7890586045778194,
    "clip_macro_f1_population_standard_deviation": 0.03176801694116027,
    "segment_accuracy_mean": 0.7199777226367361,
    "segment_accuracy_population_standard_deviation": 0.03308231014549838,
    "segment_macro_f1_mean": 0.7254016573978332,
    "segment_macro_f1_population_standard_deviation": 0.03010259254556998,
    "mean_model_size_bytes": 45450900.5,
    "classifier_batch1_p50_ns": None,
    "classifier_latency_note": "No commensurate LightGBM classifier-only timing artifact is available.",
}
SUCCESS_THRESHOLDS = (
    ("excellent", 0.010, 1_000_000),
    ("strong", 0.025, 2_000_000),
    ("promising", 0.050, 5_000_000),
)
BENCHMARK_WARMUP = 20
BENCHMARK_ITERATIONS = 100


def architecture_config(architecture_id: str) -> dict[str, Any]:
    if architecture_id == "linear":
        layers = [{"type": "dense", "units": 10, "activation": "softmax"}]
        parameter_count = 10_250
    elif architecture_id == "mlp128":
        layers = [
            {"type": "dense", "units": 128, "activation": "relu"},
            {"type": "dropout", "rate": 0.20},
            {"type": "dense", "units": 10, "activation": "softmax"},
        ]
        parameter_count = 132_490
    else:
        raise ValueError("unsupported compact architecture")
    return {
        "architecture_id": architecture_id,
        "input_features": 1024,
        "layers": layers,
        "parameter_count": parameter_count,
    }


def training_config(architecture_id: str, *, threads: int) -> dict[str, Any]:
    if threads < 1:
        raise ValueError("threads must be positive")
    document = {
        "architecture": architecture_config(architecture_id),
        "optimizer": "keras.optimizers.Adam",
        "learning_rate": 0.001,
        "batch_size": 512,
        "maximum_epochs": 30,
        "seed": 42,
        "loss": "sparse_categorical_crossentropy",
        "class_weight": {"mode": "balanced", "source": "training_segments_only"},
        "shuffle": {"training": True, "validation": False, "test": False},
        "validation_source": "official_rotating_validation_fold",
        "early_stopping": {
            "metric": "validation_clip_macro_f1",
            "patience": 5,
            "min_delta": 0.001,
            "restore_best_weights": True,
        },
        "clip_aggregation": "arithmetic_mean_of_canonical_class_probabilities",
        "canonical_classes": list(range(10)),
        "threads": threads,
        "cpu_only": True,
        "model_artifact": {
            "format": "keras_hdf5_weights_only",
            "optimizer_state_included": False,
            "architecture_reconstructed_from_fixed_config": True,
        },
        "classifier_benchmark": {
            "label": "classifier_only_cached_embedding_batch1",
            "warmup": BENCHMARK_WARMUP,
            "iterations": BENCHMARK_ITERATIONS,
            "batch_size": 1,
            "yamnet_included": False,
        },
    }
    document["config_sha256"] = document_sha256(document, excluded_fields=("config_sha256",))
    return document


def classify_compact_result(f1_loss: float, model_size_bytes: float) -> str:
    for label, maximum_loss, maximum_size in SUCCESS_THRESHOLDS:
        if f1_loss <= maximum_loss and model_size_bytes <= maximum_size:
            return label
    return "insufficient"


def pareto_front(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return architectures not dominated on F1, size, and p50 latency."""

    front: list[str] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            no_worse = (
                float(other["clip_macro_f1_mean"]) >= float(candidate["clip_macro_f1_mean"])
                and float(other["model_size_median_bytes"]) <= float(candidate["model_size_median_bytes"])
                and float(other["classifier_p50_median_ns"]) <= float(candidate["classifier_p50_median_ns"])
            )
            strictly_better = (
                float(other["clip_macro_f1_mean"]) > float(candidate["clip_macro_f1_mean"])
                or float(other["model_size_median_bytes"]) < float(candidate["model_size_median_bytes"])
                or float(other["classifier_p50_median_ns"]) < float(candidate["classifier_p50_median_ns"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(str(candidate["architecture_id"]))
    return sorted(front)


@dataclass
class ValidationClipF1Controller:
    """Framework-light state machine used by the Keras callback and unit tests."""

    clip_keys: np.ndarray
    labels: np.ndarray
    features: np.ndarray
    patience: int = 5
    min_delta: float = 0.001
    predictor: Optional[Callable[[Any, np.ndarray], np.ndarray]] = None

    def __post_init__(self) -> None:
        self.best_score = float("-inf")
        self.best_epoch: Optional[int] = None
        self.best_weights: Optional[list[Any]] = None
        self.wait = 0
        self.history: list[dict[str, Any]] = []

    def observe(self, model: Any, epoch_zero_based: int) -> float:
        predict = self.predictor or (lambda current, values: np.asarray(current(values, training=False)))
        probabilities = np.asarray(predict(model, self.features), dtype=np.float64)
        if probabilities.shape != (self.labels.shape[0], 10):
            raise ValueError("validation probability shape mismatch")
        clip_true, clip_predicted, _ = aggregate_clip_probabilities(
            self.clip_keys, self.labels, probabilities
        )
        score = float(classification_metrics(clip_true, clip_predicted)["macro_f1"])
        improved = score > self.best_score + self.min_delta
        if improved:
            self.best_score = score
            self.best_epoch = epoch_zero_based + 1
            self.best_weights = [np.array(value, copy=True) for value in model.get_weights()]
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                model.stop_training = True
        self.history.append(
            {"epoch": epoch_zero_based + 1, "validation_clip_macro_f1": score, "improved": improved}
        )
        return score

    def restore(self, model: Any) -> None:
        if self.best_weights is None or self.best_epoch is None:
            raise ValueError("no validation epoch was observed")
        model.set_weights(self.best_weights)


def _configure_tensorflow_cpu(tf: Any, threads: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()


def _build_keras_model(tf: Any, architecture_id: str, *, compile_model: bool) -> Any:
    config = architecture_config(architecture_id)
    layers: list[Any] = [tf.keras.layers.Input(shape=(1024,), dtype=tf.float32)]
    if architecture_id == "linear":
        layers.append(tf.keras.layers.Dense(10, activation="softmax"))
    else:
        layers.extend(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.20),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
    model = tf.keras.Sequential(layers, name=f"compact_{architecture_id}")
    if int(model.count_params()) != int(config["parameter_count"]):
        raise ValueError("compact model parameter count mismatch")
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="segment_accuracy")],
        )
    return model


class _KerasProbabilityAdapter:
    classes_ = np.arange(10, dtype=np.int64)

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model(features, training=False))


def _evaluate_model(model: Any, split: CacheSplit) -> dict[str, Any]:
    from urbansound_segment_task.edge_v2.models.lightgbm_legacy import _evaluate_split

    return _evaluate_split(_KerasProbabilityAdapter(model), split)


def _make_callback(tf: Any, controller: ValidationClipF1Controller) -> Any:
    class ValidationCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
            score = controller.observe(self.model, epoch)
            if logs is not None:
                logs["validation_clip_macro_f1"] = score

        def on_train_end(self, logs: Optional[dict[str, Any]] = None) -> None:
            controller.restore(self.model)

    return ValidationCallback()


def _write_history_csv(path: Path, keras_history: Mapping[str, Sequence[Any]], controller: ValidationClipF1Controller) -> None:
    fields = ["epoch", "loss", "segment_accuracy", "val_loss", "val_segment_accuracy", "validation_clip_macro_f1", "improved"]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, validation_row in enumerate(controller.history):
            writer.writerow(
                {
                    "epoch": index + 1,
                    "loss": float(keras_history["loss"][index]),
                    "segment_accuracy": float(keras_history["segment_accuracy"][index]),
                    "val_loss": float(keras_history["val_loss"][index]),
                    "val_segment_accuracy": float(keras_history["val_segment_accuracy"][index]),
                    "validation_clip_macro_f1": validation_row["validation_clip_macro_f1"],
                    "improved": validation_row["improved"],
                }
            )


def _artifact(path: Path, relative_path: str) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": streaming_file_sha256(path),
    }


def _benchmark_classifier(tf: Any, architecture_id: str, weights_path: Path, input_row: np.ndarray) -> dict[str, Any]:
    before = read_peak_rss()

    def loader() -> Any:
        loaded = _build_keras_model(tf, architecture_id, compile_model=False)
        loaded.load_weights(str(weights_path))
        return loaded

    def inference(loaded: Any, values: np.ndarray) -> np.ndarray:
        return np.asarray(loaded(values, training=False))

    lifecycle = run_lifecycle(
        loader,
        inference,
        np.asarray(input_row, dtype=np.float32).reshape(1, 1024),
        LifecycleRequest(
            name=f"compact-{architecture_id}-classifier-only",
            description="Batch-1 compact classifier inference on one cached 1024-value embedding; excludes YAMNet.",
            warmup=BENCHMARK_WARMUP,
            iterations=BENCHMARK_ITERATIONS,
            items_per_call=1,
        ),
    )
    after = read_peak_rss()
    lifecycle.update(
        {
            "label": "classifier_only_cached_embedding_batch1",
            "scope": {"yamnet_included": False, "audio_decode_included": False, "batch_size": 1},
            "peak_rss": {
                "before_bytes": _rss_bytes(before),
                "after_bytes": _rss_bytes(after),
                "approximate_incremental_bytes": (
                    _rss_bytes(after) - _rss_bytes(before)
                    if _rss_bytes(after) is not None and _rss_bytes(before) is not None
                    else None
                ),
                "source": after.get("source"),
            },
        }
    )
    return lifecycle


def build_run_manifest(
    manifests: Mapping[int, Mapping[str, Any]], verified: VerifiedCacheRecords, *, threads: int
) -> dict[str, Any]:
    configs = {name: training_config(name, threads=threads) for name in ARCHITECTURE_IDS}
    document: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "policy_id": SPLIT_POLICY_ID,
        "cache_identity": verified.cache_identity,
        "cache_index_sha256": verified.index_sha256,
        "dataset_manifest_sha256": verified.dataset_manifest_sha256,
        "yamnet_artifact_tree_sha256": verified.yamnet_artifact_tree_sha256,
        "threads": threads,
        "seed": 42,
        "class_order": manifests[1]["class_order"],
        "model_config_sha256": {name: configs[name]["config_sha256"] for name in ARCHITECTURE_IDS},
        "split_manifest_sha256": {
            str(fold): manifests[fold]["split_manifest_sha256"] for fold in range(1, 11)
        },
        "lightgbm_reference": LIGHTGBM_REFERENCE,
        "success_thresholds": [
            {"classification": label, "maximum_f1_loss": loss, "maximum_model_size_bytes": size}
            for label, loss, size in SUCCESS_THRESHOLDS
        ],
    }
    document["run_identity_sha256"] = document_sha256(
        document, excluded_fields=("created_at_utc", "run_identity_sha256")
    )
    return document


def validate_resume_manifest(existing: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if existing.get("run_identity_sha256") != expected.get("run_identity_sha256"):
        raise ValueError("resume run identity does not match compact config cache or split manifests")


def _fold_result_valid(directory: Path, run_identity: str, architecture_id: str, fold: int) -> bool:
    try:
        result_path = directory / "fold-result.json"
        if streaming_file_sha256(result_path) != (directory / "fold-result.sha256").read_text(encoding="ascii").strip():
            return False
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result["run_identity_sha256"] != run_identity
            or result["architecture_id"] != architecture_id
            or int(result["test_fold"]) != fold
            or result["status"]["outcome"] != "success"
        ):
            return False
        for artifact in result["artifacts"].values():
            path = directory / artifact["relative_path"]
            if path.stat().st_size != artifact["size_bytes"] or streaming_file_sha256(path) != artifact["sha256"]:
                return False
        return True
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _publish_fold_result(directory: Path, result: Mapping[str, Any], *, pretty: bool) -> None:
    result_path = directory / "fold-result.json"
    write_result(result_path, result, pretty=pretty)
    atomic_replace_text(directory / "fold-result.sha256", streaming_file_sha256(result_path) + "\n")


def _train_fold(
    *, tf: Any, architecture_id: str, fold: int, split_manifest: Mapping[str, Any],
    verified: VerifiedCacheRecords, directory: Path, run_identity: str, threads: int, pretty: bool,
) -> dict[str, Any]:
    train = build_cache_split("train", split_manifest["training_folds"], verified.records)
    validation = build_cache_split("validation", (int(split_manifest["validation_fold"]),), verified.records)
    test = build_cache_split("test", (fold,), verified.records)
    class_weights = balanced_class_weights(train.y)
    directory.mkdir(parents=True, exist_ok=False)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    model = _build_keras_model(tf, architecture_id, compile_model=True)
    controller = ValidationClipF1Controller(
        validation.clip_keys, validation.y, validation.X, patience=5, min_delta=0.001
    )
    callback = _make_callback(tf, controller)
    before = read_peak_rss()
    start = time.perf_counter()
    history = model.fit(
        train.X,
        train.y,
        validation_data=(validation.X, validation.y),
        epochs=30,
        batch_size=512,
        shuffle=True,
        class_weight=class_weights,
        callbacks=[callback],
        verbose=0,
    )
    training_seconds = time.perf_counter() - start
    after = read_peak_rss()
    controller.restore(model)
    validation_metrics = _evaluate_model(model, validation)
    test_metrics = _evaluate_model(model, test)
    config = training_config(architecture_id, threads=threads)
    config_path = directory / "training-config.json"
    write_result(config_path, config, pretty=True)
    history_path = directory / "epoch-history.csv"
    _write_history_csv(history_path, history.history, controller)
    per_class_path = directory / "per-class-metrics.csv"
    _write_per_class_csv(per_class_path, validation_metrics, test_metrics)
    weights_path = directory / "model.weights.h5"
    inference_model = _build_keras_model(tf, architecture_id, compile_model=False)
    inference_model.set_weights(model.get_weights())
    inference_model.save_weights(str(weights_path))
    benchmark = _benchmark_classifier(tf, architecture_id, weights_path, test.X[0])
    benchmark_path = directory / "classifier-benchmark.json"
    write_result(benchmark_path, benchmark, pretty=pretty)
    fold_identity = document_sha256(
        {
            "run_identity_sha256": run_identity,
            "architecture_id": architecture_id,
            "test_fold": fold,
            "split_manifest_sha256": split_manifest["split_manifest_sha256"],
        }
    )
    keras_version = importlib.metadata.version("keras")
    model_artifact = _artifact(weights_path, "model.weights.h5")
    model_artifact.update(
        {
            "format": "keras_hdf5_weights_only",
            "architecture_id": architecture_id,
            "config_sha256": config["config_sha256"],
            "fold_identity_sha256": fold_identity,
            "parameter_count": architecture_config(architecture_id)["parameter_count"],
            "tensorflow_version": tf.__version__,
            "keras_version": keras_version,
        }
    )
    result: dict[str, Any] = {
        "schema_version": FOLD_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "run_identity_sha256": run_identity,
        "fold_identity_sha256": fold_identity,
        "architecture_id": architecture_id,
        "architecture": architecture_config(architecture_id),
        "policy_id": SPLIT_POLICY_ID,
        "test_fold": fold,
        "validation_fold": int(split_manifest["validation_fold"]),
        "training_folds": list(split_manifest["training_folds"]),
        "split_manifest_sha256": split_manifest["split_manifest_sha256"],
        "cache_identity": verified.cache_identity,
        "config_sha256": config["config_sha256"],
        "class_weights": {str(key): value for key, value in sorted(class_weights.items())},
        "data_counts": {
            "train": _split_counts(train),
            "validation": _split_counts(validation),
            "test": _split_counts(test),
        },
        "selection": {
            "source": "official_validation_fold_only",
            "metric": "validation_clip_macro_f1",
            "best_epoch": controller.best_epoch,
            "best_validation_clip_macro_f1": controller.best_score,
            "epochs_completed": len(controller.history),
            "restore_best_weights": True,
            "test_fold_used_for_selection": False,
        },
        "training": {
            "duration_seconds": training_seconds,
            "before_training_peak_rss_bytes": _rss_bytes(before),
            "after_training_peak_rss_bytes": _rss_bytes(after),
            "approximate_incremental_peak_rss_bytes": (
                _rss_bytes(after) - _rss_bytes(before)
                if _rss_bytes(after) is not None and _rss_bytes(before) is not None
                else None
            ),
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "classifier_benchmark": benchmark,
        "runtime": {
            "tensorflow_version": tf.__version__,
            "keras_version": keras_version,
            "accelerator": "none_cpu_only",
        },
        "artifacts": {
            "model": model_artifact,
            "config": _artifact(config_path, "training-config.json"),
            "history": _artifact(history_path, "epoch-history.csv"),
            "per_class": _artifact(per_class_path, "per-class-metrics.csv"),
            "benchmark": _artifact(benchmark_path, "classifier-benchmark.json"),
        },
        "status": {"outcome": "success", "error": None},
    }
    _publish_fold_result(directory, result, pretty=pretty)
    del callback, controller, history, inference_model, model, train, validation, test
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def _median_path(rows: Sequence[Mapping[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(statistics.median(values))


def _model_aggregate(architecture_id: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = {
        "segment_accuracy": _metric_statistics(rows, ("test_metrics", "segment", "accuracy")),
        "segment_macro_f1": _metric_statistics(rows, ("test_metrics", "segment", "macro_f1")),
        "clip_accuracy": _metric_statistics(rows, ("test_metrics", "clip", "accuracy")),
        "clip_macro_f1": _metric_statistics(rows, ("test_metrics", "clip", "macro_f1")),
    }
    sizes = [int(row["artifacts"]["model"]["size_bytes"]) for row in rows]
    p50 = _median_path(rows, ("classifier_benchmark", "steady_state", "timing", "p50"))
    p95 = _median_path(rows, ("classifier_benchmark", "steady_state", "timing", "p95"))
    predictions_per_second = _median_path(
        rows, ("classifier_benchmark", "steady_state", "throughput", "items_per_second")
    )
    load_time = _median_path(rows, ("classifier_benchmark", "load_time", "duration"))
    f1_loss = LIGHTGBM_REFERENCE["clip_macro_f1_mean"] - metrics["clip_macro_f1"]["mean"]
    median_size = float(statistics.median(sizes))
    return {
        "architecture_id": architecture_id,
        "parameter_count": architecture_config(architecture_id)["parameter_count"],
        "fold_results": [
            {
                "test_fold": row["test_fold"],
                "validation_fold": row["validation_fold"],
                "best_epoch": row["selection"]["best_epoch"],
                "segment_accuracy": row["test_metrics"]["segment"]["accuracy"],
                "segment_macro_f1": row["test_metrics"]["segment"]["macro_f1"],
                "clip_accuracy": row["test_metrics"]["clip"]["accuracy"],
                "clip_macro_f1": row["test_metrics"]["clip"]["macro_f1"],
                "training_seconds": row["training"]["duration_seconds"],
                "model_size_bytes": row["artifacts"]["model"]["size_bytes"],
            }
            for row in rows
        ],
        "metrics": metrics,
        "model_size_bytes": {"median": median_size, "minimum": min(sizes), "maximum": max(sizes)},
        "classifier_only_benchmark": {
            "label": "classifier_only_cached_embedding_batch1",
            "median_load_time_ns": load_time,
            "median_p50_ns": p50,
            "median_p95_ns": p95,
            "median_predictions_per_second": predictions_per_second,
        },
        "training": {
            "total_seconds": sum(float(row["training"]["duration_seconds"]) for row in rows),
            "peak_rss_bytes": max(int(row["training"]["after_training_peak_rss_bytes"] or 0) for row in rows),
        },
        "per_class_aggregate": _per_class_aggregate(rows),
        "pooled_secondary": {
            "segment": _pooled_from_confusions(rows, "segment"),
            "clip": _pooled_from_confusions(rows, "clip"),
        },
        "zero_segment_clips": {
            "total": sum(int(row["test_metrics"]["clip"]["excluded_zero_segment_clip_count"]) for row in rows),
            "by_test_fold": {
                str(row["test_fold"]): row["test_metrics"]["clip"]["excluded_zero_segment_clip_count"]
                for row in rows
            },
        },
        "lightgbm_comparison": {
            "clip_macro_f1_loss": f1_loss,
            "size_reduction_fraction": 1.0 - median_size / LIGHTGBM_REFERENCE["mean_model_size_bytes"],
            "success_classification": classify_compact_result(f1_loss, median_size),
        },
    }


def build_aggregate(
    fold_results: Mapping[str, Sequence[Mapping[str, Any]]], run_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    missing = {
        name: sorted(set(range(1, 11)) - {int(row["test_fold"]) for row in fold_results.get(name, ())})
        for name in ARCHITECTURE_IDS
    }
    if any(missing.values()):
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_timestamp(),
            "run_identity_sha256": run_manifest["run_identity_sha256"],
            "policy_id": SPLIT_POLICY_ID,
            "missing_folds": missing,
            "status": {"outcome": "incomplete", "error": {"type": "MissingFoldResults"}},
        }
    model_results = {
        name: _model_aggregate(name, sorted(fold_results[name], key=lambda row: int(row["test_fold"])))
        for name in ARCHITECTURE_IDS
    }
    pareto_rows = [
        {
            "architecture_id": name,
            "clip_macro_f1_mean": model_results[name]["metrics"]["clip_macro_f1"]["mean"],
            "model_size_median_bytes": model_results[name]["model_size_bytes"]["median"],
            "classifier_p50_median_ns": model_results[name]["classifier_only_benchmark"]["median_p50_ns"],
        }
        for name in ARCHITECTURE_IDS
    ]
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(),
        "run_identity_sha256": run_manifest["run_identity_sha256"],
        "policy_id": SPLIT_POLICY_ID,
        "cache_identity": run_manifest["cache_identity"],
        "class_order": run_manifest["class_order"],
        "models": model_results,
        "lightgbm_reference": LIGHTGBM_REFERENCE,
        "comparison": {
            "rows": [
                {
                    "model": "lightgbm",
                    "clip_accuracy_mean": LIGHTGBM_REFERENCE["clip_accuracy_mean"],
                    "clip_macro_f1_mean": LIGHTGBM_REFERENCE["clip_macro_f1_mean"],
                    "model_size_bytes": LIGHTGBM_REFERENCE["mean_model_size_bytes"],
                    "size_reduction_fraction": 0.0,
                    "classifier_batch1_p50_ns": None,
                    "parameter_count": None,
                }
            ]
            + [
                {
                    "model": name,
                    "clip_accuracy_mean": model_results[name]["metrics"]["clip_accuracy"]["mean"],
                    "clip_macro_f1_mean": model_results[name]["metrics"]["clip_macro_f1"]["mean"],
                    "model_size_bytes": model_results[name]["model_size_bytes"]["median"],
                    "size_reduction_fraction": model_results[name]["lightgbm_comparison"]["size_reduction_fraction"],
                    "classifier_batch1_p50_ns": model_results[name]["classifier_only_benchmark"]["median_p50_ns"],
                    "parameter_count": model_results[name]["parameter_count"],
                    "clip_macro_f1_loss_vs_lightgbm": model_results[name]["lightgbm_comparison"]["clip_macro_f1_loss"],
                    "success_classification": model_results[name]["lightgbm_comparison"]["success_classification"],
                }
                for name in ARCHITECTURE_IDS
            ],
            "pareto_front_compact_models": pareto_front(pareto_rows),
            "pareto_dimensions": ["clip_macro_f1_maximize", "model_size_minimize", "classifier_p50_minimize"],
        },
        "environment": safe_environment(int(run_manifest["threads"])),
        "status": {"outcome": "success", "error": None},
    }
    document["aggregate_sha256"] = document_sha256(
        document, excluded_fields=("created_at_utc", "aggregate_sha256")
    )
    return document


def parse_models(value: str) -> tuple[str, ...]:
    models = tuple(part.strip() for part in value.split(",") if part.strip())
    if not models or len(set(models)) != len(models) or any(model not in ARCHITECTURE_IDS for model in models):
        raise ValueError("models must be a unique comma-separated subset of linear,mlp128")
    return models


def run_compact_cross_fold(
    *, cache_root: Path, split_manifest_dir: Path, output_dir: Path, models: Sequence[str],
    threads: int, fold: Optional[int], resume: bool, pretty: bool,
) -> dict[str, Any]:
    selected = tuple(models)
    if threads < 1 or (fold is not None and fold not in range(1, 11)):
        raise ValueError("threads or fold is outside the supported range")
    if not selected or len(set(selected)) != len(selected) or any(name not in ARCHITECTURE_IDS for name in selected):
        raise ValueError("unsupported or duplicate compact model selection")
    manifests = load_split_manifests(split_manifest_dir)
    verified = load_verified_cache_records(cache_root)
    expected = build_run_manifest(manifests, verified, threads=threads)
    root = Path(output_dir)
    if root.exists():
        if not resume:
            raise FileExistsError("output directory exists; --resume is required")
        existing = json.loads((root / "run-manifest.json").read_text(encoding="utf-8"))
        validate_resume_manifest(existing, expected)
        run_manifest = existing
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_result(root / "run-manifest.json", expected, pretty=True)
        run_manifest = expected

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    _configure_tensorflow_cpu(tf, threads)
    targets = [fold] if fold is not None else list(range(1, 11))
    trained: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for architecture_id in selected:
        for test_fold in targets:
            directory = root / architecture_id / f"fold-{test_fold:02d}"
            if directory.exists() and _fold_result_valid(
                directory, run_manifest["run_identity_sha256"], architecture_id, test_fold
            ):
                skipped.append({"architecture_id": architecture_id, "test_fold": test_fold})
                continue
            if directory.exists():
                os.rename(directory, root / architecture_id / f".invalid-fold-{test_fold:02d}-{time.time_ns()}")
            _train_fold(
                tf=tf,
                architecture_id=architecture_id,
                fold=test_fold,
                split_manifest=manifests[test_fold],
                verified=verified,
                directory=directory,
                run_identity=run_manifest["run_identity_sha256"],
                threads=threads,
                pretty=pretty,
            )
            trained.append({"architecture_id": architecture_id, "test_fold": test_fold})

    results: dict[str, list[dict[str, Any]]] = {name: [] for name in ARCHITECTURE_IDS}
    for architecture_id in ARCHITECTURE_IDS:
        for test_fold in range(1, 11):
            directory = root / architecture_id / f"fold-{test_fold:02d}"
            if _fold_result_valid(directory, run_manifest["run_identity_sha256"], architecture_id, test_fold):
                results[architecture_id].append(
                    json.loads((directory / "fold-result.json").read_text(encoding="utf-8"))
                )
    aggregate = build_aggregate(results, run_manifest)
    aggregate["execution"] = {
        "selected_models": list(selected),
        "requested_fold": fold,
        "resume": resume,
        "trained": trained,
        "skipped": skipped,
    }
    if aggregate["status"]["outcome"] == "success":
        aggregate["aggregate_sha256"] = document_sha256(
            aggregate, excluded_fields=("created_at_utc", "aggregate_sha256")
        )
    atomic_replace_text(
        root / "aggregate.json",
        json.dumps(aggregate, indent=2 if pretty else None, sort_keys=True, ensure_ascii=True) + "\n",
    )
    return aggregate


__all__ = [
    "ARCHITECTURE_IDS", "BENCHMARK_ITERATIONS", "BENCHMARK_WARMUP", "FOLD_SCHEMA_VERSION",
    "LIGHTGBM_REFERENCE", "RUN_SCHEMA_VERSION", "SCHEMA_VERSION", "SUCCESS_THRESHOLDS",
    "ValidationClipF1Controller", "architecture_config", "build_aggregate", "build_run_manifest",
    "classify_compact_result", "pareto_front", "parse_models", "run_compact_cross_fold",
    "training_config", "validate_resume_manifest",
]
