"""Deployment-only Linear classifier training over every verified cached segment."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import shutil
import statistics
import tempfile
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from urbansound_segment_task.edge_v2.data.manifest import document_sha256
from urbansound_segment_task.edge_v2.evaluation.result_schema import utc_timestamp, write_result
from urbansound_segment_task.edge_v2.export.onnx_linear import build_linear_keras_model
from urbansound_segment_task.edge_v2.features.cache_dataset import (
    EXPECTED_CACHE_IDENTITY,
    VerifiedCacheRecords,
    build_cache_split,
    load_verified_cache_records,
)
from urbansound_segment_task.edge_v2.models.yamnet_artifact import streaming_file_sha256


DEPLOYMENT_SCHEMA_VERSION = "edge-v2.deployment-classifier.v1"
DEPLOYMENT_ID = "linear-all-data-deployment-v1"
EXPECTED_SEGMENT_COUNT = 53_918
EXPECTED_ZERO_SEGMENT_CLIPS = 433
EXPECTED_METADATA_CLIPS = 8_732
EXPECTED_PARAMETER_COUNT = 10_250
EXPECTED_BEST_EPOCHS = (17, 12, 13, 4, 8, 11, 5, 5, 3, 5)
EXPECTED_EPOCHS = 7


def _configure_tensorflow_cpu(tf: Any, threads: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()


def balanced_class_weights_all_segments(labels: np.ndarray) -> dict[int, float]:
    values = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(values, minlength=10)
    if values.ndim != 1 or values.size == 0 or counts.shape != (10,) or np.any(counts == 0):
        raise ValueError("deployment class weights require all ten classes")
    return {
        class_id: float(values.size / (10 * counts[class_id])) for class_id in range(10)
    }


def select_deployment_epochs(best_epochs: Sequence[int]) -> dict[str, Any]:
    values = tuple(int(value) for value in best_epochs)
    if values != EXPECTED_BEST_EPOCHS:
        raise ValueError("cross-fold Linear best epochs do not match the validated source")
    median = float(statistics.median(values))
    selected = int(Decimal(str(median)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if selected != EXPECTED_EPOCHS:
        raise ValueError("deployment epoch selection did not resolve to seven")
    return {
        "source": "validated_10_fold_linear_best_epochs",
        "best_epochs": list(values),
        "median": median,
        "rounding": "round_half_up",
        "selected_epochs": selected,
    }


def validate_epoch_source(aggregate_path: Path) -> tuple[dict[str, Any], str]:
    document = json.loads(Path(aggregate_path).read_text(encoding="utf-8"))
    if document.get("schema_version") != "edge-v2.compact-classifier-cross-fold.v1":
        raise ValueError("cross-fold aggregate schema mismatch")
    if document.get("status", {}).get("outcome") != "success":
        raise ValueError("cross-fold aggregate is not successful")
    expected_hash = document_sha256(document, excluded_fields=("created_at_utc", "aggregate_sha256"))
    if document.get("aggregate_sha256") != expected_hash:
        raise ValueError("cross-fold aggregate SHA-256 mismatch")
    rows = document.get("models", {}).get("linear", {}).get("fold_results", [])
    values = tuple(int(row["best_epoch"]) for row in sorted(rows, key=lambda row: int(row["test_fold"])))
    return select_deployment_epochs(values), str(document["aggregate_sha256"])


def deployment_training_config(epoch_selection: Mapping[str, Any], *, threads: int) -> dict[str, Any]:
    if threads < 1:
        raise ValueError("threads must be positive")
    document: dict[str, Any] = {
        "deployment_id": DEPLOYMENT_ID,
        "deployment_only": True,
        "independent_test_metrics_available": False,
        "architecture": {
            "architecture_id": "linear", "input_features": 1024,
            "layers": [{"type": "dense", "units": 10, "activation": "softmax"}],
            "parameter_count": EXPECTED_PARAMETER_COUNT,
        },
        "optimizer": "keras.optimizers.Adam",
        "learning_rate": 0.001,
        "batch_size": 512,
        "seed": 42,
        "loss": "sparse_categorical_crossentropy",
        "class_weight": {"mode": "balanced", "source": "all_53918_training_segments"},
        "shuffle": True,
        "epochs": int(epoch_selection["selected_epochs"]),
        "epoch_selection": dict(epoch_selection),
        "validation_split": None,
        "test_split": None,
        "early_stopping": False,
        "threads": threads,
        "cpu_only": True,
    }
    document["config_sha256"] = document_sha256(document, excluded_fields=("config_sha256",))
    return document


def build_deployment_training_set(
    verified: VerifiedCacheRecords, *, enforce_expected_counts: bool = True
) -> Any:
    split = build_cache_split("all_verified_segments", range(1, 11), verified.records)
    if enforce_expected_counts and (
        split.segment_count != EXPECTED_SEGMENT_COUNT
        or split.zero_segment_clip_count != EXPECTED_ZERO_SEGMENT_CLIPS
        or split.metadata_clip_count != EXPECTED_METADATA_CLIPS
        or split.evaluable_clip_count != EXPECTED_METADATA_CLIPS - EXPECTED_ZERO_SEGMENT_CLIPS
    ):
        raise ValueError("all-cache deployment training counts mismatch")
    if split.X.shape != (split.segment_count, 1024) or split.y.shape != (split.segment_count,):
        raise ValueError("deployment training tensor contract mismatch")
    return split


def _artifact(path: Path, relative_path: str) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": streaming_file_sha256(path),
    }


def _write_loss_history(path: Path, losses: Sequence[Any]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "training_loss"))
        writer.writeheader()
        for index, loss in enumerate(losses, start=1):
            writer.writerow({"epoch": index, "training_loss": float(loss)})


def train_deployment_linear(
    *, cache_root: Path, cross_fold_aggregate: Path, epochs: int, threads: int,
    output_directory: Path, pretty: bool, tf_module: Optional[Any] = None,
) -> dict[str, Any]:
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("deployment classifier output directory already exists")
    epoch_selection, aggregate_identity = validate_epoch_source(cross_fold_aggregate)
    if epochs != EXPECTED_EPOCHS or epochs != int(epoch_selection["selected_epochs"]):
        raise ValueError("deployment training must use exactly seven selected epochs")
    verified = load_verified_cache_records(cache_root)
    if verified.cache_identity != EXPECTED_CACHE_IDENTITY:
        raise ValueError("deployment cache identity mismatch")
    training = build_deployment_training_set(verified)
    class_weights = balanced_class_weights_all_segments(training.y)
    config = deployment_training_config(epoch_selection, threads=threads)
    if tf_module is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import tensorflow as tf_module
    _configure_tensorflow_cpu(tf_module, threads)
    tf_module.keras.backend.clear_session()
    tf_module.keras.utils.set_random_seed(42)
    model = build_linear_keras_model(tf_module)
    if int(model.count_params()) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("deployment Linear parameter count mismatch")
    model.compile(
        optimizer=tf_module.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
    )
    start = time.perf_counter()
    history = model.fit(
        training.X, training.y, epochs=epochs, batch_size=512, shuffle=True,
        class_weight=class_weights, verbose=0,
    )
    training_seconds = time.perf_counter() - start
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".deployment-linear-", dir=output.parent))
    try:
        weights_path = temporary / "model.weights.h5"
        config_path = temporary / "training-config.json"
        history_path = temporary / "training-history.csv"
        result_path = temporary / "deployment-result.json"
        model.save_weights(str(weights_path))
        write_result(config_path, config, pretty=True)
        _write_loss_history(history_path, history.history.get("loss", ()))
        model_artifact = _artifact(weights_path, "model.weights.h5")
        model_artifact.update(
            {
                "format": "keras_hdf5_weights_only",
                "architecture_id": "linear",
                "parameter_count": EXPECTED_PARAMETER_COUNT,
                "config_sha256": config["config_sha256"],
            }
        )
        document: dict[str, Any] = {
            "schema_version": DEPLOYMENT_SCHEMA_VERSION,
            "created_at_utc": utc_timestamp(),
            "deployment_id": DEPLOYMENT_ID,
            "deployment_only": True,
            "independent_test_metrics_available": False,
            "scientific_scope": "deployment-only; not an independently evaluated test model",
            "cache_identity": verified.cache_identity,
            "cache_index_sha256": verified.index_sha256,
            "dataset_manifest_sha256": verified.dataset_manifest_sha256,
            "yamnet_artifact_tree_sha256": verified.yamnet_artifact_tree_sha256,
            "architecture": config["architecture"],
            "config_sha256": config["config_sha256"],
            "epoch_selection": dict(epoch_selection),
            "epochs": epochs,
            "class_weights": {str(key): float(value) for key, value in sorted(class_weights.items())},
            "training_data": {
                "segment_count": training.segment_count,
                "metadata_clip_count": training.metadata_clip_count,
                "evaluable_clip_count": training.evaluable_clip_count,
                "zero_segment_clip_count": training.zero_segment_clip_count,
                "folds": list(training.folds),
                "artificial_features_added": 0,
                "validation_or_test_split_created": False,
            },
            "training": {
                "duration_seconds": training_seconds,
                "batch_size": 512, "shuffle": True, "early_stopping": False,
            },
            "runtime": {
                "tensorflow_version": str(tf_module.__version__),
                "keras_version": importlib.metadata.version("keras"),
                "accelerator": "none_cpu_only", "threads": threads,
            },
            "source_cross_fold_aggregate_sha256": aggregate_identity,
            "artifacts": {
                "model": model_artifact,
                "config": _artifact(config_path, "training-config.json"),
                "history": _artifact(history_path, "training-history.csv"),
            },
            "limitations": [
                "Deployment-only model trained on all evaluable cached segments from all ten folds.",
                "No independent validation or test split remains; no accuracy or macro-F1 is available.",
                "Zero-segment clips contribute no artificial feature or label.",
            ],
            "status": {"outcome": "success", "error": None},
        }
        document["result_sha256"] = document_sha256(
            document, excluded_fields=("created_at_utc", "training", "result_sha256")
        )
        write_result(result_path, document, pretty=pretty)
        os.rename(temporary, output)
        return document
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        tf_module.keras.backend.clear_session()


__all__ = [
    "DEPLOYMENT_ID", "DEPLOYMENT_SCHEMA_VERSION", "EXPECTED_BEST_EPOCHS",
    "EXPECTED_EPOCHS", "EXPECTED_PARAMETER_COUNT", "EXPECTED_SEGMENT_COUNT",
    "EXPECTED_ZERO_SEGMENT_CLIPS", "build_deployment_training_set",
    "balanced_class_weights_all_segments", "deployment_training_config",
    "select_deployment_epochs", "train_deployment_linear",
    "validate_epoch_source",
]
