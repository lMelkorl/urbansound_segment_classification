"""Offline held-out-fold audio robustness evaluation orchestration."""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from urbansound_segment_task.edge_v2.benchmarks.schema import utc_timestamp
from urbansound_segment_task.edge_v2.data.manifest import document_sha256, serialize_json
from urbansound_segment_task.edge_v2.data.splits import SPLIT_POLICY_ID
from urbansound_segment_task.edge_v2.data.urbansound8k import detect_dataset_layout
from urbansound_segment_task.edge_v2.evaluation.robustness import (
    CLASS_NAMES, CONDITIONS, PERTURBATION_CONTRACTS, SCHEMA_VERSION, UNIT_SCHEMA_VERSION,
    aggregate_conditions, apply_perturbation, clip_probability_mean, fold_condition_metrics,
    require_clean_agreement, select_panel,
)
from urbansound_segment_task.edge_v2.evaluation.segmentation import (
    LEGACY_SEGMENTATION_POLICY, iter_segments, segment_plan,
)
from urbansound_segment_task.edge_v2.features.cache_dataset import (
    EXPECTED_CACHE_IDENTITY, EXPECTED_DATASET_MANIFEST_SHA256, EXPECTED_YAMNET_TREE_SHA256,
    VerifiedCacheRecords, load_verified_cache_records,
)
from urbansound_segment_task.edge_v2.features.extractor import (
    LocalYamnetEmbeddingBackend, load_audio_legacy,
)
from urbansound_segment_task.edge_v2.features.yamnet_cache import atomic_replace_text
from urbansound_segment_task.edge_v2.models.compact_classifier import RUN_SCHEMA_VERSION
from urbansound_segment_task.edge_v2.models.yamnet_artifact import (
    streaming_file_sha256, verify_yamnet_artifact,
)


EXPECTED_FOLD_SCHEMA = "edge-v2.compact-classifier-fold.v1"
EXPECTED_MODEL_CONFIG_SHA256 = "1161b11855620601b39d82f08b5894e4ed135df20f2ea8db38fd72b8ce173605"
EXPECTED_RUN_IDENTITY = "fcd7b71f9677ffa8e2ad41c3e7af1cf77e9893b952b6b127e74e25af39443e31"
RUNTIME_CONFIG = {
    "cpu_only": True, "threads": 8,
    "audio_loader": "librosa.load(sr=16000,mono=True,dtype=float32,res_type=soxr_hq)",
    "yamnet_loader": "tf.saved_model.load(local_path)",
    "classifier_runtime": "tensorflow_keras_weights",
    "clip_aggregation": "arithmetic_mean_of_canonical_class_probabilities",
}


@dataclass(frozen=True)
class FoldModel:
    test_fold: int
    validation_fold: int
    weights_path: Path
    weights_relative_path: str
    weights_sha256: str
    weights_size_bytes: int
    fold_identity_sha256: str
    config_sha256: str
    split_manifest_sha256: str

    def safe_identity(self) -> dict[str, Any]:
        return {
            "test_fold": self.test_fold, "validation_fold": self.validation_fold,
            "architecture_id": "linear", "parameter_count": 10_250,
            "model_sha256": self.weights_sha256, "model_size_bytes": self.weights_size_bytes,
            "fold_identity_sha256": self.fold_identity_sha256,
            "config_sha256": self.config_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
        }


def _safe_relative(value: Any) -> str:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("model artifact relative path is unsafe")
    return path.as_posix()


def resolve_fold_models(model_run: Path, folds: Sequence[int]) -> tuple[dict[str, Any], dict[int, FoldModel]]:
    root = Path(model_run)
    run_path = root / "run-manifest.json"
    if not run_path.is_file():
        candidates = (root / "artifact-manifest.json", root / "deployment-result.json")
        for candidate in candidates:
            if candidate.is_file():
                document = json.loads(candidate.read_text(encoding="utf-8"))
                if document.get("deployment_only") is True:
                    raise ValueError("deployment-only all-data model is forbidden for robustness evaluation")
        raise ValueError("compact classifier run manifest is missing")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("deployment_only") is True:
        raise ValueError("deployment-only all-data model is forbidden for robustness evaluation")
    if run.get("schema_version") != RUN_SCHEMA_VERSION or run.get("run_identity_sha256") != EXPECTED_RUN_IDENTITY:
        raise ValueError("compact classifier run identity or schema mismatch")
    if (
        run.get("cache_identity") != EXPECTED_CACHE_IDENTITY
        or run.get("dataset_manifest_sha256") != EXPECTED_DATASET_MANIFEST_SHA256
        or run.get("yamnet_artifact_tree_sha256") != EXPECTED_YAMNET_TREE_SHA256
        or run.get("model_config_sha256", {}).get("linear") != EXPECTED_MODEL_CONFIG_SHA256
        or run.get("policy_id") != SPLIT_POLICY_ID
    ):
        raise ValueError("compact classifier provenance mismatch")
    resolved: dict[int, FoldModel] = {}
    for fold in folds:
        if fold not in range(1, 11):
            raise ValueError("fold must be in 1..10")
        directory = root / "linear" / f"fold-{fold:02d}"
        result_path = directory / "fold-result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        sidecar = (directory / "fold-result.sha256").read_text(encoding="ascii").strip()
        if streaming_file_sha256(result_path) != sidecar:
            raise ValueError("fold result sidecar SHA-256 mismatch")
        expected_validation = fold % 10 + 1
        if (
            result.get("schema_version") != EXPECTED_FOLD_SCHEMA
            or result.get("status", {}).get("outcome") != "success"
            or result.get("architecture_id") != "linear"
            or int(result.get("test_fold", 0)) != fold
            or int(result.get("validation_fold", 0)) != expected_validation
            or result.get("run_identity_sha256") != run["run_identity_sha256"]
            or result.get("config_sha256") != EXPECTED_MODEL_CONFIG_SHA256
            or result.get("cache_identity") != EXPECTED_CACHE_IDENTITY
            or result.get("split_manifest_sha256") != run["split_manifest_sha256"][str(fold)]
            or int(result.get("architecture", {}).get("parameter_count", 0)) != 10_250
        ):
            raise ValueError("fold model binding or provenance mismatch")
        artifact = result.get("artifacts", {}).get("model", {})
        relative = _safe_relative(artifact.get("relative_path"))
        weights = directory / relative
        digest = streaming_file_sha256(weights)
        size = weights.stat().st_size
        if digest != artifact.get("sha256") or size != int(artifact.get("size_bytes", -1)):
            raise ValueError("fold model artifact hash or size mismatch")
        resolved[fold] = FoldModel(
            test_fold=fold, validation_fold=expected_validation, weights_path=weights,
            weights_relative_path=relative, weights_sha256=digest, weights_size_bytes=size,
            fold_identity_sha256=str(result["fold_identity_sha256"]),
            config_sha256=str(result["config_sha256"]),
            split_manifest_sha256=str(result["split_manifest_sha256"]),
        )
    return run, resolved


def read_metadata_rows(dataset_root: Path) -> list[dict[str, Any]]:
    layout = detect_dataset_layout(dataset_root)
    rows: list[dict[str, Any]] = []
    with layout.metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            fold = int(row["fold"])
            filename = Path(str(row["slice_file_name"])).name
            if filename != row["slice_file_name"]:
                raise ValueError("unsafe metadata audio filename")
            rows.append({
                "clip_key": f"fold{fold}/{filename}", "fold": fold,
                "class_id": int(row["classID"]), "class_name": str(row["class"]),
            })
    return rows


def build_robustness_panel(
    dataset_root: Path, verified_cache: VerifiedCacheRecords, clips_per_class_per_fold: int
) -> dict[str, Any]:
    evaluable = {
        str(row["clip_key"]) for row in verified_cache.records
        if int(row["embeddings"].shape[0]) > 0
    }
    return select_panel(
        read_metadata_rows(dataset_root), evaluable_clip_keys=evaluable,
        clips_per_class_per_fold=clips_per_class_per_fold,
        dataset_manifest_sha256=verified_cache.dataset_manifest_sha256,
        cache_identity=verified_cache.cache_identity,
    )


def unit_identity(
    *, panel_identity: str, condition: str, model: FoldModel,
    dataset_manifest_sha256: str, yamnet_tree_sha256: str,
) -> str:
    return document_sha256({
        "panel_identity_sha256": panel_identity,
        "perturbation_contract": PERTURBATION_CONTRACTS[condition],
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "yamnet_tree_sha256": yamnet_tree_sha256,
        "model": model.safe_identity(),
        "segmentation_policy": LEGACY_SEGMENTATION_POLICY.as_dict(),
        "runtime_config": RUNTIME_CONFIG,
    })


def _load_resume_unit(path: Path, expected_identity: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    actual_identity = document.get("unit_identity_sha256")
    if isinstance(actual_identity, str) and actual_identity != expected_identity:
        raise ValueError("resume rejected: fold/condition identity changed")
    expected_hash = document_sha256(
        document, excluded_fields=("created_at_utc", "duration_seconds", "result_sha256")
    )
    if (
        actual_identity == expected_identity
        and document.get("status", {}).get("outcome") == "success"
        and document.get("result_sha256") == expected_hash
    ):
        return document
    return None


def _replace_json(path: Path, document: Mapping[str, Any], *, pretty: bool) -> None:
    atomic_replace_text(path, serialize_json(document, pretty=pretty))


def _replace_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "fold", "condition", "class_id", "class_name", "precision", "recall", "f1", "support"
    )
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row[field]) for field in fields))
    atomic_replace_text(path, "\n".join(lines) + "\n")


def _predict(model: Any, embeddings: np.ndarray) -> tuple[np.ndarray, int, float]:
    segment_probabilities = np.asarray(model(embeddings, training=False), dtype=np.float32)
    mean = clip_probability_mean(segment_probabilities)
    predicted = int(np.argmax(mean))
    return mean, predicted, float(mean[predicted])


def evaluate_fold_condition(
    *, condition: str, clips: Sequence[Mapping[str, Any]], dataset_root: Path,
    cache_by_key: Mapping[str, Mapping[str, Any]], yamnet: Any, classifier: Any,
    clean_clips: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    layout = detect_dataset_layout(dataset_root)
    output: list[dict[str, Any]] = []
    runtime_clean: list[int] = []
    cached_clean: list[int] = []
    for clip in clips:
        key = str(clip["clip_key"])
        waveform, _ = load_audio_legacy(layout.audio_root / key)
        perturbed, perturbation = apply_perturbation(waveform, clip_key=key, condition=condition)
        plan = segment_plan(int(perturbed.shape[0]))
        if int(plan["segment_count"]) < 1:
            raise ValueError("panel clip unexpectedly produced zero segments")
        embeddings = np.stack([yamnet.extract(segment) for segment in iter_segments(perturbed)]).astype(
            np.float32, copy=False
        )
        mean, predicted, confidence = _predict(classifier, embeddings)
        row = {
            "clip_key": key, "fold": int(clip["fold"]), "class_id": int(clip["class_id"]),
            "predicted_class_id": predicted, "confidence": confidence,
            "segment_count": int(embeddings.shape[0]),
            "segment_start_samples": plan["segment_start_samples"],
            "perturbation": perturbation,
        }
        output.append(row)
        if condition == "clean":
            cached = cache_by_key[key]
            _, cached_predicted, _ = _predict(classifier, cached["embeddings"])
            runtime_clean.append(predicted)
            cached_clean.append(cached_predicted)
    control = require_clean_agreement(runtime_clean, cached_clean) if condition == "clean" else None
    return output, control


def _per_class_aggregates(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for condition in CONDITIONS:
        rows = [unit for unit in units if unit["condition"] == condition]
        if not rows:
            continue
        per_class: list[dict[str, Any]] = []
        for class_id, class_name in enumerate(CLASS_NAMES):
            fold_f1 = [float(row["metrics"]["per_class"][class_id]["f1"]) for row in rows]
            support = sum(int(row["metrics"]["per_class"][class_id]["support"]) for row in rows)
            class_clips = [clip for row in rows for clip in row["clips"] if int(clip["class_id"]) == class_id]
            flip_rate = float(np.mean([bool(clip.get("prediction_flipped", False)) for clip in class_clips]))
            per_class.append({
                "class_id": class_id, "class_name": class_name, "support": support,
                "f1_mean": statistics.fmean(fold_f1),
                "f1_population_standard_deviation": statistics.pstdev(fold_f1),
                "prediction_flip_rate": flip_rate,
            })
        grouped[condition] = per_class
    clean = {row["class_id"]: row for row in grouped.get("clean", [])}
    for condition, rows in grouped.items():
        for row in rows:
            row["absolute_f1_drop"] = clean.get(row["class_id"], row)["f1_mean"] - row["f1_mean"]
    return grouped


def _class_analysis(per_class: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    analysis: dict[str, Any] = {}
    for condition, rows in per_class.items():
        if condition == "clean":
            continue

        def summary(row: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "class_id": int(row["class_id"]), "class_name": str(row["class_name"]),
                "support": int(row["support"]), "f1_mean": float(row["f1_mean"]),
                "absolute_f1_drop": float(row["absolute_f1_drop"]),
                "prediction_flip_rate": float(row["prediction_flip_rate"]),
                "f1_population_standard_deviation": float(row["f1_population_standard_deviation"]),
            }

        analysis[condition] = {
            "most_robust_three_by_smallest_f1_drop": [
                summary(row) for row in sorted(
                    rows, key=lambda item: (float(item["absolute_f1_drop"]), int(item["class_id"]))
                )[:3]
            ],
            "largest_f1_loss_three": [
                summary(row) for row in sorted(
                    rows, key=lambda item: (-float(item["absolute_f1_drop"]), int(item["class_id"]))
                )[:3]
            ],
            "highest_prediction_flip_three": [
                summary(row) for row in sorted(
                    rows, key=lambda item: (-float(item["prediction_flip_rate"]), int(item["class_id"]))
                )[:3]
            ],
            "most_variable_across_folds_three": [
                summary(row) for row in sorted(
                    rows,
                    key=lambda item: (
                        -float(item["f1_population_standard_deviation"]), int(item["class_id"])
                    ),
                )[:3]
            ],
        }
    return analysis


def run_audio_robustness(
    *, dataset_root: Path, model_run: Path, yamnet_artifact: Path,
    clips_per_class_per_fold: int, conditions: Sequence[str], fold: int | None,
    output_dir: Path, resume: bool, pretty: bool,
) -> dict[str, Any]:
    selected_conditions = tuple(dict.fromkeys(conditions))
    if not selected_conditions or any(item not in CONDITIONS for item in selected_conditions):
        raise ValueError("conditions must be selected from the fixed allowlist")
    if "clean" not in selected_conditions:
        raise ValueError("clean condition is required for controls and degradation metrics")
    folds = (fold,) if fold is not None else tuple(range(1, 11))
    output = Path(output_dir)
    if output.exists() and not resume:
        raise FileExistsError("output directory exists; use --resume")
    run, models = resolve_fold_models(model_run, folds)
    yamnet_identity = verify_yamnet_artifact(yamnet_artifact)
    if yamnet_identity.get("tree_sha256") != EXPECTED_YAMNET_TREE_SHA256:
        raise ValueError("YAMNet artifact tree SHA-256 mismatch")
    repository_root = Path(__file__).resolve().parents[3]
    verified = load_verified_cache_records(repository_root / "cache" / "yamnet_embeddings")
    panel = build_robustness_panel(dataset_root, verified, clips_per_class_per_fold)
    run_identity = document_sha256({
        "schema_version": SCHEMA_VERSION,
        "panel_identity_sha256": panel["panel_identity_sha256"],
        "conditions": list(selected_conditions), "folds": list(folds),
        "dataset_manifest_sha256": verified.dataset_manifest_sha256,
        "cache_identity": verified.cache_identity,
        "yamnet_tree_sha256": yamnet_identity["tree_sha256"],
        "models": {str(key): value.safe_identity() for key, value in models.items()},
        "segmentation_policy": LEGACY_SEGMENTATION_POLICY.as_dict(), "runtime_config": RUNTIME_CONFIG,
    })
    manifest_path = output / "run-manifest.json"
    if manifest_path.exists():
        try:
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("resume rejected: run manifest is corrupt") from exc
        if prior_manifest.get("run_identity_sha256") != run_identity:
            raise ValueError("resume rejected: run identity changed")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("resume rejected: non-empty output lacks run manifest")
    manifest = {
        "schema_version": "edge-v2.audio-robustness-run.v1", "created_at_utc": utc_timestamp(),
        "run_identity_sha256": run_identity, "panel_identity_sha256": panel["panel_identity_sha256"],
        "conditions": list(selected_conditions), "folds": list(folds),
        "dataset_manifest_sha256": verified.dataset_manifest_sha256,
        "cache_identity": verified.cache_identity, "yamnet_tree_sha256": yamnet_identity["tree_sha256"],
        "model_identities": {str(key): value.safe_identity() for key, value in models.items()},
        "segmentation_policy": LEGACY_SEGMENTATION_POLICY.as_dict(), "runtime_config": RUNTIME_CONFIG,
    }
    output.mkdir(parents=True, exist_ok=True)
    _replace_json(manifest_path, manifest, pretty=pretty)
    _replace_json(output / "panel-manifest.json", panel, pretty=pretty)
    cache_by_key = {str(row["clip_key"]): row for row in verified.records}
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    yamnet = LocalYamnetEmbeddingBackend(Path(yamnet_artifact), int(RUNTIME_CONFIG["threads"]))
    tf = yamnet.tensorflow
    from urbansound_segment_task.edge_v2.export.onnx_linear import build_linear_keras_model
    completed: list[dict[str, Any]] = []
    skipped = 0
    for test_fold in folds:
        model_spec = models[test_fold]
        classifier = build_linear_keras_model(tf)
        classifier.load_weights(model_spec.weights_path)
        fold_clips = [clip for clip in panel["clips"] if int(clip["fold"]) == test_fold]
        clean_rows: list[dict[str, Any]] | None = None
        for condition in selected_conditions:
            identity = unit_identity(
                panel_identity=panel["panel_identity_sha256"], condition=condition, model=model_spec,
                dataset_manifest_sha256=verified.dataset_manifest_sha256,
                yamnet_tree_sha256=yamnet_identity["tree_sha256"],
            )
            unit_path = output / "raw" / f"fold-{test_fold:02d}" / f"{condition}.json"
            prior = _load_resume_unit(unit_path, identity) if resume else None
            if prior is not None:
                completed.append(prior)
                skipped += 1
                if condition == "clean":
                    clean_rows = prior["clips"]
                continue
            if condition != "clean" and clean_rows is None:
                clean_path = output / "raw" / f"fold-{test_fold:02d}" / "clean.json"
                clean_prior = _load_resume_unit(
                    clean_path,
                    unit_identity(
                        panel_identity=panel["panel_identity_sha256"], condition="clean", model=model_spec,
                        dataset_manifest_sha256=verified.dataset_manifest_sha256,
                        yamnet_tree_sha256=yamnet_identity["tree_sha256"],
                    ),
                )
                if clean_prior is None:
                    raise ValueError("clean unit must complete before perturbed conditions")
                clean_rows = clean_prior["clips"]
            started = time.perf_counter()
            rows, control = evaluate_fold_condition(
                condition=condition, clips=fold_clips, dataset_root=dataset_root,
                cache_by_key=cache_by_key, yamnet=yamnet, classifier=classifier,
                clean_clips=clean_rows,
            )
            if clean_rows is not None and condition != "clean":
                clean_by_key = {row["clip_key"]: row for row in clean_rows}
                for row in rows:
                    baseline = clean_by_key[row["clip_key"]]
                    row["prediction_flipped"] = row["predicted_class_id"] != baseline["predicted_class_id"]
                    row["confidence_change"] = row["confidence"] - baseline["confidence"]
            else:
                for row in rows:
                    row["prediction_flipped"] = False
                    row["confidence_change"] = 0.0
            metrics = fold_condition_metrics(rows, clean_clips=clean_rows if condition != "clean" else None)
            document: dict[str, Any] = {
                "schema_version": UNIT_SCHEMA_VERSION, "created_at_utc": utc_timestamp(),
                "unit_identity_sha256": identity, "run_identity_sha256": run_identity,
                "panel_identity_sha256": panel["panel_identity_sha256"],
                "fold": test_fold, "validation_fold": model_spec.validation_fold,
                "condition": condition, "model_identity": model_spec.safe_identity(),
                "metrics": metrics, "clean_control": control, "clips": rows,
                "duration_seconds": time.perf_counter() - started,
                "status": {"outcome": "success", "error": None},
            }
            document["result_sha256"] = document_sha256(
                document, excluded_fields=("created_at_utc", "duration_seconds", "result_sha256")
            )
            _replace_json(unit_path, document, pretty=pretty)
            csv_rows = [
                {"fold": test_fold, "condition": condition, "class_id": row["class_id"],
                 "class_name": CLASS_NAMES[int(row["class_id"])], "precision": row["precision"],
                 "recall": row["recall"], "f1": row["f1"], "support": row["support"]}
                for row in metrics["per_class"]
            ]
            _replace_csv(unit_path.with_suffix(".csv"), csv_rows)
            completed.append(document)
            if condition == "clean":
                clean_rows = rows
        del classifier
    condition_aggregates = aggregate_conditions(completed)
    clean_controls = [row["clean_control"] for row in completed if row["condition"] == "clean"]
    per_class = _per_class_aggregates(completed)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "created_at_utc": utc_timestamp(),
        "evaluation_protocol": {
            "held_out_only": True, "deployment_only_model_used": False,
            "test_fold_model_binding": "test_fold=k, validation_fold=k%10+1",
            "panel_selection_prediction_independent": True,
            "clip_aggregation": RUNTIME_CONFIG["clip_aggregation"],
            "macro_f1_class_order": list(range(10)),
        },
        "panel_identity": {"sha256": panel["panel_identity_sha256"], "clip_count": panel["clip_count"]},
        "cache_and_dataset_identities": {
            "cache_identity": verified.cache_identity,
            "cache_index_sha256": verified.index_sha256,
            "dataset_manifest_sha256": verified.dataset_manifest_sha256,
            "yamnet_tree_sha256": yamnet_identity["tree_sha256"],
        },
        "model_identities": {str(key): value.safe_identity() for key, value in models.items()},
        "perturbation_contracts": {key: PERTURBATION_CONTRACTS[key] for key in selected_conditions},
        "fold_results": [
            {"fold": row["fold"], "condition": row["condition"], "unit_identity_sha256": row["unit_identity_sha256"],
             "result_sha256": row["result_sha256"], "metrics": row["metrics"],
             "duration_seconds": row["duration_seconds"]}
            for row in completed
        ],
        "condition_aggregates": condition_aggregates,
        "per_class_aggregates": per_class,
        "class_analysis": _class_analysis(per_class),
        "clean_control": {
            "required_top1_agreement": 1.0,
            "fold_controls": clean_controls,
            "all_folds_passed": bool(clean_controls) and all(row["top1_agreement"] == 1.0 for row in clean_controls),
        },
        "resume": {"valid_units_skipped": skipped, "unit": "fold_plus_condition"},
        "duration_seconds": sum(float(row["duration_seconds"]) for row in completed),
        "limitations": [
            "Controlled synthetic perturbations do not represent all real-world acoustic shifts.",
            "The deterministic panel is capped per fold and class and is not the full held-out dataset.",
            "Softmax confidence is descriptive and is not treated as calibrated probability.",
            "gun_shot has limited support and must be interpreted with its reported support.",
        ],
        "status": {"outcome": "success", "error": None},
    }
    result["result_sha256"] = document_sha256(
        result, excluded_fields=("created_at_utc", "duration_seconds", "result_sha256")
    )
    _replace_json(output / "aggregate.json", result, pretty=pretty)
    return result


__all__ = [
    "EXPECTED_MODEL_CONFIG_SHA256", "EXPECTED_RUN_IDENTITY", "FoldModel", "RUNTIME_CONFIG",
    "build_robustness_panel", "evaluate_fold_condition", "read_metadata_rows",
    "resolve_fold_models", "run_audio_robustness", "unit_identity",
]
