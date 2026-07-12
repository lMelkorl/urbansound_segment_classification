"""Legacy probability-mean clip aggregation with explicit class alignment."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


CANONICAL_CLASSES = tuple(range(10))


def align_probability_columns(
    probabilities: np.ndarray, model_classes: Sequence[int], canonical_classes: Sequence[int] = CANONICAL_CLASSES
) -> tuple[np.ndarray, dict[str, Any]]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    classes = [int(value) for value in model_classes]
    canonical = [int(value) for value in canonical_classes]
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("probability matrix and model classes do not align")
    if len(set(classes)) != len(classes) or any(value not in canonical for value in classes):
        raise ValueError("model classes are not unique canonical labels")
    aligned = np.zeros((probabilities.shape[0], len(canonical)), dtype=np.float64)
    mapping = []
    for source_index, class_id in enumerate(classes):
        target_index = canonical.index(class_id)
        aligned[:, target_index] = probabilities[:, source_index]
        mapping.append({"model_column": source_index, "class_id": class_id, "canonical_column": target_index})
    return aligned, {
        "model_classes": classes, "canonical_classes": canonical,
        "column_mapping": mapping, "missing_model_classes": sorted(set(canonical) - set(classes)),
    }


def aggregate_clip_probabilities(
    clip_keys: Sequence[str], labels: Sequence[int], probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    keys = np.asarray(clip_keys)
    y = np.asarray(labels, dtype=np.int64)
    proba = np.asarray(probabilities, dtype=np.float64)
    if keys.shape[0] != y.shape[0] or y.shape[0] != proba.shape[0]:
        raise ValueError("clip keys labels and probabilities must have equal rows")
    grouped: dict[str, list[int]] = {}
    for index, key in enumerate(keys.tolist()):
        grouped.setdefault(str(key), []).append(index)
    true_labels = []
    predicted_labels = []
    ordered_keys = sorted(grouped)
    for key in ordered_keys:
        indices = grouped[key]
        clip_labels = np.unique(y[indices])
        if clip_labels.shape[0] != 1:
            raise ValueError("ground truth labels differ within a clip")
        mean_probability = proba[indices].mean(axis=0)
        true_labels.append(int(clip_labels[0]))
        predicted_labels.append(int(np.argmax(mean_probability)))
    return np.asarray(true_labels, dtype=np.int64), np.asarray(predicted_labels, dtype=np.int64), ordered_keys


__all__ = ["CANONICAL_CLASSES", "aggregate_clip_probabilities", "align_probability_columns"]
