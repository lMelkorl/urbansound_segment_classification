"""Fixed ten-class classification metrics for segment and clip predictions."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .aggregation import CANONICAL_CLASSES


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int] = CANONICAL_CLASSES
) -> dict:
    true = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    canonical = [int(value) for value in labels]
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=canonical, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=canonical, average="macro", zero_division=0)),
        "prediction_count": int(true.shape[0]),
        "class_order": canonical,
        "per_class": [
            {
                "class_id": class_id, "precision": float(precision[index]),
                "recall": float(recall[index]), "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_id in enumerate(canonical)
        ],
        "confusion_matrix": confusion_matrix(true, predicted, labels=canonical).astype(int).tolist(),
    }


__all__ = ["classification_metrics"]
