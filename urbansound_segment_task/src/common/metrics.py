import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from typing import Tuple, Dict, Any

def segment_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "segment_accuracy": float(accuracy_score(y_true, y_pred)),
        "segment_macroF1": float(f1_score(y_true, y_pred, average='macro'))
    }

def per_class_report(y_true, y_pred) -> Dict[str, Any]:
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)

def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def clip_level_from_probs(clip_ids, y_true, proba) -> Tuple[float, float, tuple]:
    df = pd.DataFrame({"clip": clip_ids, "y": y_true})
    for i in range(proba.shape[1]):
        df[f"p{i}"] = proba[:, i]
    g = df.groupby("clip").mean(numeric_only=True)
    y_clip_true = df.groupby("clip")["y"].first().reindex(g.index).values
    pcols = [c for c in g.columns if c.startswith("p")]
    y_clip_pred = g[pcols].values.argmax(axis=1)
    acc = accuracy_score(y_clip_true, y_clip_pred)
    f1m = f1_score(y_clip_true, y_clip_pred, average='macro')
    return float(acc), float(f1m), (y_clip_true, y_clip_pred)
