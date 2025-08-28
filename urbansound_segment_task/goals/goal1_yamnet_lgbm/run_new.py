#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from urbansound_segment_task.src.common.paths import detect_dataset_layout
from urbansound_segment_task.src.common.segmenter import load_mono_16k, iter_segments
from urbansound_segment_task.src.common.metrics import (
    segment_metrics, per_class_report, confusion, clip_level_from_probs
)


# ---------------------------
# GPU kontrolü
# ---------------------------
print("GPU devices:", tf.config.list_physical_devices("GPU"))
if not tf.config.list_physical_devices("GPU"):
    print("⚠️ GPU bulunamadı, işlem CPU’da yavaş sürecek.")


# ---------------------------
# Zero-pad shift (TTA için)
# ---------------------------
def shift_pad(x: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return x
    if shift > 0:
        return np.r_[np.zeros(shift, dtype=x.dtype), x[:-shift]]
    s = -shift
    return np.r_[x[s:], np.zeros(s, dtype=x.dtype)]


# ---------------------------
# YAMNet embedder (GPU + batch)
# ---------------------------
class YamnetEmbedder:
    def __init__(self, shifts=(0,)):
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        self.shifts = shifts

        @tf.function
        def _embed_fn(wf):
            # wf: (N,) tek segment
            _, embeddings, _ = self.model(wf)
            return tf.reduce_mean(embeddings, axis=0)  # (1024,)

        self._embed_tf = _embed_fn

    def embed_batch(self, seg_batch: np.ndarray) -> np.ndarray:
        """seg_batch: (B, n_samples) → (B, 1024)"""
        out = []
        for seg in seg_batch:
            preds = []
            for s in self.shifts:
                seg_shift = shift_pad(seg, s)
                wf = tf.convert_to_tensor(seg_shift, dtype=tf.float32)
                preds.append(self._embed_tf(wf))
            out.append(tf.reduce_mean(tf.stack(preds), axis=0).numpy())
        return np.vstack(out)  # (B, 1024)


def wav_path(audio_root: Path, fold: int, fname: str) -> Path:
    p = audio_root / f"fold{fold}" / fname
    if not p.exists():
        raise FileNotFoundError(f"WAV bulunamadı: {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="./urbansound_segment_task/goals/goal1_yamnet_lgbm/results_gpu")
    ap.add_argument("--win_sec", type=float, default=0.96)
    ap.add_argument("--overlap", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=64, help="GPU batch size")
    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------
    # Dataset
    # -----------------
    audio_root, meta_csv = detect_dataset_layout(args.data_dir)
    if meta_csv is None or not Path(meta_csv).exists():
        raise FileNotFoundError("UrbanSound8K.csv bulunamadı.")
    meta = pd.read_csv(meta_csv)

    train_folds = set(range(1, 9))
    val_folds = {9}
    test_folds = {10}

    embedder = YamnetEmbedder(shifts=(0, 160, -160))

    def build_split(chosen_folds):
        X, y, clip_ids = [], [], []
        rows = meta[meta["fold"].isin(chosen_folds)]
        segs, seg_labels, seg_clips = [], [], []
        for _, r in tqdm(rows.iterrows(), total=len(rows), desc=f"folds {sorted(chosen_folds)}"):
            fold = int(r["fold"])
            fn = r["slice_file_name"]
            cls = int(r["classID"])
            y_wav, sr = load_mono_16k(str(wav_path(audio_root, fold, fn)), target_sr=16000)
            for seg in iter_segments(y_wav, sr, win_sec=args.win_sec, overlap=args.overlap):
                segs.append(seg)
                seg_labels.append(cls)
                seg_clips.append(fn)

                # batch dolunca GPU’ya yolla
                if len(segs) == args.batch_size:
                    embs = embedder.embed_batch(np.stack(segs))
                    X.append(embs); y.extend(seg_labels); clip_ids.extend(seg_clips)
                    segs, seg_labels, seg_clips = [], [], []

        # kalan batch
        if segs:
            embs = embedder.embed_batch(np.stack(segs))
            X.append(embs); y.extend(seg_labels); clip_ids.extend(seg_clips)

        return np.vstack(X), np.array(y), np.array(clip_ids)

    # -----------------
    # Build splits
    # -----------------
    t0 = time.time(); X_tr, y_tr, clip_tr = build_split(train_folds); t_train_emb = time.time() - t0
    t0 = time.time(); X_va, y_va, clip_va = build_split(val_folds);   t_val_emb   = time.time() - t0
    t0 = time.time(); X_te, y_te, clip_te = build_split(test_folds);  t_test_emb  = time.time() - t0

    print(f"Embeddings hazır. Süreler (s): train={t_train_emb:.1f}, val={t_val_emb:.1f}, test={t_test_emb:.1f}")

    # -----------------
    # Class weights
    # -----------------
    classes = np.unique(y_tr)
    cw = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

    # -----------------
    # Classifiers
    # -----------------
    clf_lgbm = LGBMClassifier(
        n_estimators=2000, num_leaves=256, learning_rate=0.02,
        subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
        random_state=args.seed, class_weight=class_weight_dict
    )

    clf_xgb = XGBClassifier(
        n_estimators=1200, max_depth=12, learning_rate=0.02,
        subsample=0.9, colsample_bytree=0.9,
        tree_method="hist", n_jobs=-1, random_state=args.seed
    )

    clf_log = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)

    # -----------------
    # Train
    # -----------------
    clf_lgbm.fit(X_tr, y_tr)
    clf_xgb.fit(X_tr, y_tr)
    clf_log.fit(X_tr, y_tr)

    def eval_model(clf, name, X_va, y_va, clip_va, X_te, y_te, clip_te):
        t0 = time.time(); proba_va = clf.predict_proba(X_va); total = time.time() - t0
        seg_infer_s = total / len(X_va)
        y_pred_va = proba_va.argmax(axis=1)

        seg_m = segment_metrics(y_va, y_pred_va)
        rep   = per_class_report(y_va, y_pred_va)
        cm    = confusion(y_va, y_pred_va)

        clip_acc, clip_f1m, _ = clip_level_from_probs(clip_va, y_va, proba_va)

        # Test
        proba_te = clf.predict_proba(X_te)
        y_pred_te = proba_te.argmax(axis=1)
        seg_m_te = segment_metrics(y_te, y_pred_te)
        clip_acc_te, clip_f1m_te, _ = clip_level_from_probs(clip_te, y_te, proba_te)

        # Save
        (out_dir / f"metrics_val_{name}.json").write_text(json.dumps({
            **seg_m, "clip_accuracy": clip_acc, "clip_macroF1": clip_f1m,
            "val_segment_infer_seconds": seg_infer_s
        }, indent=2))

        (out_dir / f"metrics_test_{name}.json").write_text(json.dumps({
            **seg_m_te, "clip_accuracy": clip_acc_te, "clip_macroF1": clip_f1m_te
        }, indent=2))

        pd.DataFrame(rep).to_csv(out_dir / f"val_classification_report_{name}.csv", index=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Validation Confusion Matrix — {name}")
        fig.tight_layout(); fig.savefig(out_dir / f"val_confusion_matrix_{name}.png", dpi=150); plt.close(fig)

        return {
            "seg": seg_m, "clip_acc": clip_acc, "clip_f1": clip_f1m,
            "test": {"seg": seg_m_te, "clip_acc": clip_acc_te, "clip_f1": clip_f1m_te}
        }

    results = {}
    results["LightGBM"] = eval_model(clf_lgbm, "lgbm", X_va, y_va, clip_va, X_te, y_te, clip_te)
    results["XGB"] = eval_model(clf_xgb, "xgb", X_va, y_va, clip_va, X_te, y_te, clip_te)
    results["LogReg"] = eval_model(clf_log, "logreg", X_va, y_va, clip_va, X_te, y_te, clip_te)

    # Ensemble
    proba_va = (clf_lgbm.predict_proba(X_va) + clf_xgb.predict_proba(X_va) + clf_log.predict_proba(X_va)) / 3
    proba_te = (clf_lgbm.predict_proba(X_te) + clf_xgb.predict_proba(X_te) + clf_log.predict_proba(X_te)) / 3

    y_pred_va = proba_va.argmax(axis=1)
    seg_m = segment_metrics(y_va, y_pred_va)
    clip_acc, clip_f1m, _ = clip_level_from_probs(clip_va, y_va, proba_va)

    y_pred_te = proba_te.argmax(axis=1)
    seg_m_te = segment_metrics(y_te, y_pred_te)
    clip_acc_te, clip_f1m_te, _ = clip_level_from_probs(clip_te, y_te, proba_te)

    results["Ensemble"] = {
        "seg": seg_m, "clip_acc": clip_acc, "clip_f1": clip_f1m,
        "test": {"seg": seg_m_te, "clip_acc": clip_acc_te, "clip_f1": clip_f1m_te}
    }

    (out_dir / "metrics_all.json").write_text(json.dumps(results, indent=2))
    print("✅ Results saved to:", out_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
