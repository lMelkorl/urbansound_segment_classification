#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from urbansound_segment_task.src.common.paths import detect_dataset_layout
from urbansound_segment_task.src.common.segmenter import load_mono_16k, iter_segments
from urbansound_segment_task.src.common.metrics import (
    segment_metrics, per_class_report, confusion, clip_level_from_probs
)

def wav_path(audio_root: Path, fold: int, fname: str) -> Path:
    p = audio_root / f"fold{fold}" / fname
    if not p.exists():
        raise FileNotFoundError(f"WAV bulunamadı: {p}")
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="UrbanSound8K kökü (örn. ./data)")
    ap.add_argument("--out", default="./urbansound_segment_task/goals/1_yamnet_lgbm/results")
    ap.add_argument("--win_sec", type=float, default=0.96)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    audio_root, meta_csv = detect_dataset_layout(args.data_dir)
    if meta_csv is None or not Path(meta_csv).exists():
        raise FileNotFoundError(
            "UrbanSound8K.csv bulunamadı. `./data` içine koymalısın.\n"
            "Kaggle tek dosya indirme: \n"
            "  kaggle datasets download -d chrisfilo/urbansound8k -f UrbanSound8K.csv -p ./data --force\n"
            "  unzip -o ./data/UrbanSound8K.csv.zip -d ./data && rm ./data/UrbanSound8K.csv.zip"
        )
    meta = pd.read_csv(meta_csv)

    # deterministik split: train=1..8, val=9, test=10
    train_folds = set(range(1,9))
    val_folds   = {9}
    test_folds  = {10}

    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    def emb(segment_np: np.ndarray) -> np.ndarray:
        wf = tf.convert_to_tensor(segment_np, dtype=tf.float32)
        scores, embeddings, spectrogram = yamnet(wf)
        return tf.reduce_mean(embeddings, axis=0).numpy()  # [1024]

    def build_split(chosen_folds):
        X, y, clip_ids = [], [], []
        rows = meta[meta["fold"].isin(chosen_folds)]
        for _, r in tqdm(rows.iterrows(), total=len(rows), desc=f"folds {sorted(chosen_folds)}"):
            fold = int(r["fold"]); fn = r["slice_file_name"]; cls = int(r["classID"])
            y_wav, sr = load_mono_16k(str(wav_path(audio_root, fold, fn)), target_sr=16000)
            for seg in iter_segments(y_wav, sr, win_sec=args.win_sec, overlap=args.overlap):
                X.append(emb(seg)); y.append(cls); clip_ids.append(fn)
        return np.stack(X, axis=0), np.array(y), np.array(clip_ids)

    # Embedding çıkar
    t0=time.time(); X_tr, y_tr, clip_tr = build_split(train_folds); t_train_emb=time.time()-t0
    t0=time.time(); X_va, y_va, clip_va = build_split(val_folds);   t_val_emb  =time.time()-t0
    t0=time.time(); X_te, y_te, clip_te = build_split(test_folds);  t_test_emb =time.time()-t0

    # Sınıf ağırlıkları (dengesizliğe karşı)
    classes = np.unique(y_tr)
    cw = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

    # LightGBM
    clf = LGBMClassifier(
        n_estimators=700, num_leaves=64, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
        random_state=args.seed, class_weight=class_weight_dict
    )
    t0=time.time(); clf.fit(X_tr, y_tr); lgbm_train_s=time.time()-t0

    # Val — segment seviyesi
    t0=time.time(); proba_va = clf.predict_proba(X_va); seg_infer_s=time.time()-t0
    y_pred_va = proba_va.argmax(axis=1)
    seg_m = segment_metrics(y_va, y_pred_va)
    rep   = per_class_report(y_va, y_pred_va)
    cm    = confusion(y_va, y_pred_va)

    # Val — klip seviyesi (ortalama)
    clip_acc, clip_f1m, _ = clip_level_from_probs(clip_va, y_va, proba_va)

    # Test (rapor için)
    proba_te = clf.predict_proba(X_te)
    y_pred_te = proba_te.argmax(axis=1)
    seg_m_te = segment_metrics(y_te, y_pred_te)
    clip_acc_te, clip_f1m_te, _ = clip_level_from_probs(clip_te, y_te, proba_te)

    # Kaydet
    (out_dir/"metrics_val.json").write_text(json.dumps({
        **seg_m, "clip_accuracy": clip_acc, "clip_macroF1": clip_f1m,
        "train_embedding_seconds": t_train_emb,
        "val_embedding_seconds": t_val_emb,
        "test_embedding_seconds": t_test_emb,
        "lgbm_train_seconds": lgbm_train_s,
        "val_segment_infer_seconds": seg_infer_s
    }, indent=2))

    (out_dir/"metrics_test.json").write_text(json.dumps({
        **seg_m_te, "clip_accuracy": clip_acc_te, "clip_macroF1": clip_f1m_te
    }, indent=2))

    pd.DataFrame(rep).to_csv(out_dir/"val_classification_report.csv", index=True)

    # Confusion matrix görseli
    fig, ax = plt.subplots(figsize=(6,6))
    ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Validation Confusion Matrix (segment-level)")
    fig.tight_layout(); fig.savefig(out_dir/"val_confusion_matrix.png", dpi=150); plt.close(fig)

    print("✅ VAL:", seg_m, {"clip_accuracy": clip_acc, "clip_macroF1": clip_f1m})
    print("✅ TEST:", seg_m_te, {"clip_accuracy": clip_acc_te, "clip_macroF1": clip_f1m_te})
    print("📁 Çıktılar:", out_dir)

if __name__ == "__main__":
    main()
