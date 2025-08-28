#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Goal 2 (ESResNeXt-fbsp) — Head-only fine-tune on UrbanSound8K segments.
- 44.1 kHz mono
- window ~0.96 s, overlap 0.5
- Freeze backbone, train only a 10-class linear head
- Report segment & clip metrics (same format as Goal 1)
"""
import os, sys, time, json, subprocess, random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from urbansound_segment_task.src.common.paths import detect_dataset_layout
from urbansound_segment_task.src.common.metrics import (
    segment_metrics, per_class_report, confusion, clip_level_from_probs
)
from urbansound_segment_task.src.common.segmenter import load_mono_resample

# -------------------------
# Utils
# -------------------------
def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_esresnext_repo(repo_dir: Path):
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        print("⏬ Cloning ESResNeXt-fbsp repo...")
        code = subprocess.call(["git", "clone", "--depth", "1",
                                "https://github.com/AndreyGuzhov/ESResNeXt-fbsp.git",
                                str(repo_dir)])
        if code != 0:
            raise SystemExit("ESResNeXt-fbsp repo clone FAILED. Check internet/GitHub access.")
    # add to sys.path
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

def build_segments_index(meta: pd.DataFrame, audio_root: Path,
                         folds, sr=44100, win_sec=0.96, overlap=0.5):
    """Build flat index of (wav_path, classID, clip_id, start_sample)
       so that each dataset item is exactly one fixed-length segment."""
    rows = meta[meta["fold"].isin(folds)]
    win = int(round(win_sec * sr))
    hop = int(round(win * (1.0 - overlap)))
    index = []
    for _, r in rows.iterrows():
        fold = int(r["fold"]); fn = r["slice_file_name"]; cls = int(r["classID"])
        p = audio_root / f"fold{fold}" / fn
        if not p.exists():
            raise FileNotFoundError(f"WAV not found: {p}")
        # duration at file native sr → upscale to target sr approx
        dur = librosa.get_duration(filename=str(p))
        total = int(dur * sr)
        n_seg = (total - win) // hop + 1
        for i in range(max(0, n_seg)):
            start = i * hop
            index.append((str(p), cls, fn, start, win, sr))
    return index

class SegmentDataset(Dataset):
    def __init__(self, index_rows):
        self.rows = index_rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, cls, clip_id, start, win, sr = self.rows[i]
        # load full clip at target sr, then slice a fixed window
        y, _ = load_mono_resample(path, sr)
        # safety
        if start + win > len(y):
            pad = start + win - len(y)
            y = np.pad(y, (0, pad), mode="constant")
        seg = y[start:start+win]  # float32 [T]
        x = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0)  # [1, T]
        return x, int(cls), clip_id

# Generic adapter that learns mapping from base logits/features → 10 classes
class AdapterHead(nn.Module):
    def __init__(self, in_dim, n_classes=10):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, z):
        return self.linear(z)

def infer_base_out_dim(model, win_samples=42336, device="cuda"):
    """Run a dummy forward to detect base output dimension."""
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 1, win_samples, dtype=torch.float32, device=device)
        out = model(x)
        # some models return list/tuple
        if isinstance(out, (list, tuple)):
            out = out[0]
        # pool extra dims
        while out.dim() > 2:
            out = out.mean(dim=-1)
        return out.shape[1]

class BaseWrapper(nn.Module):
    """Wrap base model to always return [B, C] tensor (pooled if needed)."""
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        z = self.base(x)
        if isinstance(z, (list, tuple)):
            z = z[0]
        while z.dim() > 2:
            z = z.mean(dim=-1)
        return z

def freeze_all(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False

def evaluate(model, head, loader, device, class_ids, clip_ids):
    model.eval(); head.eval()
    ys, yhats, probs, clip_list = [], [], [], []
    with torch.no_grad():
        for xb, yb, cb in loader:
            xb = xb.to(device, non_blocking=True)
            z = model(xb)
            logits = head(z)
            pr = F.softmax(logits, dim=1).cpu().numpy()
            probs.append(pr)
            ys.append(yb.numpy())
            yhats.append(pr.argmax(axis=1))
            clip_list.extend(list(cb))
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(yhats)
    proba  = np.concatenate(probs, axis=0)
    seg_m  = segment_metrics(y_true, y_pred)
    cm     = confusion(y_true, y_pred)
    clip_acc, clip_f1m, _ = clip_level_from_probs(np.array(clip_list), y_true, proba)
    rep = per_class_report(y_true, y_pred)
    return seg_m, cm, rep, (clip_acc, clip_f1m), proba, y_true, np.array(clip_list)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="./urbansound_segment_task/goals/goal2_esresnext/results_headonly")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--win_sec", type=float, default=0.96)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pretrained_ckpt", type=str, default="", help="(optional) path to ESResNeXt-fbsp AudioSet checkpoint")
    args = ap.parse_args()

    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # dataset layout & meta
    audio_root, meta_csv = detect_dataset_layout(args.data_dir)
    if meta_csv is None or not Path(meta_csv).exists():
        raise FileNotFoundError("UrbanSound8K.csv not found in data_dir.")
    meta = pd.read_csv(meta_csv)

    train_folds = set(range(1,9))
    val_folds   = {9}
    test_folds  = {10}

    # build flat segment index (no leakage)
    print("🧱 Building segment indices...")
    idx_tr = build_segments_index(meta, audio_root, train_folds, sr=args.sr,
                                  win_sec=args.win_sec, overlap=args.overlap)
    idx_va = build_segments_index(meta, audio_root, val_folds,   sr=args.sr,
                                  win_sec=args.win_sec, overlap=args.overlap)
    idx_te = build_segments_index(meta, audio_root, test_folds,  sr=args.sr,
                                  win_sec=args.win_sec, overlap=args.overlap)

    ds_tr = SegmentDataset(idx_tr)
    ds_va = SegmentDataset(idx_va)
    ds_te = SegmentDataset(idx_te)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ESResNeXt-fbsp backbone
    repo_dir = Path("./external/ESResNeXt-fbsp")
    ensure_esresnext_repo(repo_dir)
    try:
        from models.esresnext_fbsp import ESResNeXtFBSP as ESX
    except Exception:
        try:
            from models.esresnext import ESResNeXtFBSP as ESX
        except Exception as e:
            raise ImportError("Could not import ESResNeXtFBSP from the external repo. "
                              "Open external/ESResNeXt-fbsp and check model file & class name.") from e

    base = ESX()  # create model (defaults from repo)
    if args.pretrained_ckpt and Path(args.pretrained_ckpt).exists():
        print(f"🔹 Loading pretrained weights: {args.pretrained_ckpt}")
        sd = torch.load(args.pretrained_ckpt, map_location="cpu")
        # accept both plain or {'state_dict': ...}
        state = sd.get("state_dict", sd)
        missing, unexpected = base.load_state_dict(state, strict=False)
        print("state_dict loaded (missing:", len(missing), "unexpected:", len(unexpected), ")")

    base.to(device)
    freeze_all(base)
    wrapped = BaseWrapper(base)

    # detect base output dim and create 10-class head
    win = int(round(args.win_sec * args.sr))
    base_out = infer_base_out_dim(wrapped, win_samples=win, device=device)
    head = AdapterHead(in_dim=base_out, n_classes=10).to(device)

    # optimizer/scheduler
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)
    best_val_f1 = -1.0; best_state = None
    t_train_start = time.time()

    print("🚂 Training (head-only)...")
    for epoch in range(1, args.epochs+1):
        head.train()
        running = 0.0; nseen = 0
        for xb, yb, _ in tqdm(dl_tr, desc=f"epoch {epoch}/{args.epochs}"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            z = wrapped(xb)
            logits = head(z)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0); nseen += xb.size(0)

        # validation
        seg_m_va, cm_va, rep_va, (clip_acc_va, clip_f1_va), proba_va, y_true_va, clip_ids_va = \
            evaluate(wrapped, head, dl_va, device, None, None)

        print(f"epoch {epoch}: train_loss={running/max(1,nseen):.4f} | "
              f"val seg_macroF1={seg_m_va['segment_macroF1']:.3f} clip_macroF1={clip_f1_va:.3f}")

        if seg_m_va["segment_macroF1"] > best_val_f1:
            best_val_f1 = seg_m_va["segment_macroF1"]
            best_state = {
                "epoch": epoch,
                "head": head.state_dict(),
                "best_val_seg_macroF1": best_val_f1,
            }

    t_train = time.time() - t_train_start
    if best_state is not None:
        torch.save(best_state, out_dir/"head_only_best.pt")

    # final eval on VAL (with best head if saved)
    if (out_dir/"head_only_best.pt").exists():
        chk = torch.load(out_dir/"head_only_best.pt", map_location="cpu")
        head.load_state_dict(chk["head"])

    t0 = time.time()
    seg_m_va, cm_va, rep_va, (clip_acc_va, clip_f1_va), proba_va, y_true_va, clip_ids_va = \
        evaluate(wrapped, head, dl_va, device, None, None)
    val_infer_s = time.time() - t0

    # test
    seg_m_te, cm_te, rep_te, (clip_acc_te, clip_f1_te), proba_te, y_true_te, clip_ids_te = \
        evaluate(wrapped, head, dl_te, device, None, None)

    # save reports (VAL focus)
    (out_dir/"metrics_val.json").write_text(json.dumps({
        **seg_m_va, "clip_accuracy": clip_acc_va, "clip_macroF1": clip_f1_va,
        "train_seconds": t_train, "val_infer_seconds": val_infer_s
    }, indent=2))

    (out_dir/"metrics_test.json").write_text(json.dumps({
        **seg_m_te, "clip_accuracy": clip_acc_te, "clip_macroF1": clip_f1_te
    }, indent=2))

    # per-class report (VAL)
    pd.DataFrame(rep_va).to_csv(out_dir/"val_classification_report.csv", index=True)

    # confusion matrix (VAL)
    fig, ax = plt.subplots(figsize=(6,6))
    ConfusionMatrixDisplay(cm_va).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Validation Confusion Matrix (segment-level) — ESResNeXt head-only")
    fig.tight_layout(); fig.savefig(out_dir/"val_confusion_matrix.png", dpi=150); plt.close(fig)

    print("✅ VAL:", seg_m_va, {"clip_accuracy": clip_acc_va, "clip_macroF1": clip_f1_va})
    print("✅ TEST:", seg_m_te, {"clip_accuracy": clip_acc_te, "clip_macroF1": clip_f1_te})
    print("📁 Saved to:", out_dir)

if __name__ == "__main__":
    main()
