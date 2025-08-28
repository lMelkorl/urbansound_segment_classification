#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Goal 2 (ESResNeXt-fbsp) — Head-warmup + Full Fine-tune on UrbanSound8K segments.
- 44.1 kHz mono
- window ~0.96 s, overlap 0.5
- Phase 1: freeze backbone, train only 10-class head (warmup)
- Phase 2: unfreeze backbone, two LR param groups (backbone low LR)
- Report segment & clip metrics (same format as Goal 1)
- Improvements: strong waveform augs, WeightedRandomSampler, CosineAnnealingLR, torch.amp, optional TTA
"""
import os, sys, time, json, subprocess, random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

from urbansound_segment_task.src.common.paths import detect_dataset_layout
from urbansound_segment_task.src.common.metrics import (
    segment_metrics, per_class_report, confusion, clip_level_from_probs
)
from urbansound_segment_task.src.common.segmenter import load_mono_44k  # mevcutta var

# -------------------------
# Utils
# -------------------------
def set_seeds(seed=42, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_esresnext_repo(repo_dir: Path):
    """
    ESResNeXt-fbsp kodlarını klonla, LFS checkpoint indirmeden kullan.
    """
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        print("⏬ Cloning ESResNeXt-fbsp repo (skip LFS smudge)...")
        code = subprocess.call([
            "env", "GIT_LFS_SKIP_SMUDGE=1",
            "git", "clone", "--depth", "1",
            "https://github.com/AndreyGuzhov/ESResNeXt-fbsp.git",
            str(repo_dir)
        ])
        if code != 0:
            raise SystemExit("❌ ESResNeXt-fbsp repo clone FAILED.")

    # 🔑 sys.path'e repo kökünü ekle
    repo_root = str(repo_dir.resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def build_segments_index(meta: pd.DataFrame, audio_root: Path,
                         folds, sr=44100, win_sec=0.96, overlap=0.5):
    """Create flat index: one item per fixed-length segment."""
    rows = meta[meta["fold"].isin(folds)]
    win = int(round(win_sec * sr))
    hop = int(round(win * (1.0 - overlap)))
    index = []
    for _, r in rows.iterrows():
        fold = int(r["fold"]); fn = r["slice_file_name"]; cls = int(r["classID"])
        p = audio_root / f"fold{fold}" / fn
        if not p.exists():
            raise FileNotFoundError(f"WAV not found: {p}")
        # hızlı: sadece süreyi al
        dur = librosa.get_duration(path=str(p))  # (filename -> path) FutureWarning fix
        total = int(dur * sr)
        if total < win:
            # tek pad'li segment
            index.append((str(p), cls, fn, 0, win, sr))
            continue
        n_seg = (total - win) // hop + 1
        for i in range(n_seg):
            start = i * hop
            index.append((str(p), cls, fn, start, win, sr))
    return index

class SegmentDataset(Dataset):
    def __init__(self, index_rows, augment=False):
        self.rows = index_rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def _augment_wave(self, y, sr, win):
        # 1) Random gain
        if np.random.rand() < 0.7:
            gain_db = np.random.uniform(-6, 6)
            y = y * (10 ** (gain_db / 20.0))

        # 2) Additive gaussian noise
        if np.random.rand() < 0.5:
            noise_level = np.random.uniform(0.001, 0.01)
            y = y + np.random.randn(*y.shape).astype(np.float32) * noise_level

        # 3) Small time shift (±2%)
        if np.random.rand() < 0.7:
            max_shift = int(0.02 * len(y))
            if max_shift > 0:
                shift = np.random.randint(-max_shift, max_shift + 1)
                y = np.roll(y, shift)

        # 4) Pitch shift (±2 semitone)
        if np.random.rand() < 0.5:
            steps = np.random.uniform(-2.0, 2.0)
            y = librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=steps)

        # 5) Time-stretch (0.9–1.1) + pad/crop
        if np.random.rand() < 0.5:
            rate = np.random.uniform(0.9, 1.1)
            y = librosa.effects.time_stretch(y.astype(np.float32), rate=rate)
            if len(y) >= win:
                y = y[:win]
            else:
                y = np.pad(y, (0, win - len(y)), mode="constant")

        return y

    def __getitem__(self, i):
        path, cls, clip_id, start, win, sr = self.rows[i]
        y, _ = load_mono_44k(path, sr)
        if start + win > len(y):
            pad = start + win - len(y)
            y = np.pad(y, (0, pad), mode="constant")
        seg = y[start:start+win]
        if self.augment:
            seg = self._augment_wave(seg, sr, win)
        x = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0)  # [1, T]
        return x, int(cls), clip_id

def freeze_all(m: nn.Module, requires_grad=False):
    for p in m.parameters():
        p.requires_grad = requires_grad

class BaseWrapper(nn.Module):
    """Always return [B, C] pooled representation."""
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

class AdapterHead(nn.Module):
    def __init__(self, in_dim, n_classes=10):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, z):
        return self.linear(z)

@torch.no_grad()
def infer_base_out_dim(model, win_samples=42336, device="cuda"):
    model.eval()
    x = torch.zeros(1, 1, win_samples, dtype=torch.float32, device=device)
    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    while out.dim() > 2:
        out = out.mean(dim=-1)
    return out.shape[1]

def _shift_right(x, pct=0.01):
    # x: [B,1,T]
    if pct <= 0: return x
    T = x.shape[-1]; s = int(T * pct)
    if s <= 0: return x
    return torch.roll(x, shifts=s, dims=-1)

@torch.no_grad()
def evaluate(model, head, loader, device, tta_shifts=0):
    model.eval(); head.eval()
    ys, yhats, probs, clip_list = [], [], [], []
    for xb, yb, cb in tqdm(loader, desc="eval"):
        xb = xb.to(device, non_blocking=True)
        # TTA: original + n small shifts
        pred_list = []
        with autocast(enabled=(device=="cuda")):
            # original
            z = model(xb)
            logits = head(z)
            pred_list.append(F.softmax(logits, dim=1).cpu().numpy())
            # shifts
            for t in range(1, tta_shifts+1):
                x_in = _shift_right(xb, pct=0.005 * t)
                z = model(x_in)
                logits = head(z)
                pred_list.append(F.softmax(logits, dim=1).cpu().numpy())
        pr = np.mean(pred_list, axis=0)

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
    return seg_m, cm, rep, (clip_acc, clip_f1m)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="./urbansound_segment_task/goals/goal2_esresnext/results_finetune")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--win_sec", type=float, default=0.96)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--warmup_epochs", type=int, default=4)   # artırıldı
    ap.add_argument("--finetune_epochs", type=int, default=32) # artırıldı
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=5e-5)  # artırıldı
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)  # eklendi
    ap.add_argument("--pretrained_ckpt", type=str, default="", help="optional ESResNeXt-fbsp AudioSet checkpoint")
    ap.add_argument("--tta_shifts", type=int, default=2, help="TTA: number of additional right shifts (0=disable)")
    args = ap.parse_args()

    set_seeds(args.seed, deterministic=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # dataset
    audio_root, meta_csv = detect_dataset_layout(args.data_dir)
    if meta_csv is None or not Path(meta_csv).exists():
        raise FileNotFoundError("UrbanSound8K.csv not found in data_dir.")
    meta = pd.read_csv(meta_csv)

    train_folds = set(range(1,9))
    val_folds   = {9}
    test_folds  = {10}

    print("🧱 Building segment indices...")
    idx_tr = build_segments_index(meta, audio_root, train_folds, sr=args.sr,
                                  win_sec=args.win_sec, overlap=args.overlap)
    idx_va = build_segments_index(meta, audio_root, val_folds,   sr=args.sr,
                                  win_sec=args.win_sec, overlap=args.overlap)
    idx_te = build_segments_index(meta, audio_root, test_folds,  sr=args.sr,
                                  win_sec=args.win_sec, overlap=args.overlap)

    # class weights (train) for CE
    y_tr = np.array([row[1] for row in idx_tr])
    classes = np.unique(y_tr)
    cw = compute_class_weight("balanced", classes=classes, y=y_tr)
    ce_weight = torch.tensor(cw, dtype=torch.float32, device=device)

    ds_tr = SegmentDataset(idx_tr, augment=True)
    ds_va = SegmentDataset(idx_va, augment=False)
    ds_te = SegmentDataset(idx_te, augment=False)

    # WeightedRandomSampler (class-balanced)
    cls_list = [row[1] for row in idx_tr]
    unique, counts = np.unique(cls_list, return_counts=True)
    freq = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.array([1.0 / freq[c] for c in cls_list], dtype=np.float32)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler,
                       num_workers=4, pin_memory=True, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=4, pin_memory=True, persistent_workers=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=4, pin_memory=True, persistent_workers=True)

    repo_dir = Path("./external/ESResNeXt_fbsp").resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    try:
        from model.esresnet_fbsp import ESResNeXtFBSP as ESX
    except ImportError:
        from model.esresnext import ESResNeXtFBSP as ESX
        
    base = ESX()
    if args.pretrained_ckpt and Path(args.pretrained_ckpt).exists():
        print(f"🔹 Loading pretrained: {args.pretrained_ckpt}")
        sd = torch.load(args.pretrained_ckpt, map_location="cpu")
        state = sd.get("state_dict", sd)

        new_state = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            # ❌ Classifier ve fbsp ile ilgili layer'ları atla
            if any(x in nk for x in ["fc", "window", "fbsp"]):
                continue
            new_state[nk] = v

        missing, unexpected = base.load_state_dict(new_state, strict=False)
        print("✅ state_dict loaded. Skipped fc/window/fbsp layers.")
        print("   missing:", len(missing), "unexpected:", len(unexpected))

    base.to(device)
    wrapped = BaseWrapper(base)

    # output dim
    win = int(round(args.win_sec * args.sr))
    base_out = infer_base_out_dim(wrapped, win_samples=win, device=device)
    head = AdapterHead(in_dim=base_out, n_classes=10).to(device)

    scaler = GradScaler(enabled=(device=="cuda"))

    def run_epoch_headonly(dloader, opt, phase_name="warmup"):
        base.eval()   # backbone frozen
        head.train()
        total_loss=0.0; nseen=0
        for xb, yb, _ in tqdm(dloader, desc=phase_name):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device=="cuda")):
                z = wrapped(xb)
                logits = head(z)
                loss = F.cross_entropy(logits, yb, weight=ce_weight, label_smoothing=args.label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            total_loss += loss.item() * xb.size(0); nseen += xb.size(0)
        return total_loss/max(1,nseen)

    # -------- Phase 1: head-only warmup --------
    freeze_all(base, requires_grad=False)
    opt_head = torch.optim.AdamW(head.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)

    best_val = -1.0
    best_chk = None
    t0 = time.time()
    for ep in range(1, args.warmup_epochs+1):
        tl = run_epoch_headonly(dl_tr, opt=opt_head, phase_name=f"warmup ep{ep}")
        seg_m_va, cm_va, rep_va, (clip_acc_va, clip_f1_va) = evaluate(wrapped, head, dl_va, device, tta_shifts=0)
        print(f"[WARMUP {ep}] train_loss={tl:.4f} | val segF1={seg_m_va['segment_macroF1']:.3f} clipF1={clip_f1_va:.3f}")
        if seg_m_va["segment_macroF1"] > best_val:
            best_val = seg_m_va["segment_macroF1"]
            best_chk = {"phase":"warmup","head":head.state_dict()}

    if best_chk is not None:
        head.load_state_dict(best_chk["head"])

    # -------- Phase 2: full fine-tune --------
    freeze_all(base, requires_grad=True)  # unfreeze
    opt_all = torch.optim.AdamW([
        {"params": base.parameters(), "lr": args.lr_backbone},
        {"params": head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    # Cosine annealing — iterasyon bazlı
    steps_per_epoch = len(dl_tr)
    T_max = max(1, args.finetune_epochs * steps_per_epoch)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(opt_all, T_max=T_max, eta_min=1e-6)

    patience = 5
    no_improve = 0
    for ep in range(1, args.finetune_epochs+1):
        base.train(); head.train()
        total_loss=0.0; nseen=0
        for xb, yb, _ in tqdm(dl_tr, desc=f"finetune ep{ep}"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt_all.zero_grad(set_to_none=True)
            with autocast(enabled=(device=="cuda")):
                z = wrapped(xb)
                logits = head(z)
                loss = F.cross_entropy(logits, yb, weight=ce_weight, label_smoothing=args.label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(opt_all); scaler.update()
            scheduler.step()  # iterasyon bazlı adım
            total_loss += loss.item() * xb.size(0); nseen += xb.size(0)

        seg_m_va, cm_va, rep_va, (clip_acc_va, clip_f1_va) = evaluate(wrapped, head, dl_va, device, tta_shifts=args.tta_shifts)
        print(f"[FT {ep}] train_loss={total_loss/max(1,nseen):.4f} | val segF1={seg_m_va['segment_macroF1']:.3f} clipF1={clip_f1_va:.3f}")

        if seg_m_va["segment_macroF1"] > best_val:
            best_val = seg_m_va["segment_macroF1"]; no_improve = 0
            torch.save({"head": head.state_dict(), "base": base.state_dict(), "epoch": ep}, out_dir/"best_finetune.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("⏹ Early stopping.")
                break

    train_seconds = time.time() - t0

    # Load best & final eval
    if (out_dir/"best_finetune.pt").exists():
        chk = torch.load(out_dir/"best_finetune.pt", map_location="cpu")
        head.load_state_dict(chk["head"]); base.load_state_dict(chk["base"])

    t_infer0 = time.time()
    seg_m_va, cm_va, rep_va, (clip_acc_va, clip_f1_va) = evaluate(wrapped, head, dl_va, device, tta_shifts=args.tta_shifts)
    val_infer_seconds = time.time() - t_infer0

    seg_m_te, cm_te, rep_te, (clip_acc_te, clip_f1_te) = evaluate(wrapped, head, dl_te, device, tta_shifts=args.tta_shifts)

    # Save reports
    (out_dir/"metrics_val.json").write_text(json.dumps({
        **seg_m_va, "clip_accuracy": clip_acc_va, "clip_macroF1": clip_f1_va,
        "train_seconds": train_seconds, "val_infer_seconds": val_infer_seconds
    }, indent=2))

    (out_dir/"metrics_test.json").write_text(json.dumps({
        **seg_m_te, "clip_accuracy": clip_acc_te, "clip_macroF1": clip_f1_te
    }, indent=2))

    pd.DataFrame(rep_va).to_csv(out_dir/"val_classification_report.csv", index=True)

    fig, ax = plt.subplots(figsize=(6,6))
    ConfusionMatrixDisplay(cm_va).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Validation Confusion Matrix (segment-level) — ESResNeXt fine-tune")
    fig.tight_layout(); fig.savefig(out_dir/"val_confusion_matrix.png", dpi=150); plt.close(fig)

    print("✅ VAL:", seg_m_va, {"clip_accuracy": clip_acc_va, "clip_macroF1": clip_f1_va})
    print("✅ TEST:", seg_m_te, {"clip_accuracy": clip_acc_te, "clip_macroF1": clip_f1_te})
    print("📁 Saved to:", out_dir)

if __name__ == "__main__":
    main()
