# UrbanSound8K Segment-Based Classification
https://colab.research.google.com/drive/1-lNzYTJY6OsZ5vqcBOOi106G2mldRUqv
This repository contains three different approaches for segment-based audio classification on the **UrbanSound8K** dataset:

1. **GOAL 1** – YAMNet Embeddings → LightGBM (Baseline)
2. **GOAL 2** – ESResNeXt-fbsp Fine-tuning  
3. **GOAL 3** – AudioCLIP Fine-tuning  

Accuracy and macro-F1 scores are reported at both segment-level and clip-level for each goal.

## Installation

### Requirements
- Python 3.9+
- GPU recommended (NVIDIA A100/V100 ideal, required for Goals 2 and 3)
- ~8GB RAM minimum

### Environment Setup
```bash
# Extract from zip file
unzip urbansound_segment_classification.zip
cd urbansound_segment_classification
pip install -r requirements.txt
```

### Dataset Setup
```bash
mkdir data
# UrbanSound8K dataset (fold1..fold10 folders + UrbanSound8K.csv) should be placed here
```

**UrbanSound8K Download:** [Kaggle link](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)

**Expected structure:**
```
data/
├── fold1/
├── fold2/
├── ...
├── fold10/
└── UrbanSound8K.csv
```

---

## GOAL 1 — YAMNet + LightGBM (Baseline)

### Working Principle
Uses YAMNet as a **fixed feature extractor**. Extracts 1024-D embeddings for each segment, then performs classification with LightGBM on these features. Not transfer learning, just feature extraction + tabular learning.

### Description
- Each audio recording is segmented into **0.96s windows** with **50% overlap**
- **YAMNet (1024-D) embeddings** are extracted per segment
- **LightGBM** based classifier predicts segment labels
- Clip-level scores are calculated by **averaging** segment probabilities
- **Fold-based split:** Train (Fold 1-8), Val (Fold 9), Test (Fold 10)

### Running
```bash
python -m urbansound_segment_task.goals.goal1_yamnet_lgbm.run \
  --data_dir ./data \
  --out ./urbansound_segment_task/goals/goal1_yamnet_lgbm/results \
  --win_sec 0.96 \
  --overlap 0.5 \
  --seed 42
```

### Results
**Source:** `results/metrics_test.json`
- **Segment-level:** accuracy = 0.744, macro-F1 = 0.758
- **Clip-level:** accuracy = 0.802, macro-F1 = 0.819
- **Inference speed:** ~220 segments/second*
- **Training time:** ~15 minutes (GPU)

**Note:** Advanced V5 version (LightGBM + MLP ensemble) is available for further research and achieves 0.828 clip F1 score. (`urbansound_segment_task/goals/goal1_yamnet_lgbm/results_v5`)

### Output Files
```
urbansound_segment_task/goals/goal1_yamnet_lgbm/results/
├── metrics_val.json              # Validation metrics
├── metrics_test.json             # Test metrics  
├── val_classification_report.csv # Class-wise details
└── val_confusion_matrix.png      # Confusion matrix
```

---

## GOAL 2 — ESResNeXt-fbsp Fine-tuning

### Working Principle
Adapts AudioSet pretrained ESResNeXt model to UrbanSound8K via **domain adaptation**. Performs transfer learning by fully fine-tuning the backbone when GPU is available. True end-to-end deep learning approach.

### Description
- **ESResNeXt-fbsp** model is used (frequency band-based attention mechanism)
- Uses pre-trained weights from **AudioSet**
- Fully fine-tunes the backbone when GPU is available

### Model and Checkpoint Setup

#### Download ESResNeXt Repository
```bash
git clone --depth 1 https://github.com/AndreyGuzhov/ESResNeXt-fbsp.git ./external/ESResNeXt_fbsp
```

#### Download Pretrained Checkpoint
```bash
bash urbansound_segment_task/scripts/download_checkpoint_goal2.sh
```

This script automatically:
- Downloads `ESResNeXtFBSP_AudioSet.pt` checkpoint
- Places it in `./urbansound_segment_task/goals/goal2_esresnext/checkpoints/` folder

### Running

#### Full Fine-tuning (GPU required) - Recommended
```bash
python -m urbansound_segment_task.goals.goal2_esresnext.run_finetune \
  --data_dir ./data \
  --out ./urbansound_segment_task/goals/goal2_esresnext/results_finetune_v2 \
  --sr 44100 --win_sec 0.96 --overlap 0.5 \
  --batch_size 128 \
  --warmup_epochs 4 \
  --finetune_epochs 32 \
  --lr_head 1e-3 \
  --lr_backbone 5e-5 \
  --weight_decay 1e-4 \
  --label_smoothing 0.05 \
  --pretrained_ckpt ./urbansound_segment_task/goals/goal2_esresnext/checkpoints/ESResNeXtFBSP_AudioSet.pt \
  --tta_shifts 2 \
  --seed 42
```

**Path Explanations:**
- `--data_dir ./data`: UrbanSound8K dataset location
- `--out ./urbansound_segment_task/goals/goal2_esresnext/results_finetune_v2`: Output folder for results
- `--pretrained_ckpt ./urbansound_segment_task/goals/goal2_esresnext/checkpoints/ESResNeXtFBSP_AudioSet.pt`: AudioSet pretrained model

### Results
**Source:** `results_finetune_v2/metrics_test.json`, `results_finetune_v2/metrics_val.json`

**Test Performance:**
- **Segment-level:** accuracy = 0.768, macro-F1 = 0.740
- **Clip-level:** accuracy = **0.841**, macro-F1 = **0.836**

**Training Details:**
- **Training time:** 2629.8 seconds (~44 minutes)
- **Validation inference:** 18.6 seconds
- **Hyperparameters:** batch_size=128, lr_head=1e-3, lr_backbone=5e-5, epochs=32
- **Hardware:** NVIDIA A100 40GB GPU
- **Highest clip-level performance** among all methods

---

## GOAL 3 — AudioCLIP Fine-tuning

### Working Principle
Adapts AudioCLIP to UrbanSound8K via **domain adaptation**. Performs transfer learning using only the audio branch of the multi-modal (audio+text) pretrained model. results_ft_q → fine-tuning quality results (best hyperparameter combination).

### Description
- **AudioCLIP** multi-modal model is used (audio branch only)
- ~134M trainable parameters
- Fully fine-tunes the model if GPU is available, otherwise only the classifier head

***If AudioCLIP is not in external/ folder, run this command:***
```
git clone --depth 1 https://github.com/AndreyGuzhov/AudioCLIP.git ./external/AudioCLIP  
```

### Running
```bash
# Full fine-tuning (Best results)
python -m urbansound_segment_task.goals.goal3_audioclip.run \
  --data_dir ./data \
  --out ./urbansound_segment_task/goals/goal3_audioclip/results_ft_q \
  --mode finetune \
  --epochs 40 \
  --warmup_epochs 6 \
  --bs 64 \
  --accum_steps 2 \
  --lr_backbone 2e-5 \
  --lr_head 3e-4 \
  --label_smoothing 0.10 \
  --early_stop 8 \
  --bn_eval 1 \
  --head_warmup_epochs 3 \
  --clip_agg logit_mean \
  --tta_shifts 2 \
  --aug_gain_db 6.0 \
  --aug_shift_frac 0.10 \
  --aug_noise_snr_low 10 \
  --aug_noise_snr_high 25 \
  --seed 42 \
  --emb_dim 1024
```

### Results
**Source:** `results_ft_q/metrics_test.json`, `results_ft_q/metrics_throughput.json`, `results_ft_q/run_args.json`

**Test Performance:**
- **Segment-level:** accuracy = 0.769, macro-F1 = 0.755
- **Clip-level:** accuracy = 0.798, macro-F1 = 0.798
- **Inference speed:** ~464 segments/second* (**fastest**)

**Training Details:**
- **Best epoch:** 5
- **Validation inference:** 11.1 seconds (5134 segments)
- **Hyperparameters:** batch_size=64, lr_head=3e-4, lr_backbone=2e-5, epochs=40
- **Hardware:** NVIDIA A100 40GB GPU

---

## Comprehensive Comparison

**All metrics are calculated on the test set. Source files are available in each goal's results folders.**

| Metric | Goal 1 (YAMNet+LGB) | Goal 2 (ESResNeXt) | Goal 3 (AudioCLIP) |
|--------|---------------------|---------------------|---------------------|
| **Test Segment Accuracy** | 74.4% | **76.8%** | 76.9% |
| **Test Segment macro-F1** | **75.8%** | 74.0% | 75.5% |
| **Test Clip Accuracy** | 80.2% | **84.1%** | 79.8% |
| **Test Clip macro-F1** | 81.9% | **83.6%** | 79.8% |
| **Inference Speed*** | ~220 seg/s | - | **~464 seg/s** |
| **Training Time** | ~15 min | ~44 min | ~60-90 min |
| **Trainable Params** | ~134K | ~25M | ~134M |
| **GPU Requirement** | No** | Yes | Yes |
| **Source File** | `results/` | `results_finetune_v2/` | `results_ft_q/` |

*Inference speed: Measured on validation set with NVIDIA A100 GPU  
**Works without GPU but takes longer

### Key Insights
1. **Best Accuracy:** Goal 2 (ESResNeXt) - 84.1% clip accuracy
2. **Best Speed/Accuracy Balance:** Goal 1 (YAMNet+LGB) - Baseline
3. **Fastest Inference:** Goal 3 (AudioCLIP) - 2x faster
4. **Practical Choice:** Goal 1 is sufficient and efficient for most applications

## Technical Details

### Data Splits
- **Training:** Fold 1-8
- **Validation:** Fold 9
- **Test:** Fold 10
- **Leak prevention:** Segments from the same clip never go to different sets

### Segmentation
- **Window length:** 0.96s (YAMNet compatible)
- **Overlap:** 50%
- **Sample rate:** 16kHz mono

### Reproducibility
- Fixed seed (42)
- Deterministic data splits  
- Pinned package versions
- Fold-based split (official UrbanSound8K protocol)

### Hardware
**Test Environment:** Tested with NVIDIA A100 40GB GPU. Also works on CPU but takes longer.
