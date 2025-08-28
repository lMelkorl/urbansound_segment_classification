# UrbanSound8K Segment-Based Classification

Bu repo, UrbanSound8K dataseti üzerinde segment tabanlı sınıflandırmayı içerir:

**Goal 1:** YAMNet → Embeddings → LightGBM

**Goal 2:** Fine-tune ESResNeXt-fbsp

## 1. Kurulum

### Gereksinimler

- Python 3.9+
- GPU (tavsiye edilir, Goal 2 için gerekli)

### Paketler:

```bash
pip install -r requirements.txt
or
!pip install -r ./urbansound_segment_task/requirements.txt
```

(örnek requirements.txt içeriği: tensorflow, tensorflow_hub, lightgbm, xgboost, scikit-learn, tqdm, pandas, matplotlib, torch, torchaudio, pytorch-ignite vb.)


```
.
├── data
│   ├── fold1
│   ├── fold10
│   ├── ...
│   └── fold9
├── external
│   └── ESResNeXt_fbsp
│       ├── ...
└── urbansound_segment_task
    ├── configs
    ├── goals
    │   ├── goal1_yamnet_lgbm
    │   ├── goal2_esresnext
    │   ├── goal3_audioclip
    │   └── __pycache__
    ├── results
    ├── scripts
    └── src
        ├── common
        └── __pycache__
```

## 2. Dataset Kurulumu

### Dataset yapısı:

```
data/
 ├── fold1/
 ├── fold2/
 ├── ...
 ├── fold10/
 └── UrbanSound8K.csv   # metadata dosyası
```

### UrbanSound8K indir:

👉 [Kaggle linki](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)

eğer google colab kullanıyorsanız:
```
!chmod 600 /content/kaggle.json
!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d chrisfilo/urbansound8k -p ./data --unzip

```


İndirdikten sonra `fold*` klasörlerini ve `metadata/UrbanSound8K.csv` dosyasını `data/` altına yerleştir.

### Beklenen yapı:

```
data/
 ├── fold1/
 ├── fold2/
 ...
 ├── fold10/
 └── UrbanSound8K.csv
```

## 3. Goal 1 — YAMNet → Embeddings → LightGBM

### Çalıştırma

```bash
python -m urbansound_segment_task.goals.goal1_yamnet_lgbm.run \
  --data_dir ./data \
  --out ./urbansound_segment_task/goals/goal1_yamnet_lgbm/results \
  --win_sec 0.96 \
  --overlap 0.5 \
  --seed 42
```

### Açıklama

- **Segment uzunluğu:** 0.96 s (YAMNet ile uyumlu)
- **Overlap:** 50%
- GPU üzerinde YAMNet ile embedding çıkarılır, Segment embedding’leri üzerinde tek bir LightGBM modeli eğitilir.
- 
### Sonuçlar:

- Segment-level ve clip-level metrikler (accuracy, macro-F1)
- Per-class rapor (CSV)
- Confusion matrix (PNG)

### Sonuç dosyaları:

```
urbansound_segment_task/goals/goal1_yamnet_lgbm/results/
 ├── metrics_val_lgbm.json
 ├── metrics_test_lgbm.json
 ├── val_classification_report_lgbm.csv
 ├── val_confusion_matrix_lgbm.png
 ├── metrics_all.json
 ...
```

## 4. Goal 2 — Fine-tune ESResNeXt-fbsp

### Checkpoint Kurulumu

Pretrained model (AudioSet):

```bash
bash scripts/download_checkpoints.sh
```

(Bu script, `ESResNeXtFBSP_AudioSet.pt` checkpoint'ini indirip
`urbansound_segment_task/goals/goal2_esresnext/checkpoints/` altına koyar.)

### Çalıştırma

```bash
!python -m urbansound_segment_task.goals.goal2_esresnext.run_finetune \
  --data_dir ./data \
  --out ./urbansound_segment_task/goals/goal2_esresnext/results_finetune \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --freeze_backbone 0   # GPU yoksa 1 yap, sadece son katman eğitilir
```

### Açıklama

- Segment tabanlı eğitim yapılır.
- GPU varsa backbone fine-tune edilir; yoksa sadece son katman eğitilir.
- Sonuçlar yine `results_finetune/` altında JSON, CSV ve PNG dosyaları olarak kaydedilir.

## 5. Çıktılar ve Raporlama

Her goal için kaydedilenler:

- `metrics_val.json` / `metrics_test.json` → segment ve clip-level accuracy, macro-F1
- `classification_report.csv` → her sınıf için precision/recall/F1
- `confusion_matrix.png` → sınıf bazlı hata görselleştirmesi

## 6. Notlar

- **Compute units:** Colab kullanıyorsan, Goal 2 (ESResNeXt) daha ağırdır → uzun sürebilir.
- Goal 1 genelde 15–20 dk'da biter (GPU + early stopping).
- Goal 2 fine-tune ise saatler sürebilir, GPU şarttır.
