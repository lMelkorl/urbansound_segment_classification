# UrbanSound Segment Task

## Quick Start
1) Veriyi indir: `python urbansound-segment-task/scripts/download_urbansound8k.py` (veya Colab hücresi)
2) Yöntem 1 (YAMNet→LightGBM): `python urbansound-segment-task/goals/1_yamnet_lgbm/run.py --data_dir ./data/UrbanSound8K --out ./urbansound-segment-task/goals/1_yamnet_lgbm/results`
3) Final rapor: `python urbansound-segment-task/scripts/make_final_report.py`

## Yapı
- `data/UrbanSound8K`: Ham veri (Kaggle).
- `urbansound-segment-task/src/common`: Ortak yardımcılar (segmentleme, metrik, aggregation).
- `urbansound-segment-task/goals/*`: 3 yaklaşımın scriptleri ve çıktıları.
- `urbansound-segment-task/results`: Toplanan tablolar/raporlar.

## Notlar
- Fold10=test, Fold1–9=train/val (sızıntı yok).
- Segment penceresi ~0.96s, %50 örtüşme. YAMNet için 16kHz mono.
- Seed ve paket sürümleri pinlenecek; raporda segment+klip metrikleri ve hız.
