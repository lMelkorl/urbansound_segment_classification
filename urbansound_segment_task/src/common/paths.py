from pathlib import Path
from typing import Tuple, Optional

def detect_dataset_layout(data_dir: str) -> Tuple[Path, Optional[Path]]:
    """
    data_dir örnekleri:
      - ./data  (fold1..fold10 doğrudan burada)
      - ./data/UrbanSound8K (klasik dağıtım)
    Döner:
      audio_root: foldX klasörlerinin olduğu yer
      meta_csv:   UrbanSound8K.csv (yoksa None)
    """
    d = Path(data_dir)

    # 1) Sende yaygın: ./data/fold1..fold10
    if (d / "fold1").exists():
        audio_root = d
        meta_csv = d / "UrbanSound8K.csv"
        if not meta_csv.exists():
            alt = d / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
            meta_csv = alt if alt.exists() else None
        return audio_root, meta_csv

    # 2) Klasik: ./data/UrbanSound8K/audio/fold1
    if (d / "UrbanSound8K" / "audio" / "fold1").exists():
        audio_root = d / "UrbanSound8K" / "audio"
        meta_csv = d / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
        meta_csv = meta_csv if meta_csv.exists() else None
        return audio_root, meta_csv

    # 3) Ender: ./data/UrbanSound8K/fold1
    if (d / "UrbanSound8K" / "fold1").exists():
        audio_root = d / "UrbanSound8K"
        meta_csv = d / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
        meta_csv = meta_csv if meta_csv.exists() else None
        return audio_root, meta_csv

    raise FileNotFoundError(f"fold klasörleri bulunamadı: {d}")
