from __future__ import annotations

from pathlib import Path
from typing import Optional


def write_fixture(root: Path) -> None:
    files = {
        "requirements.txt": "numpy>=1.26\npandas>=2.2\nlibrosa>=0.10\nsoundfile>=0.12\n",
        "urbansound_segment_task/goals/goal1_yamnet_lgbm/run.py": (
            "import tensorflow as tf\nimport tensorflow_hub as hub\n"
            "from lightgbm import LGBMClassifier\n"
            'model = hub.load("https://tfhub.dev/google/yamnet/1")\n'
        ),
        "urbansound_segment_task/goals/goal2_esresnext/run_finetune.py": (
            'device = "cuda" if torch.cuda.is_available() else "cpu"\n'
            'repo_dir = Path("./external/ESResNeXt_fbsp")\n'
            "from model.esresnext import ESResNeXtFBSP\n"
        ),
        "urbansound_segment_task/goals/goal2_esresnext/run_head_only.py": (
            'device = "cuda" if torch.cuda.is_available() else "cpu"\n'
            'repo_dir = Path("./external/ESResNeXt-fbsp")\n'
            "from models.esresnext import ESResNeXtFBSP\n"
        ),
        "urbansound_segment_task/scripts/download_checkpoint_goal2.sh": (
            "#!/bin/sh\nwget -O checkpoint.pt https://example.invalid/checkpoint.pt\n"
        ),
        "urbansound_segment_task/goals/goal3_audioclip/run.py": (
            "print('Run script placeholder. Bu dosya ileride doldurulacak.')\n"
        ),
    }
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (root / "external/ESResNeXt_fbsp").mkdir(parents=True, exist_ok=True)


def git_stage(relative_path: str) -> Optional[str]:
    if relative_path == "external/ESResNeXt_fbsp":
        return "160000 " + "a" * 40 + " 0\texternal/ESResNeXt_fbsp"
    return None


def missing_packages(_name: str) -> bool:
    return False


def no_versions(_name: str) -> None:
    return None
