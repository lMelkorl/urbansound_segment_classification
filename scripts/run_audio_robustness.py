#!/usr/bin/env python3
"""Evaluate official held-out-fold Linear models under fixed audio perturbations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.evaluation.audio_robustness import run_audio_robustness  # noqa: E402
from urbansound_segment_task.edge_v2.evaluation.robustness import CONDITIONS  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Offline deterministic robustness evaluation for held-out-fold Linear models."
    )
    value.add_argument("--dataset-root", type=Path, required=True)
    value.add_argument("--model-run", type=Path, required=True)
    value.add_argument("--yamnet-artifact", type=Path, required=True)
    value.add_argument("--clips-per-class-per-fold", type=int, default=20)
    value.add_argument("--conditions", default=",".join(CONDITIONS))
    value.add_argument("--fold", type=int, choices=range(1, 11))
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--pretty", action="store_true")
    return value


def parse_conditions(value: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selected or len(selected) != len(set(selected)) or any(item not in CONDITIONS for item in selected):
        raise ValueError("conditions must be a unique comma-separated subset of the fixed allowlist")
    return selected


def main(argv=None) -> int:
    arguments = parser().parse_args(argv)
    try:
        conditions = parse_conditions(arguments.conditions)
        result = run_audio_robustness(
            dataset_root=arguments.dataset_root,
            model_run=arguments.model_run,
            yamnet_artifact=arguments.yamnet_artifact,
            clips_per_class_per_fold=arguments.clips_per_class_per_fold,
            conditions=conditions,
            fold=arguments.fold,
            output_dir=arguments.output_dir,
            resume=arguments.resume,
            pretty=arguments.pretty,
        )
    except Exception as exc:
        print(f"error: audio robustness evaluation failed ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2
    return 0 if result.get("status", {}).get("outcome") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
