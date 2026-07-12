#!/usr/bin/env python3
"""Run resumable official-fold LightGBM evaluation from verified cache only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.models.lightgbm_cross_fold import run_cross_fold  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run official rotating-validation LightGBM folds from verified cache.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--fold", type=int, choices=range(1, 11))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_cross_fold(
            cache_root=args.cache_root, split_manifest_dir=args.split_manifest_dir,
            output_dir=args.output_dir, threads=args.threads, fold=args.fold,
            resume=args.resume, pretty=args.pretty,
        )
    except FileExistsError:
        print("error: output exists; use --resume only with matching provenance", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: cross-fold run failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0 if result["status"]["outcome"] in ("success", "incomplete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
