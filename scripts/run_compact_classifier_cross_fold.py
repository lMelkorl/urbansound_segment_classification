#!/usr/bin/env python3
"""Run fixed compact classifier baselines from verified cached embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.models.compact_classifier import (  # noqa: E402
    parse_models,
    run_compact_cross_fold,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run fixed Linear and MLP-128 rotating-fold baselines from verified cache."
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default="linear,mlp128")
    parser.add_argument("--fold", type=int, choices=range(1, 11))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        models = parse_models(args.models)
        result = run_compact_cross_fold(
            cache_root=args.cache_root,
            split_manifest_dir=args.split_manifest_dir,
            output_dir=args.output_dir,
            models=models,
            threads=args.threads,
            fold=args.fold,
            resume=args.resume,
            pretty=args.pretty,
        )
    except FileExistsError:
        print("error: output exists; use --resume only with matching provenance", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: compact cross-fold run failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0 if result["status"]["outcome"] in ("success", "incomplete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
