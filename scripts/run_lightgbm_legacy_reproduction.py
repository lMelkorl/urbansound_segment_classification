#!/usr/bin/env python3
"""Reproduce legacy Goal 1 from verified cached YAMNet embeddings only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.models.lightgbm_legacy import (  # noqa: E402
    LEGACY_DEFAULT_SEED, run_legacy_reproduction,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the legacy Fold 1-8/9/10 LightGBM experiment from verified cache only."
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=LEGACY_DEFAULT_SEED)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        result = run_legacy_reproduction(
            cache_root=args.cache_root, output_dir=args.output_dir,
            threads=args.threads, seed=args.seed, repository_root=ROOT, pretty=args.pretty,
        )
    except FileExistsError:
        print("error: output directory already exists; refusing to overwrite", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: legacy reproduction failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0 if result["status"]["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
