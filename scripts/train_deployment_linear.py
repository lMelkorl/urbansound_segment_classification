#!/usr/bin/env python3
"""Train the fixed deployment-only Linear classifier on all verified cache segments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.models.deployment_classifier import train_deployment_linear  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train deployment-only Linear softmax on all verified embeddings.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--cross-fold-aggregate", type=Path,
        default=Path("results/compact_classifier_cross_fold/fixed-baselines-v1/aggregate.json"),
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    arguments = parser.parse_args()
    try:
        train_deployment_linear(
            cache_root=arguments.cache_root,
            cross_fold_aggregate=arguments.cross_fold_aggregate,
            epochs=arguments.epochs, threads=arguments.threads,
            output_directory=arguments.output_dir, pretty=arguments.pretty,
        )
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: deployment training failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
