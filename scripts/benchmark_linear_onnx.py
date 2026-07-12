#!/usr/bin/env python3
"""Benchmark Linear Keras and FP32 ONNX classifiers in fresh CPU-only processes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.benchmarks.onnx_classifier import run_benchmark_suite  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fresh-process batch-1 Keras versus ONNX Runtime classifier-only benchmark."
    )
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--onnx-artifact", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_benchmark_suite(
            run_manifest_path=args.run_manifest,
            cache_root=args.cache_root,
            onnx_artifact_directory=args.onnx_artifact,
            warmup=args.warmup,
            iterations=args.iterations,
            repetitions=args.repetitions,
            threads=args.threads,
            output_directory=args.output_dir,
            pretty=args.pretty,
        )
    except FileExistsError:
        print("error: benchmark output directory already exists", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: classifier benchmark failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0 if result["status"]["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
