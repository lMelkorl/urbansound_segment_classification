#!/usr/bin/env python3
"""Benchmark FP32 and dynamic-INT8 Linear ONNX artifacts in fresh CPU processes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.benchmarks.onnx_int8_classifier import run_int8_benchmark_suite  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Alternating fresh-process FP32 versus dynamic-INT8 classifier benchmark.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--fp32-artifact", type=Path, required=True)
    parser.add_argument("--int8-artifact", type=Path, required=True)
    parser.add_argument("--parity-result", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_int8_benchmark_suite(
            cache_root=args.cache_root,
            fp32_artifact_directory=args.fp32_artifact,
            int8_artifact_directory=args.int8_artifact,
            parity_result_path=args.parity_result,
            warmup=args.warmup, iterations=args.iterations,
            repetitions=args.repetitions, threads=args.threads,
            output_directory=args.output_dir, pretty=args.pretty,
        )
    except FileExistsError:
        print("error: INT8 benchmark output directory already exists", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: INT8 classifier benchmark failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0 if result["status"]["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
