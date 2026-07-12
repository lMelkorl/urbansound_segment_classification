#!/usr/bin/env python3
"""Validate Keras, FP32 ONNX, and dynamic-INT8 ONNX parity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.export.onnx_int8_validation import validate_linear_int8_onnx  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate dynamic-INT8 numeric and Fold 1 metric parity on CPU.")
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--fp32-artifact", type=Path, required=True)
    parser.add_argument("--int8-artifact", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = validate_linear_int8_onnx(
            run_manifest_path=args.run_manifest, cache_root=args.cache_root,
            fp32_artifact_directory=args.fp32_artifact,
            int8_artifact_directory=args.int8_artifact,
            sample_count=args.sample_count, output_directory=args.output_dir,
            pretty=args.pretty,
        )
    except FileExistsError:
        print("error: INT8 parity output directory already exists", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: INT8 parity validation failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0 if result["status"]["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
