#!/usr/bin/env python3
"""Create the fixed dynamic-QInt8 Linear Fold 1 ONNX artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.export.onnx_quantization import quantize_linear_dynamic_int8  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Quantize the verified Linear Fold 1 FP32 ONNX weights to dynamic QInt8.")
    parser.add_argument("--fp32-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        quantize_linear_dynamic_int8(
            fp32_artifact_directory=args.fp32_artifact,
            output_path=args.output,
            manifest_path=args.manifest,
            pretty=args.pretty,
        )
    except FileExistsError:
        print("error: INT8 output or manifest already exists", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: dynamic INT8 quantization failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
