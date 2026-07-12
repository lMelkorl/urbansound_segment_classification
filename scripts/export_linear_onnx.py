#!/usr/bin/env python3
"""Export the provenance-bound Linear Fold 1 classifier to FP32 ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.export.onnx_linear import export_linear_onnx  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Export deterministic Linear Fold 1 cached-embedding classifier to FP32 ONNX."
    )
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--opset", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        export_linear_onnx(
            run_manifest_path=args.run_manifest,
            model=args.model,
            fold=args.fold,
            opset=args.opset,
            output_path=args.output,
            manifest_path=args.manifest,
            pretty=args.pretty,
        )
    except FileExistsError:
        print("error: ONNX output or manifest already exists", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: ONNX export failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
