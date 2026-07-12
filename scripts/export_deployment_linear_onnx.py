#!/usr/bin/env python3
"""Export the deployment-only Linear Keras weights to FP32 ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.export.deployment_onnx import export_deployment_linear_onnx  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export deployment-only Linear classifier to FP32 ONNX.")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--opset", type=int, required=True)
    parser.add_argument("--pretty", action="store_true")
    arguments = parser.parse_args()
    try:
        export_deployment_linear_onnx(
            source_run=arguments.source_run, output_path=arguments.output,
            manifest_path=arguments.manifest, opset=arguments.opset, pretty=arguments.pretty,
        )
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: deployment ONNX export failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

