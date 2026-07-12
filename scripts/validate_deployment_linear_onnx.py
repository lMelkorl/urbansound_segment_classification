#!/usr/bin/env python3
"""Validate Keras versus FP32 ONNX deployment classifier numeric parity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.export.deployment_onnx import validate_deployment_onnx  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate deployment Keras/ONNX numeric parity without metrics.")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--onnx-artifact", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    arguments = parser.parse_args()
    try:
        result = validate_deployment_onnx(
            source_run=arguments.source_run, cache_root=arguments.cache_root,
            onnx_artifact_directory=arguments.onnx_artifact,
            sample_count=arguments.sample_count, output_directory=arguments.output_dir,
            pretty=arguments.pretty,
        )
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: deployment ONNX parity failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0 if result["status"]["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
