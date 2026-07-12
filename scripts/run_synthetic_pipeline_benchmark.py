#!/usr/bin/env python3
"""Run a local, deterministic, multi-stage synthetic pipeline benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from urbansound_segment_task.edge_v2.benchmarks.pipeline import (  # noqa: E402
    PipelineBenchmarkRequest,
    run_pipeline_benchmark,
)
from urbansound_segment_task.edge_v2.benchmarks.schema import (  # noqa: E402
    serialize_document,
    write_document_atomic,
)
from urbansound_segment_task.edge_v2.benchmarks.stages import (  # noqa: E402
    build_synthetic_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a dataset-free synthetic decode-to-aggregate pipeline benchmark."
    )
    parser.add_argument("--warmup", type=int, default=3, help="Unmeasured runs per stage (default: 3).")
    parser.add_argument(
        "--iterations", type=int, default=20, help="Measured runs per stage (default: 20)."
    )
    parser.add_argument(
        "--items-per-call", type=int, default=1, help="Logical items per call (default: 1)."
    )
    parser.add_argument(
        "--item-duration-seconds",
        type=float,
        help="Optional represented duration of one item for real-time factor.",
    )
    parser.add_argument(
        "--input-size", type=int, default=1024, help="Synthetic sequence length (default: 1024)."
    )
    parser.add_argument("--output", type=Path, help="Atomically create a new JSON output file.")
    parser.add_argument("--pretty", action="store_true", help="Indent JSON output.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        stages, source = build_synthetic_pipeline(args.input_size)
        request = PipelineBenchmarkRequest(
            name="synthetic-pipeline",
            description="Deterministic standard-library decode-to-aggregate synthetic pipeline.",
            warmup=args.warmup,
            iterations=args.iterations,
            items_per_call=args.items_per_call,
            item_duration_seconds=args.item_duration_seconds,
            input_size=args.input_size,
        )
        document = run_pipeline_benchmark(stages, source, request)
    except ValueError as exc:
        parser.error(str(exc))

    serialized = serialize_document(document, pretty=args.pretty)
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        try:
            write_document_atomic(args.output, serialized)
        except FileExistsError:
            print("error: output file already exists; refusing to overwrite", file=sys.stderr)
            return 2
        except OSError:
            print("error: unable to create output file atomically", file=sys.stderr)
            return 2
    return 0 if document["status"]["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
