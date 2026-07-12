#!/usr/bin/env python3
"""Run verified local YAMNet lifecycle and staged CPU benchmarks offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from urbansound_segment_task.edge_v2.benchmarks.schema import (  # noqa: E402
    serialize_document,
    write_document_atomic,
)
from urbansound_segment_task.edge_v2.benchmarks.yamnet_cpu import (  # noqa: E402
    externalize_raw_runs,
    run_yamnet_cpu_benchmark,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _csv_positive_ints(value: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be comma-separated integers") from exc
    if not parsed or any(item < 1 for item in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("values must be unique positive integers")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a verified local YAMNet SavedModel on CPU without network access."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--threads", type=_positive_int, default=4)
    parser.add_argument("--warmup", type=_non_negative_int, default=10)
    parser.add_argument("--iterations", type=_positive_int, default=100)
    parser.add_argument("--repetitions", type=_positive_int, default=1)
    parser.add_argument("--sample-count", type=_positive_int, default=15_360)
    parser.add_argument("--thread-sweep", type=_csv_positive_ints)
    parser.add_argument("--input-sensitivity", type=_csv_positive_ints, default=[])
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    thread_counts = args.thread_sweep or [args.threads]
    try:
        summary = run_yamnet_cpu_benchmark(
            artifact=args.artifact,
            thread_counts=thread_counts,
            warmup=args.warmup,
            iterations=args.iterations,
            repetitions=args.repetitions,
            sample_count=args.sample_count,
            input_sensitivity=args.input_sensitivity,
            timeout_seconds=args.timeout_seconds,
            repository_root=REPOSITORY_ROOT,
        )
        if args.output is not None:
            summary = externalize_raw_runs(summary, args.output)
        serialized = serialize_document(summary, pretty=args.pretty)
        if args.output is None:
            sys.stdout.write(serialized)
        else:
            write_document_atomic(args.output, serialized)
    except FileExistsError:
        print("error: output file or raw artifact directory already exists", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: benchmark failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0 if summary["status"]["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
