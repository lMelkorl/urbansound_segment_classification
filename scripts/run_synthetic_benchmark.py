#!/usr/bin/env python3
"""Run a fast, local, single-callable synthetic benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from urbansound_segment_task.edge_v2.benchmarks.runner import (  # noqa: E402
    BenchmarkRequest,
    run_benchmark,
)
from urbansound_segment_task.edge_v2.benchmarks.schema import (  # noqa: E402
    serialize_document,
    write_document_atomic,
)


def noop_workload() -> int:
    return 1


def cpu_workload() -> int:
    """Perform a fixed amount of local integer work and return its accumulator."""

    accumulator = 0x13579BDF
    for index in range(4_000):
        accumulator = (accumulator * 1_664_525 + index + 1_013_904_223) & 0xFFFFFFFF
    return accumulator


WORKLOADS: dict[str, tuple[Callable[[], int], str]] = {
    "noop": (noop_workload, "Minimal local callable used to validate timer overhead behavior."),
    "cpu": (cpu_workload, "Fixed local integer workload used to exercise the synthetic timer."),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a dataset-free single-callable synthetic benchmark."
    )
    parser.add_argument("--warmup", type=int, default=3, help="Unmeasured warm-up calls (default: 3).")
    parser.add_argument(
        "--iterations", type=int, default=20, help="Measured callable invocations (default: 20)."
    )
    parser.add_argument(
        "--items-per-call", type=int, default=1, help="Logical items handled by each call (default: 1)."
    )
    parser.add_argument(
        "--item-duration-seconds",
        type=float,
        help="Optional represented duration of one item, used for real-time factor.",
    )
    parser.add_argument("--output", type=Path, help="Atomically create a new JSON output file.")
    parser.add_argument("--pretty", action="store_true", help="Indent JSON output.")
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="cpu")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    function, description = WORKLOADS[args.workload]
    request = BenchmarkRequest(
        name="synthetic-" + args.workload,
        description=description,
        warmup=args.warmup,
        iterations=args.iterations,
        items_per_call=args.items_per_call,
        item_duration_seconds=args.item_duration_seconds,
    )
    try:
        document = run_benchmark(function, request)
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
