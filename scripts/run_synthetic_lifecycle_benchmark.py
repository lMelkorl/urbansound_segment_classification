#!/usr/bin/env python3
"""Run a fresh-process synthetic lifecycle and peak RSS benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from urbansound_segment_task.edge_v2.benchmarks.process_runner import (  # noqa: E402
    LifecycleProcessConfig,
    run_lifecycle_in_fresh_process,
)
from urbansound_segment_task.edge_v2.benchmarks.schema import (  # noqa: E402
    serialize_document,
    write_document_atomic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a dataset-free lifecycle benchmark in a fresh spawn process."
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--items-per-call", type=int, default=1)
    parser.add_argument("--item-duration-seconds", type=float)
    parser.add_argument("--load-size", type=int, default=50_000)
    parser.add_argument("--work-size", type=int, default=1_000)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        document = run_lifecycle_in_fresh_process(
            LifecycleProcessConfig(
                threads=args.threads,
                warmup=args.warmup,
                iterations=args.iterations,
                items_per_call=args.items_per_call,
                item_duration_seconds=args.item_duration_seconds,
                load_size=args.load_size,
                work_size=args.work_size,
            ),
            timeout_seconds=args.timeout_seconds,
        )
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
