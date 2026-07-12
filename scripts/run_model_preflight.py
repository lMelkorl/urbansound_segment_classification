#!/usr/bin/env python3
"""Run offline static preflight checks for legacy model methods."""

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
from urbansound_segment_task.edge_v2.models.preflight import (  # noqa: E402
    METHOD_IDS,
    run_model_preflight,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect model benchmark readiness without imports, downloads, or inference."
    )
    parser.add_argument("--method", choices=("all",) + METHOD_IDS, default="all")
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        document = run_model_preflight(args.repo_root, method=args.method)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
