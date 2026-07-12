#!/usr/bin/env python3
"""Run one offline local YAMNet output-contract smoke test."""

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
from urbansound_segment_task.edge_v2.models.yamnet_smoke import run_yamnet_smoke  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a verified local YAMNet artifact offline.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    document = run_yamnet_smoke(args.artifact)
    serialized = serialize_document(document, pretty=args.pretty)
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        try:
            write_document_atomic(args.output, serialized)
        except FileExistsError:
            print("error: output file already exists; refusing to overwrite", file=sys.stderr)
            return 2
    return 0 if document["status"]["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
