#!/usr/bin/env python3
"""Write an offline, allowlisted Edge Audio V2 environment audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from urbansound_segment_task.edge_v2.utils.environment import (  # noqa: E402
    collect_environment,
    serialize_environment,
    write_environment_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect safe, allowlisted environment metadata without network access."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Create a new JSON file at this path; existing files are never overwritten.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Indent JSON output for human readability.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    document = collect_environment(cwd=REPOSITORY_ROOT)
    serialized = serialize_environment(document, pretty=args.pretty)
    if args.output is None:
        sys.stdout.write(serialized)
        return 0
    try:
        write_environment_file(args.output, serialized)
    except FileExistsError:
        print("error: output file already exists; refusing to overwrite", file=sys.stderr)
        return 2
    except OSError:
        print("error: unable to create output file", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

