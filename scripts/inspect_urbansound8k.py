#!/usr/bin/env python3
"""Inspect a local UrbanSound8K tree without downloading or decoding audio bodies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.data.manifest import serialize_json, write_json_atomic  # noqa: E402
from urbansound_segment_task.edge_v2.data.urbansound8k import (  # noqa: E402
    DatasetLayoutError, DatasetValidationError, inspect_urbansound8k,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build an offline UrbanSound8K dataset inventory.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        document = inspect_urbansound8k(args.dataset_root)
        serialized = serialize_json(document, pretty=args.pretty)
        if args.output:
            write_json_atomic(args.output, document, pretty=args.pretty)
        else:
            sys.stdout.write(serialized)
    except (DatasetLayoutError, DatasetValidationError) as exc:
        print(
            "error: UrbanSound8K not ready; expected UrbanSound8K.csv (or metadata/UrbanSound8K.csv) "
            "and audio/fold1..fold10 under --dataset-root",
            file=sys.stderr,
        )
        return 2
    except FileExistsError:
        print("error: output already exists; refusing to overwrite", file=sys.stderr)
        return 2
    return 0 if document["status"]["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
