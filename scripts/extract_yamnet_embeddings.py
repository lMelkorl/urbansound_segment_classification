#!/usr/bin/env python3
"""Extract resumable local YAMNet embedding cache artifacts offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.data.manifest import serialize_json, write_json_atomic  # noqa: E402
from urbansound_segment_task.edge_v2.data.urbansound8k import DatasetLayoutError  # noqa: E402
from urbansound_segment_task.edge_v2.features.extractor import extract_yamnet_embeddings  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build an offline resumable YAMNet embedding cache.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit-clips", type=int)
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.limit_clips is not None and args.limit_clips < 1):
        parser.error("--threads and --limit-clips must be positive")
    try:
        document = extract_yamnet_embeddings(
            dataset_root=args.dataset_root, artifact=args.artifact, cache_root=args.cache_root,
            threads=args.threads, limit_clips=args.limit_clips,
            confirm_full_run=args.confirm_full_run, force=args.force,
        )
        if args.output:
            write_json_atomic(args.output, document, pretty=args.pretty)
        else:
            sys.stdout.write(serialize_json(document, pretty=args.pretty))
    except DatasetLayoutError:
        print(
            "error: UrbanSound8K not found; no download attempted. Provide --dataset-root with "
            "metadata/UrbanSound8K.csv and audio/fold1..fold10, then rerun this command.",
            file=sys.stderr,
        )
        return 2
    except FileExistsError:
        print("error: output already exists; refusing to overwrite", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: extraction stopped ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0 if document["status"]["outcome"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
