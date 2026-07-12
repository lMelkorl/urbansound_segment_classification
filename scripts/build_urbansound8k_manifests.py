#!/usr/bin/env python3
"""Create ten official-fold rotating-validation manifests offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.data.manifest import write_json_atomic  # noqa: E402
from urbansound_segment_task.edge_v2.data.splits import build_all_split_manifests  # noqa: E402
from urbansound_segment_task.edge_v2.data.urbansound8k import DatasetLayoutError, inspect_urbansound8k  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build official UrbanSound8K rotating split manifests.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        dataset = inspect_urbansound8k(args.dataset_root)
        if dataset["status"]["outcome"] != "success":
            print("error: dataset inventory failed; split manifests were not written", file=sys.stderr)
            return 1
        manifests = build_all_split_manifests(dataset)
        destinations = [args.output_dir / f"test-fold-{fold}.json" for fold in range(1, 11)]
        if any(path.exists() for path in destinations):
            raise FileExistsError
        for path, manifest in zip(destinations, manifests):
            write_json_atomic(path, manifest, pretty=args.pretty)
        write_json_atomic(args.output_dir / "dataset-manifest.json", dataset, pretty=args.pretty)
    except DatasetLayoutError:
        print(
            "error: UrbanSound8K not found; provide a root containing metadata/UrbanSound8K.csv "
            "and audio/fold1..fold10",
            file=sys.stderr,
        )
        return 2
    except FileExistsError:
        print("error: output manifest already exists; refusing to overwrite", file=sys.stderr)
        return 2
    print("created 10 disjoint official-fold rotating-validation manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
