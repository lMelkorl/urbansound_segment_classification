#!/usr/bin/env python3
"""Verify every file and the deterministic tree hash of a local YAMNet artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from urbansound_segment_task.edge_v2.models.yamnet_artifact import (  # noqa: E402
    ArtifactVerificationError,
    verify_yamnet_artifact,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a local YAMNet artifact without loading it.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        identity = verify_yamnet_artifact(args.artifact)
    except ArtifactVerificationError as exc:
        print("error: " + exc.code, file=sys.stderr)
        return 1
    safe = {key: identity[key] for key in ("artifact_id", "source_model_id", "tree_sha256", "file_count")}
    print(json.dumps(safe, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
