#!/usr/bin/env python3
"""Explicitly acquire the official YAMNet TF Hub module into a local artifact."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from urbansound_segment_task.edge_v2.models.yamnet_artifact import (  # noqa: E402
    ArtifactVerificationError,
    NetworkPermissionRequired,
    YAMNET_SOURCE_ID,
    acquire_yamnet_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire YAMNet from its official model ID; network requires explicit consent."
    )
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _package_versions() -> dict[str, Optional[str]]:
    versions = {}
    for package in ("tensorflow", "tensorflow-hub", "tf-keras", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.allow_network:
        print("error: explicit --allow-network is required; no network request was made", file=sys.stderr)
        return 2

    def resolver() -> Path:
        import tensorflow_hub as hub

        return Path(hub.resolve(YAMNET_SOURCE_ID))

    try:
        manifest = acquire_yamnet_artifact(
            args.output,
            allow_network=True,
            resolver=resolver,
            package_versions=_package_versions(),
        )
    except (FileExistsError, NetworkPermissionRequired) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 2
    except ArtifactVerificationError as exc:
        print("error: " + exc.code, file=sys.stderr)
        return 1
    except Exception as exc:
        print("error: acquisition failed: " + type(exc).__name__, file=sys.stderr)
        return 1
    print(manifest["tree_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
