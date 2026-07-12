"""Result document helpers for the LightGBM legacy reproduction."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

from urbansound_segment_task.edge_v2.benchmarks.schema import utc_timestamp, write_document_atomic
from urbansound_segment_task.edge_v2.utils.environment import collect_system_info


LIGHTGBM_REPRODUCTION_SCHEMA_VERSION = "edge-v2.lightgbm-legacy-reproduction.v1"


def git_revision(repository_root: Path) -> Optional[str]:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True,
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return value if len(value) == 40 else None


def safe_environment(threads: int) -> dict[str, Any]:
    return {
        "system": collect_system_info(),
        "python_version": platform.python_version(),
        "configured_threads": threads,
        "accelerator": "none_cpu_only",
    }


def serialize_result(document: Mapping[str, Any], *, pretty: bool) -> str:
    if pretty:
        return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    return json.dumps(document, separators=(",", ":"), sort_keys=True, ensure_ascii=True) + "\n"


def write_result(path: Path, document: Mapping[str, Any], *, pretty: bool) -> None:
    write_document_atomic(path, serialize_result(document, pretty=pretty))


__all__ = [
    "LIGHTGBM_REPRODUCTION_SCHEMA_VERSION", "git_revision", "safe_environment",
    "serialize_result", "utc_timestamp", "write_result",
]
