"""Deterministic, path-safe manifest serialization helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from urbansound_segment_task.edge_v2.benchmarks.schema import write_document_atomic


def canonical_json_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        document, separators=(",", ":"), sort_keys=True, ensure_ascii=True
    ).encode("utf-8")


def document_sha256(document: Mapping[str, Any], *, excluded_fields: tuple[str, ...] = ()) -> str:
    payload = {key: value for key, value in document.items() if key not in excluded_fields}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def serialize_json(document: Mapping[str, Any], *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    return canonical_json_bytes(document).decode("utf-8") + "\n"


def write_json_atomic(path: Path, document: Mapping[str, Any], *, pretty: bool = False) -> None:
    write_document_atomic(Path(path), serialize_json(document, pretty=pretty))


__all__ = ["canonical_json_bytes", "document_sha256", "serialize_json", "write_json_atomic"]
