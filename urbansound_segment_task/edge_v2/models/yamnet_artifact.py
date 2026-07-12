"""Acquisition and integrity verification for a local YAMNet SavedModel tree."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional


ARTIFACT_SCHEMA_VERSION = "edge-v2.yamnet-artifact.v1"
YAMNET_SOURCE_ID = "https://tfhub.dev/google/yamnet/1"
TREE_HASH_METHOD = "sha256(sorted(relative_path\\0size\\0file_sha256_bytes))"


class ArtifactVerificationError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__("YAMNet artifact verification failed: " + code)
        self.code = code


class NetworkPermissionRequired(RuntimeError):
    pass


def _utc_timestamp(now: Optional[datetime] = None) -> str:
    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def streaming_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_model_files(model_directory: Path) -> list[dict]:
    root = Path(model_directory)
    if not root.is_dir():
        raise ArtifactVerificationError("MODEL_DIRECTORY_MISSING")
    records = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ArtifactVerificationError("SYMLINK_NOT_ALLOWED")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "relative_path": "model/" + relative,
                "size_bytes": path.stat().st_size,
                "sha256": streaming_file_sha256(path),
            }
        )
    if not records:
        raise ArtifactVerificationError("MODEL_TREE_EMPTY")
    return records


def deterministic_tree_sha256(file_records: list[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(file_records, key=lambda item: str(item["relative_path"])):
        relative_path = str(record["relative_path"])
        size = str(int(record["size_bytes"]))
        file_hash = str(record["sha256"])
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(size.encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash))
    return digest.hexdigest()


def build_manifest(
    model_directory: Path,
    *,
    package_versions: Mapping[str, Optional[str]],
    now: Optional[datetime] = None,
) -> dict:
    files = collect_model_files(model_directory)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": "yamnet-tfhub-v1",
        "source_model_id": YAMNET_SOURCE_ID,
        "acquired_at_utc": _utc_timestamp(now),
        "license": {
            "status": "unverified",
            "identifier": None,
            "notes": "License was not inferred by the acquisition tool; verify official model metadata manually.",
        },
        "tree_hash_method": TREE_HASH_METHOD,
        "tree_sha256": deterministic_tree_sha256(files),
        "files": files,
        "package_versions": {
            name: version for name, version in sorted(package_versions.items())
        },
    }


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def acquire_yamnet_artifact(
    output_directory: Path,
    *,
    allow_network: bool,
    resolver: Callable[[], Path],
    package_versions: Mapping[str, Optional[str]],
    now: Optional[datetime] = None,
) -> dict:
    """Resolve into a temporary tree and atomically publish a new artifact."""

    if not allow_network:
        raise NetworkPermissionRequired("explicit --allow-network is required")
    output = Path(output_directory)
    if output.exists() or os.path.lexists(output):
        raise FileExistsError("artifact output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".yamnet-acquire-", dir=output.parent))
    try:
        resolved = Path(resolver())
        if not resolved.is_dir():
            raise ArtifactVerificationError("RESOLVED_MODEL_DIRECTORY_MISSING")
        model_output = temporary / "model"
        shutil.copytree(resolved, model_output, symlinks=False)
        manifest = build_manifest(
            model_output,
            package_versions=package_versions,
            now=now,
        )
        _write_json(temporary / "artifact-manifest.json", manifest)
        os.rename(temporary, output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_manifest(artifact_directory: Path) -> dict:
    manifest_path = Path(artifact_directory) / "artifact-manifest.json"
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ArtifactVerificationError("MANIFEST_MISSING") from None
    except (OSError, json.JSONDecodeError):
        raise ArtifactVerificationError("MANIFEST_INVALID") from None
    if not isinstance(document, dict):
        raise ArtifactVerificationError("MANIFEST_INVALID")
    return document


def _reject_url(value: object) -> None:
    normalized = str(value).lower()
    if normalized.startswith(("http://", "https://")):
        raise ArtifactVerificationError("LOCAL_PATH_REQUIRED")


def verify_yamnet_artifact(artifact_directory: Path) -> dict:
    _reject_url(artifact_directory)
    artifact = Path(artifact_directory)
    manifest = _load_manifest(artifact)
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactVerificationError("SCHEMA_VERSION_MISMATCH")
    if manifest.get("source_model_id") != YAMNET_SOURCE_ID:
        raise ArtifactVerificationError("SOURCE_ID_MISMATCH")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, list):
        raise ArtifactVerificationError("FILE_MANIFEST_INVALID")
    actual_files = collect_model_files(artifact / "model")
    expected_by_path = {str(item.get("relative_path")): item for item in expected_files}
    actual_by_path = {item["relative_path"]: item for item in actual_files}
    if set(expected_by_path) != set(actual_by_path):
        raise ArtifactVerificationError("FILE_SET_MISMATCH")
    for relative_path, actual in actual_by_path.items():
        expected = expected_by_path[relative_path]
        if int(expected.get("size_bytes", -1)) != actual["size_bytes"]:
            raise ArtifactVerificationError("FILE_SIZE_MISMATCH")
        if str(expected.get("sha256")) != actual["sha256"]:
            raise ArtifactVerificationError("FILE_HASH_MISMATCH")
    tree_hash = deterministic_tree_sha256(actual_files)
    if manifest.get("tree_sha256") != tree_hash:
        raise ArtifactVerificationError("TREE_HASH_MISMATCH")
    return {
        "artifact_id": str(manifest.get("artifact_id")),
        "source_model_id": str(manifest.get("source_model_id")),
        "tree_sha256": tree_hash,
        "file_count": len(actual_files),
        "model_relative_path": "model",
        "manifest": manifest,
    }
