"""Offline orchestration and safe filesystem/package inspection."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

from urbansound_segment_task.edge_v2.benchmarks.schema import (
    environment_section,
    safe_environment_summary,
    utc_timestamp,
)

from .base import ArtifactSpec, DependencySpec


MODEL_PREFLIGHT_SCHEMA_VERSION = "edge-v2.model-preflight.v1"
METHOD_IDS = ("yamnet_lgbm", "esresnext", "audioclip")
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+!_\-]{0,127}$")
_SAFE_REVISION = re.compile(r"^[0-9a-fA-F]{7,64}$")


def _default_find_spec(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _default_version(package_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def streaming_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _default_git_stage(repo_root: Path, relative_path: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--stage", "--", relative_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


class PreflightInspector:
    def __init__(
        self,
        repo_root: Path,
        *,
        find_spec: Callable[[str], bool] = _default_find_spec,
        version_lookup: Callable[[str], Optional[str]] = _default_version,
        git_stage_lookup: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self._find_spec = find_spec
        self._version_lookup = version_lookup
        self._git_stage_lookup = git_stage_lookup

    @staticmethod
    def _validate_relative_path(relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("preflight paths must be repository-relative")
        return path

    def path(self, relative_path: str) -> Path:
        return self.repo_root / self._validate_relative_path(relative_path)

    def exists(self, relative_path: str) -> bool:
        return os.path.lexists(self.path(relative_path))

    def dependency(self, spec: DependencySpec) -> dict:
        try:
            import_available = bool(self._find_spec(spec.import_name))
        except Exception:
            import_available = False
        try:
            raw_version = self._version_lookup(spec.package_name)
        except Exception:
            raw_version = None
        version = (
            str(raw_version)
            if raw_version is not None and _SAFE_VERSION.fullmatch(str(raw_version))
            else "redacted" if raw_version is not None else None
        )
        notes = spec.notes
        if import_available and version is None:
            notes = (notes + " Import is discoverable but distribution version is unavailable.").strip()
        return {
            "package_name": spec.package_name,
            "import_name": spec.import_name,
            "installed": import_available,
            "version": version,
            "required": spec.required,
            "notes": notes,
        }

    def gitlink_info(self, relative_path: str) -> dict:
        self._validate_relative_path(relative_path)
        stage = (
            self._git_stage_lookup(relative_path)
            if self._git_stage_lookup is not None
            else _default_git_stage(self.repo_root, relative_path)
        )
        if not stage:
            return {"is_gitlink": False, "revision": None}
        parts = stage.split()
        if len(parts) >= 2 and parts[0] == "160000" and _SAFE_REVISION.fullmatch(parts[1]):
            return {"is_gitlink": True, "revision": parts[1].lower()}
        return {"is_gitlink": False, "revision": None}

    def artifact(self, spec: ArtifactSpec) -> dict:
        path = self.path(spec.relative_path)
        gitlink = self.gitlink_info(spec.relative_path)
        exists = os.path.lexists(path)
        kind = spec.kind
        size_bytes: Optional[int] = None
        sha256: Optional[str] = None
        provenance = "missing"
        notes = spec.notes

        if gitlink["is_gitlink"]:
            kind = "gitlink"
            provenance = "present_unverified" if exists else "broken_reference"
        elif path.is_symlink():
            kind = "symlink"
            provenance = "present_unverified" if path.exists() else "broken_reference"
            try:
                size_bytes = path.lstat().st_size
            except OSError:
                size_bytes = None
        elif path.is_file():
            kind = "file"
            try:
                size_bytes = path.stat().st_size
                sha256 = streaming_sha256(path)
                provenance = "present_unverified"
            except OSError:
                provenance = "broken_reference"
        elif path.is_dir():
            kind = "directory"
            provenance = "present_unverified"
        elif exists:
            provenance = "broken_reference"

        return {
            "artifact_id": spec.artifact_id,
            "relative_path": spec.relative_path,
            "exists": exists,
            "kind": kind,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "required": spec.required,
            "provenance_status": provenance,
            "notes": notes,
            "source_revision": gitlink["revision"],
        }

    def read_text(self, relative_path: str) -> Optional[str]:
        path = self.path(relative_path)
        try:
            if not path.is_file() or path.stat().st_size > 2_000_000:
                return None
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None


def _method_inspectors() -> dict[str, Callable[[PreflightInspector], dict]]:
    from .audioclip import inspect_audioclip
    from .esresnext import inspect_esresnext
    from .yamnet_lgbm import inspect_yamnet_lgbm

    return {
        "yamnet_lgbm": inspect_yamnet_lgbm,
        "esresnext": inspect_esresnext,
        "audioclip": inspect_audioclip,
    }


def _summary(methods: Sequence[dict]) -> dict:
    counts = {status: 0 for status in ("ready", "blocked", "unavailable", "unverified")}
    for method in methods:
        counts[method["benchmark_readiness"]] += 1

    candidates = [
        method
        for method in methods
        if method["implementation_status"] in ("implemented", "partial")
        and method["cpu_support"]["code_path_detected"]
        and method["benchmark_readiness"] != "unavailable"
    ]
    readiness_rank = {"ready": 0, "unverified": 1, "blocked": 2, "unavailable": 3}
    candidates.sort(
        key=lambda method: (
            readiness_rank[method["benchmark_readiness"]],
            sum(1 for item in method["issues"] if item["severity"] == "blocker"),
            METHOD_IDS.index(method["method_id"]),
        )
    )
    return {
        "ready_count": counts["ready"],
        "blocked_count": counts["blocked"],
        "unavailable_count": counts["unavailable"],
        "unverified_count": counts["unverified"],
        "recommended_first_method": candidates[0]["method_id"] if candidates else None,
    }


def run_model_preflight(
    repo_root: Path,
    *,
    method: str = "all",
    now: Optional[datetime] = None,
    find_spec: Callable[[str], bool] = _default_find_spec,
    version_lookup: Callable[[str], Optional[str]] = _default_version,
    git_stage_lookup: Optional[Callable[[str], Optional[str]]] = None,
) -> dict:
    if method not in ("all",) + METHOD_IDS:
        raise ValueError("unknown model preflight method")
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ValueError("repo_root must be an existing directory")
    inspector = PreflightInspector(
        root,
        find_spec=find_spec,
        version_lookup=version_lookup,
        git_stage_lookup=git_stage_lookup,
    )
    registry = _method_inspectors()
    selected = METHOD_IDS if method == "all" else (method,)
    methods = [registry[method_id](inspector) for method_id in selected]
    return {
        "schema_version": MODEL_PREFLIGHT_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(now),
        "environment": environment_section(safe_environment_summary()),
        "methods": methods,
        "summary": _summary(methods),
    }
