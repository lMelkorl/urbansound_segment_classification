"""Shared static model-preflight contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


IMPLEMENTATION_STATUSES = ("implemented", "partial", "placeholder", "missing")
READINESS_STATUSES = ("ready", "blocked", "unavailable", "unverified")
PROVENANCE_STATUSES = (
    "verified_local",
    "present_unverified",
    "missing",
    "broken_reference",
    "placeholder",
)


@dataclass(frozen=True)
class DependencySpec:
    package_name: str
    import_name: str
    required: bool = True
    notes: str = ""


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    relative_path: str
    kind: str
    required: bool = True
    notes: str = ""


class InspectorProtocol(Protocol):
    def dependency(self, spec: DependencySpec) -> dict: ...

    def artifact(self, spec: ArtifactSpec) -> dict: ...

    def read_text(self, relative_path: str) -> Optional[str]: ...

    def gitlink_info(self, relative_path: str) -> dict: ...

    def exists(self, relative_path: str) -> bool: ...


def issue(code: str, severity: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def cpu_support(
    *,
    declared: bool,
    code_path_detected: bool,
    notes: str,
) -> dict:
    return {
        "declared": declared,
        "code_path_detected": code_path_detected,
        "runtime_verified": False,
        "notes": notes,
    }


def method_status() -> dict:
    return {"outcome": "success", "error": None}
