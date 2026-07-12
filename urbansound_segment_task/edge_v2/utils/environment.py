"""Offline, allowlisted environment metadata collection.

The collector deliberately avoids hostnames, user information, network details,
repository paths, and bulk environment-variable capture. Optional dependencies
are inspected through package metadata and never installed or downloaded.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


SCHEMA_VERSION = "edge-v2.environment.v1"

PACKAGE_DISTRIBUTIONS: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
    ("tensorflow", "tensorflow"),
    ("tensorflow-hub", "tensorflow-hub"),
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
)

THREAD_ENVIRONMENT_NAMES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)

_SAFE_THREAD_VALUE = re.compile(r"^[0-9]{1,5}$")
_SAFE_VERSION_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+!_\-]{0,127}$")
_SAFE_GIT_COMMIT = re.compile(r"^[0-9a-fA-F]{7,64}$")
_SAFE_GIT_BRANCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]{0,127}$")

CommandRunner = Callable[[Sequence[str], Optional[Path]], Optional[str]]


def _run_command(command: Sequence[str], cwd: Optional[Path] = None) -> Optional[str]:
    """Run a small local introspection command and return stripped stdout."""

    try:
        result = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _safe_text(value: object, *, limit: int = 200) -> Optional[str]:
    """Normalize one-line allowlisted descriptive values."""

    if value is None:
        return None
    normalized = " ".join(str(value).split())
    if not normalized:
        return None
    return normalized[:limit]


def _positive_int(value: Optional[str]) -> Optional[int]:
    try:
        parsed = int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and parsed >= 0 else None


def _linux_cpuinfo() -> Optional[str]:
    try:
        return Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _cpu_name(system: str, machine: str, run_command: CommandRunner) -> Optional[str]:
    if system == "Darwin":
        return (
            _safe_text(run_command(("sysctl", "-n", "machdep.cpu.brand_string"), None))
            or _safe_text(platform.processor())
            or _safe_text(machine)
        )
    if system == "Linux":
        cpuinfo = _linux_cpuinfo()
        if cpuinfo:
            for line in cpuinfo.splitlines():
                if line.lower().startswith(("model name", "hardware", "processor")) and ":" in line:
                    value = _safe_text(line.split(":", 1)[1])
                    if value:
                        return value
    return _safe_text(platform.processor()) or _safe_text(machine)


def _physical_core_count(system: str, run_command: CommandRunner) -> Optional[int]:
    if system == "Darwin":
        return _positive_int(run_command(("sysctl", "-n", "hw.physicalcpu"), None))
    if system == "Linux":
        cpuinfo = _linux_cpuinfo()
        if cpuinfo:
            physical_id: Optional[str] = None
            core_id: Optional[str] = None
            cores: set[tuple[str, str]] = set()
            for line in cpuinfo.splitlines() + [""]:
                if not line.strip():
                    if physical_id is not None and core_id is not None:
                        cores.add((physical_id, core_id))
                    physical_id = core_id = None
                elif ":" in line:
                    key, value = (part.strip() for part in line.split(":", 1))
                    if key == "physical id":
                        physical_id = value
                    elif key == "core id":
                        core_id = value
            if cores:
                return len(cores)
    return None


def _total_ram_bytes(system: str, run_command: CommandRunner) -> Optional[int]:
    if system == "Darwin":
        sysctl_value = _positive_int(run_command(("sysctl", "-n", "hw.memsize"), None))
        if sysctl_value is not None:
            return sysctl_value
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    total = page_size * page_count
    return total if total >= 0 else None


def collect_system_info(run_command: CommandRunner = _run_command) -> dict[str, Any]:
    """Collect a narrow allowlist of OS and hardware fields."""

    system = platform.system() or "unknown"
    if system == "Darwin":
        os_version = platform.mac_ver()[0] or platform.release()
    elif system == "Windows":
        os_version = platform.version() or platform.release()
    else:
        os_version = platform.release()
    machine = platform.machine() or "unknown"
    return {
        "os": _safe_text(system),
        "os_version": _safe_text(os_version),
        "architecture": _safe_text(machine),
        "cpu_name": _cpu_name(system, machine, run_command),
        "physical_cores": _physical_core_count(system, run_command),
        "logical_cores": os.cpu_count(),
        "total_ram_bytes": _total_ram_bytes(system, run_command),
    }


def collect_python_info() -> dict[str, Optional[str]]:
    """Collect Python details without exposing an executable path."""

    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable_name": Path(sys.executable).name or None,
    }


def collect_thread_environment(environ: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Read only approved thread variables and retain only numeric values."""

    result: dict[str, dict[str, Any]] = {}
    for name in THREAD_ENVIRONMENT_NAMES:
        raw_value = environ.get(name)
        if raw_value is None:
            result[name] = {"configured": False, "value": None}
        elif _SAFE_THREAD_VALUE.fullmatch(raw_value):
            result[name] = {"configured": True, "value": raw_value}
        else:
            result[name] = {"configured": True, "value": "redacted_invalid_value"}
    return result


def collect_packages(
    version_lookup: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, dict[str, Any]]:
    """Inspect selected package distributions without importing them."""

    result: dict[str, dict[str, Any]] = {}
    for output_name, distribution_name in PACKAGE_DISTRIBUTIONS:
        try:
            version = version_lookup(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            result[output_name] = {"installed": False, "version": None}
        except Exception:
            # Broken optional metadata must not make environment collection fail.
            result[output_name] = {"installed": False, "version": None}
        else:
            safe_version = version if _SAFE_VERSION_VALUE.fullmatch(str(version)) else "redacted"
            result[output_name] = {"installed": True, "version": safe_version}
    return result


def collect_pytorch_runtime(
    torch_package: Mapping[str, Any],
    import_module: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    """Probe accelerator availability only when the torch package is installed."""

    base = {
        "installed": bool(torch_package.get("installed")),
        "version": torch_package.get("version"),
        "importable": False,
        "mps_available": None,
        "cuda_available": None,
    }
    if not base["installed"]:
        return base
    try:
        torch = import_module("torch")
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        cuda_backend = getattr(torch, "cuda", None)
        base["importable"] = True
        base["mps_available"] = bool(mps_backend and mps_backend.is_available())
        base["cuda_available"] = bool(cuda_backend and cuda_backend.is_available())
    except Exception:
        # Do not serialize exception messages: they can contain local paths.
        pass
    return base


def collect_git_info(
    cwd: Optional[Path] = None,
    run_command: CommandRunner = _run_command,
) -> dict[str, Any]:
    """Collect repository state without paths, remotes, authors, or messages."""

    git_version = run_command(("git", "--version"), cwd)
    result = {
        "available": git_version is not None,
        "repository": False,
        "commit": None,
        "branch": None,
        "clean": None,
    }
    if git_version is None:
        return result
    if run_command(("git", "rev-parse", "--is-inside-work-tree"), cwd) != "true":
        return result

    result["repository"] = True
    commit = run_command(("git", "rev-parse", "HEAD"), cwd)
    branch = run_command(("git", "branch", "--show-current"), cwd)
    status = run_command(("git", "status", "--porcelain", "--untracked-files=normal"), cwd)
    result["commit"] = commit.lower() if commit and _SAFE_GIT_COMMIT.fullmatch(commit) else None
    result["branch"] = branch if branch and _SAFE_GIT_BRANCH.fullmatch(branch) else None
    result["clean"] = status == "" if status is not None else None
    return result


def _utc_timestamp(now: Optional[datetime]) -> str:
    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.isoformat(timespec="seconds").replace("+00:00", "Z")


def collect_environment(
    *,
    now: Optional[datetime] = None,
    environ: Optional[Mapping[str, str]] = None,
    cwd: Optional[Path] = None,
    system_collector: Callable[[], dict[str, Any]] = collect_system_info,
    python_collector: Callable[[], dict[str, Any]] = collect_python_info,
    package_collector: Callable[[], dict[str, dict[str, Any]]] = collect_packages,
    pytorch_collector: Callable[[Mapping[str, Any]], dict[str, Any]] = collect_pytorch_runtime,
    git_collector: Optional[Callable[[Optional[Path]], dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Build a schema-versioned environment document from mockable collectors."""

    packages = package_collector()
    git_fn = git_collector or collect_git_info
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_timestamp(now),
        "system": system_collector(),
        "python": python_collector(),
        "thread_environment": collect_thread_environment(environ if environ is not None else os.environ),
        "packages": packages,
        "pytorch": pytorch_collector(packages["torch"]),
        "git": git_fn(cwd),
    }


def serialize_environment(document: Mapping[str, Any], *, pretty: bool = False) -> str:
    """Serialize deterministically for stable artifacts and tests."""

    if pretty:
        return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    return json.dumps(document, separators=(",", ":"), sort_keys=True, ensure_ascii=True) + "\n"


def write_environment_file(path: Path, serialized: str) -> None:
    """Create parent directories and a new file, refusing any overwrite."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
