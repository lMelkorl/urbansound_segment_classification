"""Versioned JSON schema helpers for single-callable benchmark results."""

from __future__ import annotations

import json
import os
import platform
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .timer import PERCENTILE_METHOD, TIMER_SOURCE, TimingStatistics


SCHEMA_VERSION = "edge-v2.benchmark.v1"
_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SAFE_ERROR_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_ABSOLUTE_PATH = re.compile(r"(^|[\s(])(?:/|[A-Za-z]:[\\/])")
_SAFE_ENVIRONMENT_FIELDS = ("os", "os_version", "architecture", "python_version")


def utc_timestamp(now: Optional[datetime] = None) -> str:
    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def safe_environment_summary() -> dict[str, Optional[str]]:
    """Return a path-free standard-library-only environment summary."""

    return {
        "os": platform.system() or None,
        "os_version": platform.mac_ver()[0] or platform.release() or None,
        "architecture": platform.machine() or None,
        "python_version": platform.python_version() or None,
    }


def validate_benchmark_metadata(name: str, description: str) -> None:
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError("benchmark name must use lowercase safe-name characters")
    if not description or len(description) > 200 or "\n" in description or "\r" in description:
        raise ValueError("description must be a non-empty single line of at most 200 characters")
    if _ABSOLUTE_PATH.search(description) or "file://" in description.lower():
        raise ValueError("description cannot contain an absolute local path")


def environment_section(environment: Mapping[str, Any]) -> dict[str, Optional[str]]:
    """Reduce injected environment data to a fixed, path-free allowlist."""

    result: dict[str, Optional[str]] = {}
    for field in _SAFE_ENVIRONMENT_FIELDS:
        raw_value = environment.get(field)
        value = str(raw_value) if raw_value is not None else None
        if (
            value is None
            or not value
            or len(value) > 100
            or "\n" in value
            or "\r" in value
            or _ABSOLUTE_PATH.search(value)
            or "file://" in value.lower()
        ):
            result[field] = None
        else:
            result[field] = value
    return result


def timing_section(statistics: Optional[TimingStatistics]) -> dict[str, Any]:
    if statistics is None:
        return {
            "unit": "nanoseconds",
            "timer_source": TIMER_SOURCE,
            "percentile_method": PERCENTILE_METHOD,
            "raw_samples": [],
            "p50": None,
            "p95": None,
            "p99": None,
            "mean": None,
            "minimum": None,
            "maximum": None,
            "standard_deviation": None,
        }
    return {
        "unit": "nanoseconds",
        "timer_source": TIMER_SOURCE,
        "percentile_method": PERCENTILE_METHOD,
        "raw_samples": list(statistics.raw_samples_ns),
        "p50": statistics.p50,
        "p95": statistics.p95,
        "p99": statistics.p99,
        "mean": statistics.mean,
        "minimum": statistics.minimum,
        "maximum": statistics.maximum,
        "standard_deviation": statistics.standard_deviation,
    }


def build_document(
    *,
    name: str,
    description: str,
    warmup: int,
    iterations: int,
    items_per_call: int,
    item_duration_seconds: Optional[float],
    statistics: Optional[TimingStatistics],
    throughput: Mapping[str, Optional[float]],
    status: Mapping[str, Any],
    environment: Mapping[str, Any],
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    validate_benchmark_metadata(name, description)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(now),
        "benchmark": {
            "name": name,
            "description": description,
            "warmup_runs": warmup,
            "measured_iterations": iterations,
            "items_per_call": items_per_call,
            "item_duration_seconds": item_duration_seconds,
        },
        "timing": timing_section(statistics),
        "throughput": dict(throughput),
        "status": dict(status),
        "environment": environment_section(environment),
    }


def safe_error(error_type: str, stage: str, iteration_index: int) -> dict[str, Any]:
    normalized_type = error_type if _SAFE_ERROR_TYPE.fullmatch(error_type) else "CallableError"
    return {
        "type": normalized_type,
        "stage": stage,
        "iteration_index": iteration_index,
        "iteration_index_base": 0,
        "message": "benchmark callable did not complete",
    }


def serialize_document(document: Mapping[str, Any], *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    return json.dumps(document, separators=(",", ":"), sort_keys=True, ensure_ascii=True) + "\n"


def write_document_atomic(path: Path, serialized: str) -> None:
    """Atomically publish a new file and refuse to replace an existing one."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=".benchmark-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        if os.name == "nt":
            os.rename(temporary_name, destination)
        else:
            os.link(temporary_name, destination)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
