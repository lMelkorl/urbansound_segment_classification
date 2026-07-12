"""Allowlisted, temporary thread environment policy."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import Any, Iterator, MutableMapping, Optional


THREAD_ENV_ALLOWLIST = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_SAFE_VALUE = re.compile(r"^[0-9]{1,5}$")


def validate_thread_count(thread_count: int) -> None:
    if isinstance(thread_count, bool) or not isinstance(thread_count, int) or thread_count < 1:
        raise ValueError("thread_count must be a positive integer")


def safe_thread_snapshot(environ: MutableMapping[str, str]) -> dict[str, dict[str, Any]]:
    """Capture only allowlisted names and numeric values."""

    result: dict[str, dict[str, Any]] = {}
    for name in THREAD_ENV_ALLOWLIST:
        value = environ.get(name)
        result[name] = {
            "configured": value is not None,
            "value": (
                value
                if value is not None and _SAFE_VALUE.fullmatch(value)
                else "redacted_invalid_value" if value is not None else None
            ),
        }
    return result


@contextmanager
def temporary_thread_policy(
    thread_count: int,
    environ: Optional[MutableMapping[str, str]] = None,
) -> Iterator[dict[str, Any]]:
    """Apply thread variables for a scope and restore the exact prior state."""

    validate_thread_count(thread_count)
    target = environ if environ is not None else os.environ
    missing = object()
    previous_raw: dict[str, object] = {
        name: target.get(name, missing) for name in THREAD_ENV_ALLOWLIST
    }
    previous_safe = safe_thread_snapshot(target)
    requested_value = str(thread_count)
    for name in THREAD_ENV_ALLOWLIST:
        target[name] = requested_value
    try:
        yield {
            "requested_thread_count": thread_count,
            "effective_thread_environment": {
                name: target[name] for name in THREAD_ENV_ALLOWLIST
            },
            "previous_thread_environment": previous_safe,
        }
    finally:
        for name, old_value in previous_raw.items():
            if old_value is missing:
                target.pop(name, None)
            else:
                target[name] = str(old_value)

