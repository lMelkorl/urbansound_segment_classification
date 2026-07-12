"""Platform-aware process-lifetime peak RSS helpers."""

from __future__ import annotations

import platform
from typing import Any, Callable, Mapping, Optional

try:
    import resource
except ImportError:  # pragma: no cover - exercised through injected unsupported platform tests
    resource = None  # type: ignore[assignment]


def normalize_peak_rss_bytes(raw_value: int, platform_name: str) -> Optional[int]:
    if raw_value < 0:
        return None
    if platform_name == "Darwin":
        return int(raw_value)
    if platform_name == "Linux":
        return int(raw_value) * 1024
    return None


def read_peak_rss(
    *,
    platform_name: Optional[str] = None,
    usage_reader: Optional[Callable[[], int]] = None,
) -> dict[str, Any]:
    """Read and normalize ``ru_maxrss`` without estimating unsupported systems."""

    current_platform = platform_name or platform.system()
    if current_platform not in ("Darwin", "Linux"):
        return {
            "supported": False,
            "source": None,
            "unit": "bytes",
            "peak_rss_bytes": None,
            "reason": "peak RSS is unsupported on this platform by the standard-library collector",
        }
    if usage_reader is None:
        if resource is None:
            return {
                "supported": False,
                "source": None,
                "unit": "bytes",
                "peak_rss_bytes": None,
                "reason": "resource module is unavailable",
            }
        usage_reader = lambda: int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    try:
        raw_value = int(usage_reader())
        normalized = normalize_peak_rss_bytes(raw_value, current_platform)
    except (OSError, TypeError, ValueError):
        normalized = None
    if normalized is None:
        return {
            "supported": False,
            "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "unit": "bytes",
            "peak_rss_bytes": None,
            "reason": "peak RSS reading could not be normalized",
        }
    limitation = (
        "Darwin ru_maxrss is a process-lifetime high-water mark reported in bytes."
        if current_platform == "Darwin"
        else "Linux ru_maxrss is a process-lifetime high-water mark reported in KiB and normalized to bytes."
    )
    return {
        "supported": True,
        "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "unit": "bytes",
        "peak_rss_bytes": normalized,
        "reason": None,
        "platform_limitation": limitation,
    }


def summarize_peak_rss(
    baseline: Mapping[str, Any],
    final: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe approximate incremental process high-water memory."""

    if not baseline.get("supported") or not final.get("supported"):
        reason = final.get("reason") or baseline.get("reason") or "peak RSS unavailable"
        return {
            "supported": False,
            "source": final.get("source") or baseline.get("source"),
            "unit": "bytes",
            "baseline_peak_rss_bytes": None,
            "final_peak_rss_bytes": None,
            "approximate_incremental_peak_rss_bytes": None,
            "reason": reason,
            "platform_limitation": (
                "Memory unavailability does not invalidate successful latency measurements."
            ),
        }
    baseline_bytes = int(baseline["peak_rss_bytes"])
    final_bytes = int(final["peak_rss_bytes"])
    return {
        "supported": True,
        "source": final.get("source"),
        "unit": "bytes",
        "baseline_peak_rss_bytes": baseline_bytes,
        "final_peak_rss_bytes": final_bytes,
        "approximate_incremental_peak_rss_bytes": final_bytes - baseline_bytes,
        "reason": None,
        "platform_limitation": (
            str(final.get("platform_limitation"))
            + " Incremental RSS is the difference between two process high-water marks, not exact model allocation."
        ),
    }

