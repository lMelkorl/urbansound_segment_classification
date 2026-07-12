"""Fresh-spawn lifecycle execution with safe child-process reporting."""

from __future__ import annotations

import math
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing.connection import Connection
from typing import Any, Callable, Mapping, Optional

from .lifecycle import LifecycleRequest, run_lifecycle
from .memory import read_peak_rss, summarize_peak_rss
from .schema import (
    environment_section,
    safe_environment_summary,
    timing_section,
    utc_timestamp,
    validate_benchmark_metadata,
)
from .thread_policy import THREAD_ENV_ALLOWLIST, temporary_thread_policy, validate_thread_count


LIFECYCLE_SCHEMA_VERSION = "edge-v2.lifecycle-benchmark.v1"
_ALLOWED_WORKLOADS = ("synthetic", "test_busy", "test_crash", "test_nonzero")


@dataclass(frozen=True)
class LifecycleProcessConfig:
    name: str = "synthetic-lifecycle"
    description: str = "Deterministic standard-library synthetic lifecycle benchmark."
    workload: str = "synthetic"
    threads: int = 1
    warmup: int = 3
    iterations: int = 20
    items_per_call: int = 1
    item_duration_seconds: Optional[float] = None
    load_size: int = 50_000
    work_size: int = 1_000
    test_busy_seconds: float = 1.0

    def validate(self) -> None:
        validate_benchmark_metadata(self.name, self.description)
        validate_thread_count(self.threads)
        LifecycleRequest(
            name=self.name,
            description=self.description,
            warmup=self.warmup,
            iterations=self.iterations,
            items_per_call=self.items_per_call,
            item_duration_seconds=self.item_duration_seconds,
        ).validate()
        if self.workload not in _ALLOWED_WORKLOADS:
            raise ValueError("workload is not in the predefined safe registry")
        for field_name, value, limit in (
            ("load_size", self.load_size, 5_000_000),
            ("work_size", self.work_size, 1_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > limit:
                raise ValueError(field_name + " is outside the supported positive range")
        if (
            not isinstance(self.test_busy_seconds, (int, float))
            or isinstance(self.test_busy_seconds, bool)
            or not math.isfinite(self.test_busy_seconds)
            or self.test_busy_seconds <= 0
            or self.test_busy_seconds > 10
        ):
            raise ValueError("test_busy_seconds is outside the safe range")


def _duration_section(duration_ns: Optional[int], success: bool) -> dict[str, Any]:
    return {
        "unit": "nanoseconds",
        "timer_source": "time.perf_counter_ns",
        "duration": duration_ns,
        "measurement": "parent_observed_spawn_to_child_entry",
        "status": {
            "outcome": "success" if success else "failure",
            "error": None if success else {"type": "ProcessStartupUnavailable"},
        },
    }


def _empty_steady(error: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "timing": timing_section(None),
        "throughput": {
            "calls_per_second": None,
            "items_per_second": None,
            "real_time_factor": None,
        },
        "status": {"outcome": "failure", "error": dict(error)},
    }


def _empty_lifecycle(startup_ns: Optional[int], error: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "process_startup_time": _duration_section(startup_ns, startup_ns is not None),
        "load_time": {
            "unit": "nanoseconds",
            "timer_source": "time.perf_counter_ns",
            "duration": None,
            "status": {"outcome": "failure", "error": dict(error)},
        },
        "first_call_latency": {
            "unit": "nanoseconds",
            "timer_source": "time.perf_counter_ns",
            "duration": None,
            "status": {"outcome": "failure", "error": dict(error)},
        },
        "steady_state": _empty_steady(error),
        "status": {"outcome": "failure", "error": dict(error)},
    }


def _unavailable_memory(reason: str) -> dict[str, Any]:
    return {
        "supported": False,
        "source": None,
        "unit": "bytes",
        "baseline_peak_rss_bytes": None,
        "final_peak_rss_bytes": None,
        "approximate_incremental_peak_rss_bytes": None,
        "reason": reason,
        "platform_limitation": "Memory unavailability does not invalidate latency results.",
    }


def _base_execution_policy(config: LifecycleProcessConfig) -> dict[str, Any]:
    return {
        "requested_thread_count": config.threads,
        "effective_thread_environment": {name: None for name in THREAD_ENV_ALLOWLIST},
        "previous_thread_environment": {name: None for name in THREAD_ENV_ALLOWLIST},
        "process_isolation": "fresh_child_process",
        "multiprocessing_start_method": "spawn",
        "gc_manipulation_applied": False,
        "affinity_applied": False,
    }


def _assemble_document(
    config: LifecycleProcessConfig,
    *,
    startup_ns: Optional[int],
    child_payload: Optional[Mapping[str, Any]],
    process_error: Optional[Mapping[str, Any]],
    now: Optional[datetime],
) -> dict[str, Any]:
    if child_payload is None:
        error = dict(process_error or {"stage": "child_process", "type": "ChildResultMissing"})
        lifecycle = _empty_lifecycle(startup_ns, error)
        memory = _unavailable_memory("child process did not return memory data")
        execution_policy = _base_execution_policy(config)
        environment = safe_environment_summary()
        status = {"outcome": "failure", "error": error}
    else:
        lifecycle = dict(child_payload["lifecycle"])
        lifecycle["process_startup_time"] = _duration_section(startup_ns, startup_ns is not None)
        memory = dict(child_payload["memory"])
        execution_policy = dict(child_payload["execution_policy"])
        environment = dict(child_payload["environment"])
        status = dict(child_payload["status"])
    return {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "created_at_utc": utc_timestamp(now),
        "benchmark": {
            "name": config.name,
            "description": config.description,
            "warmup_runs": config.warmup,
            "steady_state_iterations": config.iterations,
            "items_per_call": config.items_per_call,
            "item_duration_seconds": config.item_duration_seconds,
            "workload": {
                "name": config.workload,
                "load_size": config.load_size,
                "work_size": config.work_size,
            },
        },
        "execution_policy": execution_policy,
        "lifecycle": lifecycle,
        "memory": memory,
        "environment": environment_section(environment),
        "status": status,
    }


def _child_entry(connection: Connection, config_data: dict[str, Any]) -> None:
    """Spawn target. Only predefined workloads can reach this function."""

    connection.send({"kind": "started"})
    config = LifecycleProcessConfig(**config_data)
    try:
        with temporary_thread_policy(config.threads) as policy:
            execution_policy = dict(policy)
            execution_policy.update(
                {
                    "process_isolation": "fresh_child_process",
                    "multiprocessing_start_method": "spawn",
                    "gc_manipulation_applied": False,
                    "affinity_applied": False,
                }
            )
            if config.workload == "test_crash":
                os._exit(70)
            if config.workload == "test_nonzero":
                os._exit(7)
            if config.workload == "test_busy":
                deadline = time.perf_counter() + config.test_busy_seconds
                accumulator = 0
                while time.perf_counter() < deadline:
                    accumulator = (accumulator * 33 + 17) & 0xFFFFFFFF
                connection.send({"kind": "unexpected", "value": accumulator})
                return

            # Imported only after the child thread policy has been applied.
            from .synthetic_lifecycle import build_synthetic_lifecycle

            loader, inference, input_data = build_synthetic_lifecycle(
                config.load_size, config.work_size
            )
            baseline = read_peak_rss()
            lifecycle = run_lifecycle(
                loader,
                inference,
                input_data,
                LifecycleRequest(
                    name=config.name,
                    description=config.description,
                    warmup=config.warmup,
                    iterations=config.iterations,
                    items_per_call=config.items_per_call,
                    item_duration_seconds=config.item_duration_seconds,
                ),
            )
            final = read_peak_rss()
            memory = summarize_peak_rss(baseline, final)
            payload = {
                "kind": "result",
                "execution_policy": execution_policy,
                "lifecycle": lifecycle,
                "memory": memory,
                "environment": safe_environment_summary(),
                "status": dict(lifecycle["status"]),
            }
            connection.send(payload)
    except Exception as exc:
        connection.send(
            {
                "kind": "child_failure",
                "error": {
                    "stage": "child_setup",
                    "type": type(exc).__name__,
                    "message": "child process setup did not complete",
                },
            }
        )
    finally:
        connection.close()


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _exit_error_type(config: LifecycleProcessConfig, exit_code: Optional[int]) -> str:
    if config.workload == "test_crash" or (exit_code is not None and exit_code < 0):
        return "ChildProcessCrash"
    return "ChildProcessNonZeroExit"


def run_lifecycle_in_fresh_process(
    config: LifecycleProcessConfig,
    *,
    timeout_seconds: float,
    now: Optional[datetime] = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Run one lifecycle benchmark in a new explicit ``spawn`` process."""

    config.validate()
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive")

    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(target=_child_entry, args=(send_connection, asdict(config)))
    start_ns = int(clock_ns())
    deadline = time.monotonic() + float(timeout_seconds)
    startup_ns: Optional[int] = None
    child_payload: Optional[Mapping[str, Any]] = None
    process_error: Optional[dict[str, Any]] = None
    try:
        process.start()
        send_connection.close()
        if not receive_connection.poll(_remaining(deadline)):
            process_error = {"stage": "process_startup", "type": "ChildProcessTimeout"}
        else:
            try:
                started = receive_connection.recv()
            except EOFError:
                started = None
            if not isinstance(started, dict) or started.get("kind") != "started":
                process.join(timeout=0.2)
                process_error = {
                    "stage": "process_startup",
                    "type": _exit_error_type(config, process.exitcode)
                    if process.exitcode
                    else "ChildProtocolError",
                    "exit_code": process.exitcode,
                }
            else:
                startup_ns = int(clock_ns()) - start_ns
                if not receive_connection.poll(_remaining(deadline)):
                    process_error = {"stage": "child_process", "type": "ChildProcessTimeout"}
                else:
                    try:
                        message = receive_connection.recv()
                    except EOFError:
                        message = None
                    process.join(timeout=_remaining(deadline))
                    if process.is_alive():
                        process_error = {"stage": "child_process", "type": "ChildProcessTimeout"}
                    elif process.exitcode not in (0, None):
                        process_error = {
                            "stage": "child_process",
                            "type": _exit_error_type(config, process.exitcode),
                            "exit_code": process.exitcode,
                        }
                    elif isinstance(message, dict) and message.get("kind") == "result":
                        child_payload = message
                    elif isinstance(message, dict) and message.get("kind") == "child_failure":
                        process_error = dict(message["error"])
                    else:
                        process_error = {"stage": "child_process", "type": "ChildProtocolError"}
    finally:
        receive_connection.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=1.0)

    return _assemble_document(
        config,
        startup_ns=startup_ns,
        child_payload=child_payload,
        process_error=process_error,
        now=now,
    )
