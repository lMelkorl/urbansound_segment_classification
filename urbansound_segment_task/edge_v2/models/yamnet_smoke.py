"""Schema generation for one-shot local YAMNet contract smoke tests."""

from __future__ import annotations

import importlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from urbansound_segment_task.edge_v2.benchmarks.schema import utc_timestamp

from .yamnet_runtime import LoadedYamnet, load_local_yamnet, validate_yamnet_outputs


YAMNET_SMOKE_SCHEMA_VERSION = "edge-v2.yamnet-smoke.v1"
INPUT_CASES = (
    (15_360, 0.960),
    (15_600, 0.975),
    (16_000, 1.000),
)


def _lightgbm_smoke(import_module: Callable[[str], Any]) -> dict:
    try:
        lightgbm = import_module("lightgbm")
    except Exception as exc:
        return {
            "importable": False,
            "version": None,
            "cpu_available": False,
            "model_artifact_present": False,
            "error_type": type(exc).__name__,
        }
    return {
        "importable": True,
        "version": getattr(lightgbm, "__version__", None),
        "cpu_available": True,
        "model_artifact_present": False,
        "error_type": None,
    }


def run_yamnet_smoke(
    artifact_directory: Path,
    *,
    runtime_loader: Callable[[Path], LoadedYamnet] = load_local_yamnet,
    import_module: Callable[[str], Any] = importlib.import_module,
    now: Optional[datetime] = None,
) -> dict:
    created_at = utc_timestamp(now)
    try:
        loaded = runtime_loader(Path(artifact_directory))
    except Exception as exc:
        return {
            "schema_version": YAMNET_SMOKE_SCHEMA_VERSION,
            "created_at_utc": created_at,
            "artifact_identity": None,
            "artifact_tree_sha256": None,
            "tensorflow_version": None,
            "loader": None,
            "visible_devices": [],
            "input_cases": [],
            "lightgbm": _lightgbm_smoke(import_module),
            "status": {
                "outcome": "failure",
                "error": {"stage": "load", "type": type(exc).__name__},
            },
        }
    cases = []
    for sample_count, duration_seconds in INPUT_CASES:
        try:
            waveform = loaded.tensorflow.zeros([sample_count], dtype=loaded.tensorflow.float32)
            outputs = loaded.model(waveform)
            contract = validate_yamnet_outputs(outputs)
            cases.append(
                {
                    "sample_count": sample_count,
                    "duration_seconds": duration_seconds,
                    "success": True,
                    **contract,
                    "error_type": None,
                }
            )
        except Exception as exc:
            cases.append(
                {
                    "sample_count": sample_count,
                    "duration_seconds": duration_seconds,
                    "success": False,
                    "scores_shape": None,
                    "embeddings_shape": None,
                    "spectrogram_shape": None,
                    "frame_count": None,
                    "error_type": type(exc).__name__,
                }
            )
    all_cases_succeeded = all(case["success"] for case in cases)
    return {
        "schema_version": YAMNET_SMOKE_SCHEMA_VERSION,
        "created_at_utc": created_at,
        "artifact_identity": loaded.artifact_identity["artifact_id"],
        "artifact_tree_sha256": loaded.artifact_identity["tree_sha256"],
        "tensorflow_version": loaded.tensorflow_version,
        "loader": loaded.loader_method,
        "visible_devices": list(loaded.visible_devices),
        "input_cases": cases,
        "lightgbm": _lightgbm_smoke(import_module),
        "status": {
            "outcome": "success" if all_cases_succeeded else "failure",
            "error": None if all_cases_succeeded else {"type": "InputContractFailure"},
        },
    }
