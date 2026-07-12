"""Static YAMNet plus LightGBM readiness inspection."""

from __future__ import annotations

from .base import (
    ArtifactSpec,
    DependencySpec,
    InspectorProtocol,
    cpu_support,
    issue,
    method_status,
)


def inspect_yamnet_lgbm(inspector: InspectorProtocol) -> dict:
    dependencies = [
        inspector.dependency(spec)
        for spec in (
            DependencySpec("tensorflow", "tensorflow"),
            DependencySpec("tensorflow-hub", "tensorflow_hub"),
            DependencySpec("lightgbm", "lightgbm"),
            DependencySpec("librosa", "librosa"),
            DependencySpec("soundfile", "soundfile"),
            DependencySpec("numpy", "numpy"),
            DependencySpec("pandas", "pandas"),
            DependencySpec("scikit-learn", "sklearn"),
        )
    ]
    artifacts = [
        inspector.artifact(
            ArtifactSpec(
                "yamnet_model_directory",
                "artifacts/yamnet/tfhub-v1/model",
                "directory",
                notes="Expected immutable local YAMNet model directory.",
            )
        ),
        inspector.artifact(
            ArtifactSpec(
                "yamnet_manifest",
                "artifacts/yamnet/tfhub-v1/artifact-manifest.json",
                "file",
                notes="Expected provenance manifest containing source, license, and checksum.",
            )
        ),
        inspector.artifact(
            ArtifactSpec(
                "lightgbm_model",
                "artifacts/models/yamnet_lgbm/model.txt",
                "file",
                notes="A trained classifier artifact is required for inference-only benchmarking.",
            )
        ),
    ]
    entry_path = "urbansound_segment_task/goals/goal1_yamnet_lgbm/run.py"
    source = inspector.read_text(entry_path) or ""
    entry_exists = bool(source)
    runtime_url_detected = "https://tfhub.dev/google/yamnet/1" in source and "hub.load" in source
    code_path_detected = entry_exists and "LGBMClassifier" in source and "tensorflow" in source

    issues = []
    if runtime_url_detected:
        issues.append(
            issue(
                "YAMNET_RUNTIME_DOWNLOAD",
                "blocker",
                "Legacy code loads YAMNet from a remote TF Hub URL at runtime; offline benchmarks require a reviewed local immutable artifact.",
            )
        )
    missing_dependencies = [item["package_name"] for item in dependencies if item["required"] and not item["installed"]]
    if missing_dependencies:
        issues.append(
            issue(
                "MISSING_DEPENDENCIES",
                "blocker",
                "Required dependency group is incomplete: " + ", ".join(missing_dependencies) + ".",
            )
        )
    missing_artifacts = [item["artifact_id"] for item in artifacts if item["required"] and not item["exists"]]
    if missing_artifacts:
        issues.append(
            issue(
                "MISSING_LOCAL_ARTIFACTS",
                "blocker",
                "Required local artifacts are missing: " + ", ".join(missing_artifacts) + ".",
            )
        )
    if not entry_exists:
        issues.append(issue("IMPLEMENTATION_MISSING", "blocker", "Legacy YAMNet implementation entry point is missing."))

    required_actions = []
    if runtime_url_detected or any(item["artifact_id"].startswith("yamnet_") and not item["exists"] for item in artifacts):
        required_actions.append(
            "Create a reviewed local YAMNet artifact acquisition step with URL, license, immutable identity, and SHA-256."
        )
    if not artifacts[-1]["exists"]:
        required_actions.append(
            "Provide a traceable LightGBM model artifact or approve a separate reproducible training step before inference benchmarking."
        )
    if missing_dependencies:
        required_actions.append(
            "Install the approved YAMNet benchmark dependency group in a project virtual environment."
        )

    blockers = any(item["severity"] == "blocker" for item in issues)
    return {
        "method_id": "yamnet_lgbm",
        "display_name": "YAMNet embeddings + LightGBM",
        "implementation_status": "implemented" if entry_exists else "missing",
        "benchmark_readiness": "blocked" if blockers else "ready",
        "cpu_support": cpu_support(
            declared=True,
            code_path_detected=code_path_detected,
            notes="TensorFlow/YAMNet and LightGBM have a CPU candidate path, but this preflight does not import or execute them.",
        ),
        "dependencies": dependencies,
        "artifacts": artifacts,
        "source_code": {
            "entry_points": [entry_path],
            "entry_point_exists": entry_exists,
            "runtime_url_detected": runtime_url_detected,
            "runtime_urls": ["https://tfhub.dev/google/yamnet/1"] if runtime_url_detected else [],
            "static_checks_only": True,
        },
        "issues": issues,
        "required_actions": required_actions,
        "status": method_status(),
    }
