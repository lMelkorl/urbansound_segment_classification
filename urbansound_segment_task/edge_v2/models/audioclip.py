"""Static AudioCLIP readiness inspection."""

from __future__ import annotations

from .base import (
    ArtifactSpec,
    DependencySpec,
    InspectorProtocol,
    cpu_support,
    issue,
    method_status,
)


def inspect_audioclip(inspector: InspectorProtocol) -> dict:
    dependencies = [
        inspector.dependency(
            DependencySpec(
                "torch",
                "torch",
                notes="Expected by the README-described AudioCLIP architecture, but the implementation is absent.",
            )
        )
    ]
    artifacts = [
        inspector.artifact(
            ArtifactSpec(
                "audioclip_external_source",
                "external/AudioCLIP",
                "directory",
                notes="Expected external AudioCLIP source checkout.",
            )
        ),
        inspector.artifact(
            ArtifactSpec(
                "audioclip_checkpoint",
                "artifacts/models/audioclip/audio_model.pt",
                "file",
                notes="Expected immutable local model weights.",
            )
        ),
        inspector.artifact(
            ArtifactSpec(
                "audioclip_metrics_test",
                "urbansound_segment_task/goals/goal3_audioclip/results_ft_q/metrics_test.json",
                "file",
                required=False,
                notes="README-referenced legacy result artifact.",
            )
        ),
        inspector.artifact(
            ArtifactSpec(
                "audioclip_metrics_throughput",
                "urbansound_segment_task/goals/goal3_audioclip/results_ft_q/metrics_throughput.json",
                "file",
                required=False,
                notes="README-referenced legacy throughput artifact.",
            )
        ),
        inspector.artifact(
            ArtifactSpec(
                "audioclip_run_args",
                "urbansound_segment_task/goals/goal3_audioclip/results_ft_q/run_args.json",
                "file",
                required=False,
                notes="README-referenced legacy run configuration.",
            )
        ),
    ]
    entry_path = "urbansound_segment_task/goals/goal3_audioclip/run.py"
    source = inspector.read_text(entry_path) or ""
    entry_exists = bool(source)
    placeholder = entry_exists and (
        "placeholder" in source.lower() or "ileride doldurulacak" in source.lower()
    )
    missing_results = [
        item["artifact_id"]
        for item in artifacts
        if item["artifact_id"].startswith("audioclip_metrics") or item["artifact_id"] == "audioclip_run_args"
        if not item["exists"]
    ]
    issues = [
        issue("IMPLEMENTATION_PLACEHOLDER", "blocker", "Legacy AudioCLIP run.py is only a placeholder."),
        issue("EXTERNAL_SOURCE_MISSING", "blocker", "Expected AudioCLIP external source is unavailable."),
        issue("CHECKPOINT_IDENTITY_MISSING", "blocker", "No local AudioCLIP checkpoint identity or SHA-256 is available."),
    ]
    if missing_results:
        issues.append(
            issue(
                "README_RESULT_ARTIFACTS_MISSING",
                "blocker",
                "README-referenced artifacts are missing: " + ", ".join(missing_results) + ".",
            )
        )
    required_actions = [
        "Recover or explicitly retire the missing AudioCLIP implementation and its immutable source revision.",
        "Recover traceable checkpoint and README result provenance before any benchmark attempt.",
    ]
    return {
        "method_id": "audioclip",
        "display_name": "AudioCLIP audio branch",
        "implementation_status": "placeholder" if placeholder else "missing",
        "benchmark_readiness": "unavailable",
        "cpu_support": cpu_support(
            declared=True,
            code_path_detected=False,
            notes="README text describes a fallback, but the repository has no implementation from which CPU compatibility can be verified.",
        ),
        "dependencies": dependencies,
        "artifacts": artifacts,
        "source_code": {
            "entry_points": [entry_path],
            "entry_point_exists": entry_exists,
            "placeholder_detected": placeholder,
            "external_source_exists": artifacts[0]["exists"],
            "missing_readme_artifacts": missing_results,
            "static_checks_only": True,
        },
        "issues": issues,
        "required_actions": required_actions,
        "status": method_status(),
    }
