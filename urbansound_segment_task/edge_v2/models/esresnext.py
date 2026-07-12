"""Static ESResNeXt-fbsp readiness inspection."""

from __future__ import annotations

from .base import (
    ArtifactSpec,
    DependencySpec,
    InspectorProtocol,
    cpu_support,
    issue,
    method_status,
)


def inspect_esresnext(inspector: InspectorProtocol) -> dict:
    dependencies = [
        inspector.dependency(spec)
        for spec in (
            DependencySpec("torch", "torch", notes="Imported by both legacy Goal 2 scripts but absent from legacy requirements.txt."),
            DependencySpec("torchaudio", "torchaudio", required=False, notes="Not imported by the current Goal 2 scripts; retained as an optional upstream compatibility check."),
            DependencySpec("librosa", "librosa"),
            DependencySpec("soundfile", "soundfile"),
            DependencySpec("numpy", "numpy"),
            DependencySpec("pandas", "pandas"),
        )
    ]
    source_artifact = inspector.artifact(
        ArtifactSpec(
            "esresnext_source",
            "external/ESResNeXt_fbsp",
            "directory",
            notes="Expected external ESResNeXt-fbsp source checkout.",
        )
    )
    gitmodules_artifact = inspector.artifact(
        ArtifactSpec(
            "gitmodules",
            ".gitmodules",
            "file",
            notes="Required to map the recorded gitlink to an upstream URL.",
        )
    )
    checkpoint_artifact = inspector.artifact(
        ArtifactSpec(
            "esresnext_audioset_checkpoint",
            "urbansound_segment_task/goals/goal2_esresnext/checkpoints/ESResNeXtFBSP_AudioSet.pt",
            "file",
            notes="Expected AudioSet checkpoint; never deserialized during preflight.",
        )
    )
    download_script = inspector.artifact(
        ArtifactSpec(
            "checkpoint_download_script",
            "urbansound_segment_task/scripts/download_checkpoint_goal2.sh",
            "file",
            required=False,
            notes="Legacy acquisition script is inspected only as text.",
        )
    )
    artifacts = [source_artifact, gitmodules_artifact, checkpoint_artifact, download_script]

    fine_path = "urbansound_segment_task/goals/goal2_esresnext/run_finetune.py"
    head_path = "urbansound_segment_task/goals/goal2_esresnext/run_head_only.py"
    fine_source = inspector.read_text(fine_path) or ""
    head_source = inspector.read_text(head_path) or ""
    requirements = inspector.read_text("requirements.txt") or ""
    checkpoint_script = inspector.read_text(
        "urbansound_segment_task/scripts/download_checkpoint_goal2.sh"
    ) or ""
    gitmodules_text = inspector.read_text(".gitmodules") or ""

    entry_points_exist = bool(fine_source) and bool(head_source)
    cpu_path_detected = all(
        'device = "cuda" if torch.cuda.is_available() else "cpu"' in source
        for source in (fine_source, head_source)
    )
    underscore_path = "external/ESResNeXt_fbsp" in fine_source
    hyphen_path = "external/ESResNeXt-fbsp" in head_source
    singular_import = "from model." in fine_source
    plural_import = "from models." in head_source
    path_import_inconsistent = (underscore_path and hyphen_path) or (singular_import and plural_import)
    gitlink = inspector.gitlink_info("external/ESResNeXt_fbsp")
    gitmodules_mapping_present = (
        bool(gitmodules_text)
        and "external/ESResNeXt_fbsp" in gitmodules_text
        and "url" in gitmodules_text
    )
    expected_source_candidates = (
        "external/ESResNeXt_fbsp/model/esresnet_fbsp.py",
        "external/ESResNeXt_fbsp/model/esresnext.py",
        "external/ESResNeXt-fbsp/models/esresnext_fbsp.py",
        "external/ESResNeXt-fbsp/models/esresnext.py",
    )
    external_source_files_present = any(inspector.exists(path) for path in expected_source_candidates)
    checksum_verification_detected = any(
        token in checkpoint_script.lower() for token in ("sha256", "shasum", "checksum")
    )
    torch_declared_in_requirements = any(
        line.strip().split("=", 1)[0].strip().lower() in ("torch", "pytorch")
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )

    issues = []
    if gitlink["is_gitlink"] and not gitmodules_mapping_present:
        source_artifact["provenance_status"] = "broken_reference"
        issues.append(
            issue(
                "BROKEN_GITLINK_MAPPING",
                "blocker",
                "external/ESResNeXt_fbsp is a gitlink but .gitmodules does not provide a usable mapping.",
            )
        )
    if not external_source_files_present:
        issues.append(
            issue(
                "EXTERNAL_SOURCE_UNAVAILABLE",
                "blocker",
                "Expected ESResNeXt implementation files are unavailable in the external source directories.",
            )
        )
    if path_import_inconsistent:
        issues.append(
            issue(
                "EXTERNAL_PATH_IMPORT_INCONSISTENCY",
                "blocker",
                "Head-only and fine-tune scripts use inconsistent external directory and import namespaces.",
            )
        )
    if not torch_declared_in_requirements:
        issues.append(
            issue(
                "TORCH_UNDECLARED",
                "blocker",
                "PyTorch is imported by Goal 2 but is not declared in legacy requirements.txt.",
            )
        )
    if checkpoint_script and not checksum_verification_detected:
        issues.append(
            issue(
                "CHECKPOINT_CHECKSUM_MISSING",
                "blocker",
                "The legacy checkpoint download script does not verify a checksum.",
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
    if not checkpoint_artifact["exists"]:
        issues.append(issue("CHECKPOINT_MISSING", "blocker", "The required AudioSet checkpoint is missing."))
    if not entry_points_exist:
        issues.append(issue("IMPLEMENTATION_MISSING", "blocker", "One or more Goal 2 entry points are missing."))

    required_actions = []
    if gitlink["is_gitlink"] and not gitmodules_mapping_present or not external_source_files_present:
        required_actions.append(
            "Repair and review ESResNeXt source provenance, URL, immutable revision, and local checkout before inference."
        )
    if path_import_inconsistent:
        required_actions.append(
            "Define one reviewed external directory and import contract for the future adapter without rewriting legacy scripts."
        )
    if not checkpoint_artifact["exists"] or not checksum_verification_detected:
        required_actions.append(
            "Provide the checkpoint through an approved manual acquisition step with license, size, and SHA-256 verification."
        )
    if missing_dependencies or not torch_declared_in_requirements:
        required_actions.append(
            "Define and install a reviewed ESResNeXt benchmark dependency group in a project virtual environment."
        )

    return {
        "method_id": "esresnext",
        "display_name": "ESResNeXt-fbsp",
        "implementation_status": "implemented" if entry_points_exist else "partial",
        "benchmark_readiness": "blocked",
        "cpu_support": cpu_support(
            declared=True,
            code_path_detected=cpu_path_detected,
            notes="Both legacy scripts contain a CUDA-or-CPU device fallback; no model import or runtime verification was performed.",
        ),
        "dependencies": dependencies,
        "artifacts": artifacts,
        "source_code": {
            "entry_points": [fine_path, head_path],
            "entry_points_exist": entry_points_exist,
            "gitlink_detected": gitlink["is_gitlink"],
            "expected_source_revision": gitlink["revision"],
            "gitmodules_mapping_present": gitmodules_mapping_present,
            "external_source_files_present": external_source_files_present,
            "path_import_inconsistent": path_import_inconsistent,
            "torch_declared_in_legacy_requirements": torch_declared_in_requirements,
            "checkpoint_checksum_verification_detected": checksum_verification_detected,
            "static_checks_only": True,
        },
        "issues": issues,
        "required_actions": required_actions,
        "status": method_status(),
    }
