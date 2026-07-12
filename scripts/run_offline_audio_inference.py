#!/usr/bin/env python3
"""Run one local WAV through verified YAMNet and the FP32 ONNX classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.runtime.pipeline import run_offline_audio_inference  # noqa: E402
from urbansound_segment_task.edge_v2.runtime.result_schema import write_inference_result  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Offline local-WAV YAMNet + FP32 ONNX inference smoke.")
    value.add_argument("--audio", type=Path, required=True)
    value.add_argument("--yamnet-artifact", type=Path, required=True)
    value.add_argument("--classifier-artifact", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--stable-clip-key")
    value.add_argument("--ground-truth-class-id", type=int)
    value.add_argument("--threads", type=int, default=1)
    value.add_argument("--pretty", action="store_true")
    return value


def main() -> int:
    arguments = parser().parse_args()
    if arguments.output.exists():
        parser().error("output already exists; refusing to overwrite")
    document = run_offline_audio_inference(
        audio_path=arguments.audio,
        yamnet_artifact=arguments.yamnet_artifact,
        classifier_artifact=arguments.classifier_artifact,
        stable_clip_key=arguments.stable_clip_key,
        ground_truth_class_id=arguments.ground_truth_class_id,
        threads=arguments.threads,
    )
    write_inference_result(arguments.output, document, pretty=arguments.pretty)
    return 0 if document["status"]["outcome"] in ("success", "no_prediction") else 1


if __name__ == "__main__":
    raise SystemExit(main())

