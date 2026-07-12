from __future__ import annotations

import json
import unittest

from urbansound_segment_task.edge_v2.demo.presenter import (
    CONFIDENCE_NOTICE,
    DEPLOYMENT_NOTICE,
    OFFICIAL_METRICS_CONTEXT,
    empty_presentation,
    present_result,
    presentation_outputs,
)


def result_fixture(outcome: str = "success") -> dict:
    clip = {
        "class_id": 3, "class_name": "dog_bark", "confidence": 0.885,
        "segment_count": 2,
        "top3": [
            {"class_id": 3, "class_name": "dog_bark", "confidence": 0.885},
            {"class_id": 2, "class_name": "children_playing", "confidence": 0.08},
            {"class_id": 6, "class_name": "gun_shot", "confidence": 0.035},
        ],
    }
    return {
        "status": {
            "outcome": outcome,
            "reason": "zero_segments_under_legacy_policy" if outcome == "no_prediction" else None,
        },
        "clip_prediction": clip if outcome == "success" else None,
        "segment_predictions": (
            [
                {"start_sample": 0, "class_name": "dog_bark", "confidence": 0.9},
                {"start_sample": 7680, "class_name": "children_playing", "confidence": 0.6},
            ] if outcome == "success" else []
        ),
        "audio_metadata": {
            "original_sample_rate": 48_000, "original_channel_count": 1,
            "original_duration_seconds": 1.5,
        },
        "preprocessing": {"resampled_sample_count": 24_000},
        "segmentation": {
            "target_sample_rate": 16_000, "window_samples": 15_360,
            "segment_count": 2 if outcome == "success" else 0,
            "dropped_tail_samples": 960, "policy_id": "yamnet-legacy-full-window-v1",
        },
        "timing": {
            "audio_decode_ns": 1_000_000, "resampling_and_preprocessing_ns": 2_000_000,
            "segment_inference_total_ns": 3_000_000, "aggregation_ns": 4_000,
            "end_to_end_ns": 7_000_000,
        },
        "classifier_artifact": {"onnx_sha256": "o" * 64},
        "yamnet_artifact": {"tree_sha256": "y" * 64},
        "providers": ["CPUExecutionProvider"],
        "runtime": {
            "tensorflow_version": "2.15.1", "onnxruntime_version": "1.20.1",
            "model_load_counts": {"yamnet": 1, "onnx_classifier": 1},
        },
    }


class DemoPresenterTests(unittest.TestCase):
    def test_top3_order_confidence_notice_and_json_round_trip(self) -> None:
        presentation = present_result(result_fixture(), {"startup_end_to_end_ns": 10_000_000})
        self.assertEqual([row[1] for row in presentation["top3"]], ["dog bark", "children playing", "gun shot"])
        self.assertIn("88.5%", presentation["prediction"])
        self.assertIn(CONFIDENCE_NOTICE, presentation["prediction"])
        json.loads(json.dumps(presentation))

    def test_segment_timeline_start_and_end_seconds(self) -> None:
        timeline = present_result(result_fixture(), {})["timeline"]
        self.assertEqual(timeline[0][:2], [0.0, 0.96])
        self.assertEqual(timeline[1][:2], [0.48, 1.44])

    def test_short_clip_no_prediction_presentation(self) -> None:
        presentation = present_result(result_fixture("no_prediction"), {})
        self.assertEqual(
            presentation["prediction"],
            "No prediction — audio is shorter than one 0.96 s segment.",
        )
        self.assertEqual(presentation["timeline"], [])

    def test_deployment_and_official_metric_context_are_exact(self) -> None:
        technical = present_result(result_fixture(), {})["technical"]
        self.assertIn(DEPLOYMENT_NOTICE, technical)
        self.assertIn(OFFICIAL_METRICS_CONTEXT, technical)
        self.assertIn("0.789331 ± 0.033921", technical)
        self.assertIn("0.780485 ± 0.036271", technical)

    def test_absolute_path_is_not_presented(self) -> None:
        result = result_fixture()
        result["audio_identity"] = {"unsafe_path": "/Users/private/audio.wav"}
        serialized = json.dumps(present_result(result, {}), sort_keys=True)
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("private", serialized)

    def test_empty_and_output_order(self) -> None:
        empty = empty_presentation("Select audio")
        outputs = presentation_outputs(empty)
        self.assertEqual(outputs[0], "Select audio")
        self.assertEqual(len(outputs), 7)


if __name__ == "__main__":
    unittest.main()

