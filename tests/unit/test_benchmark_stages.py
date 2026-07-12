from __future__ import annotations

import unittest

from urbansound_segment_task.edge_v2.benchmarks.stages import (
    PipelineStages,
    build_synthetic_pipeline,
    execute_pipeline,
)


class BenchmarkStagesTests(unittest.TestCase):
    def test_stages_run_in_order_and_use_previous_outputs(self) -> None:
        calls: list[str] = []

        def decode(source: str) -> str:
            self.assertEqual(source, "source")
            calls.append("decode")
            return "decoded"

        def preprocess(decoded: str) -> str:
            self.assertEqual(decoded, "decoded")
            calls.append("preprocess")
            return "processed"

        def inference(processed: str) -> str:
            self.assertEqual(processed, "processed")
            calls.append("inference")
            return "prediction"

        def aggregate(prediction: str) -> str:
            self.assertEqual(prediction, "prediction")
            calls.append("aggregate")
            return "result"

        result = execute_pipeline(
            PipelineStages(decode, preprocess, inference, aggregate),
            "source",
        )

        self.assertEqual(result, "result")
        self.assertEqual(calls, ["decode", "preprocess", "inference", "aggregate"])

    def test_synthetic_pipeline_is_deterministic_and_data_dependent(self) -> None:
        stages, source = build_synthetic_pipeline(16)

        first = execute_pipeline(stages, source)
        second = execute_pipeline(stages, source)
        smaller = execute_pipeline(stages, 8)

        self.assertEqual(first, second)
        self.assertNotEqual(first, smaller)
        self.assertEqual(len(stages.decode(source)), 16)

    def test_invalid_input_size_is_rejected(self) -> None:
        for invalid in (0, -1, True, 1_000_001):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    build_synthetic_pipeline(invalid)


if __name__ == "__main__":
    unittest.main()
