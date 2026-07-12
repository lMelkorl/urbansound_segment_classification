from __future__ import annotations

import unittest

import numpy as np

from urbansound_segment_task.edge_v2.evaluation.aggregation import (
    aggregate_clip_probabilities, align_probability_columns,
)


class ClipAggregationTests(unittest.TestCase):
    def test_probability_columns_follow_model_classes(self) -> None:
        raw = np.asarray([[0.7, 0.2, 0.1]])
        aligned, metadata = align_probability_columns(raw, [2, 0, 1])
        self.assertEqual(aligned[0, :3].tolist(), [0.2, 0.1, 0.7])
        self.assertEqual(metadata["model_classes"], [2, 0, 1])
        self.assertEqual(metadata["missing_model_classes"], list(range(3, 10)))

    def test_clip_aggregation_uses_arithmetic_probability_mean(self) -> None:
        probabilities = np.zeros((3, 10))
        probabilities[0, :2] = [0.9, 0.1]
        probabilities[1, :2] = [0.0, 1.0]
        probabilities[2, :2] = [0.2, 0.8]
        true, predicted, keys = aggregate_clip_probabilities(
            ["clip-a", "clip-a", "clip-b"], [0, 0, 1], probabilities
        )
        self.assertEqual(keys, ["clip-a", "clip-b"])
        self.assertEqual(true.tolist(), [0, 1])
        self.assertEqual(predicted.tolist(), [1, 1])

    def test_inconsistent_ground_truth_within_clip_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ground truth"):
            aggregate_clip_probabilities(["a", "a"], [0, 1], np.ones((2,10)))


if __name__ == "__main__":
    unittest.main()
