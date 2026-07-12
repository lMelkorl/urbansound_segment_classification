from __future__ import annotations

import unittest

from urbansound_segment_task.edge_v2.evaluation.segmentation import segment_plan


class LegacySegmentationTests(unittest.TestCase):
    def test_exact_window_and_hop_counts(self) -> None:
        self.assertEqual(segment_plan(15_360)["segment_start_samples"], [0])
        self.assertEqual(segment_plan(23_040)["segment_start_samples"], [0, 7_680])

    def test_short_clip_emits_zero_segments(self) -> None:
        plan = segment_plan(15_359)
        self.assertEqual(plan["segment_count"], 0)
        self.assertTrue(plan["short_clip_zero_segment"])
        self.assertEqual(plan["dropped_tail_samples"], 15_359)

    def test_tail_is_dropped_and_reported(self) -> None:
        plan = segment_plan(24_000)
        self.assertEqual(plan["segment_start_samples"], [0, 7_680])
        self.assertEqual(plan["dropped_tail_samples"], 960)


if __name__ == "__main__":
    unittest.main()
