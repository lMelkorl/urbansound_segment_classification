from __future__ import annotations

import unittest

from urbansound_segment_task.edge_v2.benchmarks.yamnet_cpu import (
    median_repetition_summary,
    select_thread_configuration,
)


def configuration(threads: int, throughput: float, p95: float, rss: float = 100.0):
    return {
        "thread_count": threads,
        "status": "success",
        "median_legacy_feature_pipeline_segments_per_second": throughput,
        "median_legacy_pipeline_p95_ns": p95,
        "median_incremental_peak_rss_bytes": rss,
    }


class YamnetThreadSelectionTests(unittest.TestCase):
    def test_repetition_summary_uses_median(self) -> None:
        values = []
        for number in (10.0, 30.0, 20.0):
            values.append(
                {
                    "steady_state_model_p50_ns": number,
                    "steady_state_model_p95_ns": number + 1,
                    "model_inference_segments_per_second": number + 2,
                    "legacy_pipeline_p50_ns": number + 3,
                    "legacy_pipeline_p95_ns": number + 4,
                    "legacy_feature_pipeline_segments_per_second": number + 5,
                    "real_time_factor": number + 6,
                    "incremental_peak_rss_bytes": number + 7,
                }
            )

        summary = median_repetition_summary(values)

        self.assertEqual(summary["repetition_count"], 3)
        self.assertEqual(summary["median_steady_state_model_p50_ns"], 20.0)
        self.assertEqual(
            summary["median_legacy_feature_pipeline_segments_per_second"], 25.0
        )

    def test_more_than_three_percent_prefers_throughput(self) -> None:
        selected = select_thread_configuration(
            [configuration(2, 96.0, 1.0), configuration(4, 100.0, 10.0)]
        )
        self.assertEqual(selected["thread_count"], 4)

    def test_within_three_percent_prefers_lower_p95(self) -> None:
        selected = select_thread_configuration(
            [configuration(2, 97.0, 5.0), configuration(4, 100.0, 10.0)]
        )
        self.assertEqual(selected["thread_count"], 2)

    def test_equal_p95_prefers_lower_thread_count(self) -> None:
        selected = select_thread_configuration(
            [configuration(4, 100.0, 5.0), configuration(2, 99.0, 5.0)]
        )
        self.assertEqual(selected["thread_count"], 2)

    def test_selection_is_deterministic_across_input_order(self) -> None:
        candidates = [configuration(8, 100.0, 7.0), configuration(4, 99.0, 6.0)]
        self.assertEqual(
            select_thread_configuration(candidates)["thread_count"],
            select_thread_configuration(list(reversed(candidates)))["thread_count"],
        )


if __name__ == "__main__":
    unittest.main()
