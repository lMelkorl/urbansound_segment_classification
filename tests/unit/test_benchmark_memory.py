from __future__ import annotations

import unittest

from urbansound_segment_task.edge_v2.benchmarks.memory import (
    normalize_peak_rss_bytes,
    read_peak_rss,
    summarize_peak_rss,
)


class BenchmarkMemoryTests(unittest.TestCase):
    def test_darwin_peak_rss_is_already_bytes(self) -> None:
        self.assertEqual(normalize_peak_rss_bytes(123_456, "Darwin"), 123_456)

    def test_linux_peak_rss_kib_is_normalized_to_bytes(self) -> None:
        self.assertEqual(normalize_peak_rss_bytes(123, "Linux"), 123 * 1024)

    def test_unsupported_platform_is_structured(self) -> None:
        result = read_peak_rss(platform_name="Windows", usage_reader=lambda: 100)

        self.assertFalse(result["supported"])
        self.assertIsNone(result["peak_rss_bytes"])
        self.assertIn("unsupported", result["reason"])

    def test_incremental_peak_rss_is_high_water_difference(self) -> None:
        baseline = read_peak_rss(platform_name="Darwin", usage_reader=lambda: 1_000)
        final = read_peak_rss(platform_name="Darwin", usage_reader=lambda: 1_600)
        result = summarize_peak_rss(baseline, final)

        self.assertTrue(result["supported"])
        self.assertEqual(result["baseline_peak_rss_bytes"], 1_000)
        self.assertEqual(result["final_peak_rss_bytes"], 1_600)
        self.assertEqual(result["approximate_incremental_peak_rss_bytes"], 600)


if __name__ == "__main__":
    unittest.main()
