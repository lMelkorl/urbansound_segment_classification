from __future__ import annotations

import unittest

from urbansound_segment_task.edge_v2.benchmarks.thread_policy import (
    THREAD_ENV_ALLOWLIST,
    temporary_thread_policy,
    validate_thread_count,
)


class BenchmarkThreadPolicyTests(unittest.TestCase):
    def test_positive_thread_count_is_applied_to_allowlist(self) -> None:
        environ = {"UNRELATED": "keep"}
        with temporary_thread_policy(4, environ) as policy:
            self.assertEqual(
                policy["effective_thread_environment"],
                {name: "4" for name in THREAD_ENV_ALLOWLIST},
            )
            self.assertTrue(all(environ[name] == "4" for name in THREAD_ENV_ALLOWLIST))
            self.assertEqual(environ["UNRELATED"], "keep")

    def test_environment_is_restored_exactly_after_context(self) -> None:
        environ = {"OMP_NUM_THREADS": "2", "UNRELATED": "keep"}
        original = dict(environ)

        with temporary_thread_policy(3, environ):
            self.assertEqual(environ["OMP_NUM_THREADS"], "3")

        self.assertEqual(environ, original)

    def test_non_allowlisted_values_are_never_changed(self) -> None:
        environ = {"SECRET_TOKEN": "private", "ANOTHER": "value"}
        with temporary_thread_policy(2, environ):
            self.assertEqual(environ["SECRET_TOKEN"], "private")
            self.assertEqual(environ["ANOTHER"], "value")

    def test_invalid_thread_counts_are_rejected(self) -> None:
        for invalid in (0, -1, True, 1.5, "4"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_thread_count(invalid)


if __name__ == "__main__":
    unittest.main()
