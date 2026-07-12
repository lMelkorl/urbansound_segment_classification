from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from urbansound_segment_task.edge_v2.models.lightgbm_legacy import (
    LEGACY_DEFAULT_SEED, balanced_class_weights, legacy_training_config,
    validate_legacy_source,
)


ROOT = Path(__file__).resolve().parents[2]


class LightgbmLegacyConfigTests(unittest.TestCase):
    def test_exact_explicit_legacy_config_and_defaults(self) -> None:
        config = legacy_training_config(threads=8)
        self.assertEqual(
            config["explicit_legacy_parameters"],
            {"n_estimators":700,"num_leaves":64,"learning_rate":0.05,"subsample":0.9,"colsample_bytree":0.9,"n_jobs":-1,"random_state":42},
        )
        self.assertEqual(config["library_defaults"]["subsample_freq"], 0)
        self.assertFalse(config["early_stopping"])
        self.assertFalse(config["validation_used_during_fit"])
        self.assertFalse(config["seed_deviation_from_legacy"])

    def test_seed_override_is_recorded_as_protocol_deviation(self) -> None:
        self.assertTrue(legacy_training_config(seed=7, threads=8)["seed_deviation_from_legacy"])
        self.assertEqual(LEGACY_DEFAULT_SEED, 42)

    def test_class_weights_use_only_provided_training_labels(self) -> None:
        weights = balanced_class_weights(np.asarray([0, 0, 0, 1], dtype=np.int64))
        self.assertAlmostEqual(weights[0], 4 / 6)
        self.assertAlmostEqual(weights[1], 2.0)
        self.assertNotIn(2, weights)

    def test_legacy_source_matches_extracted_configuration(self) -> None:
        digest = validate_legacy_source(ROOT)
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
