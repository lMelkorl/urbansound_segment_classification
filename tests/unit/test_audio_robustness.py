from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from urbansound_segment_task.edge_v2.data.manifest import document_sha256
from urbansound_segment_task.edge_v2.evaluation.audio_robustness import (
    FoldModel, _class_analysis, _load_resume_unit, resolve_fold_models, unit_identity,
)
from urbansound_segment_task.edge_v2.evaluation.robustness import (
    CLASS_NAMES, CONDITIONS, PERTURBATION_CONTRACTS, SCHEMA_VERSION,
    aggregate_conditions, apply_perturbation, clip_probability_mean,
    degradation_classification, deterministic_seed, fold_condition_metrics,
    measured_snr_db, require_clean_agreement, select_panel,
)
from urbansound_segment_task.edge_v2.evaluation.segmentation import segment_plan


ROOT = Path(__file__).resolve().parents[2]
MODEL_RUN = ROOT / "results" / "compact_classifier_cross_fold" / "fixed-baselines-v1"


def _metadata_rows(per_group: int = 2):
    rows = []
    for fold in range(1, 11):
        for class_id, class_name in enumerate(CLASS_NAMES):
            for index in range(per_group):
                rows.append({
                    "clip_key": f"fold{fold}/{fold}-{class_id}-{index}.wav",
                    "fold": fold, "class_id": class_id, "class_name": class_name,
                })
    return rows


def _clip(key: str, class_id: int, predicted: int, confidence: float = 0.8):
    return {
        "clip_key": key, "class_id": class_id, "predicted_class_id": predicted,
        "confidence": confidence, "segment_count": 2,
    }


def _model() -> FoldModel:
    return FoldModel(
        test_fold=1, validation_fold=2, weights_path=Path("model.weights.h5"),
        weights_relative_path="model.weights.h5", weights_sha256="a" * 64,
        weights_size_bytes=123, fold_identity_sha256="b" * 64,
        config_sha256="c" * 64, split_manifest_sha256="d" * 64,
    )


class PanelTests(unittest.TestCase):
    def test_deterministic_stratified_panel_coverage_and_no_duplicates(self):
        rows = _metadata_rows(3)
        evaluable = {row["clip_key"] for row in rows if not row["clip_key"].endswith("-0.wav")}
        first = select_panel(
            rows, evaluable_clip_keys=evaluable, clips_per_class_per_fold=2,
            dataset_manifest_sha256="1" * 64, cache_identity="2" * 64,
        )
        second = select_panel(
            rows, evaluable_clip_keys=evaluable, clips_per_class_per_fold=2,
            dataset_manifest_sha256="1" * 64, cache_identity="2" * 64,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["clip_count"], 200)
        self.assertEqual(len({row["clip_key"] for row in first["clips"]}), 200)
        self.assertEqual({row["fold"] for row in first["clips"]}, set(range(1, 11)))
        self.assertEqual({row["class_id"] for row in first["clips"]}, set(range(10)))
        self.assertEqual(
            first["panel_identity_sha256"],
            document_sha256(first, excluded_fields=("panel_identity_sha256",)),
        )

    def test_panel_shortfall_is_reported(self):
        rows = _metadata_rows(1)
        panel = select_panel(
            rows, evaluable_clip_keys={row["clip_key"] for row in rows},
            clips_per_class_per_fold=2, dataset_manifest_sha256="1" * 64,
            cache_identity="2" * 64,
        )
        self.assertTrue(all(row["shortfall"] == 1 for row in panel["coverage"]))

    def test_duplicate_metadata_clip_is_rejected(self):
        rows = _metadata_rows(1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_panel(
                rows + [dict(rows[0])], evaluable_clip_keys={row["clip_key"] for row in rows},
                clips_per_class_per_fold=1, dataset_manifest_sha256="1" * 64,
                cache_identity="2" * 64,
            )

    def test_panel_serialization_has_no_absolute_path_or_user(self):
        rows = _metadata_rows(1)
        panel = select_panel(
            rows, evaluable_clip_keys={row["clip_key"] for row in rows},
            clips_per_class_per_fold=1, dataset_manifest_sha256="1" * 64,
            cache_identity="2" * 64,
        )
        serialized = json.dumps(panel, sort_keys=True)
        self.assertNotIn(str(Path.home()), serialized)
        self.assertNotIn("dataset_root", serialized)
        self.assertNotIn("username", serialized)


class PerturbationTests(unittest.TestCase):
    def setUp(self):
        time = np.arange(32_000, dtype=np.float32) / np.float32(16_000)
        self.waveform = (0.15 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)

    def test_white_noise_seed_and_output_are_deterministic(self):
        self.assertEqual(
            deterministic_seed("fold1/a.wav", "white_noise_snr_10db"),
            deterministic_seed("fold1/a.wav", "white_noise_snr_10db"),
        )
        first, _ = apply_perturbation(
            self.waveform, clip_key="fold1/a.wav", condition="white_noise_snr_10db"
        )
        second, _ = apply_perturbation(
            self.waveform, clip_key="fold1/a.wav", condition="white_noise_snr_10db"
        )
        np.testing.assert_array_equal(first, second)

    def test_rms_snr_targets(self):
        for condition, expected in (
            ("white_noise_snr_20db", 20.0), ("white_noise_snr_10db", 10.0),
            ("white_noise_snr_0db", 0.0),
        ):
            output, metadata = apply_perturbation(
                self.waveform, clip_key="fold1/a.wav", condition=condition
            )
            self.assertAlmostEqual(measured_snr_db(self.waveform, output), expected, places=4)
            self.assertAlmostEqual(metadata["measured_snr_db"], expected, places=4)

    def test_silent_waveform_stays_finite_and_is_special_cased(self):
        output, metadata = apply_perturbation(
            np.zeros(15_360, dtype=np.float32), clip_key="fold1/silent.wav",
            condition="white_noise_snr_0db",
        )
        self.assertTrue(metadata["silent_input"])
        self.assertIsNone(metadata["measured_snr_db"])
        self.assertTrue(np.all(np.isfinite(output)))

    def test_gain_minus_12db_exact_factor(self):
        output, metadata = apply_perturbation(
            self.waveform, clip_key="fold1/a.wav", condition="gain_minus_12db"
        )
        expected = 10.0 ** (-12.0 / 20.0)
        np.testing.assert_allclose(output, self.waveform * expected, rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(metadata["gain_factor"], expected)

    def test_roundtrip_preserves_length_and_segment_starts(self):
        def fake_resample(values, *, orig_sr, target_sr, res_type):
            length = round(len(values) * target_sr / orig_sr)
            return np.linspace(values[0], values[-1], length, dtype=np.float32)

        output, metadata = apply_perturbation(
            self.waveform, clip_key="fold1/a.wav", condition="bandlimit_8khz_roundtrip",
            resample=fake_resample,
        )
        self.assertEqual(output.shape, self.waveform.shape)
        self.assertTrue(metadata["segment_start_samples_unchanged"])
        self.assertEqual(segment_plan(len(output))["segment_start_samples"], segment_plan(len(self.waveform))["segment_start_samples"])

    def test_nonfinite_input_is_rejected_and_output_is_float32(self):
        with self.assertRaises(ValueError):
            apply_perturbation(
                np.array([np.nan], dtype=np.float32), clip_key="fold1/a.wav", condition="clean"
            )
        output, metadata = apply_perturbation(
            np.array([2.0], dtype=np.float32), clip_key="fold1/a.wav", condition="clean"
        )
        self.assertEqual(output.dtype, np.float32)
        self.assertEqual(float(output[0]), 2.0)
        self.assertFalse(metadata["hard_clipping_applied"])


class MetricTests(unittest.TestCase):
    def test_arithmetic_probability_mean(self):
        values = np.array([[0.9, 0.1] + [0.0] * 8, [0.1, 0.9] + [0.0] * 8])
        np.testing.assert_allclose(clip_probability_mean(values), values.mean(axis=0))

    def test_macro_f1_uses_fixed_ten_classes(self):
        metrics = fold_condition_metrics([_clip("a", 0, 0)])
        self.assertEqual(metrics["class_order"], list(range(10)))
        self.assertAlmostEqual(metrics["macro_f1"], 0.1)

    def test_flip_rate_and_confidence_change(self):
        clean = [_clip("a", 0, 0, 0.8), _clip("b", 1, 1, 0.7)]
        perturbed = [_clip("a", 0, 1, 0.6), _clip("b", 1, 1, 0.75)]
        metrics = fold_condition_metrics(perturbed, clean_clips=clean)
        self.assertEqual(metrics["prediction_flip_rate"], 0.5)
        self.assertAlmostEqual(metrics["confidence_change_mean"], -0.075)

    def test_fold_mean_population_std_and_drops(self):
        def row(condition, accuracy, f1, flip=0.0):
            return {"condition": condition, "metrics": {
                "accuracy": accuracy, "macro_f1": f1, "prediction_flip_rate": flip,
            }}
        aggregate = aggregate_conditions([
            row("clean", 0.8, 0.8), row("clean", 1.0, 1.0),
            row("white_noise_snr_0db", 0.5, 0.6, 0.4),
            row("white_noise_snr_0db", 0.7, 0.6, 0.2),
        ])
        self.assertAlmostEqual(aggregate["clean"]["clip_accuracy_mean"], 0.9)
        self.assertAlmostEqual(aggregate["clean"]["clip_accuracy_population_standard_deviation"], 0.1)
        noisy = aggregate["white_noise_snr_0db"]
        self.assertAlmostEqual(noisy["absolute_accuracy_drop"], 0.3)
        self.assertAlmostEqual(noisy["absolute_macro_f1_drop"], 0.3)
        self.assertAlmostEqual(noisy["prediction_flip_rate_mean"], 0.3)

    def test_degradation_thresholds_are_fixed(self):
        self.assertEqual(degradation_classification(0.025), "minor")
        self.assertEqual(degradation_classification(0.0250001), "moderate")
        self.assertEqual(degradation_classification(0.075), "moderate")
        self.assertEqual(degradation_classification(0.0750001), "major")

    def test_class_analysis_ranks_drop_flip_and_variability_with_support(self):
        rows = [
            {"class_id": index, "class_name": CLASS_NAMES[index], "support": index + 1,
             "f1_mean": 0.8 - index / 100, "absolute_f1_drop": index / 100,
             "prediction_flip_rate": index / 20,
             "f1_population_standard_deviation": index / 200}
            for index in range(10)
        ]
        analysis = _class_analysis({"white_noise_snr_0db": rows})["white_noise_snr_0db"]
        self.assertEqual(analysis["most_robust_three_by_smallest_f1_drop"][0]["class_id"], 0)
        self.assertEqual(analysis["largest_f1_loss_three"][0]["class_id"], 9)
        self.assertEqual(analysis["highest_prediction_flip_three"][0]["class_id"], 9)
        self.assertEqual(analysis["most_variable_across_folds_three"][0]["support"], 10)

    def test_clean_agreement_is_strictly_one_hundred_percent(self):
        self.assertEqual(require_clean_agreement([0, 1], [0, 1])["top1_agreement"], 1.0)
        with self.assertRaisesRegex(ValueError, "100%"):
            require_clean_agreement([0, 1], [0, 2])


class ProvenanceAndResumeTests(unittest.TestCase):
    def test_correct_fold_model_is_bound_to_held_out_fold(self):
        _, models = resolve_fold_models(MODEL_RUN, (1, 10))
        self.assertEqual((models[1].test_fold, models[1].validation_fold), (1, 2))
        self.assertEqual((models[10].test_fold, models[10].validation_fold), (10, 1))
        self.assertTrue(models[1].weights_path.name.endswith("weights.h5"))

    def test_deployment_only_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifact-manifest.json").write_text(
                json.dumps({"deployment_only": True}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "deployment-only"):
                resolve_fold_models(root, (1,))

    def test_unit_identity_changes_with_contract_or_model_hash(self):
        first = unit_identity(
            panel_identity="p", condition="clean", model=_model(),
            dataset_manifest_sha256="d", yamnet_tree_sha256="y",
        )
        changed = FoldModel(**{**_model().__dict__, "weights_sha256": "e" * 64})
        second = unit_identity(
            panel_identity="p", condition="clean", model=changed,
            dataset_manifest_sha256="d", yamnet_tree_sha256="y",
        )
        self.assertNotEqual(first, second)

    def test_valid_resume_unit_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.json"
            document = {
                "unit_identity_sha256": "u", "status": {"outcome": "success"},
                "created_at_utc": "ignored", "duration_seconds": 1.0,
            }
            document["result_sha256"] = document_sha256(
                document, excluded_fields=("created_at_utc", "duration_seconds", "result_sha256")
            )
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(_load_resume_unit(path, "u"), document)

    def test_resume_identity_mismatch_is_rejected_and_corrupt_is_rerunnable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.json"
            path.write_text(json.dumps({"unit_identity_sha256": "old"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                _load_resume_unit(path, "new")
            path.write_text("not json", encoding="utf-8")
            self.assertIsNone(_load_resume_unit(path, "new"))

    def test_schema_and_condition_contract_are_fixed(self):
        self.assertEqual(SCHEMA_VERSION, "edge-v2.audio-robustness-cross-fold.v1")
        self.assertEqual(tuple(PERTURBATION_CONTRACTS), CONDITIONS)


class CliTests(unittest.TestCase):
    def test_help_smoke(self):
        completed = subprocess.run(
            [str(ROOT / ".venv-edge-runtime" / "bin" / "python"),
             str(ROOT / "scripts" / "run_audio_robustness.py"), "--help"],
            cwd=ROOT, check=False, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for option in (
            "--dataset-root", "--model-run", "--yamnet-artifact",
            "--clips-per-class-per-fold", "--conditions", "--fold", "--output-dir",
            "--resume", "--pretty",
        ):
            self.assertIn(option, completed.stdout)


if __name__ == "__main__":
    unittest.main()
