from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from urbansound_segment_task.edge_v2.benchmarks.stages import execute_pipeline
from urbansound_segment_task.edge_v2.benchmarks.yamnet_cpu import (
    YamnetRunConfig,
    real_time_factor,
    run_yamnet_in_fresh_process,
)
from urbansound_segment_task.edge_v2.models.yamnet_benchmark import (
    aggregate_outputs,
    build_yamnet_pipeline,
    generate_pcm16_input,
    input_identity,
    load_benchmark_runtime,
    materialized_inference,
    WarmupTimingRecorder,
)


class FakeTensor:
    def __init__(self, values: np.ndarray, events: list[str] | None = None) -> None:
        self.values = values
        self.shape = values.shape
        self.events = events

    def numpy(self) -> np.ndarray:
        if self.events is not None:
            self.events.append("materialize")
        return self.values


def fake_outputs(events: list[str] | None = None):
    return (
        FakeTensor(np.ones((1, 521), dtype=np.float32), events),
        FakeTensor(np.arange(1024, dtype=np.float32).reshape(1, 1024), events),
        FakeTensor(np.ones((96, 64), dtype=np.float32), events),
    )


class YamnetCpuOperationTests(unittest.TestCase):
    def test_deterministic_non_silent_input_and_hash(self) -> None:
        first = generate_pcm16_input(15_360)
        second = generate_pcm16_input(15_360)

        self.assertEqual(first, second)
        self.assertNotEqual(first, bytes(len(first)))
        identity = input_identity(first, 15_360)
        self.assertEqual(identity["sha256"], hashlib.sha256(first).hexdigest())
        self.assertEqual(identity["duration_seconds"], 0.960)

    def test_sample_length_metadata(self) -> None:
        for count, duration in ((15_360, 0.960), (15_600, 0.975), (16_000, 1.0)):
            with self.subTest(count=count):
                pcm = generate_pcm16_input(count)
                identity = input_identity(pcm, count)
                self.assertEqual(identity["sample_count"], count)
                self.assertEqual(identity["duration_seconds"], duration)

    def test_inference_materializes_all_outputs_inside_call(self) -> None:
        events: list[str] = []

        def model(waveform):
            events.append("model")
            return fake_outputs(events)

        result = materialized_inference(model, np.zeros(15_360, dtype=np.float32))

        self.assertEqual(events, ["model", "materialize", "materialize", "materialize"])
        self.assertEqual(result["contract"]["frame_count"], 1)

    def test_aggregate_is_embedding_mean_with_1024_features(self) -> None:
        result = materialized_inference(lambda _: fake_outputs(), np.zeros(1, dtype=np.float32))
        aggregate = aggregate_outputs(result, np)

        self.assertEqual(aggregate["feature_size"], 1024)
        np.testing.assert_array_equal(aggregate["feature_vector"], np.arange(1024, dtype=np.float32))

    def test_pipeline_end_to_end_calls_all_stages_and_excludes_preprocess_from_inference(self) -> None:
        events: list[str] = []

        def model(waveform):
            events.append("inference")
            self.assertEqual(waveform.dtype, np.float32)
            return fake_outputs()

        stages = build_yamnet_pipeline(model, 32, np)
        wrapped = type(stages)(
            lambda value: events.append("decode") or stages.decode(value),
            lambda value: events.append("preprocess") or stages.preprocess(value),
            stages.inference,
            lambda value: events.append("aggregate") or stages.aggregate(value),
        )
        result = execute_pipeline(wrapped, generate_pcm16_input(32))

        self.assertEqual(events, ["decode", "preprocess", "inference", "aggregate"])
        self.assertEqual(result["feature_size"], 1024)

    def test_real_time_factor(self) -> None:
        self.assertAlmostEqual(real_time_factor(10.0, 0.960), 9.6)

    def test_warmup_recorder_excludes_first_and_steady_calls(self) -> None:
        clock = iter([100, 110, 200, 230])
        recorder = WarmupTimingRecorder(
            lambda model, value: value + 1,
            2,
            clock_ns=lambda: next(clock),
        )

        results = [recorder(None, value) for value in range(5)]

        self.assertEqual(results, [1, 2, 3, 4, 5])
        self.assertEqual(recorder.raw_samples_ns, [10, 30])
        self.assertEqual(recorder.call_count, 5)

    def test_artifact_is_verified_before_tensorflow_import_and_threads_before_load(self) -> None:
        events: list[str] = []

        class Threading:
            intra = 0
            inter = 0

            @classmethod
            def set_intra_op_parallelism_threads(cls, value):
                events.append("set_intra")
                cls.intra = value

            @classmethod
            def set_inter_op_parallelism_threads(cls, value):
                events.append("set_inter")
                cls.inter = value

            @classmethod
            def get_intra_op_parallelism_threads(cls):
                return cls.intra

            @classmethod
            def get_inter_op_parallelism_threads(cls):
                return cls.inter

        class SavedModel:
            @staticmethod
            def load(path):
                events.append("load")
                return lambda waveform: fake_outputs()

        fake_tf = type(
            "TF",
            (),
            {
                "__version__": "test",
                "saved_model": SavedModel,
                "config": type(
                    "Config",
                    (),
                    {
                        "threading": Threading,
                        "get_visible_devices": staticmethod(lambda: []),
                    },
                ),
            },
        )

        def importer(name):
            events.append("import_" + name)
            return fake_tf if name == "tensorflow" else np

        identity = {
            "artifact_id": "yamnet-tfhub-v1",
            "tree_sha256": "a" * 64,
            "model_relative_path": "model",
        }
        with patch(
            "urbansound_segment_task.edge_v2.models.yamnet_benchmark.verify_yamnet_artifact",
            side_effect=lambda path: events.append("verify") or identity,
        ):
            loaded = load_benchmark_runtime(Path("safe-artifact"), 4, import_module=importer)

        self.assertEqual(events[:5], ["verify", "import_tensorflow", "set_intra", "set_inter", "import_numpy"])
        self.assertLess(events.index("set_inter"), events.index("load"))
        self.assertEqual(loaded.effective_intra_op_threads, 4)

    def test_thread_configuration_failure_is_not_silent(self) -> None:
        class BadThreading:
            set_intra_op_parallelism_threads = staticmethod(lambda value: None)
            set_inter_op_parallelism_threads = staticmethod(lambda value: None)
            get_intra_op_parallelism_threads = staticmethod(lambda: 0)
            get_inter_op_parallelism_threads = staticmethod(lambda: 0)

        fake_tf = type("TF", (), {"config": type("C", (), {"threading": BadThreading})})
        identity = {"model_relative_path": "model"}
        with patch(
            "urbansound_segment_task.edge_v2.models.yamnet_benchmark.verify_yamnet_artifact",
            return_value=identity,
        ):
            with self.assertRaises(RuntimeError):
                load_benchmark_runtime(
                    Path("safe-artifact"),
                    4,
                    import_module=lambda name: fake_tf if name == "tensorflow" else np,
                )

    def test_invalid_artifact_child_returns_structured_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = run_yamnet_in_fresh_process(
                YamnetRunConfig(directory, 1, 0, 1, 15_360), timeout_seconds=10
            )

        self.assertEqual(document["status"]["outcome"], "failure")
        self.assertIn(document["status"]["error"]["type"], {"ArtifactVerificationError", "ChildProcessCrash"})
        self.assertNotIn(directory, json.dumps(document))

    def test_method_specific_child_timeout_is_structured(self) -> None:
        class Receive:
            def poll(self, timeout):
                return False

            def close(self):
                pass

        class Send:
            def close(self):
                pass

        class Process:
            exitcode = None

            def start(self):
                pass

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return True

            def terminate(self):
                self.exitcode = -15

        class Context:
            def Pipe(self, duplex=False):
                return Receive(), Send()

            def Process(self, target, args):
                return Process()

        with patch(
            "urbansound_segment_task.edge_v2.benchmarks.yamnet_cpu.multiprocessing.get_context",
            return_value=Context(),
        ):
            document = run_yamnet_in_fresh_process(
                YamnetRunConfig("safe-artifact", 1, 0, 1, 15_360), timeout_seconds=0.001
            )

        self.assertEqual(document["status"]["outcome"], "failure")
        self.assertEqual(document["status"]["error"]["type"], "ChildProcessTimeout")

    def test_method_specific_child_crash_is_distinct(self) -> None:
        class Receive:
            calls = 0

            def poll(self, timeout):
                return True

            def recv(self):
                self.calls += 1
                if self.calls == 1:
                    return {"kind": "started"}
                raise EOFError

            def close(self):
                pass

        class Send:
            def close(self):
                pass

        class Process:
            exitcode = -9

            def start(self):
                pass

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return False

            def terminate(self):
                pass

        class Context:
            def Pipe(self, duplex=False):
                return Receive(), Send()

            def Process(self, target, args):
                return Process()

        with patch(
            "urbansound_segment_task.edge_v2.benchmarks.yamnet_cpu.multiprocessing.get_context",
            return_value=Context(),
        ):
            document = run_yamnet_in_fresh_process(
                YamnetRunConfig("safe-artifact", 1, 0, 1, 15_360), timeout_seconds=1
            )

        self.assertEqual(document["status"]["error"]["type"], "ChildProcessCrash")


if __name__ == "__main__":
    unittest.main()
