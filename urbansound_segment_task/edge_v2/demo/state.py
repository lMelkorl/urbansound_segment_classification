"""Process-local runtime lifecycle and safe demo handlers."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from urbansound_segment_task.edge_v2.runtime.pipeline import (
    AudioDurationLimitError,
    ReusableOfflineAudioRuntime,
)

from .presenter import empty_presentation, present_result


LOGGER = logging.getLogger(__name__)


class DemoRuntimeState:
    def __init__(
        self,
        *,
        yamnet_artifact: Path,
        classifier_artifact: Path,
        max_duration_seconds: float = 30.0,
        runtime_factory: Callable[..., Any] = ReusableOfflineAudioRuntime,
    ) -> None:
        if max_duration_seconds <= 0:
            raise ValueError("maximum duration must be positive")
        self.yamnet_artifact = Path(yamnet_artifact)
        self.classifier_artifact = Path(classifier_artifact)
        self.max_duration_seconds = float(max_duration_seconds)
        self.runtime_factory = runtime_factory
        self.runtime: Optional[Any] = None
        self.ready = False
        self.initialization_error = ""
        self._initialized = False
        self._inference_lock = threading.Lock()

    def initialize(self) -> bool:
        if self._initialized:
            return self.ready
        self._initialized = True
        try:
            self.runtime = self.runtime_factory(
                yamnet_artifact=self.yamnet_artifact,
                classifier_artifact=self.classifier_artifact,
                threads=1,
            )
            status = self.runtime.safe_status()
            if status.get("active_onnx_providers") != ["CPUExecutionProvider"]:
                raise ValueError("CPU provider verification failed")
            self.ready = True
        except Exception as exc:
            LOGGER.error("Demo runtime initialization failed: %s", type(exc).__name__)
            self.initialization_error = "Runtime initialization failed. Verify local model artifacts."
            self.ready = False
        return self.ready

    def status_markdown(self) -> str:
        if not self.ready or self.runtime is None:
            return "**Offline · CPU · Runtime not ready**"
        status = self.runtime.safe_status()
        loads = status["model_load_counts"]
        provider = ", ".join(status["active_onnx_providers"])
        return (
            "**Offline · CPU · YAMNet · FP32 ONNX · Runtime ready**  \n"
            f"Model loads — YAMNet: `{loads['yamnet']}`, ONNX: `{loads['onnx_classifier']}` · "
            f"Provider: `{provider}`"
        )

    def analyze_audio(self, audio_path: Optional[str]) -> dict[str, Any]:
        if not audio_path:
            return empty_presentation("Select or record a WAV file before analysis.")
        normalized = str(audio_path).lower()
        if normalized.startswith(("http:/", "https:/")):
            return empty_presentation("Audio URLs are not supported. Choose a local WAV file.")
        if not self.ready or self.runtime is None:
            return empty_presentation(self.initialization_error or "Runtime is not ready.")
        if not self._inference_lock.acquire(blocking=False):
            return empty_presentation("Runtime busy. Wait for the current analysis to finish.")
        try:
            result = self.runtime.analyze(
                Path(audio_path), max_duration_seconds=self.max_duration_seconds
            )
            return present_result(result, self.runtime.startup_timing)
        except AudioDurationLimitError:
            return empty_presentation(
                f"Audio is longer than the {self.max_duration_seconds:g}-second local limit."
            )
        except FileNotFoundError:
            return empty_presentation("The selected WAV file is no longer available.")
        except ValueError:
            return empty_presentation("The WAV could not be decoded or did not satisfy the audio contract.")
        except Exception as exc:
            LOGGER.error("Safe demo analysis failure: %s", type(exc).__name__)
            return empty_presentation("Analysis failed locally. No audio was uploaded or retained.")
        finally:
            self._inference_lock.release()

    def clear(self) -> dict[str, Any]:
        return empty_presentation()


__all__ = ["DemoRuntimeState"]

