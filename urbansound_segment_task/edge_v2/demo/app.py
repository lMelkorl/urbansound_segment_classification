"""Lazy Gradio Blocks construction for the local-only audio demo."""

from __future__ import annotations

import os
from typing import Any

from .presenter import empty_presentation, presentation_outputs
from .state import DemoRuntimeState


LOCAL_ENVIRONMENT = {
    "GRADIO_ANALYTICS_ENABLED": "False",
    "GRADIO_SHARE": "False",
    "GRADIO_SERVER_NAME": "127.0.0.1",
    "GRADIO_FLAGGING_MODE": "never",
}


def configure_local_environment() -> dict[str, str]:
    for name, value in LOCAL_ENVIRONMENT.items():
        os.environ[name] = value
    return dict(LOCAL_ENVIRONMENT)


def build_demo(state: DemoRuntimeState) -> Any:
    import gradio as gr

    initial = empty_presentation()

    def analyze(audio_path: str | None):
        return presentation_outputs(state.analyze_audio(audio_path))

    def clear():
        return (None, *presentation_outputs(state.clear()))

    css = """
    .gradio-container { max-width: 980px !important; margin: 0 auto; }
    .runtime-status { border-left: 3px solid #4b728f; padding-left: 0.8rem; }
    """
    with gr.Blocks(title="EdgeSound — Offline Urban Audio Classifier") as demo:
        gr.Markdown("# EdgeSound — Offline Urban Audio Classifier")
        gr.Markdown("Runs locally on CPU. Audio is not uploaded to a cloud service.")
        status = gr.Markdown(state.status_markdown(), elem_classes=["runtime-status"])
        audio = gr.Audio(
            sources=["upload", "microphone"], type="filepath", format="wav",
            label="Upload a WAV or record up to 30 seconds",
        )
        with gr.Row():
            analyze_button = gr.Button("Analyze Audio", variant="primary")
            clear_button = gr.Button("Clear")
        message = gr.Markdown(initial["message"])
        prediction = gr.Markdown(initial["prediction"])
        gr.Markdown("Raw softmax confidence is shown without calibration.")
        gr.Markdown("### Top-3")
        top3 = gr.Dataframe(
            headers=["Rank", "Class", "Confidence"], datatype=["number", "str", "number"],
            value=initial["top3"], interactive=False,
        )
        gr.Markdown("### Segment timeline")
        timeline = gr.Dataframe(
            headers=["Start (s)", "End (s)", "Prediction", "Confidence"],
            datatype=["number", "number", "str", "number"],
            value=initial["timeline"], interactive=False,
        )
        gr.Markdown("### Audio information")
        audio_info = gr.JSON(value=initial["audio_info"], label="Audio")
        with gr.Accordion("Runtime timing", open=False):
            request_timing = gr.JSON(value=initial["request_timing"], label="Request and startup timing")
        with gr.Accordion("Technical information", open=False):
            technical = gr.Markdown(initial["technical"])
        outputs = [message, prediction, top3, timeline, audio_info, request_timing, technical]
        analyze_button.click(
            analyze, inputs=[audio], outputs=outputs, concurrency_limit=1,
            api_name=None, api_visibility="private", show_progress="minimal",
        )
        clear_button.click(
            clear, inputs=[], outputs=[audio, *outputs], concurrency_limit=1,
            api_name=None, api_visibility="private", show_progress="hidden",
        )
        demo.queue(default_concurrency_limit=1, max_size=4, api_open=False)
    demo.edge_runtime_state = state
    demo.edge_status_component = status
    demo.edge_css = css
    return demo


__all__ = ["LOCAL_ENVIRONMENT", "build_demo", "configure_local_environment"]
