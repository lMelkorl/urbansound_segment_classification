#!/usr/bin/env python3
"""Launch the local-only EdgeSound Gradio demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urbansound_segment_task.edge_v2.demo.app import build_demo, configure_local_environment  # noqa: E402
from urbansound_segment_task.edge_v2.demo.state import DemoRuntimeState  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local-only EdgeSound Gradio demo.")
    parser.add_argument("--yamnet-artifact", type=Path, required=True)
    parser.add_argument("--classifier-artifact", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-duration-seconds", type=float, default=30.0)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.host != "127.0.0.1":
        print("security warning: only 127.0.0.1 is allowed; refusing non-localhost binding", file=sys.stderr)
        return 2
    if not 1 <= arguments.port <= 65535 or arguments.max_duration_seconds <= 0:
        print("error: invalid port or duration limit", file=sys.stderr)
        return 2
    configure_local_environment()
    state = DemoRuntimeState(
        yamnet_artifact=arguments.yamnet_artifact,
        classifier_artifact=arguments.classifier_artifact,
        max_duration_seconds=arguments.max_duration_seconds,
    )
    if not state.initialize():
        print("error: runtime initialization failed; UI was not started", file=sys.stderr)
        return 1
    demo = build_demo(state)
    demo.launch(
        server_name="127.0.0.1", server_port=arguments.port, share=False,
        inbrowser=not arguments.no_browser, css=demo.edge_css, footer_links=[],
        enable_monitoring=False, mcp_server=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
