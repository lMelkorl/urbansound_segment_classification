# Edge Audio V2 — Local Demo Guide

## What the demo does

The demo runs the complete local inference chain for a WAV upload or a completed browser microphone recording:

```text
local audio → mono/16 kHz → legacy segmentation → local YAMNet
→ FP32 ONNX Linear classifier → clip probability mean → UI
```

It displays the clip prediction, raw softmax confidence, top three classes, per-segment timeline, audio metadata, and request/startup timing. It is not continuous microphone streaming.

## Prerequisites

- macOS arm64;
- Python 3.11 available as `python3.11`;
- a verified local YAMNet artifact;
- the provenance-bound deployment FP32 ONNX classifier;
- no cloud API key.

Model binaries are ignored by Git. Their expected hashes and contracts are recorded in the [YAMNet manifest](../artifacts/yamnet/tfhub-v1/artifact-manifest.json) and [deployment classifier manifest](../artifacts/compact_classifier/linear-all-data-fp32-onnx-v1/artifact-manifest.json).

## Create `.venv-demo`

The exact environment is locked in [`requirements/demo-macos-arm64-py311.lock`](../requirements/demo-macos-arm64-py311.lock):

```bash
python3.11 -m venv .venv-demo
.venv-demo/bin/python -m pip install --upgrade pip wheel setuptools
.venv-demo/bin/python -m pip install \
  -r requirements/demo-macos-arm64-py311.lock
.venv-demo/bin/python -m pip check
```

The shorter dependency intent is in [`requirements/demo-macos-arm64-py311.in`](../requirements/demo-macos-arm64-py311.in). Do not mix this environment with the legacy repository-wide `requirements.txt`.

## Start the demo

From the repository root:

```bash
GRADIO_ANALYTICS_ENABLED=False \
GRADIO_SHARE=False \
.venv-demo/bin/python scripts/run_audio_demo.py \
  --yamnet-artifact artifacts/yamnet/tfhub-v1 \
  --classifier-artifact artifacts/compact_classifier/linear-all-data-fp32-onnx-v1 \
  --host 127.0.0.1 \
  --port 7860
```

Open `http://127.0.0.1:7860`. The CLI refuses non-loopback hosts. Gradio sharing, analytics, flagging, and public API access are disabled by the application as defense in depth.

## File upload

1. Choose **Upload** in the audio input.
2. Select a local WAV file no longer than 30 seconds.
3. Select **Analyze Audio**.
4. Wait until the current request completes; the runtime processes one request at a time.

Only local WAV input is supported. URLs and unsupported formats are rejected. The selected file is decoded locally and is not retained by the application after analysis.

## Microphone use

1. Choose **Microphone**.
2. Start and stop a recording in the browser.
3. Select **Analyze Audio** after the recording is complete.

This is completed-recording inference, not a streaming listener. Keep recordings under the 30-second limit.

## macOS microphone permission

The first recording attempt may trigger a macOS permission prompt for the browser. Allow microphone access only for the browser you are using. If access was denied:

1. open **System Settings → Privacy & Security → Microphone**;
2. enable the relevant browser;
3. reload the local demo page;
4. record again.

The terminal normally does not need microphone permission because capture occurs in the browser.

## Local-only and privacy settings

The command and application enforce:

```text
GRADIO_ANALYTICS_ENABLED=False
GRADIO_SHARE=False
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_FLAGGING_MODE=never
```

The demo does not create a Gradio share link and does not expose its analysis handlers as a public API. Audio content is not written to result artifacts by Edge Audio V2. Browser and operating-system temporary-file behavior still applies; do not analyze sensitive recordings on an untrusted machine.

No application-level network call is needed after model artifacts and dependencies are installed. Firewall or offline testing can be used to verify this operationally.

## Analyze Audio flow

At process startup, the application verifies artifact manifests and hashes, loads YAMNet once, creates one ONNX Runtime CPU session, and performs a warm-up. Each request then:

1. validates local WAV metadata and the 30-second limit;
2. decodes float32 audio and averages channels to mono;
3. resamples to 16 kHz with `soxr_hq` when necessary;
4. creates 15,360-sample windows with a 7,680-sample hop;
5. runs local YAMNet and the FP32 ONNX classifier;
6. averages segment probability vectors;
7. renders the result without exposing the absolute source path.

YAMNet and the ONNX classifier are reused rather than reloaded for every request.

## Understanding the result screen

- **Prediction:** highest mean clip probability and canonical class ID.
- **Confidence:** raw, uncalibrated softmax output; not a guaranteed probability of correctness.
- **Top-3:** three highest clip-level values after segment aggregation.
- **Segment timeline:** prediction and confidence for each complete 0.96-second window.
- **Audio information:** safe sample-rate, channel, duration, segment, and dropped-tail metadata.
- **Runtime timing:** descriptive request and one-time startup timings; not the controlled benchmark protocol.
- **Technical information:** verified artifact hashes, runtime versions, provider, segmentation policy, and model-load counts.

Audio shorter than 0.96 seconds produces no prediction. Incomplete final tails are reported and dropped.

## Stop and clear

**Clear** resets the visible request result but intentionally keeps models loaded. Stop the local server with `Ctrl-C` in the terminal. Closing only the browser tab does not necessarily terminate the Python process.

## Troubleshooting

### Runtime initialization failed

- Confirm both artifact directories exist.
- Verify that ignored model binaries are present locally.
- Run the YAMNet verification command:

  ```bash
  .venv-demo/bin/python scripts/verify_yamnet_artifact.py \
    --artifact artifacts/yamnet/tfhub-v1
  ```

- Confirm the ONNX file hash matches the deployment manifest.
- Run `.venv-demo/bin/python -m pip check`.

### Microphone is unavailable

- Check browser and macOS microphone permissions.
- Use `http://127.0.0.1:7860`, not a remote hostname.
- Reload the page after changing permissions.

### Audio is rejected

- Use a local WAV file.
- Keep duration at or below 30 seconds.
- Confirm the file still exists and can be decoded by SoundFile.

### No prediction is shown

The preprocessed audio is shorter than one 15,360-sample window. Padding is deliberately not added.

### Runtime busy

Only one request runs at a time. Wait for the current analysis to finish. The queue is deliberately bounded.

### Port 7860 is already in use

Choose another local port, for example `--port 7861`, and open the matching localhost URL.

## Known limitations

- Ten UrbanSound8K classes only; no unknown or open-world class.
- Softmax confidence is uncalibrated.
- No continuous streaming or background microphone listener.
- No sub-window event localization.
- YAMNet remains the dominant startup, memory, and inference component.
- CPU measurements depend on hardware, thermals, process state, and thread configuration.
- The deployment-only classifier has no independent held-out test metric.
- The macOS arm64 lock does not guarantee compatibility on Linux, Windows, or Intel macOS.
