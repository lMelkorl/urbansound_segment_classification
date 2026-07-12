# Edge Audio V2 — Offline Urban Sound Classification

## Project overview

Edge Audio V2 is a reproducible, CPU-first environmental sound classification system built on the UrbanSound8K case study in this repository. The deployable path is:

```text
local WAV or microphone recording
→ mono 16 kHz preprocessing
→ 0.96 s legacy windows with 50% overlap
→ local YAMNet embeddings
→ 10-class Linear softmax classifier
→ arithmetic mean of segment probabilities
→ clip prediction and segment timeline
```

Inference is local and does not require a cloud API. The final classifier is exported to FP32 ONNX and runs with ONNX Runtime's CPU provider. The original YAMNet + LightGBM, ESResNeXt, and AudioCLIP implementations remain documented in the [historical task README](urbansound_segment_task/README.md) and preserved under `urbansound_segment_task/goals/`.

## Why this revision exists

The original case study demonstrated three GPU-oriented approaches, primarily with a legacy Fold 1–8 training, Fold 9 validation, Fold 10 test split. Edge Audio V2 adds:

- official rotating 10-fold evaluation without clip leakage;
- a compact 10,250-parameter classifier over cached YAMNet embeddings;
- reproducible CPU and ONNX Runtime measurements;
- a provenance-bound deployment-only model trained on all available embeddings;
- offline file-upload and microphone demo flows;
- a deterministic, held-out-fold audio robustness evaluation;
- machine-readable manifests, hashes, dependency locks, and result artifacts.

The revision does not rewrite or relabel the historical results. Scientific performance claims come from official cross-fold results; the all-data deployment artifact has no independent test metric.

## System architecture

The canonical audio contract is mono float32 at 16 kHz. Legacy segmentation uses 15,360-sample windows, a 7,680-sample hop, and drops incomplete tails. Clips shorter than one full window produce no prediction; no artificial padding is introduced.

YAMNet is loaded from a verified local SavedModel. Each segment produces a 1,024-dimensional mean embedding. The FP32 ONNX classifier maps `[batch, 1024]` embeddings to `[batch, 10]` softmax outputs. Clip aggregation is the arithmetic mean of canonical-class probabilities.

The ten classes, in fixed output order, are: `air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `siren`, and `street_music`.

## Main scientific results

Official evaluation uses each UrbanSound8K fold exactly once as test data. For test fold `k`, validation fold is `k % 10 + 1`; the remaining eight folds are training data. Mean and population standard deviation are reported across ten held-out folds.

| Model | Clip accuracy | Clip macro-F1 | Segment accuracy | Segment macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Linear softmax | **0.780485 ± 0.036271** | **0.789331 ± 0.033921** | 0.716043 ± 0.036224 | 0.708456 ± 0.030906 |
| LightGBM | 0.778559 ± 0.034277 | 0.789059 ± 0.031768 | 0.719978 ± 0.033083 | 0.725402 ± 0.030103 |

Sources: [compact classifier aggregate](results/compact_classifier_cross_fold/fixed-baselines-v1/aggregate.json) and [LightGBM aggregate](results/lightgbm_cross_fold/legacy-config-v1/aggregate.json).

These results are not directly comparable with the historical single-split numbers without an explicit protocol warning. See [BENCHMARKS.md](docs/BENCHMARKS.md) for scope separation.

## Compact classifier result

The selected architecture is one Dense softmax layer over 1,024-dimensional YAMNet embeddings:

- 10,250 parameters;
- 47,920-byte Keras weights artifact during fold evaluation;
- 0.789331 ± 0.033921 official clip macro-F1;
- Pareto-preferred over the tested MLP-128 on macro-F1, serialized size, and classifier-only latency.

The deployment classifier uses the same architecture but is trained on all 53,918 verified segments after the evaluation protocol and epoch-selection rule were frozen. It is marked `deployment_only: true`; `independent_test_metrics_available` is `false`. Its 41,795-byte FP32 ONNX artifact is for local inference, not a new scientific test result. Provenance is recorded in the [deployment artifact manifest](artifacts/compact_classifier/linear-all-data-fp32-onnx-v1/artifact-manifest.json).

## CPU and ONNX benchmark

The representative Fold 1 classifier-only benchmark measured:

| Runtime | Batch-1 p50 | Batch-1 p95 | Artifact size |
| --- | ---: | ---: | ---: |
| Keras | 0.161917 ms | 0.174936 ms | 47,920 bytes |
| ONNX Runtime CPU | 0.003916 ms | 0.009500 ms | 41,795 bytes |

The p50 ratio is **41.35×**, but this is strictly a classifier-only comparison on precomputed 1,024-dimensional embeddings. It is not a 41× full-pipeline speedup. YAMNet remains the dominant latency and memory component.

On the documented Apple M3 Pro CPU run, the selected eight-thread YAMNet configuration recorded a 2.424563 ms steady-state model p50 and approximately 591 MB incremental peak RSS. These are benchmark-process measurements, not interactive demo request guarantees. See the [authoritative YAMNet summary](results/yamnet_cpu_benchmarks/final-m3pro-run/summary.json) and [ONNX benchmark summary](results/onnx_benchmark/linear-fold1-fp32-v1/summary.json).

Dynamic INT8 reduced the classifier artifact by about 70.46%, but it was slower, did not reduce measured memory, and failed the parity gate. FP32 ONNX therefore remains the deployment candidate. See [BENCHMARKS.md](docs/BENCHMARKS.md).

## Robustness results

The authoritative robustness panel contains 1,978 evaluable clips, selected deterministically with up to 20 clips per fold and class. Each fold-specific Linear model is evaluated only on its own held-out fold. Perturbations are applied to canonical waveform audio before segmentation and YAMNet extraction.

| Condition | Clip accuracy | Clip macro-F1 | Accuracy drop | Macro-F1 drop | Flip rate | Degradation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Clean | 0.833140 ± 0.060407 | 0.826161 ± 0.063502 | 0.000000 | 0.000000 | 0.000000 | Minor |
| White noise, 20 dB SNR | 0.783637 ± 0.069998 | 0.778345 ± 0.070533 | 0.049503 | 0.047816 | 0.119815 | Moderate |
| White noise, 10 dB SNR | 0.646313 ± 0.084660 | 0.636696 ± 0.090411 | 0.186827 | 0.189465 | 0.306698 | Major |
| White noise, 0 dB SNR | 0.395738 ± 0.086018 | 0.347563 ± 0.086023 | 0.437402 | 0.478598 | 0.572808 | Major |
| Gain −12 dB | 0.787730 ± 0.065254 | 0.775732 ± 0.071825 | 0.045410 | 0.050429 | 0.138535 | Moderate |
| 8 kHz bandlimit roundtrip | 0.797119 ± 0.075748 | 0.787245 ± 0.080990 | 0.036022 | 0.038916 | 0.162312 | Moderate |

Source: [authoritative 10-fold robustness aggregate](results/audio_robustness/linear-cross-fold-v1/aggregate.json).

Clean runtime predictions agree 100% with cached-pipeline top-1 predictions in every fold. The clean panel score is not the official full-fold score: the robustness panel is class-capped, excludes zero-segment clips, and has a different sample composition. A general claim that the model is “robust” is not justified; degradation must be interpreted condition by condition.

## Offline demo

The Gradio demo accepts a local WAV upload or a completed microphone recording, displays the current clip prediction, top three classes, segment timeline, audio metadata, and request timing. It binds only to localhost, disables sharing and analytics, uses `CPUExecutionProvider`, and does not upload audio to an application cloud service.

The demo is request-based, not continuous streaming. Softmax confidence is not calibrated. See [DEMO.md](docs/DEMO.md) for setup, privacy behavior, permissions, and troubleshooting.

## Installation

The validated local environments target macOS arm64 and Python 3.11. Other platforms require compatible TensorFlow and ONNX Runtime wheels and have not been validated by the included lock files.

```bash
python3.11 -m venv .venv-yamnet
.venv-yamnet/bin/python -m pip install --upgrade pip wheel setuptools
.venv-yamnet/bin/python -m pip install \
  -r requirements/yamnet-macos-arm64-py311.lock

python3.11 -m venv .venv-onnx
.venv-onnx/bin/python -m pip install --upgrade pip wheel setuptools
.venv-onnx/bin/python -m pip install \
  -r requirements/onnx-export-macos-arm64-py311.lock

python3.11 -m venv .venv-edge-runtime
.venv-edge-runtime/bin/python -m pip install --upgrade pip wheel setuptools
.venv-edge-runtime/bin/python -m pip install \
  -r requirements/edge-runtime-macos-arm64-py311.lock

python3.11 -m venv .venv-demo
.venv-demo/bin/python -m pip install --upgrade pip wheel setuptools
.venv-demo/bin/python -m pip install \
  -r requirements/demo-macos-arm64-py311.lock
```

UrbanSound8K must remain outside the repository. Model binaries are intentionally ignored by Git; only their small provenance manifests are trackable. Acquire and verify YAMNet locally with:

```bash
.venv-yamnet/bin/python scripts/acquire_yamnet_artifact.py \
  --allow-network \
  --output artifacts/yamnet/tfhub-v1

.venv-yamnet/bin/python scripts/verify_yamnet_artifact.py \
  --artifact artifacts/yamnet/tfhub-v1
```

Network access is required only for the explicit acquisition command. Evaluation and demo commands use local artifacts.

## Running the demo

The deployment ONNX binary must exist under the directory bound by its checked manifest. Then run:

```bash
GRADIO_ANALYTICS_ENABLED=False \
GRADIO_SHARE=False \
.venv-demo/bin/python scripts/run_audio_demo.py \
  --yamnet-artifact artifacts/yamnet/tfhub-v1 \
  --classifier-artifact artifacts/compact_classifier/linear-all-data-fp32-onnx-v1 \
  --host 127.0.0.1 \
  --port 7860
```

Open `http://127.0.0.1:7860`, choose upload or microphone input, and select **Analyze Audio**. Stop the demo with `Ctrl-C` in its terminal.

## Reproducing evaluations

Set a local dataset path without writing it into result artifacts:

```bash
export URBANSOUND8K_ROOT=../datasets/UrbanSound8K
```

Inventory and official split manifests:

```bash
.venv-yamnet/bin/python scripts/inspect_urbansound8k.py \
  --dataset-root "$URBANSOUND8K_ROOT" \
  --output results/reproduction/dataset-inventory.json --pretty

.venv-yamnet/bin/python scripts/build_urbansound8k_manifests.py \
  --dataset-root "$URBANSOUND8K_ROOT" \
  --output-dir results/reproduction/splits --pretty
```

Embedding extraction is a full-dataset operation and requires explicit confirmation:

```bash
.venv-yamnet/bin/python scripts/extract_yamnet_embeddings.py \
  --dataset-root "$URBANSOUND8K_ROOT" \
  --artifact artifacts/yamnet/tfhub-v1 \
  --cache-root cache/yamnet_embeddings \
  --threads 8 --confirm-full-run \
  --output results/reproduction/yamnet-extraction.json --pretty
```

Official cross-fold classifiers:

```bash
.venv-yamnet/bin/python scripts/run_lightgbm_cross_fold.py \
  --cache-root cache/yamnet_embeddings \
  --split-manifest-dir results/reproduction/splits \
  --output-dir results/reproduction/lightgbm \
  --threads 8 --resume --pretty

.venv-yamnet/bin/python scripts/run_compact_classifier_cross_fold.py \
  --cache-root cache/yamnet_embeddings \
  --split-manifest-dir results/reproduction/splits \
  --output-dir results/reproduction/compact \
  --models linear,mlp128 --threads 8 --resume --pretty
```

Fold 1 FP32 ONNX export and parity:

```bash
.venv-onnx/bin/python scripts/export_linear_onnx.py \
  --run-manifest results/reproduction/compact/run-manifest.json \
  --model linear --fold 1 --opset 15 \
  --output artifacts/compact_classifier/linear-fold1-fp32-onnx-reproduction/model.onnx \
  --manifest artifacts/compact_classifier/linear-fold1-fp32-onnx-reproduction/artifact-manifest.json \
  --pretty

.venv-onnx/bin/python scripts/validate_linear_onnx.py \
  --run-manifest results/reproduction/compact/run-manifest.json \
  --cache-root cache/yamnet_embeddings \
  --onnx-artifact artifacts/compact_classifier/linear-fold1-fp32-onnx-reproduction \
  --sample-count 1000 \
  --output-dir results/reproduction/onnx-parity --pretty
```

Offline inference and robustness evaluation:

```bash
.venv-edge-runtime/bin/python scripts/run_offline_audio_inference.py \
  --audio local-example.wav \
  --yamnet-artifact artifacts/yamnet/tfhub-v1 \
  --classifier-artifact artifacts/compact_classifier/linear-all-data-fp32-onnx-v1 \
  --output results/reproduction/offline-inference.json --pretty

.venv-edge-runtime/bin/python scripts/run_audio_robustness.py \
  --dataset-root "$URBANSOUND8K_ROOT" \
  --model-run results/reproduction/compact \
  --yamnet-artifact artifacts/yamnet/tfhub-v1 \
  --clips-per-class-per-fold 20 \
  --conditions clean,white_noise_snr_20db,white_noise_snr_10db,white_noise_snr_0db,gain_minus_12db,bandlimit_8khz_roundtrip \
  --output-dir results/reproduction/robustness \
  --resume --pretty
```

Long-running extraction, training, and full evaluation commands should be reviewed before execution. Detailed benchmark scope and artifact references are in [BENCHMARKS.md](docs/BENCHMARKS.md).

## Repository structure

```text
urbansound_segment_task/
  edge_v2/                  # CPU-first data, models, export, runtime, demo, evaluation
  goals/                    # preserved historical case-study implementations
scripts/                    # public CLI entry points
requirements/               # platform-specific inputs and exact environment locks
artifacts/                  # small manifests; large model binaries are ignored
results/                    # machine-readable scientific and benchmark outputs
docs/                       # audit, methodology-oriented guides, model card
tests/unit/                 # framework-light and runtime unit tests
cache/                      # ignored local YAMNet embedding cache
```

## Limitations

- UrbanSound8K contains only ten urban sound classes and does not represent open-world audio.
- The deployment all-data classifier has no independent held-out test metric.
- Softmax confidence is uncalibrated and must not be interpreted as guaranteed correctness.
- YAMNet dominates end-to-end latency and memory despite the tiny classifier.
- The demo analyzes completed uploads or recordings; it is not streaming inference.
- Audio shorter than 0.96 seconds has zero legacy segments and receives no prediction.
- Incomplete tails are dropped rather than padded.
- Robustness perturbations are controlled synthetic shifts, not a complete model of real environments.
- Class support is limited for some fold/class combinations, especially `gun_shot`.
- Included lock files are validated for macOS arm64 with Python 3.11, not every OS/CPU.

## Scientific integrity notes

- Historical Goal results use a legacy single split unless explicitly stated otherwise.
- Official Linear and LightGBM values are ten-fold means with population standard deviations.
- Segments from one clip never cross train, validation, and test partitions.
- Test folds are not used for model selection.
- The robustness evaluation uses fold-specific models only on their own held-out folds.
- The deployment-only all-data model is never used for scientific robustness metrics.
- Classifier-only latency values are not presented as end-to-end audio-pipeline speedups.
- Every final numeric claim above points to a machine-readable result artifact.

## Artifact and license notes

Datasets, embedding caches, pretrained checkpoints, Keras weights, ONNX binaries, and other large generated artifacts are intentionally excluded from Git. Trackable manifests record their expected hashes, sizes, contracts, and provenance.

UrbanSound8K, YAMNet, TensorFlow, ONNX Runtime, Gradio, ESResNeXt, AudioCLIP, and other third-party components remain subject to their respective upstream terms. Verify those terms before redistribution or commercial use. This repository currently has no top-level `LICENSE` file; choosing and adding a project license is a release blocker that requires an explicit owner decision.
