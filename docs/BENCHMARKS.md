# Edge Audio V2 — Benchmark and Evaluation Guide

## Reading the results correctly

Each value below belongs to a specific scope. Numbers with different inputs, hardware, split protocols, or included stages are not direct speed or quality comparisons. Machine-readable artifacts are authoritative; rounded documentation values are derived from them.

Population standard deviation is used for cross-fold metrics. Historical Goal results remain legacy single-split experiments unless explicitly marked as official ten-fold evaluations.

## YAMNet CPU benchmark

The [final Apple M3 Pro summary](../results/yamnet_cpu_benchmarks/final-m3pro-run/summary.json) measures local YAMNet with fresh child processes and several thread counts. The selected eight-thread configuration reports medians across three repetitions for a 15,360-sample input:

| Measurement | Result |
| --- | ---: |
| Model load | 959.0375 ms |
| First call | 76.4701 ms |
| Steady-state model p50 | 2.424563 ms |
| Steady-state model p95 | 2.674858 ms |
| Legacy feature pipeline p50 | 2.332854 ms |
| Legacy feature pipeline throughput | 422.787 segments/s |
| Approximate incremental peak RSS | 591,216,640 bytes |

The RSS value is based on Darwin process high-water marks. It is not exact model allocation. The feature-pipeline and model-only measurements are separate benchmark scopes and their small timing difference should not be treated as stage subtraction.

## Legacy LightGBM reproduction

The legacy reproduction preserves the original Fold 1–8 train, Fold 9 validation, Fold 10 test protocol and exact LightGBM configuration. It verifies provenance and historical behavior but is not a complete cross-validation result. See the tracked [run 1 result](../results/lightgbm_legacy_reproduction/m3pro-run-1/result.json).

Do not compare this single held-out Fold 10 result directly with a ten-fold mean without labeling the protocol difference.

## Official LightGBM ten-fold evaluation

The [official LightGBM aggregate](../results/lightgbm_cross_fold/legacy-config-v1/aggregate.json) uses rotating validation and every official fold once as test data:

| Metric | Mean ± population std |
| --- | ---: |
| Clip accuracy | 0.778559 ± 0.034277 |
| Clip macro-F1 | 0.789059 ± 0.031768 |
| Segment accuracy | 0.719978 ± 0.033083 |
| Segment macro-F1 | 0.725402 ± 0.030103 |

The classifier artifacts are roughly 45 MB each. Their ignored `model.txt` files are not required in Git because result JSON, configs, and hashes preserve the evaluated record.

## Linear and MLP ten-fold evaluation

The [compact classifier aggregate](../results/compact_classifier_cross_fold/fixed-baselines-v1/aggregate.json) evaluates both fixed candidates on the same cached embeddings and split policy:

| Model | Parameters | Clip accuracy | Clip macro-F1 | Median Keras weight size | Classifier-only p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear | 10,250 | 0.780485 ± 0.036271 | 0.789331 ± 0.033921 | 47,920 bytes | 0.165187 ms |
| MLP-128 | 132,490 | 0.777972 ± 0.034418 | 0.784742 ± 0.032904 | 541,272 bytes | 0.342906 ms |

The p50 values in this table are Keras classifier-only measurements on cached embeddings. YAMNet, audio decode, resampling, segmentation, and clip aggregation are excluded. Linear is the compact-model Pareto choice across macro-F1, size, and classifier-only latency.

## Compact model size comparison

The evaluated LightGBM mean size is 45,450,900.5 bytes. The 47,920-byte Linear Keras weight file is approximately 99.89% smaller while matching LightGBM's ten-fold clip macro-F1 within the observed rounding. Serialized formats differ, so this is an artifact-size comparison, not parameter equivalence.

The final deployment FP32 ONNX classifier is 41,795 bytes. It contains only the Linear classifier; YAMNet is not included in that size.

## Keras–ONNX parity

The Fold 1 FP32 export uses ONNX opset 15 and a dynamic batch dimension. The deterministic parity result is stored in [the FP32 parity artifact](../results/onnx_export/linear-fold1-fp32-v1/parity-result.json). The deployment-only artifact has its own [parity result](../results/deployment_onnx/linear-all-data-fp32-v1/parity/parity-result.json):

- top-1 agreement: 1.0;
- maximum absolute error: `4.470348358154297e-07`;
- total numeric samples: 1,168;
- status: success.

Parity establishes numerical consistency between native and exported classifiers. It does not create a new accuracy estimate for the deployment-only model.

## Classifier-only ONNX benchmark

The [five-repetition benchmark](../results/onnx_benchmark/linear-fold1-fp32-v1/summary.json) uses the same deterministic 1,024-element embedding input in fresh child processes:

| Runtime | p50 | p95 | Predictions/s | Approx. incremental peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Keras | 0.161917 ms | 0.174936 ms | 4,700.91 | 338,509,824 bytes |
| ONNX Runtime CPU | 0.003916 ms | 0.009500 ms | 113,456.61 | 15,581,184 bytes |

The Keras-p50/ONNX-p50 ratio is 41.35×. This applies only to the tiny classifier after embeddings already exist. It is not an end-to-end audio speedup, and the process high-water RSS comparison is approximate. YAMNet remains the dominant end-to-end cost.

## Dynamic INT8 rejection

Dynamic INT8 reduced the ONNX classifier from 41,795 to 12,347 bytes, a 70.46% reduction. It was nevertheless rejected:

- parity threshold gate failed;
- p50 latency was about 142.59% worse rather than at least 5% better;
- measured incremental RSS was about 1.06% worse;
- throughput was lower;
- final deployment decision: `fp32_onnx`.

Sources: [INT8 parity result](../results/onnx_int8/linear-fold1-dynamic-v1/parity/parity-result.json) and [five-repetition INT8 benchmark](../results/onnx_int8/linear-fold1-dynamic-v1/benchmark/summary.json). A smaller binary alone is not sufficient evidence for deployment.

## Audio robustness evaluation

The [authoritative aggregate](../results/audio_robustness/linear-cross-fold-v1/aggregate.json) evaluates 1,978 deterministic panel clips with fold-specific models. Perturbed audio is not cached or persisted. Clean predictions are checked against verified cached embeddings before robustness results are accepted.

| Condition | Accuracy mean ± std | Macro-F1 mean ± std | Accuracy drop | Macro-F1 drop | Flip rate | Degradation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Clean | 0.833140 ± 0.060407 | 0.826161 ± 0.063502 | 0.000000 | 0.000000 | 0.000000 | Minor |
| White noise, 20 dB SNR | 0.783637 ± 0.069998 | 0.778345 ± 0.070533 | 0.049503 | 0.047816 | 0.119815 | Moderate |
| White noise, 10 dB SNR | 0.646313 ± 0.084660 | 0.636696 ± 0.090411 | 0.186827 | 0.189465 | 0.306698 | Major |
| White noise, 0 dB SNR | 0.395738 ± 0.086018 | 0.347563 ± 0.086023 | 0.437402 | 0.478598 | 0.572808 | Major |
| Gain −12 dB | 0.787730 ± 0.065254 | 0.775732 ± 0.071825 | 0.045410 | 0.050429 | 0.138535 | Moderate |
| 8 kHz bandlimit roundtrip | 0.797119 ± 0.075748 | 0.787245 ± 0.080990 | 0.036022 | 0.038916 | 0.162312 | Moderate |

Degradation thresholds were fixed before results: minor at macro-F1 drop ≤ 0.025, moderate at drop ≤ 0.075, and major above 0.075. The clean robustness score is not the official full-fold score because the panel is class-capped and excludes zero-segment clips.

The aggregate semantic hash, 60 raw JSON result hashes, and all JSON/CSV parses have been verified. Each of the 60 CSV files exactly matches its corresponding JSON per-class metrics. CSV files do not currently have independent committed sidecar hashes; their content is cross-validated against the provenance-bound JSON units.

## Reproduction command index

The primary commands and exact options are documented in the root [README](../README.md#reproducing-evaluations):

- dataset inventory and official split manifests;
- local YAMNet artifact acquisition and verification;
- full embedding extraction;
- official LightGBM and compact-model ten-fold evaluation;
- FP32 ONNX export and parity;
- offline WAV inference;
- local demo;
- full robustness evaluation.

Model acquisition, full cache extraction, training, and full robustness runs can be compute- or storage-intensive. Review their output locations and resource requirements before execution.
