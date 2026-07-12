# Edge Audio V2 Linear Classifier — Model Card

## Model overview

Edge Audio V2 combines a fixed local YAMNet feature extractor with a compact Linear softmax classifier. YAMNet converts each 0.96-second audio segment to a 1,024-dimensional mean embedding. A single Dense layer maps that embedding to ten UrbanSound8K class probabilities. Clip probabilities are the arithmetic mean of segment probabilities.

There are two deliberately separate artifact families:

1. ten fold-specific Linear models used for official held-out evaluation; and
2. one all-data FP32 ONNX classifier used only for local deployment.

The classifier has 10,250 parameters. The deployment ONNX artifact is 41,795 bytes and is described by the [deployment manifest](../artifacts/compact_classifier/linear-all-data-fp32-onnx-v1/artifact-manifest.json). YAMNet is a separate, substantially larger runtime component.

## Intended use

- Offline, CPU-only classification of short urban environmental recordings.
- Local demonstrations using uploaded WAV files or completed microphone recordings.
- Reproducible research on UrbanSound8K's fixed ten-class label space.
- Engineering evaluation of a small classifier layered on verified YAMNet embeddings.

Human review is appropriate whenever a prediction could influence a consequential decision. The model is a demonstration and research artifact, not a safety system.

## Out-of-scope use

- Open-world acoustic event detection.
- Surveillance, identity inference, speaker recognition, or biometric analysis.
- Emergency, medical, legal, industrial-safety, or law-enforcement decisions.
- Continuous streaming detection or event localization.
- Treating raw softmax output as calibrated likelihood.
- Claiming performance for classes, microphones, environments, or populations not evaluated here.

## Input and output contract

Input audio is decoded locally, converted to mono float32, and resampled to 16 kHz with `soxr_hq`. Legacy segmentation uses:

- window: 15,360 samples (0.96 seconds);
- hop: 7,680 samples (50% overlap);
- incomplete tail: dropped;
- clip shorter than one window: zero segments and no prediction;
- padding: none.

For each segment, local YAMNet produces a 1,024-dimensional mean embedding. The classifier contract is:

```text
input:  embedding     float32 [batch, 1024]
output: probabilities float32 [batch, 10]
```

The deployment graph uses ONNX opset 15 and ONNX Runtime `CPUExecutionProvider`. Clip aggregation is an arithmetic probability mean, not a vote or logit mean.

## Canonical ten classes

| Class ID | Class name |
| ---: | --- |
| 0 | `air_conditioner` |
| 1 | `car_horn` |
| 2 | `children_playing` |
| 3 | `dog_bark` |
| 4 | `drilling` |
| 5 | `engine_idling` |
| 6 | `gun_shot` |
| 7 | `jackhammer` |
| 8 | `siren` |
| 9 | `street_music` |

Output-column order is fixed and must not be changed without creating a new artifact contract.

## Training protocol

Scientific evaluation uses the official UrbanSound8K folds. For test fold `k`, validation fold is `k % 10 + 1`, with the other eight folds used for training. Segments from one original clip remain within the same partition. Class weights are derived only from training segments. Seed, optimizer, early stopping, and architecture are fixed in the machine-readable fold configurations.

The evaluated Linear models are selected using validation clip macro-F1; the held-out test fold is not used for model selection. Full provenance is in the [compact classifier run manifest](../results/compact_classifier_cross_fold/fixed-baselines-v1/run-manifest.json).

## Official ten-fold evaluation

Mean and population standard deviation across ten held-out folds:

| Metric | Result |
| --- | ---: |
| Clip accuracy | 0.780485 ± 0.036271 |
| Clip macro-F1 | 0.789331 ± 0.033921 |
| Segment accuracy | 0.716043 ± 0.036224 |
| Segment macro-F1 | 0.708456 ± 0.030906 |

Source: [compact classifier aggregate](../results/compact_classifier_cross_fold/fixed-baselines-v1/aggregate.json).

## Deployment-only model distinction

After architecture choice and evaluation were complete, the same Linear architecture was trained for seven fixed epochs on all 53,918 verified cached segments. This artifact is explicitly marked:

```text
deployment_only: true
independent_test_metrics_available: false
```

It must not be evaluated on or described with the held-out predictions used for model development. The official ten-fold numbers characterize the evaluated architecture and protocol; they are not independent metrics for the particular all-data weight file. Native/ONNX numerical parity passed with 100% top-1 agreement on the deterministic parity fixture, as recorded in the [deployment parity result](../results/deployment_onnx/linear-all-data-fp32-v1/parity/parity-result.json).

## Robustness summary

A deterministic panel selected up to 20 evaluable clips per fold and class, producing 1,978 held-out clips. Each fold-specific model was used only on its own test fold. Clean top-1 predictions agreed 100% with the cached reference pipeline.

| Condition | Clip macro-F1 | Drop from clean | Flip rate | Classification |
| --- | ---: | ---: | ---: | --- |
| Clean | 0.826161 ± 0.063502 | 0.000000 | 0.000000 | Minor |
| White noise, 20 dB SNR | 0.778345 ± 0.070533 | 0.047816 | 0.119815 | Moderate |
| White noise, 10 dB SNR | 0.636696 ± 0.090411 | 0.189465 | 0.306698 | Major |
| White noise, 0 dB SNR | 0.347563 ± 0.086023 | 0.478598 | 0.572808 | Major |
| Gain −12 dB | 0.775732 ± 0.071825 | 0.050429 | 0.138535 | Moderate |
| 8 kHz bandlimit roundtrip | 0.787245 ± 0.080990 | 0.038916 | 0.162312 | Moderate |

Source: [authoritative robustness aggregate](../results/audio_robustness/linear-cross-fold-v1/aggregate.json). The panel is class-capped and excludes zero-segment clips, so its clean score must not replace the official full-fold result.

## Limitations

- The ten labels are a closed subset of real environmental sounds.
- UrbanSound8K does not establish performance for other cities, devices, recording conditions, or recent real-world distributions.
- Some fold/class cells have limited support; `gun_shot` requires particular caution.
- Controlled noise, gain, and bandlimit perturbations do not cover reverberation, mixtures, clipping, codecs, or adversarial inputs.
- YAMNet remains the main memory and latency cost.
- Short clips receive no prediction and incomplete tails are discarded.
- The interactive demo is not continuous streaming.
- The macOS arm64 dependency locks do not certify other platforms.

## Confidence calibration warning

The output is a raw softmax vector. It has not been calibrated with temperature scaling, isotonic regression, or an independent calibration set. A displayed value such as 80% must not be interpreted as an 80% guaranteed probability of correctness. The current deployment contract has no unknown class or calibrated rejection threshold.

## Ethical and operational notes

Audio can contain sensitive speech or contextual information even though this model does not target speech. Obtain consent, minimize retention, and follow applicable policy and law. The demo binds to localhost and avoids application-level cloud upload, but operating-system temporary files and browser microphone handling still apply. Do not retain or commit recordings without explicit permission.

Model, dataset, and framework components retain their upstream licensing obligations. The repository currently lacks a top-level project license; redistribution should wait for an explicit license decision.
