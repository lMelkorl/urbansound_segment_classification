# Phase 0 Repository Audit

Audit date: 2026-07-12

## Scope and evidence level

This audit is a static review of the repository and its committed result artifacts. No dataset or checkpoint was downloaded, no model was trained, and no result file was changed. The Python sources were byte-compiled, committed JSON artifacts were parsed, the checkpoint shell script passed `bash -n`, and the public Python entry points were probed with `--help`. Dataset-dependent behavior and numerical results were not re-executed.

The current branch is `feat/edge-audio-v2`. The original state is recoverable through the existing `v1-case-study` tag. The working tree was clean before this documentation was added.

## Executive summary

The repository preserves two substantive implementations and one placeholder:

- Goal 1 implements YAMNet embedding extraction followed by LightGBM.
- Goal 2 implements ESResNeXt-fbsp head-only and full fine-tuning variants.
- Goal 3 contains only a placeholder; the AudioCLIP implementation and its reported artifacts are absent.

The implemented goals use a fixed Fold 1-8 train, Fold 9 validation, Fold 10 test policy. This is a valid fold-separated legacy split, but it is not a complete UrbanSound8K 10-fold evaluation. Segment-to-clip aggregation for Goals 1 and 2 is the arithmetic mean of segment probability vectors, followed by `argmax`. The AudioCLIP README claim of `logit_mean` aggregation cannot be verified.

The committed Goal 1 and Goal 2 accuracy/F1 values are traceable to JSON files and agree with the README after rounding. Most speed, hardware, parameter-count, and Goal 3 claims are not traceable to a committed machine-readable artifact. Existing artifacts also omit the run configuration, package versions, hardware, Git revision, checkpoint identity, split manifest, raw timing samples, and test per-class results required for a reproducible benchmark.

The safest V2 migration is additive: keep `urbansound_segment_task/goals/` and its result files immutable as legacy evidence, and build a separate `urbansound_segment_task/edge_v2/` pipeline with shared data, segmentation, aggregation, evaluation, benchmark, export, and runtime contracts. New outputs should be run-scoped under versioned `results/` directories and should never default to a legacy output directory.

## Repository inventory

### Top level

| Path | Purpose | Audit finding |
| --- | --- | --- |
| `AGENTS.md` | V2 scientific and engineering rules | Complete Phase 0/Phase 1-6 constraints; currently the only V2 change relative to the tagged legacy state. |
| `README.md` | Original case-study report and commands | Contains useful legacy context but several missing or ambiguous sources and overstates reproducibility. |
| `requirements.txt` | Shared dependencies | Uses unbounded lower constraints, omits PyTorch, and does not separate legacy/training/benchmark/demo dependencies. |
| `.gitignore` | Data and binary exclusions | Ignores `data/`, `external`, `*.pt`, arrays and archives. It does not establish versioned result-output conventions. |
| `external/ESResNeXt_fbsp` | Gitlink to upstream source | Recorded as Git mode `160000` at commit `7c1488a...`, but `.gitmodules` is absent; a fresh checkout cannot initialize it as a normal submodule. The directory is empty in this checkout. |
| `.config/` | Local gcloud configuration files | Tracked environment-specific state unrelated to the case; may contain machine/user configuration and reduces portability. It should be reviewed separately before any cleanup, not deleted during Phase 0. |

### Python package

| Path | Purpose | Audit finding |
| --- | --- | --- |
| `urbansound_segment_task/src/common/paths.py` | Detects three UrbanSound8K layouts | Relative-path friendly, but validates only the presence of `fold1` when selecting a layout and does not validate schema, all ten folds, duplicates, or split invariants. |
| `urbansound_segment_task/src/common/segmenter.py` | Audio loading and full-window segmentation | Goal 1 uses this segmenter. It drops short clips and final partial windows; overlap/range inputs are not validated. |
| `urbansound_segment_task/src/common/metrics.py` | Segment metrics and probability-mean clip aggregation | Shared by Goals 1 and 2. It assumes clip labels are internally consistent and class IDs align with probability columns; neither invariant is asserted. |
| `goals/goal1_yamnet_lgbm/run.py` | YAMNet plus LightGBM legacy run | Complete single-file implementation. It downloads YAMNet at runtime and materializes all embeddings in memory. |
| `goals/goal2_esresnext/run_head_only.py` | Frozen-backbone ESResNeXt training | Complete legacy implementation, but duplicates data/index/model/evaluation logic and uses a different external directory/import convention from full fine-tuning. |
| `goals/goal2_esresnext/run_finetune.py` | Warm-up and full ESResNeXt fine-tuning | Complete legacy implementation with CPU device fallback, augmentation, TTA, and checkpoint writing. It assumes external model code already exists. |
| `goals/goal3_audioclip/run.py` | Claimed AudioCLIP approach | Placeholder only; it prints a message and accepts no real CLI contract. |
| `scripts/download_checkpoint_goal2.sh` | Downloads ESResNeXt weights | Uses `wget` and a fixed URL, but has no checksum, size validation, retry policy, provenance manifest, or platform fallback. |
| `urbansound_segment_task/README.md` | Old package notes | Commands and paths use the nonexistent name `urbansound-segment-task`, refer to nonexistent scripts, and conflict with the actual split description. |

There is no test directory, CI configuration, package metadata (`pyproject.toml`/`setup.py`), lock file, environment file, benchmark harness, export validation, model card, or machine-readable run schema.

## The three legacy methods

### Goal 1: YAMNet embeddings and LightGBM

Flow:

1. Read official metadata and choose rows by fold.
2. Decode each file as 16 kHz mono.
3. Emit complete 0.96 second windows with 50% overlap; discard the tail.
4. Invoke TF Hub YAMNet on each window and mean-pool YAMNet's internal frame embeddings to one 1024-dimensional vector.
5. Train a 700-tree LightGBM classifier with segment-derived class weights.
6. Compute segment predictions and arithmetic probability-mean clip predictions.

CPU status: TensorFlow/YAMNet and LightGBM can conceptually execute on CPU, and the code does not explicitly require CUDA. It is not an offline pipeline because `hub.load("https://tfhub.dev/google/yamnet/1")` fetches the model at runtime unless an external cache happens to exist. `n_jobs=-1` and unspecified TensorFlow thread settings make benchmark conditions uncontrolled. Embeddings for every split are held in RAM and are recomputed on every run. In the audited Python 3.13 environment, TensorFlow, TensorFlow Hub, LightGBM, librosa, SoundFile, and scikit-learn are missing, so even `--help` fails at import time.

### Goal 2: ESResNeXt-fbsp

Both scripts decode 44.1 kHz mono audio, build a flat segment index, run an AudioSet-pretrained backbone, add a ten-class linear adapter, and use the shared probability-mean clip aggregation.

- `run_head_only.py` freezes the backbone and trains only the adapter.
- `run_finetune.py` first warms the head, then unfreezes the backbone, applies waveform augmentation and weighted sampling, and optionally averages TTA probability vectors before clip aggregation.

CPU status: both scripts select `cpu` when CUDA is unavailable, load checkpoints through CPU memory, and disable CUDA autocast on CPU. Therefore a CPU execution path exists in source. It is not currently runnable from a fresh checkout: PyTorch is undeclared in `requirements.txt`, the external source checkout is broken, and model import paths/directories differ between the two scripts. Full training is likely impractical for ordinary CPU-only hardware, but inference feasibility must be measured rather than assumed. Fixed worker counts, unconditional pinned-memory settings, `persistent_workers=True` in full fine-tuning, repeated full-file decoding per segment, and no thread controls make current code unsuitable as a controlled cross-platform CPU benchmark.

The two segment indexers also differ: full fine-tuning pads clips shorter than one window, while head-only emits no segment for them. This changes the evaluated population.

### Goal 3: AudioCLIP

No implementation is present. The committed file is a three-line placeholder, `external/AudioCLIP` is absent, and none of `results_ft_q/metrics_test.json`, `metrics_throughput.json`, or `run_args.json` exists. Fold policy, segmentation, aggregation, checkpoint handling, CPU compatibility, parameter count, training time, accuracy, and throughput claims cannot be audited from this checkout. Goal 3 should remain a historical claim marked unverified until its exact source revision and artifacts are recovered; V2 must not reconstruct or present those values as newly validated legacy evidence.

## Existing metrics and provenance

All values below are existing legacy values, not newly generated metrics.

| README claim | Committed source | Verification |
| --- | --- | --- |
| Goal 1 test segment accuracy 0.744 and macro-F1 0.758 | `goals/goal1_yamnet_lgbm/results/metrics_test.json` | Exact underlying values are present and round as reported. |
| Goal 1 test clip accuracy 0.802 and macro-F1 0.819 | Same JSON | Exact underlying values are present and round as reported. |
| Goal 1 validation metrics and timings | `metrics_val.json`; class support in `val_classification_report.csv` | Present. Timings use wall-clock `time.time`; no hardware, warm-up, repeats, raw samples, or timing scope manifest is recorded. |
| Goal 1 approximately 220 segments/s | No direct artifact | Not traceable. The JSON records seconds but not segment counts in the same artifact or a benchmark protocol. The script's `val_segment_infer_seconds` times only LightGBM, not YAMNet or end-to-end inference. |
| Goal 1 approximately 15 minute training | No direct normalized artifact | The file records embedding and LightGBM components, but README does not define which components are included and the total is not stored. |
| Goal 1 V5 clip F1 0.828 | Claimed `results_v5`, absent | Not verifiable. |
| Goal 2 v2 test segment accuracy 0.768 and macro-F1 0.740 | `goals/goal2_esresnext/results_finetune_v2/metrics_test.json` | Exact underlying values are present and round as reported. |
| Goal 2 v2 test clip accuracy 0.841 and macro-F1 0.836 | Same JSON | Exact underlying values are present and round as reported. |
| Goal 2 training 2629.8 s and validation inference 18.6 s | `results_finetune_v2/metrics_val.json` | Present, but timing protocol and hardware are not embedded. Validation inference includes data loading, model, TTA, and Python loop, but scope is not declared in the artifact. |
| Goal 2 NVIDIA A100 40 GB and approximately 25M parameters | No machine-readable artifact | Not verifiable from committed evidence. |
| Goal 3 accuracy/F1, approximately 464 segments/s, 11.1 s, 60-90 min, approximately 134M parameters | Referenced `results_ft_q` artifacts are absent | Not verifiable. |

Only validation per-class CSV reports and validation confusion-matrix PNGs are committed for Goals 1 and 2. Test per-class reports and test confusion matrices are absent. The PNGs do not carry source arrays or run manifests. The result JSON files do not contain schema versions, sample/clip counts, class ordering, configuration, seed, package/runtime versions, hardware, Git revision, checkpoint/model hash, exact split, or provenance links. Git history shows when artifacts were committed, but not a reproducible command or environment.

## Fold split and leakage review

### What is correct

- Goal 1 and both Goal 2 scripts filter metadata rows before segment generation using disjoint fold sets: train 1-8, validation 9, and test 10.
- All segments from a metadata row therefore stay in that row's official fold.
- Training-only class weights/sampling are derived from the training segment index.
- Validation controls model selection in Goal 2; the test fold is evaluated after training.

### Limitations and risks

- This is a legacy single split, not ten-fold cross-validation. README wording such as “official UrbanSound8K protocol” is ambiguous and can be read as complete official fold evaluation.
- No executable assertion checks fold disjointness, clip-key disjointness, metadata uniqueness, class schema, or the presence of all official folds.
- Clip identity is only `slice_file_name`. This is normally sufficient for the expected dataset, but V2 should use an explicit stable key such as fold plus filename and validate uniqueness rather than rely on an assumption.
- Segment generation occurs after fold filtering, which prevents segment-level random-split leakage. However, no machine-readable split manifest exists to prove the exact rows used by a historical run.
- Model selection is based on validation segment macro-F1 while headline comparison emphasizes clip metrics. This is not leakage, but the selection target must be recorded and kept consistent in future comparisons.
- Reusing fixed output directories allows stale checkpoints or mixed artifacts to affect a later run. This can invalidate provenance even when the data split itself is correct.
- The code evaluates the test fold on every invocation. A disciplined cross-fold runner must isolate configuration/model selection from held-out test evaluation and record the policy to reduce accidental test-guided iteration.

## Segmentation and clip aggregation

The shared aggregator groups segments by clip ID, averages each class probability arithmetically, and chooses the largest mean probability. Clip ground truth is the first segment label in the group. It does not validate that all labels in a group agree. Goal 2 TTA first averages probability vectors per segment and then applies the same clip-level averaging.

Important inconsistencies:

- Goal 1/shared segmenter emits only full windows and drops clips shorter than 0.96 seconds as well as final partial tails.
- Goal 2 head-only also drops short clips, using duration-derived segment counts.
- Goal 2 full fine-tuning pads short clips to one window but drops ordinary final tails.
- Goal 1 uses `int(sr * win_sec)` while Goal 2 uses rounded sizes. At current sample rates and 0.96 seconds these are integral, but the implementations can diverge for other configurations.
- Invalid overlap/window values are not rejected consistently.
- README globally states 16 kHz mono, while Goal 2 actually uses 44.1 kHz.
- AudioCLIP's documented `logit_mean` behavior cannot be matched to source.

V2 should define one tested segmentation policy, including tail/short-clip behavior, stable clip identity, sample rounding, and aggregation type. Any change from legacy behavior must be versioned and should not be used to silently reinterpret legacy metrics.

## Risk register

| Priority | Area | Risk | Consequence | Recommended control |
| --- | --- | --- | --- | --- |
| Critical | Provenance | Goal 3 code/results and Goal 1 V5 results are absent. | README comparisons cannot be reproduced or fully audited. | Mark claims as unverified legacy evidence; recover original immutable sources if available, otherwise use `TBD` for V2. |
| Critical | External source | ESResNeXt is a gitlink without `.gitmodules`; scripts disagree on underscore/hyphen paths and import namespaces. | Fresh checkout cannot run Goal 2 reliably. | Record upstream URL and immutable commit in an explicit dependency manifest or valid submodule; add an offline preflight. |
| High | Result integrity | Legacy commands write directly into historical result directories; no atomic/run-scoped output or overwrite guard exists. | Historical artifacts can be overwritten or mixed with stale files. | Make legacy results read-only by convention; default all V2 outputs to unique versioned run directories and fail on collision. |
| High | Checkpoints | Checkpoint download has no checksum; `torch.load` consumes pickle-compatible files; identity is not logged. | Corruption, supply-chain exposure, or irreproducible model identity. | Publish URL, license, SHA-256 and size; verify before load; prefer weights-only/safe formats where compatible. |
| High | Checkpoints | Full fine-tuning may load an old `best_finetune.pt` already present in the output directory. If fine-tuning never beats warm-up, no best full checkpoint is written and final weights may be evaluated. In-memory `state_dict()` snapshots are shallow. | Reported metrics may correspond to stale or non-best weights. | Use fresh run directories, deep/serialized snapshots, explicit best-stage metadata, and checkpoint/run compatibility validation. |
| High | Dependencies | `>=` constraints are not pins; PyTorch is undeclared; external dependencies are unspecified; no lock or supported Python matrix exists. | Non-repeatable installs and resolver/runtime incompatibilities. | Define a supported Python version and lock tested environments; separate core, legacy, training, benchmark, export, and demo extras. |
| High | Offline behavior | Goal 1 downloads TF Hub YAMNet during execution; Goal 2 scripts may clone/download dependencies. | Runtime is not offline or deterministic. | Make model acquisition an explicit verified setup step; runtime accepts local immutable artifacts only. |
| High | Evaluation | No split manifest or invariant tests; fixed split is described too broadly. | Accidental leakage or scientifically invalid comparisons may go unnoticed. | Build deterministic fold manifests and assert clip/fold disjointness before training/evaluation. |
| High | Benchmarking | Timings use `time.time`, lack warm-up/repeats/raw samples/thread settings, and mix different scopes. | Throughput and latency claims are not comparable or reproducible. | Use a shared `perf_counter_ns` harness with explicit stages, metadata, warm-up, distributions, and raw samples. |
| Medium | Segmentation | Short/tail behavior differs across methods and long clips contribute more training segments. | Evaluation populations and weighting differ across approaches. | Centralize and test segmentation; record counts and policy; report both segment- and clip-weighted views as specified. |
| Medium | CPU portability | Hard-coded worker counts, `n_jobs=-1`, unconditional pinned memory, persistent workers, and no thread manifest. | Oversubscription, platform issues, and misleading CPU measurements. | Provide explicit thread/worker configuration with conservative portable defaults and record effective values. |
| Medium | Memory/I/O | Goal 1 stores all embeddings in RAM; Goal 2 decodes the complete clip for every segment item. | Excess memory and repeated CPU/audio decoding cost. | Add content-addressed feature caching and clip-aware decoding without changing the legacy code. |
| Medium | Metrics | Macro-F1 does not pass an explicit full class list; probability columns/class IDs are assumed aligned; clip labels are not validated. | Missing classes or noncontiguous labels can silently change results. | Pass and persist canonical class order; validate labels and probability shape/order. |
| Medium | Packaging | Scripts assume repository-root CWD and package availability; nested README commands are stale. | Commands fail from other directories and on fresh installs. | Add package metadata and tested module/console entry points; resolve defaults from the repository/config. |
| Medium | Testing | No unit, smoke, integration, or CI tests exist. | Split, segmentation, aggregation, schema, and CLI regressions are undetected. | Add dataset-free unit tests first, then marked dataset/model smoke tests. |
| Low | Repository hygiene | `.DS_Store` and local `.config` files are tracked; nested `.gitignore` is empty; `external` ignore naming is broad. | Noise, potential local-state exposure, and unclear ownership. | Review deliberately in a separate approved cleanup; do not delete without inspection. |

## Dependency and environment findings

The current requirements declare TensorFlow, TensorFlow Hub, LightGBM, librosa, SoundFile, scikit-learn, pandas, NumPy, tqdm, and matplotlib using only lower bounds. Goal 2 imports PyTorch but PyTorch is not declared. No dependency file covers AudioCLIP, ESResNeXt, ONNX Runtime, benchmarking, testing, or the future demo.

The audited machine is Darwin arm64 with Python 3.13.5. The active interpreter has pandas 2.3.1, NumPy 2.2.6, tqdm 4.67.1, matplotlib 3.10.3, and torch 2.7.1, but lacks TensorFlow, TensorFlow Hub, LightGBM, librosa, SoundFile, scikit-learn, torchaudio, and ONNX Runtime. `pip check` reports no broken installed packages, but that does not mean repository requirements are installed. Goal 1 and Goal 2 fail before argparse can display `--help`; Goal 3 merely prints its placeholder message.

Python 3.9+ in README is too broad without a tested framework/platform matrix, especially for TensorFlow and arm64. A reproducible baseline should choose and test a narrower Python version, then generate locked files per supported platform or document resolution constraints. This should be proposed and reviewed before dependency changes; no mass upgrade is justified.

## Duplicate code and maintainability

The two Goal 2 scripts duplicate seed setup, segment indexing, dataset loading, external-model setup, wrapper/head classes, dimension inference, device selection, evaluation, checkpointing, metric writing, and plotting. Goal 1 and Goal 2 also contain method-local orchestration around shared but underspecified segmentation/metric behavior. These legacy scripts should remain intact for history. Duplication should be removed only in new `edge_v2` modules, not by refactoring the historical implementations in place.

The future single sources of truth should be:

- a validated dataset index and fold manifest;
- one versioned segmentation/preprocessing specification;
- one aggregation and metric contract with canonical class order;
- model adapters exposing load/preprocess/infer metadata without owning evaluation;
- one result schema used by evaluation, benchmark, export validation, CLI, and demo.

## README statements requiring clarification

1. Label all current numbers as **Legacy single-split results (train folds 1-8, validation fold 9, test fold 10)**, not a complete official cross-fold benchmark.
2. Replace “Pinned package versions” with the actual state: lower bounds only and no lock file.
3. Qualify “Goals 2 and 3 GPU required”: Goal 2 has a CPU code path but feasibility is unbenchmarked; Goal 3 is unavailable. Training practicality and inference compatibility are different claims.
4. Remove or mark `TBD/unverified` for Goal 3 metrics, timings, parameter count, throughput, and “fastest” status until artifacts are restored.
5. Remove or mark unverified the missing Goal 1 V5 result.
6. Define Goal 1's 220 segments/s and 15 minute scopes or link a benchmark artifact; current JSON does not directly support those claims.
7. Embed a machine-readable source for hardware and parameter counts before repeating A100/25M/134M statements as measured facts.
8. State method-specific sample rates: 16 kHz for YAMNet and 44.1 kHz for current ESResNeXt code.
9. Clarify that Goal 1 arithmetic-means probabilities; AudioCLIP `logit_mean` is not verifiable here.
10. Repair the nested README's nonexistent hyphenated paths and download/report scripts, or label it historical.
11. Replace the zip-oriented installation instructions with a reproducible checkout/environment workflow after dependency policy is approved.

README should not be edited until the result/provenance labels and target environment are reviewed; Phase 0 preserves the original report.

## Recommended V2 architecture and migration path

Keep all legacy code and artifacts where they are. Add V2 only when a component has a real consumer:

```text
urbansound_segment_task/
  edge_v2/
    configs/        # validated, versioned experiment/runtime configuration
    data/           # metadata validation and deterministic fold manifests
    features/       # local, keyed, non-committed feature cache contracts
    models/         # adapters plus compact candidates
    evaluation/     # canonical aggregation, metrics, cross-fold orchestration
    benchmarks/     # staged CPU timers and result schema
    export/         # ONNX export and native/export parity checks
    runtime/        # one offline preprocessing/inference pipeline
    demo/           # thin UI over runtime
    utils/          # seeds, environment and structured logging
configs/edge_v2/
scripts/
results/
  legacy/           # references/manifests for immutable historical artifacts
  cpu_benchmarks/
  cross_fold/
  compact_model/
  quantization/
  robustness/
tests/
```

Data flow should be explicit and one-directional:

1. Validate metadata/audio inventory and create a versioned split manifest.
2. Apply one serialized preprocessing/segmentation configuration.
3. Call a model adapter that declares artifact identity, input contract, runtime and precision.
4. Aggregate through one canonical, tested evaluator.
5. Write an immutable run directory containing configuration, environment, artifact hashes, raw outputs/timings, and summaries.
6. Let CLI and demo consume the same offline runtime rather than reimplement preprocessing.

The initial compact-model path should follow the preferred practical route in `AGENTS.md`: cache verified YAMNet embeddings, compare LightGBM with small temporal candidates, select on the accuracy/latency/size/memory Pareto frontier, then export the selected compact classifier to ONNX. This is an architectural direction, not a claim that YAMNet extraction itself satisfies offline latency or size targets; Phase 1 benchmarks must establish that.

## Validation performed and current limitations

Performed:

- Read the complete `AGENTS.md`.
- Inspected all tracked Python, shell, README, dependency, ignore, metric JSON, and report CSV files.
- Verified the branch/tag and clean pre-documentation working tree.
- Parsed all committed JSON metrics successfully.
- Byte-compiled all Python files successfully.
- Validated checkpoint script shell syntax.
- Probed public entry points and installed-package state without downloads.
- Inspected Git history and the malformed external gitlink/submodule state.

Not performed:

- Dataset inventory or metadata-level disjointness validation, because the dataset is absent.
- Model import/forward-pass validation, because required dependencies/external sources/checkpoints are absent.
- Any CPU/GPU timing, accuracy, memory, or throughput benchmark.
- Checkpoint download, checksum discovery, or model-size verification.
- AudioCLIP validation, because code and artifacts are absent.

Therefore the existing accuracy numbers are only verified as committed legacy artifacts that match README text, not independently reproduced measurements.
