# ML README

This document covers the ML-specific part of LoLvlance:

- current runtime model status and training results
- model and export architecture
- degradation-based training path
- evaluation and CI gating
- input-integrity test coverage
- model history reference

Korean version: `ML_README.ko.md`

## Quick Links

[![Architecture](https://img.shields.io/badge/Model%20Architecture-2563EB?style=for-the-badge&logo=pytorch&logoColor=white)](#model-architecture)
[![Training](https://img.shields.io/badge/Training%20Pipeline-7C3AED?style=for-the-badge&logo=python&logoColor=white)](#training-pipeline)
[![Evaluation](https://img.shields.io/badge/Evaluation%20System-0891B2?style=for-the-badge&logo=target&logoColor=white)](#evaluation-system)
[![Integrity](https://img.shields.io/badge/Input%20Integrity-DC2626?style=for-the-badge&logo=pytest&logoColor=white)](#input-integrity)
[![Main README](https://img.shields.io/badge/Main-README-111827?style=for-the-badge&logo=gitbook&logoColor=white)](README.md)
[![Handover](https://img.shields.io/badge/Handover-111827?style=for-the-badge&logo=files&logoColor=white)](HANDOVER.md)

## Runtime Note

- The browser runtime analyzes a rolling `3.0` second window.
- Monitoring passes run on a `1.0` second stride in `src/app/hooks/useMonitoring.ts`.
- The active runtime model is `v0.1-real-data` — the first real-data checkpoint, trained on MUSAN + OpenMIC-2018.
- v0.2 and v0.3 uncovered two training pipeline bugs (resampler mismatch + always-positive collapse) and were not promoted.
- v0.4 (`v0.4-clean-ratio`) is currently training with all root causes fixed; 40 epochs, PID 51567, log: `ml/train_v04.log`.
- Product positioning should remain continuous monitoring and live session analysis, not instant-response AI.

<a id="model-architecture"></a>
## 1. Model Architecture

### Active Browser Runtime Contract

The browser ONNX contract remains:

- `issue_probs`
- `source_probs`
- `eq_freq`
- `eq_gain_db`

This is a compatibility contract used by the frontend.

### Python Model Code

The current Python model implementation lives in `ml/model.py` and supports:

- `AudioIntelligenceNet`
- `ProductionAudioIntelligenceNet`
- `LightweightAudioAnalysisNet`

### Encoder Variants

- `student`
  lightweight CNN encoder intended for browser-oriented deployment
- `teacher`
  transformer-style spectrogram encoder intended for stronger offline training and distillation workflows

### Current Learned Heads

The current model code supports:

- source classification head
- source-conditioned issue classification head
- learned multi-band EQ head

The learned EQ head predicts:

```text
eq_params: (batch, N_bands, 2)
```

where each band contains:

- frequency in Hz
- gain in dB

Default band count in code:

```text
N_bands = 5
```

### Important Distinction

The Python training model now supports learned multi-band EQ, but the browser-facing ONNX export currently emits a compatibility summary:

- `eq_freq`
- `eq_gain_db`

That is a runtime contract decision, not a statement that the internal model remains single-band.

### Label Schema

Issue labels:

```text
[muddy, harsh, buried, boomy, thin, boxy, nasal, sibilant, dull]
```

Source labels:

```text
[vocal, guitar, bass, drums, keys]
```

Schema source-of-truth:

- Python: `ml/label_schema.py`
- Frontend mirror: `src/app/audio/mlSchema.ts`

<a id="training-pipeline"></a>
## 2. Training Pipeline

### Training History

See `ml/model_history.md` for the complete record of all trained models with full metrics, artifact paths, and dataset details.

Summary:

| Version | Status | Clips | Macro Issue F1 | Macro Source F1 | Notes |
|---|---|---|---|---|---|
| `v0.0-pipeline-check` | Archived | Synthetic | N/A | N/A | Pipeline validation |
| `v0.1-real-data` | **Active** | 65,738 | 0.531 | 0.612 | CI gate passed |
| `v0.2-fsd50k-extended` | Not promoted | ~85K | — | — | Resampler bug + always-positive |
| `v0.3-resampler-fix` | Not promoted | ~85K | 0.156 | 0.511 | Resampler fixed; dataset bug |
| `v0.4-clean-ratio` | **In training** | ~85K | TBD | TBD | All root causes fixed |

### Pipeline Bugs Discovered and Fixed (Apr 16–17, 2026)

#### Bug 1: Resampler Mismatch

- **Where**: `ml/preprocessing.py → resample_audio`
- **What**: Python used linear interpolation for downsampling. The browser (`audioUtils.ts → resampleMonoBuffer`) uses block averaging for downsampling.
- **Impact**: Feature divergence between training and inference. All 10 parity tests failed (MSE 0.50–2.34 vs. threshold 0.0001).
- **Fix**: Vectorized cumsum block averaging for downsampling; vectorized `np.interp` for upsampling. 23/23 tests pass.

#### Bug 2: Always-Positive Collapse

- **Where**: `ml/degradation.py → RealAudioDegradationDataset`
- **What**: Every training sample had degradation applied. The model never saw a clean input and learned "always predict positive" as the loss-minimizing strategy.
- **Impact**: CI eval shows predicted-positive ratio ≈ 1.0 for 7/9 issue labels on all golden samples.
- **Fix**: `clean_ratio=0.25` field in `DegradationConfig`. 25% of samples are returned unmodified with all-zero issue targets. Focal loss `alpha` also corrected from `0.25` (which downweighted positives) to `0.75`.

#### Additional Training Improvements

- `OneCycleLR` scheduler: prevents the issue F1 collapse seen in v0.3 (0.525 → 0.005 from epoch 1 to 2)
- Gradient clipping (`max_norm=1.0`): prevents early training instability
- Epoch-level logging: `ml/train.py` now prints per-epoch progress during training (was silent until completion)

### Current Training Direction

The ML stack is now structured around a real-audio-plus-degradation path rather than a pure rule-projection path.

Key files:

- `ml/train.py`
- `ml/degradation.py`
- `ml/losses.py`
- `ml/model.py`

### Acquiring Real Training Data

Use `ml/download_datasets.py` to download supported public datasets:

```bash
# List available datasets
python ml/download_datasets.py --list

# Download MUSAN (~11 GB, public domain, recommended starting point)
python ml/download_datasets.py --datasets musan --output-root data/datasets

# Download all (MUSAN + FSD50K + OpenMIC)
python ml/download_datasets.py --datasets musan fsd50k openmic --output-root data/datasets
```

Then train:

```bash
python -m ml.train \
  --audio-root data/datasets/musan \
  --musan-root data/datasets/musan \
  --rebuild-manifest --epochs 20 --export-onnx
```

`--audio-root` accepts any folder of WAV/FLAC files. Files with `vocal`, `guitar`, `drums`, etc. in their names automatically get source labels via `infer_source_targets_from_path`.

### Ingesting User Feedback

After deployment, users can export per-analysis feedback from the browser. Ingest it with:

```bash
python ml/ingest_feedback.py \
    --feedback lolvlance_feedback_2024-01-15.jsonl \
    --output ml/artifacts/feedback_manifest.jsonl

# Merge and retrain
cat ml/artifacts/public_dataset_manifest.jsonl \
    ml/artifacts/feedback_manifest.jsonl \
    > ml/artifacts/merged_manifest.jsonl

python -m ml.train \
    --manifest-path ml/artifacts/merged_manifest.jsonl \
    --epochs 10 --export-onnx
```

Feedback entries with `verdict=correct` become weak soft labels; entries with `verdict=wrong` and user-selected corrected labels become reviewed hard labels.

### Data Strategy in Code

The repo now supports a degradation-based dataset path:

- start from real source audio manifests (via `--audio-root` or public dataset roots)
- apply controlled degradations at training time via `RealAudioDegradationDataset`
- train issue/source heads on degraded audio
- train the EQ head against inverse EQ targets

### Implemented Degradation Types

The current degradation pipeline includes:

- EQ distortion
- compression
- synthetic reverb
- low-pass / high-pass filter errors

The generated recipe stores:

- inverse EQ bands
- issue labels
- optional compression metadata
- optional reverb metadata
- optional filter-error metadata

### Training Losses

Current loss modules in `ml/losses.py`:

- focal loss for issue prediction (`alpha=0.75`, `gamma=2.0` — positive-upweighted)
- BCE or softmax source loss depending on head mode
- Smooth L1 loss for EQ regression
- KL-based distillation loss for student training

### Distillation Path

The current code supports teacher-student training:

- teacher checkpoint can be provided to `ml/train.py`
- student can distill issue logits
- student can distill source logits
- student can distill normalized EQ outputs

### Current Reality Check

The training code is ahead of the active model artifact.

The repo supports a more advanced ML path, but the checked-in browser model is still the isolated experimental checkpoint.

<a id="evaluation-system"></a>
## 3. Evaluation System

### Golden Evaluation

Golden samples live under:

```text
eval/goldens/
```

Evaluator:

```text
ml/eval/evaluate.py
```

The evaluator:

- loads golden sample metadata
- runs model inference
- computes precision, recall, and F1 for issue and source labels
- prints a confusion summary
- compares results against a stored baseline

### Baseline Tracking

Baseline file:

```text
ml/eval/baseline.json
```

Used for:

- previous checkpoint comparison
- per-label regression detection
- CI gating

### CI Gate

Workflow:

```text
.github/workflows/eval.yml
```

Current gate:

- evaluates `public/models/lightweight_audio_model.production.onnx` (always the current promoted model)
- compares against `ml/eval/baseline.json` (written from v0.1 metrics)
- allows small baseline-relative movement with configurable epsilons
- fails on macro regression or per-label F1 collapse vs baseline
- `--max-ratio-per-label 1.0` disables the absolute distribution ceiling (too strict with only 3 golden samples); re-enable after golden set is expanded

When a new model is promoted:
1. Run `evaluate.py --write-baseline` to record new baseline
2. Update `ml/checkpoints/label_thresholds.json` to new thresholds
3. Commit both files alongside the new ONNX

### Important Limitation

The current golden set is still small. It is useful for regression detection, but it is not yet enough to justify strong model-quality claims.

<a id="input-integrity"></a>
## 4. Input Integrity

The repo now includes explicit tests for preprocessing consistency and dataset hygiene.

### Tests

- `ml/tests/test_waveform_parity.py`
- `ml/tests/test_feature_parity.py`
- `ml/tests/test_resampler.py`
- `ml/tests/test_manifest_leakage.py`

### Coverage

These tests check:

- waveform parity between Python and browser-equivalent preprocessing
- feature parity for final model inputs
- resampler behavior across common browser capture rates
- train/validation leakage in the manifest

### Why This Exists

Model quality can degrade because of input inconsistency even when the architecture is unchanged.

These tests are meant to make that failure mode visible.

<a id="current-status"></a>
## 5. Current Status

### What Is Production-Like

- browser ML loading path works end to end
- ONNX export path works
- `v0.1-real-data` is trained on real public audio and deployed as default
- source detection is functional: guitar F1=0.82, drums AUROC=0.93, vocal AUROC=0.92
- monitoring pipeline uses a rolling window with no intentional blind gap
- golden evaluation and integrity tests exist
- CI gate validates model regressions before merging
- real dataset download pipeline exists (`ml/download_datasets.py`)
- browser feedback collection and ingestion pipeline exists (`FeedbackWidget` → `ml/ingest_feedback.py`)

### What Still Needs Improvement

- `v0.4-clean-ratio` is in training — until it completes and passes CI, `v0.1-real-data` remains the only promoted model
- CI golden set has only 3 samples — not a real production benchmark; must be expanded
- feedback ingestion is manual (export → script → retrain), not automated
- browser EQ output is still single-band compatibility contract, not full multi-band

### Correct Interpretation

Today the repo proves:

- the pipeline can run on real audio end to end
- source detection generalizes to real recordings (v0.1)
- issue detection is meaningful on common labels when the training dataset is balanced (v0.4 target)
- regressions can be caught via CI

It does **not** yet prove:

- production accuracy across all 9 issue labels (v0.4 is the first attempt with a correctly designed training set)
- reliable prediction on `boxy`, `nasal`, `thin` (sparse training data, hard to fix without more labeled audio)
- production-ready EQ intelligence (single-band only in browser)

### v0.4 Training Parameters (current run)

```bash
.venv-ml/bin/python -m ml.train \
  --musan-root data/datasets/musan/musan \
  --openmic-root data/datasets/openmic/openmic-2018 \
  --fsd50k-root data/datasets/fsd50k/FSD50K.dev_audio \
  --manifest-path ml/artifacts/manifest_musan_openmic_fsd50k.jsonl \
  --clips-per-file 2 \
  --epochs 40 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --clean-ratio 0.25 \
  --grad-clip 1.0 \
  --num-workers 0 \
  --export-onnx \
  --onnx-output public/models/lightweight_audio_model.production.onnx
```
