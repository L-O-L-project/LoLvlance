# ML README

This document covers the ML-specific part of LoLvlance:

- current runtime model status
- model and export architecture
- degradation-based training path
- evaluation and CI gating
- input-integrity test coverage

The most important framing is:

- the **active browser checkpoint** is still `v0.0-pipeline-check`
- the **Python ML codebase** now supports a more advanced learning-based path
- the existence of that code does not change the current product honesty requirements

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
- The active runtime model is isolated as `v0.0-pipeline-check`.
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

### Current Training Direction

The ML stack is now structured around a real-audio-plus-degradation path rather than a pure rule-projection path.

Key files:

- `ml/train.py`
- `ml/degradation.py`
- `ml/losses.py`
- `ml/model.py`

### Data Strategy in Code

The repo now supports a degradation-based dataset path:

- start from real source audio manifests
- apply controlled degradations
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

- focal loss for issue prediction
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

Current threshold:

- `min_f1 = 0.65`

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

- browser ML loading path works
- ONNX export path works
- model isolation exists
- monitoring pipeline uses a rolling window with no intentional blind gap
- golden evaluation and integrity tests exist

### What Is Not Production-Ready

- the active browser checkpoint
- synthetic-data bias in the current runtime artifact
- source reliability on real audio
- real-world benchmark coverage

### Correct Interpretation

Today the repo proves:

- the pipeline can run
- the system can be evaluated
- regressions can be caught

It does **not** prove:

- production accuracy
- production trustworthiness
- production-ready EQ intelligence
