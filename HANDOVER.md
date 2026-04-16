# LoLvlance Handover

This document is the technical handover for the current LoLvlance codebase.

It is written to help the next engineer understand what is actually implemented, what is isolated for safety, and what still should not be treated as production-ready.

The short version is:

- the browser monitoring path is working end to end, with a mobile-first full-screen UI
- the active browser model is `v0.1-real-data` — the first real-data trained checkpoint, trained on MUSAN + OpenMIC-2018 (65K clips)
- three follow-up training runs (v0.2, v0.3, v0.4) uncovered and fixed two major pipeline bugs: a Python/JS resampler mismatch and a training dataset design flaw
- the runtime uses rolling-window monitoring with no intentional blind gap
- golden evaluation and input-integrity test infrastructure exist and are CI-gated
- v0.4 training is currently in progress with all known bugs fixed; expected to be the first promotable successor to v0.1

Korean version: `HANDOVER.ko.md`

## Quick Links

[![Overview](https://img.shields.io/badge/Project%20Overview-2563EB?style=for-the-badge&logo=bookstack&logoColor=white)](#project-overview)
[![Architecture](https://img.shields.io/badge/System%20Architecture-7C3AED?style=for-the-badge&logo=appveyor&logoColor=white)](#system-architecture)
[![ML](https://img.shields.io/badge/ML%20System-DC2626?style=for-the-badge&logo=pytorch&logoColor=white)](#ml-system)
[![Monitoring](https://img.shields.io/badge/Monitoring%20System-0891B2?style=for-the-badge&logo=webaudio&logoColor=white)](#monitoring-system)
[![Evaluation](https://img.shields.io/badge/Evaluation%20System-0F766E?style=for-the-badge&logo=githubactions&logoColor=white)](#evaluation-system)
[![Tests](https://img.shields.io/badge/Test%20Infrastructure-D97706?style=for-the-badge&logo=pytest&logoColor=white)](#test-infrastructure)
[![Workflow](https://img.shields.io/badge/Development%20Workflow-16A34A?style=for-the-badge&logo=git&logoColor=white)](#development-workflow)
[![Roadmap](https://img.shields.io/badge/Roadmap-1D4ED8?style=for-the-badge&logo=roadmapdotsh&logoColor=white)](#roadmap)

<a id="project-overview"></a>
## 1. Project Overview

LoLvlance is a browser-first audio monitoring product for live-session operators who need guidance, not a full autonomous mix system.

The intended use case is:

- church sound teams
- worship volunteers
- small live setups
- operators who need fast suggestions while handling other responsibilities

The core product questions remain:

1. What sounds wrong right now?
2. Which source is most likely involved?
3. What EQ move should the user try first?

### Product Positioning

Current product language should be:

- continuous monitoring
- live session analysis
- updated every few seconds
- assistive guidance

Avoid:

- real-time
- instant-response AI engineer
- production-grade diagnosis language

<a id="system-architecture"></a>
## 2. System Architecture

### Runtime Flow

```text
Mic Input
  -> Browser Capture Node
  -> Rolling Buffers
  -> Feature Extraction
  -> ONNX Inference (optional)
  -> Source Enrichment
  -> Rule Fallback / Post-Processing
  -> EMA Smoothing
  -> UI
```

### Main Runtime Files

- `src/app/hooks/useMonitoring.ts`
  Main orchestration layer for microphone capture, rolling buffers, inference scheduling, smoothing, and final result updates.
- `src/app/audio/audioUtils.ts`
  Resampling, circular buffer helpers, RMS/peak helpers, and buffered snapshot creation.
- `src/app/audio/featureExtraction.ts`
  Frontend feature extraction for browser inference.
- `src/app/audio/mlInference.ts`
  Browser-side ONNX runtime setup, output parsing, model-version-aware logging, and fallback behavior.
- `src/app/audio/diagnosisPostProcessing.ts`
  Derived diagnosis logic.
- `src/app/audio/sourceAwareEq.ts`
  Human-readable EQ guidance.
- `src/app/audio/ruleBasedAnalysis.ts`
  Fallback rule engine when ML is disabled or unavailable.
- `src/app/config/modelRuntime.ts`
  Model version routing, experimental/production split, and kill switch.

### Current Runtime Responsibilities

- microphone capture happens continuously
- inference is scheduled separately from capture
- source enrichment can come from the local stem service or browser fallback tagging
- final UI outputs are smoothed before display

<a id="ml-system"></a>
## 3. ML System

There are now two different ML layers to understand:

### Active Browser Model

The current browser model is:

```text
v0.1-real-data
```

This model:

- was trained on real public audio (MUSAN + OpenMIC-2018, 65,738 clips)
- is the first real-data checkpoint deployed to the browser
- has been validated through the CI golden evaluation gate
- produces meaningful predictions: guitar F1=0.82, drums AUROC=0.93, vocal AUROC=0.92, muddy AUROC=0.87
- has known weak spots: `boxy` F1=0.0, `nasal`/`thin`/`sibilant` AUROC near 0.56–0.60

See `ml/model_history.md` for full per-label metrics and artifact paths.

### Current Browser ONNX Contract

The frontend currently reads:

- `issue_probs`
- `source_probs`
- `eq_freq`
- `eq_gain_db`

This remains the browser compatibility contract.

### Python Training Path

The Python ML stack has moved beyond the original prototype and now supports:

- teacher/student variants in `ml/model.py`
- a frozen-or-trainable transformer-like spectrogram encoder for the teacher path
- a lightweight CNN student path for browser-oriented deployment
- source-conditioned issue prediction
- a learned multi-band EQ head producing `eq_params`
- self-supervised degradation-based training in `ml/degradation.py`
- distillation-ready losses in `ml/losses.py`

### Training Pipeline Bugs Fixed (Apr 16–17, 2026)

Two deep bugs were discovered and fixed during v0.2–v0.4:

**Bug 1 — Resampler mismatch** (`ml/preprocessing.py`):
- `resample_audio` used linear interpolation for downsampling
- Browser (`audioUtils.ts → resampleMonoBuffer`) uses block averaging for downsampling
- Result: train/inference feature divergence; all 10 parity tests failed with MSE 0.50–2.34
- Fix: rewrote downsampling path with vectorized numpy cumsum block averaging; upsampling uses vectorized `np.interp`

**Bug 2 — Always-positive predictions** (`ml/degradation.py`):
- `RealAudioDegradationDataset` applied degradation to 100% of training samples
- Model never saw a clean example → learned "always predict positive" as the optimal strategy
- Fix: `clean_ratio=0.25` parameter — 25% of samples are returned unmodified with all-zero issue targets

**Additional training improvements**:
- Focal loss `alpha` corrected from `0.25` to `0.75` (was downweighting positive class, opposite of intent)
- `OneCycleLR` scheduler added (warmup 10% + cosine decay); prevents issue F1 collapse seen in v0.3
- Gradient clipping (`max_norm=1.0`) added
- Epoch-level logging added to `ml/train.py` (was silent until completion)

### Important Operational Distinction

The codebase can support a more advanced learning-based model, but the active user-facing checkpoint does not yet justify production claims.

This distinction must remain explicit in product copy, demos, and handoff conversations.

<a id="model-versioning"></a>
## 4. Model Versioning

The browser now has a model isolation layer.

### Runtime Defaults

- active default version: `v0.1-real-data` (PRODUCTION)
- archived experimental version: `v0.0-pipeline-check`
- active production model path: `public/models/lightweight_audio_model.production.onnx`
- archived experimental model path: `public/models/lightweight_audio_model.onnx`

Source of truth:

- `src/app/config/modelRuntime.ts`

### Model History

Three model generations have been trained. See `ml/model_history.md` for full details.

| Version | Status | Data | Notes |
|---|---|---|---|
| `v0.0-pipeline-check` | Archived | Synthetic | Pipeline validation only |
| `v0.1-real-data` | **Active (browser default)** | MUSAN + OpenMIC (65K) | Passed CI gate |
| `v0.2-fsd50k-extended` | Not promoted | + FSD50K (85K) | Resampler bug + always-positive |
| `v0.3-resampler-fix` | Not promoted | Same as v0.2 | Resampler fixed; training dataset bug |
| `v0.4-clean-ratio` | **In training** | Same as v0.2 | All root causes fixed; 40 epochs |

### Runtime Controls

- `MODEL_VERSION`
- `ENABLE_MODEL`

Behavior:

- `ENABLE_MODEL=false` disables ML inference entirely
- `MODEL_VERSION === "v0.0-pipeline-check"` routes to the archived experimental artifact
- `MODEL_VERSION === "v0.1-real-data"` (default) routes to the production artifact path
- any other non-experimental version also routes to the production artifact path

### User-Facing Safety Layer

The experimental safety banner (`Experimental Mode`, `Not production-ready`) is only shown when `MODEL_VERSION === "v0.0-pipeline-check"`. With the default `v0.1-real-data`, the banner is not shown.

<a id="audio-pipeline"></a>
## 5. Audio Pipeline

### Browser Preprocessing Contract

Current contract:

- sample rate: `16_000`
- clip window: `3.0` seconds
- window size: `25 ms`
- hop size: `10 ms`
- FFT size: `512`
- mel bins: `64`

### Buffer Layout

The runtime keeps two rolling buffers:

- native-sample-rate buffer for stem/source paths
- resampled `16 kHz` buffer for ML feature extraction

### Why This Matters

This split avoids forcing every downstream consumer to share a single resampling path while still preserving the ML input contract.

### Current Risk Area

Preprocessing consistency remains a real risk area. The repo now has explicit tests for it, but those tests are there because this failure mode is subtle and easy to miss.

<a id="monitoring-system"></a>
## 6. Monitoring System

The current monitoring system is a rolling-window pipeline, not a fixed-window batch loop.

### Current Runtime Values

- monitoring window: `3.0` seconds
- monitoring stride: `1.0` second
- EMA alpha: `0.3`
- minimum warm-up buffer: `750 ms`

### What Changed

Old behavior:

- capture a fixed clip
- analyze
- wait
- repeat

Current behavior:

- capture continuously
- maintain a circular buffer
- re-analyze the latest `3.0` seconds every `1.0` second

### Result

- no intentional blind gap between analyses
- smoother updates
- better session coverage

### Implementation Notes

- `AudioWorklet` is used when available
- `ScriptProcessorNode` remains as a fallback
- capture is separated from inference scheduling
- if inference is still running when the next stride arrives, a pending pass is queued

### Stability Layer

`useMonitoring.ts` applies EMA smoothing to:

- problem confidence
- detected source confidence
- source-specific EQ recommendations
- raw ML-derived diagnosis scores

### Edge Cases Already Handled

- insufficient initial buffer
- silence short-circuiting
- microphone permission errors
- microphone interruption
- model disabled mode
- ML inference failure with rule fallback

<a id="evaluation-system"></a>
## 7. Evaluation System

LoLvlance now has a real evaluation path with CI gating.

### Golden Set

Location:

```text
eval/goldens/
```

Each sample contains:

- a `.wav` file
- a `metadata.json`

Metadata includes:

- `file`
- `expected_source`
- `expected_issue`
- `severity`

### Evaluator

Main script:

```text
ml/eval/evaluate.py
```

It:

- loads golden samples
- runs the ONNX model
- computes issue/source precision, recall, and F1
- prints confusion details
- checks performance against thresholds and baseline

### Baseline Tracking

Baseline file:

```text
ml/eval/baseline.json
```

Current baseline stores:

- overall F1
- per-label F1
- critical labels
- regression tolerance

### CI Gate

Workflow:

```text
.github/workflows/eval.yml
```

Current gate behavior:

- triggers on `ml/**`, `eval/**`, and workflow changes
- uses Python 3.11
- installs `ml/requirements-test.txt`
- runs the golden evaluator
- compares against `ml/eval/baseline.json`
- blocks macro regression, per-label collapse, and obvious prediction-bias regressions

### Important Limitation

The current golden set is still small. It is useful for regression detection, but it is not yet a complete real-world benchmark.

<a id="test-infrastructure"></a>
## 8. Test Infrastructure

### Input Integrity Tests

- `ml/tests/test_waveform_parity.py`
  Waveform-level parity between Python preprocessing and browser-equivalent preprocessing.
- `ml/tests/test_feature_parity.py`
  End-to-end feature parity for model inputs.
- `ml/tests/test_resampler.py`
  Resampling behavior for common browser capture rates and output safety checks.
- `ml/tests/test_manifest_leakage.py`
  Train/validation leakage checks for duplicate files, overlapping groups, and naming collisions.

### Existing ML Pipeline Tests

- `ml/tests/test_export_to_onnx.py`
- `ml/tests/test_training_pipeline.py`
- `ml/tests/test_legacy_onnx_adapter.py`

### Interpretation

These tests are there to catch failure modes that can silently damage model quality:

- preprocessing drift
- resampler drift
- dataset leakage
- ONNX export/schema breakage

Do not assume that because the tests exist, the problem is solved permanently. Their job is to keep these regressions visible.

<a id="limitations"></a>
## 9. Limitations

### Model Limitations

- active browser model is `v0.1-real-data` — real-data trained but not perfect on all labels
- `boxy` F1=0.0 (only 23 val samples); `nasal`, `thin`, `sibilant` AUROC near chance (0.56–0.60)
- guitar threshold is low (0.25) due to dataset imbalance — may over-predict on non-guitar audio
- CI golden set has only 3 samples; threshold calibration is conservative for this small set
- browser runtime still uses a compatibility single-band EQ output contract

### Data Limitations

- browser-side user feedback collection is now in place (`FeedbackWidget` + `feedbackStore.ts` + `ml/ingest_feedback.py`), but it requires manual export and retraining — not yet automated
- no human-reviewed production dataset yet
- real public datasets must be downloaded separately via `ml/download_datasets.py` before real-data training is possible
- golden evaluation set is still too small

### Inference Limitations

- browser-only inference is device-dependent
- preprocessing consistency remains a release risk
- local sidecar services still affect source enrichment quality
- silence handling intentionally suppresses some output paths

### Product Limitations

- the system should not be sold as an authoritative diagnosis engine
- guidance quality is not yet sufficient to justify strong AI performance claims

<a id="product-positioning"></a>
## 10. Product Positioning

Current product language should stay aligned with the actual runtime:

- continuous monitoring
- live session analysis
- updated every few seconds
- guidance over authority

Recommended explanation in demos:

> LoLvlance listens continuously, reviews the latest few seconds of audio on a rolling basis, and suggests what to check next.

Do not position it as:

- real-time AI diagnosis
- instant-response autonomous EQ
- production-grade audio intelligence

<a id="development-workflow"></a>
## 11. Development Workflow

### Frontend

```bash
npm install
npm run dev
npm run build
```

### Python Environment

```powershell
python -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install --upgrade pip
.\.venv-ml\Scripts\python.exe -m pip install -r ml\requirements-test.txt
```

### Run Golden Evaluation

```powershell
.\.venv-ml\Scripts\python.exe ml\eval\evaluate.py `
  --goldens-dir eval\goldens `
  --model-path ml\checkpoints\lightweight_audio_model.onnx `
  --thresholds-path ml\checkpoints\label_thresholds.json `
  --baseline-path ml\eval\baseline.json `
  --macro-epsilon 0.02 `
  --per-label-epsilon 0.03 `
  --weak-label-f1-threshold 0.40 `
  --weak-label-epsilon 0.02 `
  --max-ratio-per-label 0.50 `
  --distribution-slack 0.20 `
  --entropy-epsilon 0.12
```

### Run ML Tests

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv-ml\Scripts\python.exe -m unittest discover -s ml/tests -p 'test_*.py' -v
```

### Optional Stem Service

```powershell
.\.venv-ml\Scripts\python.exe -m pip install -r ml\requirements-stem-service.txt
.\.venv-ml\Scripts\python.exe ml\stem_separation_service.py
```

### Download Datasets and Train on Real Audio

```bash
# Download real audio (MUSAN is the simplest starting point)
python ml/download_datasets.py --datasets musan --output-root data/datasets

# Train on real audio (--audio-root accepts any folder of WAV/FLAC)
python -m ml.train \
  --audio-root data/datasets/musan \
  --rebuild-manifest --epochs 20 --export-onnx

# Update the browser model after training
cp ml/checkpoints/lightweight_audio_model.onnx public/models/
```

### Ingest User Feedback into Training

```bash
# 1. User exports feedback from browser → lolvlance_feedback_*.jsonl
# 2. Convert to manifest
python ml/ingest_feedback.py \
    --feedback lolvlance_feedback_2024-01-15.jsonl \
    --output ml/artifacts/feedback_manifest.jsonl
# 3. Merge and retrain
cat ml/artifacts/public_dataset_manifest.jsonl \
    ml/artifacts/feedback_manifest.jsonl \
    > ml/artifacts/merged_manifest.jsonl
python -m ml.train \
    --manifest-path ml/artifacts/merged_manifest.jsonl \
    --epochs 10 --export-onnx
```

### Model Promotion Checklist

1. train or export a candidate checkpoint under `ml/checkpoints/`
2. run evaluation and tests
3. copy the selected ONNX file into `public/models/`
4. confirm `MODEL_VERSION` routing
5. verify browser behavior manually
6. only then consider the artifact user-facing

<a id="roadmap"></a>
## 12. Roadmap

### Immediate

- monitor `v0.4-clean-ratio` training (PID 51567, log: `ml/train_v04.log`, 40 epochs)
- when complete: run `ml/eval/evaluate.py` with new ONNX and confirm CI gate passes
- if gate passes: run `--write-baseline` to lock new baseline, then promote to browser

### Short Term

- evaluate v0.4 against golden set; promote if issue macro F1 ≥ 0.40 and gate passes
- expand the golden evaluation set beyond 3 samples — CI gate with only 3 samples is not a real benchmark
- run feedback ingestion cycle: export browser feedback → `ml/ingest_feedback.py` → merge manifest → retrain
- improve threshold calibration with a larger golden evaluation set
- promote learned EQ outputs further into runtime-facing contracts

### Longer Term

- graduate from single-band browser EQ contract to multi-band EQ output
- consider server inference if browser-only constraints become limiting
- add human review step to the feedback ingestion cycle

<a id="handover-notes"></a>
## 13. Handover Notes

### Where to Start

- `README.md`
  Product-facing overview and current runtime summary.
- `src/app/hooks/useMonitoring.ts`
  Most important runtime file.
- `src/app/config/modelRuntime.ts`
  Model isolation and routing.
- `ml/eval/evaluate.py`
  Regression gate logic.
- `ml/tests/`
  Input-integrity and export/training regression tests.

### Practical Rules for the Next Engineer

- do not promote a new model checkpoint without running `ml/eval/evaluate.py` and confirming CI passes
- do not remove the experimental banner code — it will auto-activate if `MODEL_VERSION` is set back to `v0.0-pipeline-check`
- do not assume preprocessing parity without running the tests
- do not replace the public ONNX artifact without rerunning evaluation and updating `ml/model_history.md`

### Most Important Mental Model

There is a difference between:

- **the pipeline works**
- **the model is trustworthy**

As of Apr 17, 2026, `v0.1-real-data` is still the active browser default. The model is functional and passed CI evaluation, but three follow-up runs (v0.2, v0.3, v0.4) revealed that the training pipeline had two serious bugs that prevented the model from learning to discriminate issues correctly. Both bugs are now fixed in v0.4. Until v0.4 is evaluated and promoted, `v0.1-real-data` remains the production checkpoint.
