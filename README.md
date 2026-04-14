# LoLvlance

LoLvlance is a browser-based audio monitoring system for church sound teams, worship volunteers, and small live setups.

The current product goal is simple:

**continuous monitoring + issue guidance + first-step EQ suggestions during a live session**

LoLvlance listens to microphone audio in the browser, analyzes a rolling audio window, updates guidance every few seconds, and helps the user decide what to check next. It is an assistive system, not an authority system.

## Quick Links

[![Project Overview](https://img.shields.io/badge/Project%20Overview-2563EB?style=for-the-badge&logo=bookstack&logoColor=white)](#project-overview)
[![System Architecture](https://img.shields.io/badge/System%20Architecture-7C3AED?style=for-the-badge&logo=appveyor&logoColor=white)](#system-architecture)
[![Model Versioning](https://img.shields.io/badge/Model%20Versioning-DC2626?style=for-the-badge&logo=onnx&logoColor=white)](#model-versioning)
[![Monitoring](https://img.shields.io/badge/Monitoring%20System-0891B2?style=for-the-badge&logo=webaudio&logoColor=white)](#monitoring-system)
[![Evaluation](https://img.shields.io/badge/Evaluation%20System-0F766E?style=for-the-badge&logo=githubactions&logoColor=white)](#evaluation-system)
[![Tests](https://img.shields.io/badge/Test%20Infrastructure-D97706?style=for-the-badge&logo=pytest&logoColor=white)](#test-infrastructure)
[![Workflow](https://img.shields.io/badge/Development%20Workflow-16A34A?style=for-the-badge&logo=vite&logoColor=white)](#development-workflow)
[![Roadmap](https://img.shields.io/badge/Roadmap-1D4ED8?style=for-the-badge&logo=roadmapdotsh&logoColor=white)](#roadmap)

[![Korean README](https://img.shields.io/badge/Korean-README-111827?style=for-the-badge&logo=readme&logoColor=white)](README.ko.md)
[![Developer Handover](https://img.shields.io/badge/Developer-Handover-111827?style=for-the-badge&logo=gitbook&logoColor=white)](HANDOVER.md)
[![ML README](https://img.shields.io/badge/ML-README-111827?style=for-the-badge&logo=pytorch&logoColor=white)](ML_README.md)

## Language Versions

- English:
  - `README.md`
  - `HANDOVER.md`
  - `ML_README.md`
- Korean:
  - `README.ko.md`
  - `HANDOVER.ko.md`
  - `ML_README.ko.md`

<a id="project-overview"></a>
## 1. Project Overview

LoLvlance is designed to help users answer three questions during a session:

1. What sounds off right now?
2. Which source is most likely involved?
3. What EQ move should the user try first?

The intended user is not a full-time mix engineer. The product is aimed at people running sound while also doing other jobs, especially church teams and small live crews that need practical guidance under time pressure.

### Product Positioning

- LoLvlance is **continuous monitoring**, not instant-response analysis.
- The system analyzes a rolling audio window and updates guidance every few seconds.
- The output should be treated as **assistive guidance**, not as a guaranteed diagnosis.

### What the Product Does Today

- captures microphone audio in the browser
- maintains a rolling analysis buffer
- extracts audio features locally
- runs an ONNX model locally when enabled
- merges ML output with rule-based diagnostics, source detection, and EQ guidance

### What the Product Does Not Claim

- it is not a production-grade AI diagnosis system yet
- it is not a fully learned EQ system in the browser path
- it does not provide sub-second response guarantees

<a id="system-architecture"></a>
## 2. System Architecture

### End-to-End Flow

```text
Mic Input
  -> Browser Audio Capture
  -> Native Rolling Buffer + Resampled Rolling Buffer
  -> Feature Extraction
  -> ONNX Inference (optional)
  -> Source Enrichment / Rule Fallback
  -> EMA Smoothing
  -> UI
```

### Runtime Responsibilities

- `src/app/hooks/useMonitoring.ts`
  Main runtime orchestrator for microphone capture, rolling buffers, inference scheduling, smoothing, stem service enrichment, and UI updates.
- `src/app/audio/audioUtils.ts`
  Buffer utilities, resampling helpers, buffered snapshot creation, and related audio math.
- `src/app/audio/featureExtraction.ts`
  Frontend feature extraction from buffered audio.
- `src/app/audio/mlInference.ts`
  Browser ONNX loading, output parsing, model-version-aware logging, and ML fallback behavior.
- `src/app/audio/diagnosisPostProcessing.ts`
  Derived diagnoses such as `vocal_buried` and other schema-level post-processing.
- `src/app/audio/sourceAwareEq.ts`
  Human-readable EQ guidance.
- `src/app/audio/ruleBasedAnalysis.ts`
  Deterministic fallback issue analysis when ML is disabled or unavailable.

### Runtime Split of Responsibilities

- **ML**
  Estimates issue probabilities and source probabilities from log-mel input.
- **Rules and post-processing**
  Handle silence gating, interpret predictions, merge source evidence, stabilize outputs, and generate user-facing guidance.

<a id="model-versioning"></a>
## 3. Model Versioning

LoLvlance now has an explicit model isolation layer.

### Current Runtime Defaults

- active default model version: `v0.0-pipeline-check`
- config file: `src/app/config/modelRuntime.ts`
- default browser model path: `public/models/lightweight_audio_model.onnx`
- reserved production path: `public/models/lightweight_audio_model.production.onnx`

### Current Model Status

The active browser model is intentionally marked as:

```text
v0.0-pipeline-check
```

This means:

- it is a pipeline-validation artifact
- it is not production-ready
- it was trained on synthetic data with known bias issues
- its outputs may be inaccurate on real voice and music

### Kill Switch and Routing

The frontend runtime supports:

- `MODEL_VERSION`
- `ENABLE_MODEL`

Behavior:

- if `ENABLE_MODEL=false`, browser ML is skipped and the app uses fallback analysis only
- if `MODEL_VERSION === "v0.0-pipeline-check"`, the experimental model path is used
- otherwise, the runtime can route to a future production model artifact

### UI Status

The current UI explicitly warns when the experimental model is active:

- `Experimental Mode`
- `Not production-ready`
- `Results may be inaccurate`

<a id="ml-system"></a>
## 4. ML System

There are now two different ways to think about the ML system:

### Active Browser Runtime

The browser currently serves an **experimental checkpoint** and consumes a compatibility ONNX contract:

- `issue_probs`
- `source_probs`
- `eq_freq`
- `eq_gain_db`

That runtime path is isolated for safety and should not be treated as production AI.

### Python Training Path

The Python-side ML system has evolved beyond the original two-head CNN prototype.

Current code supports:

- teacher/student model variants in `ml/model.py`
- a transformer-style spectrogram encoder for the teacher path
- a lightweight CNN encoder for the student path
- source-conditioned issue prediction
- a learned multi-band EQ head producing `eq_params`
- self-supervised degradation training in `ml/degradation.py`
- distillation-ready losses in `ml/losses.py`

### Important Distinction

The training code is ahead of the active browser model.

Today:

- the **codebase** supports a more capable learning-based path
- the **active browser checkpoint** is still `v0.0-pipeline-check`
- the **runtime product claim** must therefore stay conservative

<a id="audio-pipeline"></a>
## 5. Audio Pipeline

### Browser Audio Path

The browser monitoring path now uses continuous capture rather than isolated fixed windows.

The main pieces are:

- microphone capture through `AudioWorklet` when available
- `ScriptProcessorNode` fallback when needed
- native-sample-rate rolling buffer for source/stem paths
- resampled `16 kHz` rolling buffer for ML feature extraction

### Preprocessing Contract

The current preprocessing contract is still:

- sample rate: `16_000`
- clip window: `3.0` seconds
- window size: `25 ms`
- hop size: `10 ms`
- FFT size: `512`
- mel bins: `64`

### Input Integrity Note

A test suite now exists to validate that browser-side preprocessing and Python-side preprocessing stay aligned.

That suite covers:

- waveform parity
- feature parity
- resampler consistency
- manifest leakage safety

The existence of these tests does **not** mean preprocessing consistency can be assumed forever. They are release safeguards and should stay green.

<a id="monitoring-system"></a>
## 6. Monitoring System

The monitoring system no longer uses the old "capture 3 seconds, wait, repeat" pattern.

### Current Monitoring Behavior

- analysis window: `3.0` seconds
- inference stride: `1.0` second
- smoothing: EMA with `alpha = 0.3`
- initial minimum buffer before analysis: `750 ms`

### What Changed

The runtime now keeps a **rolling buffer** and re-analyzes the most recent `3.0` seconds on a fixed stride.

That means:

- there is no intentional blind gap between analysis passes
- each pass covers the latest available audio
- the UI is updated on a rolling basis rather than in isolated chunks

### Why This Matters

The old fixed-window/cadence mismatch created unanalyzed audio gaps.

The current design fixes that by separating:

- **continuous capture**
- **interval-based inference**

This keeps coverage continuous without blocking the audio capture path.

### Output Stabilization

To reduce flicker, the runtime applies EMA smoothing to:

- issue confidences
- detected source confidences
- source EQ recommendations
- raw ML-derived diagnosis scores

### Edge Cases

The monitoring runtime explicitly handles:

- insufficient buffer during warm-up
- silence short-circuiting
- microphone permission failures
- microphone interruption
- ML disabled mode
- ML inference failure with rule-based fallback

<a id="evaluation-system"></a>
## 7. Evaluation System

LoLvlance now includes a basic golden-set evaluation and CI gate.

### Golden Dataset

Golden samples live under:

```text
eval/goldens/
```

Each sample directory contains:

- one audio file
- one `metadata.json`

Example metadata fields:

- `file`
- `expected_source`
- `expected_issue`
- `severity`

### Evaluation Script

Main evaluator:

```text
ml/eval/evaluate.py
```

It:

- loads golden samples
- runs model inference
- computes issue/source precision, recall, and F1
- prints a confusion summary
- compares results against a stored baseline

### Baseline Tracking

Baseline file:

```text
ml/eval/baseline.json
```

This stores:

- previous overall performance
- per-label F1 values
- critical labels
- regression tolerance

### CI Gate

Workflow:

```text
.github/workflows/eval.yml
```

Current gate:

- runs on pushes and pull requests affecting `ml/**`, `eval/**`, and the workflow itself
- uses `ml/eval/evaluate.py`
- compares the current run against `ml/eval/baseline.json`
- fails on macro regression, per-label collapse, or distribution/entropy bias violations

### Important Limitation

The current golden set is still small. It is a useful regression detector, not yet a robust production benchmark.

<a id="test-infrastructure"></a>
## 8. Test Infrastructure

The repo now has dedicated tests for input integrity and ML export behavior.

### Input Integrity Tests

- `ml/tests/test_waveform_parity.py`
  Validates waveform-level parity between Python preprocessing and browser-equivalent preprocessing.
- `ml/tests/test_feature_parity.py`
  Validates end-to-end model-input feature parity.
- `ml/tests/test_resampler.py`
  Validates resampling behavior for common browser capture rates such as `48 kHz` and `44.1 kHz`.
- `ml/tests/test_manifest_leakage.py`
  Detects split leakage, duplicate files, track-group overlap, and weak naming collisions.

### Existing ML Pipeline Tests

- `ml/tests/test_export_to_onnx.py`
- `ml/tests/test_training_pipeline.py`
- `ml/tests/test_legacy_onnx_adapter.py`

### What These Tests Mean

These tests are meant to catch silent reliability failures:

- preprocessing drift
- resampling drift
- export contract breakage
- dataset split contamination

They should be treated as part of the release criteria for future production models.

<a id="product-positioning"></a>
## 9. Product Positioning

LoLvlance should currently be described as:

- continuous monitoring
- live session analysis
- updated every few seconds
- assistive guidance for non-expert operators

It should **not** be described as:

- real-time diagnosis
- instant-response audio intelligence
- production-grade AI sound engineer

Recommended product interpretation:

> LoLvlance watches the session continuously, updates guidance on a rolling basis, and suggests what to check next.

<a id="development-workflow"></a>
## 10. Development Workflow

### Frontend

```bash
npm install
npm run dev
```

Build check:

```bash
npm run build
```

### Python Environment

PowerShell example:

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

### Download Public Datasets

```bash
# List available datasets
python ml/download_datasets.py --list

# Download MUSAN (~11 GB, easiest starting point)
python ml/download_datasets.py --datasets musan --output-root data/datasets

# Download all supported datasets
python ml/download_datasets.py --datasets musan fsd50k openmic --output-root data/datasets
```

After download, the script prints the exact `ml/train.py` command to run.

### Train on Real Audio

```bash
python -m ml.train \
  --audio-root data/datasets/musan \
  --musan-root data/datasets/musan \
  --rebuild-manifest --epochs 20 --export-onnx
```

### Collect and Ingest User Feedback

```bash
# 1. User clicks "Export" in the browser → lolvlance_feedback_*.jsonl

# 2. Convert to training manifest
python ml/ingest_feedback.py \
    --feedback lolvlance_feedback_2024-01-15.jsonl \
    --output ml/artifacts/feedback_manifest.jsonl

# 3. Merge with existing manifest and retrain
cat ml/artifacts/public_dataset_manifest.jsonl \
    ml/artifacts/feedback_manifest.jsonl \
    > ml/artifacts/merged_manifest.jsonl

python -m ml.train \
    --manifest-path ml/artifacts/merged_manifest.jsonl \
    --epochs 10 --export-onnx
```

### Model Promotion Flow

1. Train or export a checkpoint under `ml/checkpoints/`
2. Run golden evaluation and relevant tests
3. Copy the promoted ONNX artifact into `public/models/`
4. Update `MODEL_VERSION` / runtime config if needed
5. Verify browser behavior manually before treating the model as user-facing

<a id="limitations"></a>
## 11. Limitations

### Model Limitations

- the active browser model is `v0.0-pipeline-check`
- it is synthetic-data-trained and not production-ready
- source predictions remain especially vulnerable to dataset bias
- browser runtime still uses a compatibility EQ output contract rather than a full multi-band browser schema

### Data Limitations

- the browser now collects per-analysis user feedback (thumbs up / wrong label) via `FeedbackWidget`, stored locally and exportable as JSONL — but this is a collection mechanism, not a closed loop yet
- the current golden dataset is too small to act as a complete benchmark
- synthetic fallback data remains in the repo and can still mislead if treated as a quality signal
- real public dataset audio must still be downloaded separately before real-data training is possible

### Inference Limitations

- browser inference quality depends on local preprocessing consistency
- optional source enrichment still depends on separate local services or browser fallbacks
- there is no server-side inference path yet
- silence and low-level inputs intentionally short-circuit parts of the pipeline

### Product Limitations

- LoLvlance helps users decide what to try next; it does not guarantee that the diagnosis is correct
- the current system should not be marketed as a fully trustworthy AI mix assistant

<a id="roadmap"></a>
## 12. Roadmap

### Immediate

- keep product copy aligned with the actual monitoring behavior
- maintain model isolation until a real-data checkpoint exists
- expand the golden dataset beyond the current starter set
- keep parity and leakage tests green

### Short Term

- train on real-source audio using `ml/download_datasets.py` + `ml/train.py --audio-root`
- close the feedback loop: periodically run `ml/ingest_feedback.py` on exported browser feedback and merge into the training manifest
- promote learned EQ outputs beyond the internal training path
- expand CI coverage for input integrity and browser-facing regression checks
- improve threshold calibration on real evaluation audio

### Longer Term

- graduate from `v0.0-pipeline-check` to a true production model
- decide whether browser-only inference remains sufficient
- add a server inference path if larger models become necessary
- add human review step into the feedback ingestion loop

<a id="handover-notes"></a>
## 13. Handover Notes

If you are taking over the project, start here:

- `HANDOVER.md` for operational and architectural context
- `src/app/hooks/useMonitoring.ts` for browser runtime behavior
- `src/app/config/modelRuntime.ts` for model isolation and version routing
- `ml/eval/evaluate.py` for regression gating
- `ml/tests/` for preprocessing and data-integrity checks

The most important thing to preserve is the distinction between:

- **pipeline works**
- **model is trustworthy**

Right now, the first statement is true. The second is not yet true.

## Related Docs

- `HANDOVER.md`
  Developer-oriented technical handover and operational guidance.
- `ML_README.md`
  ML-specific notes, model details, and training/export context.
- `HANDOVER.ko.md`
  Korean developer handover.
- `ML_README.ko.md`
  Korean ML-specific documentation.
