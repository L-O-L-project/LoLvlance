# LoLvlance Handover

This document is the technical handover for the current LoLvlance codebase.

The short version is:

- the browser ML path is working end to end
- the ONNX contract is aligned between Python and frontend
- the current checkpoint is synthetic and only suitable for pipeline validation
- EQ is still deterministic and rule-derived

Korean version: `HANDOVER.ko.md`

## Quick Links

[![Completed](https://img.shields.io/badge/Completed%20Work-2563EB?style=for-the-badge&logo=checkmarx&logoColor=white)](#completed-work)
[![Not Done](https://img.shields.io/badge/Not%20Done-DC2626?style=for-the-badge&logo=verizon&logoColor=white)](#not-done)
[![Critical Files](https://img.shields.io/badge/Critical%20Files-7C3AED?style=for-the-badge&logo=files&logoColor=white)](#critical-files)
[![ML Pipeline](https://img.shields.io/badge/ML%20Pipeline-0891B2?style=for-the-badge&logo=pytorch&logoColor=white)](#ml-pipeline-details)

[![ONNX](https://img.shields.io/badge/ONNX%20Integration-0F766E?style=for-the-badge&logo=onnx&logoColor=white)](#onnx-integration)
[![Rule EQ](https://img.shields.io/badge/Rule--Based%20EQ-D97706?style=for-the-badge&logo=beatsbydre&logoColor=white)](#rule-based-eq-system)
[![Retrain](https://img.shields.io/badge/Retrain%20Guide-16A34A?style=for-the-badge&logo=python&logoColor=white)](#how-to-retrain-the-model)
[![Failure Points](https://img.shields.io/badge/Failure%20Points-1D4ED8?style=for-the-badge&logo=sentry&logoColor=white)](#common-failure-points)

<a id="completed-work"></a>
## 1. What Has Been Completed

### ML and Export Pipeline

- A shared CNN encoder with two learned heads is implemented in `ml/model.py`.
- The training loop in `ml/train.py` supports:
  - manifest creation
  - weak issue/source targets
  - masked multi-label training
  - threshold tuning
  - checkpoint export
- ONNX export is implemented in `ml/export_to_onnx.py`.
- The ONNX export returns the stable browser schema:
  - `issue_probs`
  - `source_probs`
  - `eq_freq`
  - `eq_gain_db`
- `onnxruntime` verification is implemented during export.

### Frontend Integration

- The frontend consumes the new schema directly in `src/app/audio/mlInference.ts`.
- Legacy output parsing for `problem_probs` and `instrument_probs` has been removed from the frontend path.
- Browser inference is running through `onnxruntime-web`.
- Vite-specific WASM asset resolution is configured explicitly in `mlInference.ts`. This is required for the current setup.
- Browser ML failure still falls back cleanly to `ruleBasedAnalysis.ts`.

### Browser Runtime Pipeline

- Microphone capture and rolling buffers are implemented in `src/app/hooks/useMonitoring.ts`.
- Feature extraction is implemented in `src/app/audio/featureExtraction.ts`.
- Derived diagnoses are implemented in `src/app/audio/diagnosisPostProcessing.ts`.
- Source-aware EQ recommendations are implemented in `src/app/audio/sourceAwareEq.ts`.
- Optional local stem separation and fallback source tagging are integrated into the runtime pipeline.

### Synthetic Training Validation

- A synthetic fallback dataset generator exists at `ml/generate_synthetic_public_datasets.py`.
- The current synthetic manifest exists at `ml/artifacts/public_dataset_manifest.jsonl`.
- The current training run produced these artifacts in `ml/checkpoints/`:
  - `model.pt`
  - `best_sound_issue_model.pt`
  - `last_sound_issue_model.pt`
  - `config.json`
  - `thresholds.json`
  - `label_thresholds.json`
  - `training_history.json`
  - `lightweight_audio_model.onnx`
  - `lightweight_audio_model.metadata.json`
- The current browser model was updated from the exported checkpoint:
  - `ml/checkpoints/lightweight_audio_model.onnx`
  - `public/models/lightweight_audio_model.onnx`

<a id="not-done"></a>
## 2. What Is NOT Done

- No real public datasets are installed locally in this workspace.
- No real-dataset checkpoint has been trained yet.
- No production-quality accuracy claim is justified.
- No data collection or human review loop exists.
- No learned EQ head exists.
- No formal latency benchmark across device classes exists.
- No automated browser regression suite is committed for mic-driven end-to-end validation.

Treat the current checkpoint as an integration artifact, not a production model.

<a id="critical-files"></a>
## 3. Critical Files

- `ml/model.py`
  Trainable model definition. Shared CNN encoder plus `issue_head` and `source_head`.
- `ml/train.py`
  End-to-end training loop, checkpoint writing, threshold tuning, and optional ONNX export trigger.
- `ml/export_to_onnx.py`
  ONNX export wrapper. Adds deterministic EQ outputs on top of the two-head model.
- `ml/onnx_schema_adapter.py`
  Deterministic EQ projection and legacy-adaptation utilities.
- `ml/label_schema.py`
  Authoritative schema source for labels, thresholds, fallback EQ mappings, and schema versioning.
- `ml/dataset.py`
  Public-style dataset scanning, weak label generation, manifest writing, and runtime dataset loading.
- `ml/preprocessing.py`
  Audio loading, resampling, log-mel extraction, and spectral feature computation used by training.
- `ml/metrics.py`
  Multilabel evaluation and threshold tuning.
- `src/app/hooks/useMonitoring.ts`
  Main browser runtime orchestrator. Start here if you need to understand the end-to-end UI pipeline.
- `src/app/audio/mlInference.ts`
  Browser ONNX session creation, schema parsing, inference error handling, and temporary raw tensor logging.
- `src/app/audio/mlSchema.ts`
  Frontend mirror of the Python label schema. Keep this aligned with `ml/label_schema.py`.
- `src/app/audio/diagnosisPostProcessing.ts`
  Derived diagnosis logic such as `vocal_buried` and `guitar_harsh`.
- `src/app/audio/sourceAwareEq.ts`
  User-facing source-aware EQ recommendation logic.
- `src/app/audio/ruleBasedAnalysis.ts`
  Fallback issue detector if browser ML is unavailable.

<a id="ml-pipeline-details"></a>
## 4. ML Pipeline Details

### Input Format

The trainable model consumes:

- input name: `log_mel_spectrogram`
- dtype: `float32`
- shape: `(batch, time_steps, 64)`

The preprocessing contract in `ml/preprocessing.py` is:

- sample rate: `16_000`
- clip duration: `3.0` seconds
- window size: `25 ms`
- hop size: `10 ms`
- FFT size: `512`
- mel bins: `64`

For a typical 3-second clip, the time dimension is about `298` frames.

### Dataset and Manifest

`ml/dataset.py` is designed around four public-style dataset roots:

- OpenMIC
- Slakh
- MUSAN
- FSD50K

The manifest format stores:

- audio path
- clip start and duration
- split
- issue targets
- source targets
- target masks
- label quality metadata
- feature-derived metadata
- `track_group_id` for split hygiene

Source supervision is intentionally mask-aware. If a source label is unavailable for a clip, the mask is `0` and the loss does not force a positive or negative target for that label.

### Model Structure

The current model in `ml/model.py` is a lightweight CNN:

- stacked convolutional encoder blocks
- mean pooling + max pooling
- concatenated embedding of size `192`
- `issue_head` output size `9`
- `source_head` output size `5`

Trainable issue labels:

```text
[muddy, harsh, buried, boomy, thin, boxy, nasal, sibilant, dull]
```

Trainable source labels:

```text
[vocal, guitar, bass, drums, keys]
```

Derived diagnosis labels generated in post-processing:

```text
[vocal_buried, guitar_harsh, bass_muddy, drums_overpower, keys_masking]
```

Schema source-of-truth:

- Python: `ml/label_schema.py`
- Frontend mirror: `src/app/audio/mlSchema.ts`

There is no learned EQ head in the current architecture.

### Training Loop

`ml/train.py` currently does the following:

1. Resolves dataset roots and builds or loads the manifest.
2. Creates train and validation datasets.
3. Extracts log-mel features dynamically in `__getitem__`.
4. Computes class-wise positive weights from the training split.
5. Trains with masked BCE losses for:
   - issue head
   - source head
6. Tunes thresholds on the best validation epoch.
7. Writes checkpoints, threshold JSON, config JSON, and training history.
8. Optionally exports ONNX.

Current training defaults in code:

- optimizer: `AdamW`
- learning rate: `1e-3`
- batch size: `16`
- weight decay: `1e-4`

### Current Synthetic Training Run

The current checked-in checkpoint was trained on synthetic data with:

- total clips: `44`
- train split: `22`
- val split: `22`
- best epoch: `10`
- train loss: `1.4123 -> 0.5785`
- val loss: `1.3208 -> 1.1534`
- selection score: `0.7264`

Important caveat:

- `keys` has support in all `44` synthetic samples, so the current source model is biased.
- That bias is visible in browser tests where real voice and music do not separate cleanly by source.

<a id="onnx-integration"></a>
## 5. ONNX Integration

### Export Contract

`ml/export_to_onnx.py` wraps the trainable model in `OnnxExportWrapper`.

That wrapper returns:

```text
(issue_probs, source_probs, eq_freq, eq_gain_db)
```

Current export properties:

- opset: `18`
- dynamic axes:
  - batch
  - time steps
- input name:
  - `log_mel_spectrogram`
- output names:
  - `issue_probs`
  - `source_probs`
  - `eq_freq`
  - `eq_gain_db`

### Browser Consumption

`src/app/audio/mlInference.ts` expects only the new schema. It reads:

- `issue_probs`
- `source_probs`
- `eq_freq`
- `eq_gain_db`

If any required tensor is missing, parsing throws and the system falls back to the existing rule-based engine.

### Vite and ONNX Runtime Web

This point is critical.

`onnxruntime-web` must be given explicit WASM asset URLs under Vite. Without that, the browser can request the wrong resource and ORT initialization fails before inference starts.

The current code handles this by importing:

- `onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url`
- `onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url`

and then setting:

- `ort.env.wasm.numThreads = 1`
- `ort.env.wasm.wasmPaths = { mjs, wasm }`

If browser ML suddenly starts falling back everywhere, check this first.

### Active Model Paths

- Python export artifact:
  - `ml/checkpoints/lightweight_audio_model.onnx`
- Active browser artifact:
  - `public/models/lightweight_audio_model.onnx`

`public/models/lightweight_audio_model.onnx.data` is still in the repo from an older external-data export. The current browser model is the standalone `.onnx` file.

### Metadata Companion File

`ml/export_to_onnx.py` also writes a companion metadata file:

- `ml/checkpoints/lightweight_audio_model.metadata.json`

This includes:

- schema version
- issue labels
- primary issue labels
- source labels
- derived diagnosis labels
- thresholds
- issue-to-cause mappings
- issue-to-source-affinity mappings
- fallback EQ mappings

<a id="rule-based-eq-system"></a>
## 6. Rule-Based EQ System

EQ is not currently learned.

There are two separate EQ layers in the project:

### ONNX-Level Deterministic EQ Projection

Defined in `ml/onnx_schema_adapter.py` as `HierarchicalEqProjection`.

It takes:

- issue probabilities
- source probabilities

and produces:

- `eq_freq`
- `eq_gain_db`

This projection uses:

- per-issue fallback EQ mappings
- issue/source pair overrides
- weighted blending across active issue and source probabilities

This is how the ONNX model can expose the full four-output browser schema while the trainable network remains two-head only.

### Frontend Source-Aware EQ Recommendations

Defined in `src/app/audio/sourceAwareEq.ts`.

This layer builds human-readable recommendations using:

- detected sources
- issue labels
- optional stem metrics

Examples:

- cut vocal mud around `180-320Hz`
- tame drum harshness around `4500-8000Hz`
- add presence to buried vocals around `1500-3000Hz`

### Fallback Rule Engine

Defined in `src/app/audio/ruleBasedAnalysis.ts`.

If browser ML fails entirely, the app can still:

- detect basic issues like `muddy`, `harsh`, and `buried`
- provide basic EQ suggestions

<a id="how-to-retrain-the-model"></a>
## 7. How to Retrain the Model

### Step 1: Create the Python Environment

PowerShell example:

```powershell
python -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install --upgrade pip
.\.venv-ml\Scripts\python.exe -m pip install -r ml\requirements-test.txt
```

### Step 2A: Train From Real Public Datasets

If you have the dataset roots locally, run:

```powershell
.\.venv-ml\Scripts\python.exe -m ml.train `
  --openmic-root C:\path\to\openmic `
  --slakh-root C:\path\to\slakh `
  --musan-root C:\path\to\musan `
  --fsd50k-root C:\path\to\fsd50k `
  --manifest-path ml\artifacts\public_dataset_manifest.jsonl `
  --rebuild-manifest `
  --epochs 10 `
  --batch-size 16 `
  --learning-rate 1e-3 `
  --checkpoint-dir ml\checkpoints `
  --export-onnx `
  --onnx-output ml\checkpoints\lightweight_audio_model.onnx
```

### Step 2B: Synthetic Fallback Training

If real datasets are not available, rebuild the synthetic set:

```powershell
.\.venv-ml\Scripts\python.exe -m ml.generate_synthetic_public_datasets `
  --output-root ml\artifacts\synthetic_public_datasets
```

Then train:

```powershell
.\.venv-ml\Scripts\python.exe -m ml.train `
  --openmic-root ml\artifacts\synthetic_public_datasets\openmic `
  --slakh-root ml\artifacts\synthetic_public_datasets\slakh `
  --musan-root ml\artifacts\synthetic_public_datasets\musan `
  --fsd50k-root ml\artifacts\synthetic_public_datasets\fsd50k `
  --manifest-path ml\artifacts\public_dataset_manifest.jsonl `
  --rebuild-manifest `
  --epochs 10 `
  --batch-size 16 `
  --learning-rate 1e-3 `
  --checkpoint-dir ml\checkpoints `
  --export-onnx `
  --onnx-output ml\checkpoints\lightweight_audio_model.onnx
```

### Step 3: Standalone ONNX Export

If you already have a checkpoint and only want to export:

```powershell
.\.venv-ml\Scripts\python.exe -m ml.export_to_onnx `
  --checkpoint ml\checkpoints\best_sound_issue_model.pt `
  --output ml\checkpoints\lightweight_audio_model.onnx `
  --time-steps 298 `
  --verify
```

### Step 4: Replace the Browser Model

```powershell
Copy-Item ml\checkpoints\lightweight_audio_model.onnx public\models\lightweight_audio_model.onnx -Force
```

### Step 5: Verify in the Browser

```bash
npm run dev
```

Then verify:

- `[audio-ml]` logs `status: 'ready'`
- raw outputs are finite and non-constant
- no missing output-key errors appear
- no `[audio-ml] Inference failed...` warnings appear during normal inference

<a id="common-failure-points"></a>
## 8. Common Failure Points

### 1. Model Fails to Load in Browser

Symptoms:

- `[audio-ml] Model warm-up skipped`
- ORT backend initialization errors
- immediate rule-based fallback

Checks:

- confirm `public/models/lightweight_audio_model.onnx` exists
- confirm `mlInference.ts` still sets `ort.env.wasm.wasmPaths`
- confirm the model URL resolves under the current Vite base path

### 2. Missing ONNX Outputs

Symptoms:

- errors mentioning `Missing tensor output: issue_probs`
- errors mentioning `Missing scalar output: eq_freq`

Cause:

- exported model did not go through `OnnxExportWrapper`
- output names were changed
- wrong artifact was copied into `public/models/`

### 3. Constant or Meaningless Outputs

Symptoms:

- `issue_probs` do not move between silence, voice, and music
- `source_probs` collapse onto one source

Checks:

- confirm you are not in the silence gate path
- inspect `[audio-ml:raw]` logs
- remember the current checkpoint is synthetic and biased

Relevant automated checks already exist in:

- `ml/tests/test_export_to_onnx.py`
- `ml/tests/test_training_pipeline.py`
- `ml/tests/test_legacy_onnx_adapter.py`

### 4. Silence Produces No Raw ML Output

This is expected if RMS is below the silence threshold.

Current browser silence gating in `mlInference.ts` short-circuits before ONNX inference when:

- `features.rms < 0.012`

### 5. Schema Drift Between Python and Frontend

If you add or rename labels:

- update `ml/label_schema.py`
- update `src/app/audio/mlSchema.ts`
- update any post-processing thresholds and mappings

Do not change only one side.

### 6. Confusing Stem-Service Fallback With ML Fallback

The UI can show a stem-service fallback even when browser ML is healthy.

Those are different paths:

- ML fallback means ONNX failed and the app used `ruleBasedAnalysis.ts`.
- Stem-service fallback means the separate local source-separation sidecar was unavailable, so the app used browser-side tagging instead.

### 7. Microphone Permission Problems

Checks:

- browser permission state
- device availability
- secure context / local dev environment
- whether `getUserMedia` is blocked by browser settings

### 8. Stale Public Model After Retraining

The training pipeline writes to `ml/checkpoints/`.
The browser reads from `public/models/`.

Retraining alone does not update the runtime model unless you copy the exported ONNX file into `public/models/`.

## Recommended Next Work

If a new engineer is taking over now, the highest-value next steps are:

1. Replace synthetic training data with real public datasets.
2. Build a proper evaluation report on real audio.
3. Add a data collection and review loop.
4. Decide whether EQ should remain deterministic or become a learned head later.
5. Add automated browser regression coverage for schema, loading, and output variability.
