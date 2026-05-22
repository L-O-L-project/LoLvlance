# LoLvlance Model Card

## Active Browser Model

- Version: `v0.1-real-data`
- Artifact: `public/models/lightweight_audio_model.production.onnx`
- Metadata: `public/models/lightweight_audio_model.production.metadata.json`
- Training checkpoint: `ml/checkpoints/best_sound_issue_model.pt`
- Validation status: MVP regression-gated, not production-validated

## Training Data

The active model is documented as trained on MUSAN and OpenMIC-2018 weak labels. The repository contains training and dataset ingestion code, but it does not contain the full public datasets required to retrain the active model from scratch.

The current golden regression set contains only three labeled audio clips under `eval/goldens/`. These clips are useful for detecting obvious regressions and class-bias failures, but they are not enough to claim real-world accuracy.

Dataset collection and evaluation workflow are documented in:

- `eval/goldens/README.md`
- `eval/goldens/labels.json`
- `eval/goldens/labels.schema.json`
- `ml/eval/EVALUATION_PROTOCOL.md`

## Browser ONNX Contract

Input:

- `log_mel_spectrogram`
- dtype: `float32`
- shape: `[batch_size, time_steps, 64]`
- preprocessing: mono audio resampled to 16 kHz, log-mel features, 64 mel bins

Outputs:

- `issue_probs`: `float32`, runtime shape `[1, 9]`
- `source_probs`: `float32`, runtime shape `[1, 5]`
- `eq_freq`: `float32`, runtime shape `[1, 1]`
- `eq_gain_db`: `float32`, runtime shape `[1, 1]`

Issue class order:

1. `muddy`
2. `harsh`
3. `buried`
4. `boomy`
5. `thin`
6. `boxy`
7. `nasal`
8. `sibilant`
9. `dull`

Source class order:

1. `vocal`
2. `guitar`
3. `bass`
4. `drums`
5. `keys`

## Confidence Interpretation

Outputs are weak-label probabilities from an MVP model. They should be interpreted as ranking and triage signals, not calibrated production probabilities.

The browser must continue to distinguish:

- ML result
- low-confidence ML result
- rule-based fallback result
- invalid or insufficient audio result

## Known Limitations

- The golden evaluation set is too small for reliable accuracy claims.
- `nasal` and `guitar` over-prediction is visible in the current public golden evaluation.
- Some labels have weak or sparse validation support, especially source labels in live-mix contexts.
- The active model depends on weak public labels and does not prove venue/live-sound generalization.
- The archived checkpoint `ml/checkpoints/lightweight_audio_model.onnx` currently regresses badly against the active public model and must not be promoted.

## Required Promotion Checks

Before replacing `public/models/lightweight_audio_model.production.onnx`:

1. Run `python ml/validate_onnx_contract.py --model-path <candidate.onnx>`.
2. Run `python ml/eval/evaluate.py --model-path <candidate.onnx> --thresholds-path ml/checkpoints/label_thresholds.json`.
3. Confirm the gate passes against `ml/eval/baseline.json`.
4. Compare per-label precision/recall/F1 and prediction distribution against the active model.
5. Update `public/models/lightweight_audio_model.production.metadata.json`.
6. Update `ml/model_history.md`.
7. Run `npm install`, `npm audit`, and `npm run build`.

Do not promote a model based only on training loss or a single aggregate score.
