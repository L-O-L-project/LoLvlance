# LoLvlance Model Evaluation Protocol

This protocol describes how to compare future model candidates against the current production browser model.

It does not claim that the current model is production accurate. The current golden set is too small for accuracy claims.

## Required Inputs

- Candidate ONNX model
- Candidate metadata JSON
- `eval/goldens/labels.json`
- Per-sample `metadata.json` files for compatibility with `ml/eval/evaluate.py`
- Current baseline: `ml/eval/baseline.json`
- Current production model: `public/models/lightweight_audio_model.production.onnx`

## Preflight Checks

Run:

```bash
python ml/validate_onnx_contract.py --model-path <candidate.onnx>
```

The candidate must expose:

- input `log_mel_spectrogram`, float32, `[batch, time, 64]`
- output `issue_probs`, float32, `[1, 9]` at runtime
- output `source_probs`, float32, `[1, 5]` at runtime
- output `eq_freq`, float32, `[1, 1]` at runtime
- output `eq_gain_db`, float32, `[1, 1]` at runtime

## Metrics

Primary regression metrics:

- combined macro F1
- issue macro F1
- source macro F1
- per-class precision, recall, and F1

Bias and safety metrics:

- predicted-positive ratio per class
- dominant predicted class ratio
- prediction entropy
- false positive rate per class
- false negative rate per class
- low-confidence result rate

Calibration metrics to add when the dataset is large enough:

- expected calibration error
- confidence bucket accuracy
- per-class threshold curves
- false positive rate at fixed recall

Operational metrics:

- preprocessing latency
- ONNX inference latency
- end-to-end analysis latency
- invalid audio rejection rate
- silence/clipping/noise handling rate

## Current Evaluator

Run:

```bash
python ml/eval/evaluate.py \
  --model-path <candidate.onnx> \
  --thresholds-path ml/checkpoints/label_thresholds.json \
  --report-json-path ml/eval/<candidate>_report.json
```

The evaluator currently uses per-sample `metadata.json` files under `eval/goldens/`. Keep `labels.json` synchronized until the evaluator is extended to consume the central manifest directly.

## Candidate Comparison Rules

A candidate can be considered for promotion only if:

1. ONNX contract validation passes.
2. The evaluation gate passes against `ml/eval/baseline.json`.
3. Combined macro F1 does not regress beyond the configured epsilon.
4. Supported per-class F1 values do not regress beyond their configured epsilon.
5. Predicted-positive ratios do not show obvious class collapse.
6. The candidate does not over-predict unsupported classes on clean/no-issue clips.
7. Known weak labels are documented, not hidden by aggregate scores.

If a candidate improves aggregate F1 but collapses a critical class, do not promote it.

## Dataset Split Guidance

Recommended splits:

- `regression`: small, stable set used in CI and quick checks
- `validation`: larger set used for threshold tuning and candidate comparison
- `production_eval`: locked holdout for release decisions
- `holdout`: optional future blind set

Never tune thresholds on `production_eval`.

## Reporting

Every candidate report should include:

- model path and version
- dataset version
- threshold file used
- ONNX contract summary
- aggregate metrics
- per-class metrics
- confusion summary
- prediction distribution
- gate status and failures
- notes about excluded or low-confidence labels

Do not publish accuracy claims from the MVP regression set.
