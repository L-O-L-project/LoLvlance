# LoLvlance Golden Audio Dataset

This directory contains labeled audio used to evaluate model candidates before they are promoted to the browser app.

The current set has only three samples. It is useful for smoke/regression checks, not for accuracy claims.

## Directory Layout

Recommended layout:

```text
eval/goldens/
  labels.schema.json
  labels.json
  README.md
  <sample_id>/
    <sample_id>.wav
    metadata.json
```

`metadata.json` is kept for compatibility with `ml/eval/evaluate.py`. `labels.json` is the central manifest for richer dataset management and future tooling.

## Label Sets

Issue labels:

- `muddy`
- `harsh`
- `buried`
- `boomy`
- `thin`
- `boxy`
- `nasal`
- `sibilant`
- `dull`

Source labels:

- `vocal`
- `guitar`
- `bass`
- `drums`
- `keys`

## Required Metadata

Every new sample should record:

- expected issue labels
- expected source labels
- severity: `none`, `low`, `medium`, `high`, or `unknown`
- label quality: `reviewed`, `single_reviewer`, `unreviewed`, `synthetic`, or `unknown`
- recording environment: live venue, rehearsal room, studio, synthetic fixture, or unknown
- content type: full mix, stem, solo source, silence, noise, or unknown
- device and microphone type
- browser and OS when captured through the web app
- clipping, silence, high-noise-floor, and low-confidence-label flags
- reviewer notes when labels are uncertain

Do not infer unknown recording metadata. Use `unknown` and mark `low_confidence_label` when needed.

## Collection Guidance

Capture clips that represent the browser use case:

- 3-10 seconds per clip
- mono or stereo WAV is acceptable, but preserve original sample rate until preprocessing
- avoid post-processing after capture
- include realistic room noise and live/rehearsal playback levels
- include clean or no-issue examples, not only degraded audio
- include silence, clipped input, and high-noise-floor negative/edge cases

Suggested review process:

1. One engineer imports the clip and fills metadata.
2. A second reviewer confirms labels where possible.
3. Mark `label_quality` as `reviewed` only after review.
4. Keep uncertain clips, but set `low_confidence_label: true` and exclude them from strict promotion gates if needed.

## Minimum Dataset Sizes

MVP regression set:

- Minimum: 30 clips
- Goal: catch obvious regressions and class-bias failures
- At least 3 positive clips per issue class
- At least 5 positive clips per source class
- At least 5 clean/no-issue clips
- At least 3 silence/noise/clipping edge cases

Validation set:

- Minimum: 250 clips
- Recommended: at least 25 positive clips per issue class
- Recommended: at least 40 positive clips per source class
- Include balanced clean/no-issue examples
- Include multiple devices, rooms, and browser capture paths

Production-grade evaluation set:

- Minimum: 1000 clips
- Recommended: at least 100 positive clips per issue class
- Recommended: at least 150 positive clips per source class
- Separate holdout data from training and threshold tuning
- Include class co-occurrence cases such as vocal+keys, bass+drums, and full-band mixes

## Per-Class Targets

Issue classes:

| Label | MVP | Validation | Production eval |
|---|---:|---:|---:|
| muddy | 3 | 25 | 100 |
| harsh | 3 | 25 | 100 |
| buried | 3 | 25 | 100 |
| boomy | 3 | 25 | 100 |
| thin | 3 | 25 | 100 |
| boxy | 3 | 25 | 100 |
| nasal | 3 | 25 | 100 |
| sibilant | 3 | 25 | 100 |
| dull | 3 | 25 | 100 |

Source classes:

| Label | MVP | Validation | Production eval |
|---|---:|---:|---:|
| vocal | 5 | 40 | 150 |
| guitar | 5 | 40 | 150 |
| bass | 5 | 40 | 150 |
| drums | 5 | 40 | 150 |
| keys | 5 | 40 | 150 |

These are positive-label targets. Multi-label clips can count for multiple labels, but avoid letting one source or issue dominate the set.

## Promotion Rule

Do not replace `public/models/lightweight_audio_model.production.onnx` unless a candidate:

1. Passes ONNX contract validation.
2. Passes the regression gate against the current baseline.
3. Does not introduce obvious class-bias regressions.
4. Improves or preserves per-class precision/recall for supported labels.
5. Is documented in `ml/model_history.md` and `ml/model-card.md`.
