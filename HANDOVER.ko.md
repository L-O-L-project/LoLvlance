# LoLvlance Handover

이 문서는 현재 LoLvlance 코드베이스에 대한 기술 handover 문서입니다.

짧게 요약하면 다음과 같습니다.

- 브라우저 ML 경로는 end-to-end로 동작합니다.
- ONNX 계약은 Python과 frontend 사이에서 정렬되어 있습니다.
- 현재 체크포인트는 synthetic 기반이며 pipeline validation 용도로만 적합합니다.
- EQ는 아직 결정론적이며 rule-derived 방식입니다.

영문 버전: `HANDOVER.md`

## 1. 완료된 항목

### ML 및 Export 파이프라인

- `ml/model.py`에 shared CNN encoder와 두 개의 learned head가 구현되어 있습니다.
- `ml/train.py`는 다음을 지원합니다.
  - manifest 생성
  - weak issue/source target 생성
  - masked multi-label training
  - threshold tuning
  - checkpoint export
- ONNX export는 `ml/export_to_onnx.py`에 구현되어 있습니다.
- ONNX export는 아래 안정적인 브라우저 스키마를 반환합니다.
  - `issue_probs`
  - `source_probs`
  - `eq_freq`
  - `eq_gain_db`
- export 단계에서 `onnxruntime` 검증이 구현되어 있습니다.

### Frontend 통합

- 프론트엔드는 `src/app/audio/mlInference.ts`에서 새 스키마를 직접 소비합니다.
- `problem_probs`, `instrument_probs`에 대한 legacy output parsing은 frontend 경로에서 제거되었습니다.
- 브라우저 추론은 `onnxruntime-web`을 통해 동작합니다.
- Vite 전용 WASM asset resolution이 `mlInference.ts`에 명시적으로 설정되어 있습니다. 현재 구조에서는 이것이 필요합니다.
- 브라우저 ML이 실패해도 시스템은 기존 `ruleBasedAnalysis.ts`로 clean fallback됩니다.

### 브라우저 런타임 파이프라인

- 마이크 캡처와 rolling buffer는 `src/app/hooks/useMonitoring.ts`에 구현되어 있습니다.
- feature extraction은 `src/app/audio/featureExtraction.ts`에 구현되어 있습니다.
- 파생 diagnosis는 `src/app/audio/diagnosisPostProcessing.ts`에 구현되어 있습니다.
- source-aware EQ recommendation은 `src/app/audio/sourceAwareEq.ts`에 구현되어 있습니다.
- optional local stem separation과 fallback source tagging이 런타임 파이프라인에 통합되어 있습니다.

### Synthetic Training 검증

- synthetic fallback dataset generator가 `ml/generate_synthetic_public_datasets.py`에 있습니다.
- 현재 synthetic manifest는 `ml/artifacts/public_dataset_manifest.jsonl`에 있습니다.
- 현재 학습 결과로 `ml/checkpoints/`에 아래 아티팩트가 존재합니다.
  - `model.pt`
  - `best_sound_issue_model.pt`
  - `last_sound_issue_model.pt`
  - `config.json`
  - `thresholds.json`
  - `label_thresholds.json`
  - `training_history.json`
  - `lightweight_audio_model.onnx`
  - `lightweight_audio_model.metadata.json`
- 현재 브라우저 모델은 export 결과에서 업데이트되었습니다.
  - `ml/checkpoints/lightweight_audio_model.onnx`
  - `public/models/lightweight_audio_model.onnx`

## 2. 아직 완료되지 않은 항목

- 이 워크스페이스에는 실제 public dataset root가 설치되어 있지 않습니다.
- 실제 public dataset 기반 체크포인트는 아직 학습되지 않았습니다.
- 프로덕션 품질의 정확도를 주장할 수 없습니다.
- 데이터 수집이나 human review 루프가 없습니다.
- learned EQ head가 없습니다.
- 디바이스 클래스별 공식 latency benchmark가 없습니다.
- 마이크 기반 end-to-end 검증을 위한 자동화된 브라우저 회귀 테스트가 저장소에 포함되어 있지 않습니다.

현재 체크포인트는 integration artifact로 취급해야 하며, production model로 간주하면 안 됩니다.

## 3. 핵심 파일

- `ml/model.py`
  trainable model 정의. shared CNN encoder와 `issue_head`, `source_head`가 있습니다.
- `ml/train.py`
  end-to-end training loop, checkpoint 저장, threshold tuning, optional ONNX export trigger를 담당합니다.
- `ml/export_to_onnx.py`
  ONNX export wrapper입니다. 두-head model 위에 deterministic EQ output을 추가합니다.
- `ml/onnx_schema_adapter.py`
  deterministic EQ projection과 legacy adaptation utility를 담고 있습니다.
- `ml/label_schema.py`
  라벨, threshold, fallback EQ mapping, schema version의 authoritative source입니다.
- `ml/dataset.py`
  public-style dataset scanning, weak label 생성, manifest 저장, runtime dataset loading을 담당합니다.
- `ml/preprocessing.py`
  audio loading, resampling, log-mel extraction, spectral feature computation을 담당합니다.
- `ml/metrics.py`
  multilabel evaluation과 threshold tuning을 담당합니다.
- `src/app/hooks/useMonitoring.ts`
  메인 브라우저 런타임 orchestration 파일입니다. 전체 UI 파이프라인을 이해하려면 여기서 시작하는 것이 좋습니다.
- `src/app/audio/mlInference.ts`
  브라우저 ONNX session 생성, schema parsing, inference error handling, temporary raw tensor logging을 담당합니다.
- `src/app/audio/mlSchema.ts`
  Python label schema의 frontend mirror입니다. `ml/label_schema.py`와 반드시 맞춰야 합니다.
- `src/app/audio/diagnosisPostProcessing.ts`
  `vocal_buried`, `guitar_harsh` 같은 derived diagnosis를 계산합니다.
- `src/app/audio/sourceAwareEq.ts`
  사용자에게 보여주는 source-aware EQ recommendation 로직입니다.
- `src/app/audio/ruleBasedAnalysis.ts`
  브라우저 ML이 unavailable일 때 사용하는 fallback issue detector입니다.

## 4. ML 파이프라인 상세

### 입력 형식

trainable model은 다음 입력을 사용합니다.

- input name: `log_mel_spectrogram`
- dtype: `float32`
- shape: `(batch, time_steps, 64)`

`ml/preprocessing.py`의 preprocessing 계약은 다음과 같습니다.

- sample rate: `16_000`
- clip duration: `3.0`초
- window size: `25 ms`
- hop size: `10 ms`
- FFT size: `512`
- mel bins: `64`

일반적인 3초 clip의 경우 time dimension은 약 `298` frame입니다.

### Dataset과 Manifest

`ml/dataset.py`는 아래 네 가지 public-style dataset root를 기준으로 설계되어 있습니다.

- OpenMIC
- Slakh
- MUSAN
- FSD50K

manifest 형식에는 다음이 저장됩니다.

- audio path
- clip start와 duration
- split
- issue targets
- source targets
- target masks
- label quality metadata
- feature-derived metadata
- split hygiene를 위한 `track_group_id`

source supervision은 mask-aware하게 설계되어 있습니다. 특정 clip에 source label이 없으면 mask가 `0`이 되고, loss는 그 label에 대해 positive나 negative를 강제하지 않습니다.

### 모델 구조

현재 `ml/model.py`의 모델은 경량 CNN 구조입니다.

- stacked convolutional encoder blocks
- mean pooling + max pooling
- concatenated embedding size `192`
- `issue_head` output size `9`
- `source_head` output size `5`

학습 issue label:

```text
[muddy, harsh, buried, boomy, thin, boxy, nasal, sibilant, dull]
```

학습 source label:

```text
[vocal, guitar, bass, drums, keys]
```

post-processing에서 생성하는 derived diagnosis label:

```text
[vocal_buried, guitar_harsh, bass_muddy, drums_overpower, keys_masking]
```

schema source-of-truth:

- Python: `ml/label_schema.py`
- Frontend mirror: `src/app/audio/mlSchema.ts`

현재 아키텍처에는 learned EQ head가 없습니다.

### Training Loop

`ml/train.py`는 현재 다음 순서로 동작합니다.

1. dataset root를 해석하고 manifest를 생성하거나 로드합니다.
2. train / validation dataset을 만듭니다.
3. `__getitem__`에서 log-mel feature를 동적으로 추출합니다.
4. train split 기준으로 class-wise positive weight를 계산합니다.
5. 다음 두 head에 대해 masked BCE loss로 학습합니다.
   - issue head
   - source head
6. best validation epoch 기준으로 threshold를 tuning합니다.
7. checkpoint, threshold JSON, config JSON, training history를 저장합니다.
8. 필요하면 ONNX를 export합니다.

현재 코드 기본값:

- optimizer: `AdamW`
- learning rate: `1e-3`
- batch size: `16`
- weight decay: `1e-4`

### 현재 Synthetic Training Run

현재 체크인된 체크포인트는 아래 synthetic data 설정으로 학습되었습니다.

- total clips: `44`
- train split: `22`
- val split: `22`
- best epoch: `10`
- train loss: `1.4123 -> 0.5785`
- val loss: `1.3208 -> 1.1534`
- selection score: `0.7264`

중요한 주의사항:

- `keys`는 synthetic manifest의 `44`개 sample 전체에서 support를 갖습니다.
- 이 편향은 브라우저 테스트에서 실제 voice/music에 대해 source separation이 부정확하게 보이는 원인 중 하나입니다.

## 5. ONNX 통합

### Export 계약

`ml/export_to_onnx.py`는 trainable model을 `OnnxExportWrapper`로 감쌉니다.

이 wrapper는 다음을 반환합니다.

```text
(issue_probs, source_probs, eq_freq, eq_gain_db)
```

현재 export 속성:

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

### Browser 소비 방식

`src/app/audio/mlInference.ts`는 새 스키마만 기대합니다. 즉 다음 output만 읽습니다.

- `issue_probs`
- `source_probs`
- `eq_freq`
- `eq_gain_db`

필수 tensor가 하나라도 없으면 parsing이 throw되고, 시스템은 기존 rule-based engine으로 fallback됩니다.

### Vite와 ONNX Runtime Web

이 부분은 중요합니다.

Vite 환경에서는 `onnxruntime-web`에 explicit WASM asset URL을 넘겨야 합니다. 그렇지 않으면 브라우저가 잘못된 리소스를 가져와 ORT 초기화가 inference 시작 전부터 실패할 수 있습니다.

현재 코드는 아래 asset을 import합니다.

- `onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url`
- `onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url`

그리고 다음을 설정합니다.

- `ort.env.wasm.numThreads = 1`
- `ort.env.wasm.wasmPaths = { mjs, wasm }`

브라우저 ML이 갑자기 전부 fallback되기 시작하면 가장 먼저 이 부분을 확인해야 합니다.

### 현재 활성 모델 경로

- Python export artifact:
  - `ml/checkpoints/lightweight_audio_model.onnx`
- Active browser artifact:
  - `public/models/lightweight_audio_model.onnx`

`public/models/lightweight_audio_model.onnx.data`는 과거 external-data export에서 남아 있는 파일입니다. 현재 브라우저 모델은 standalone `.onnx` 파일입니다.

### Metadata Companion File

`ml/export_to_onnx.py`는 companion metadata 파일도 생성합니다.

- `ml/checkpoints/lightweight_audio_model.metadata.json`

이 파일에는 다음 정보가 들어 있습니다.

- schema version
- issue labels
- primary issue labels
- source labels
- derived diagnosis labels
- thresholds
- issue-to-cause mappings
- issue-to-source-affinity mappings
- fallback EQ mappings

## 6. Rule-Based EQ 시스템

EQ는 현재 학습되지 않습니다.

프로젝트에는 서로 다른 두 계층의 EQ 로직이 있습니다.

### ONNX 수준 Deterministic EQ Projection

`ml/onnx_schema_adapter.py`의 `HierarchicalEqProjection`에 정의되어 있습니다.

이 레이어는 다음 입력을 받아:

- issue probabilities
- source probabilities

다음을 생성합니다.

- `eq_freq`
- `eq_gain_db`

이 projection은 다음 정보를 사용합니다.

- issue별 fallback EQ mapping
- issue/source pair override
- active issue/source 확률에 대한 weighted blending

즉, trainable network가 여전히 2-head 구조여도 ONNX contract는 4출력을 유지할 수 있습니다.

### Frontend Source-Aware EQ Recommendation

`src/app/audio/sourceAwareEq.ts`에 정의되어 있습니다.

이 레이어는 다음 정보를 이용해 사람이 읽기 쉬운 추천을 만듭니다.

- detected sources
- issue labels
- optional stem metrics

예시:

- `180-320Hz`에서 vocal mud cut
- `4500-8000Hz`에서 drum harshness 완화
- `1500-3000Hz`에서 buried vocal presence 보강

### Fallback Rule Engine

`src/app/audio/ruleBasedAnalysis.ts`에 정의되어 있습니다.

브라우저 ML이 완전히 실패해도 앱은 다음을 계속 수행할 수 있습니다.

- `muddy`, `harsh`, `buried` 같은 기본 issue 탐지
- 기본 EQ suggestion 제공

## 7. 모델 재학습 방법

### Step 1: Python 환경 구성

PowerShell 예시:

```powershell
python -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install --upgrade pip
.\.venv-ml\Scripts\python.exe -m pip install -r ml\requirements-test.txt
```

### Step 2A: 실제 Public Dataset으로 학습

실제 dataset root가 있다면 다음처럼 실행합니다.

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

### Step 2B: Synthetic Fallback 학습

실제 dataset이 없으면 synthetic set을 다시 생성합니다.

```powershell
.\.venv-ml\Scripts\python.exe -m ml.generate_synthetic_public_datasets `
  --output-root ml\artifacts\synthetic_public_datasets
```

그 다음 학습합니다.

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

이미 checkpoint가 있고 export만 다시 하고 싶다면:

```powershell
.\.venv-ml\Scripts\python.exe -m ml.export_to_onnx `
  --checkpoint ml\checkpoints\best_sound_issue_model.pt `
  --output ml\checkpoints\lightweight_audio_model.onnx `
  --time-steps 298 `
  --verify
```

### Step 4: 브라우저 모델 교체

```powershell
Copy-Item ml\checkpoints\lightweight_audio_model.onnx public\models\lightweight_audio_model.onnx -Force
```

### Step 5: 브라우저 검증

```bash
npm run dev
```

그 다음 아래를 확인합니다.

- `[audio-ml]` 로그에 `status: 'ready'`가 찍히는지
- raw output이 finite하고 constant하지 않은지
- missing output-key error가 없는지
- 정상 추론 중 `[audio-ml] Inference failed...` 경고가 뜨지 않는지

## 8. 자주 발생하는 실패 지점

### 1. 브라우저에서 모델이 로드되지 않음

증상:

- `[audio-ml] Model warm-up skipped`
- ORT backend initialization error
- 즉시 rule-based fallback

체크할 것:

- `public/models/lightweight_audio_model.onnx`가 존재하는지
- `mlInference.ts`가 `ort.env.wasm.wasmPaths`를 계속 설정하고 있는지
- 현재 Vite base path 기준으로 모델 URL이 정상 해석되는지

### 2. ONNX Output 누락

증상:

- `Missing tensor output: issue_probs`
- `Missing scalar output: eq_freq`

원인:

- export가 `OnnxExportWrapper`를 거치지 않았음
- output name이 바뀌었음
- 잘못된 artifact를 `public/models/`에 복사했음

### 3. Constant하거나 의미 없는 출력

증상:

- `issue_probs`가 silence, voice, music 사이에서 거의 변하지 않음
- `source_probs`가 한 source로 collapse됨

체크할 것:

- silence gate 경로에 걸린 것이 아닌지
- `[audio-ml:raw]` 로그를 볼 것
- 현재 체크포인트가 synthetic이며 편향되어 있다는 점을 항상 고려할 것

이미 존재하는 자동 검증 파일:

- `ml/tests/test_export_to_onnx.py`
- `ml/tests/test_training_pipeline.py`
- `ml/tests/test_legacy_onnx_adapter.py`

### 4. Silence에서 Raw ML Output이 안 나옴

이것은 expected behavior일 수 있습니다.

현재 `mlInference.ts`는 다음 조건이면 ONNX inference 전에 short-circuit합니다.

- `features.rms < 0.012`

### 5. Python과 Frontend 사이 Schema Drift

label을 추가하거나 이름을 바꿀 때는 반드시 함께 업데이트해야 합니다.

- `ml/label_schema.py`
- `src/app/audio/mlSchema.ts`
- post-processing threshold와 mapping

한쪽만 바꾸면 안 됩니다.

### 6. Stem-Service Fallback과 ML Fallback 혼동

UI는 브라우저 ML이 정상이어도 stem-service fallback을 표시할 수 있습니다.

두 경로는 다릅니다.

- ML fallback은 ONNX가 실패해서 `ruleBasedAnalysis.ts`를 사용한 경우입니다.
- stem-service fallback은 별도 local source-separation sidecar가 unavailable해서 브라우저 source tagging으로 대체한 경우입니다.

### 7. 마이크 권한 문제

확인할 것:

- 브라우저 permission state
- 장치 존재 여부
- secure context / local dev 환경
- 브라우저 설정에서 `getUserMedia`가 막혀 있지 않은지

### 8. 재학습 후 Public Model이 갱신되지 않음

학습 파이프라인은 `ml/checkpoints/`에 저장합니다.
브라우저는 `public/models/`를 읽습니다.

즉, 재학습만으로는 런타임 모델이 바뀌지 않으며, export한 ONNX를 `public/models/`로 복사해야 합니다.

## 권장 다음 작업

새 엔지니어가 지금 인수받는다면 우선순위가 높은 다음 단계는 아래와 같습니다.

1. synthetic data를 실제 public dataset으로 교체하기
2. 실제 오디오 기준 평가 리포트 만들기
3. 데이터 수집 및 review loop 구축하기
4. EQ를 계속 deterministic하게 둘지, 나중에 learned head로 옮길지 결정하기
5. schema, loading, output variability를 자동 검증하는 브라우저 회귀 테스트 추가하기
