# LoLvlance

LoLvlance는 실시간 오디오 입력을 분석해 믹스 문제를 진단하고, 악기/보컬 존재 신호와 결합해 실용적인 EQ 가이드를 제공하는 오디오 진단 시스템입니다. 현재 저장소는 브라우저 실시간 분석 UI, local stem separation sidecar, 그리고 Python 기반 hierarchical multi-head 학습/ONNX export 파이프라인까지 연결된 상태입니다.

핵심 목표는 "완벽한 오디오 이해"가 아니라, 다음을 일관된 계약으로 묶는 것입니다.

- 브라우저에서 3초 내외의 오디오를 안정적으로 수집
- log-mel 기반 경량 모델로 글로벌 이슈와 소스 신호를 추론
- 직접 학습하지 않은 source-specific diagnosis는 후처리에서 파생
- ONNX Runtime Web과 호환되는 추론 계약 유지
- public dataset 기반으로 재학습 가능한 Python 파이프라인 제공

## 현재 상태 요약

- 브라우저 실시간 마이크 입력 분석 동작
- ONNX Runtime Web 기반 ML 추론 동작
- rule-based fallback 분석 동작
- local stem separation sidecar 연동 동작
- MediaPipe YAMNet fallback source tagging 동작
- hierarchical multi-head training pipeline 구현 완료
- ONNX export 및 `onnxruntime` 검증 구현 완료
- ML 스키마 버전 관리 도입 완료: `2.0.0`

이 저장소는 더 이상 flat 3-label classifier 기준 문서가 아닙니다. 현재 ML 구조는 다음과 같습니다.

- Head A: 글로벌 acoustic issue multi-label
- Head B: source presence/dominance multi-label
- Head C: 직접 학습하지 않고 후처리로 파생하는 source-specific diagnoses

## 시스템 아키텍처

### 1. 브라우저 실시간 분석 레이어

- 마이크 권한 요청 및 상태 추적
- native sample rate 오디오 캡처
- 16kHz 분석용 버퍼 유지
- 약 3초 rolling circular buffer 유지
- log-mel spectrogram + RMS 추출
- ONNX Runtime Web 기반 경량 ML 추론
- rule-based issue fallback
- stem separation sidecar 결과 병합
- YAMNet 기반 open-source fallback source tagging
- source-aware EQ recommendation 및 결과 카드 UI

### 2. Python 학습 레이어

- public dataset 스캔 및 manifest 생성
- weak label 기반 hierarchical target 생성
- song / session / track group 중심 split
- shared encoder + issue head + source head 학습
- masked BCE 손실로 unavailable source supervision 처리
- per-label metric / threshold tuning
- ONNX export + metadata export + runtime verification

### 3. 추론 후처리 레이어

- issue head 출력과 source head 출력을 결합
- source-specific diagnosis를 규칙 기반으로 파생
- confidence / reasons / explanation 포함 구조화 출력 생성
- 프론트 UI와 rule-based/source-aware EQ가 이해하기 쉬운 contract 유지

## 현재 ML taxonomy

### Head A: 직접 학습하는 글로벌 이슈

- `muddy`
- `harsh`
- `buried`
- `boomy`
- `thin`
- `boxy`
- `nasal`
- `sibilant`
- `dull`

### Head B: 별도 학습하는 source presence / dominance

- `vocal`
- `guitar`
- `bass`
- `drums`
- `keys`

### Head C: v1에서 직접 학습하지 않는 파생 진단

- `vocal_buried`
- `guitar_harsh`
- `bass_muddy`
- `drums_overpower`
- `keys_masking`

### Detailed causes

다음 계층은 현재 primary training target이 아니라 explanatory metadata입니다.

- `low_mid_overlap`
- `sibilance`
- `competing_sources`
- `boxy_resonance`
- `guitar_presence_peak`
- `mid_range_masking`
- 기타 세부 cause labels

Python의 authoritative schema는 [ml/label_schema.py](ml/label_schema.py) 에 있고, 프론트엔드 mirror는 [src/app/audio/mlSchema.ts](src/app/audio/mlSchema.ts) 에 있습니다.

## 런타임 분석 플로우

1. 사용자가 UI에서 분석 시작
2. [src/app/hooks/useMonitoring.ts](src/app/hooks/useMonitoring.ts) 가 마이크 입력을 확보
3. native sample rate 원본 버퍼와 16kHz 분석 버퍼를 동시에 유지
4. [src/app/audio/featureExtraction.ts](src/app/audio/featureExtraction.ts) 에서 log-mel spectrogram / RMS 계산
5. [src/app/audio/mlInference.ts](src/app/audio/mlInference.ts) 가 `public/models/lightweight_audio_model.onnx` 로 추론
6. 모델 출력 `issue_probs` / `source_probs` 를 구조화된 `ml_output` 으로 변환
7. [src/app/audio/diagnosisPostProcessing.ts](src/app/audio/diagnosisPostProcessing.ts) 가 source-specific diagnosis 파생
8. local stem service가 가능하면 [src/app/audio/stemSeparationClient.ts](src/app/audio/stemSeparationClient.ts) 결과 병합
9. stem 결과가 부족하면 [src/app/audio/openSourceAudioTagging.ts](src/app/audio/openSourceAudioTagging.ts) fallback source tagging 수행
10. [src/app/audio/sourceAwareEq.ts](src/app/audio/sourceAwareEq.ts) 가 감지된 소스와 문제를 바탕으로 EQ recommendation 생성
11. [src/app/components/ResultCards.tsx](src/app/components/ResultCards.tsx) 와 [src/app/components/EQVisualization.tsx](src/app/components/EQVisualization.tsx) 가 결과를 렌더링

## Python 학습 플로우

1. [ml/train.py](ml/train.py) 가 public dataset root를 받음
2. [ml/dataset.py](ml/dataset.py) 가 manifest를 생성하거나 로드
3. 각 샘플을 16kHz mono, 3초, log-mel 입력으로 정규화
4. weak issue label과 weak/partial source label을 생성
5. source label이 불가능한 샘플은 mask로 학습에서 제외
6. [ml/model.py](ml/model.py) 의 shared encoder가 embedding 생성
7. issue head와 source head에 대해 별도 BCEWithLogitsLoss 계산
8. validation에서 per-label metric 계산
9. best epoch 이후 label threshold tuning 수행
10. checkpoint, threshold JSON, training history, optional ONNX artifact 저장

## 데이터셋 및 weak labeling 전략

현재 파이프라인은 public dataset만 대상으로 설계되어 있습니다.

- OpenMIC-2018
- Slakh2100
- MUSAN
- FSD50K

현재 라벨 품질은 명시적으로 구분됩니다.

- `weak`: 휴리스틱 또는 공개 annotation 기반 추정
- `reviewed`: 향후 human-reviewed label이 추가될 때 사용
- `derived`: 후처리로만 생성되는 label
- `unavailable`: 현재 데이터셋/샘플에서 supervision 불가

[ml/dataset.py](ml/dataset.py) 는 각 manifest entry에 다음 구조를 기록합니다.

- `issue_targets.values`
- `issue_targets.mask`
- `issue_targets.quality`
- `source_targets.values`
- `source_targets.mask`
- `source_targets.quality`
- `metadata.issue_reasons`
- `metadata.source_evidence`
- `metadata.source_support`
- `track_group_id`

핵심 원칙은 다음과 같습니다.

- source label을 확실히 알 수 없으면 억지로 positive/negative를 만들지 않음
- training code가 mask를 통해 unavailable supervision을 자연스럽게 처리
- future reviewed dataset이 들어와도 schema를 깨지 않고 연결 가능

## 모델 구조

[ml/model.py](ml/model.py) 의 현재 모델은 경량 CNN 기반 shared encoder 구조입니다.

- 입력: `log_mel_spectrogram` shape `(batch, time_steps, 64)`
- backbone: shared convolutional encoder
- pooled embedding: `192` 차원
- issue head 출력: `(batch, 9)`
- source head 출력: `(batch, 5)`

forward output contract:

- `issue_logits`
- `issue_probs`
- `source_logits`
- `source_probs`
- `embedding`
- `problem_probs`

`problem_probs` 는 legacy compatibility alias이며 `issue_probs` 와 동일합니다.

## ONNX contract

현재 export는 [ml/export_to_onnx.py](ml/export_to_onnx.py) 에서 수행합니다.

### 입력

- name: `log_mel_spectrogram`
- dtype: `float32`
- shape: `(batch, time_steps, 64)`
- dynamic axis: `batch`, `time_steps`

### 출력

- `issue_probs`: `(batch, 9)`
- `source_probs`: `(batch, 5)`

### companion metadata

export 시 `*.metadata.json` 도 함께 생성됩니다.

포함 정보:

- `schema_version`
- `issue_labels`
- `primary_issue_labels`
- `source_labels`
- `derived_diagnosis_labels`
- `thresholds`
- `issue_to_causes`
- `issue_to_source_affinity`
- `issue_fallback_eq`

주의할 점:

- 현재 저장소에 체크인된 browser artifact는 external data ONNX 포맷이므로 `public/models/lightweight_audio_model.onnx` 와 `public/models/lightweight_audio_model.onnx.data` 를 함께 유지해야 합니다.
- export 파이프라인은 metadata JSON도 생성하지만, 현재 프론트 런타임은 이를 자동 로드하지 않고 TS mirror schema를 사용합니다.

## 프론트엔드 ML 출력 구조

브라우저 추론 결과는 [src/app/types.ts](src/app/types.ts) 의 `MlInferenceOutput` 구조로 정규화됩니다.

```json
{
  "schema_version": "2.0.0",
  "issues": {
    "muddy": 0.82,
    "harsh": 0.14
  },
  "sources": {
    "vocal": 0.91,
    "guitar": 0.35
  },
  "derived_diagnoses": {
    "vocal_buried": {
      "score": 0.78,
      "reasons": ["buried_high", "vocal_present", "relative_presence_low"],
      "explanation": "Buried vocal likelihood comes from buried issue evidence plus strong vocal presence."
    }
  },
  "metadata": {
    "thresholds_used": {},
    "label_quality": {}
  }
}
```

실제 UI 호환성을 위해 다음도 유지합니다.

- `AnalysisResult.issues` 는 현재도 `muddy / harsh / buried` primary subset만 사용
- legacy ONNX outputs `problem_probs` / `instrument_probs` 도 프론트에서 허용
- source detection이 비어 있을 때는 model-predicted source를 fallback으로 활용 가능

## 주요 파일 맵

### 프론트엔드

- [src/app/App.tsx](src/app/App.tsx): 앱 진입점
- [src/app/hooks/useMonitoring.ts](src/app/hooks/useMonitoring.ts): 실시간 캡처, 추론, stem/fallback 병합의 중심
- [src/app/audio/featureExtraction.ts](src/app/audio/featureExtraction.ts): log-mel / RMS 추출
- [src/app/audio/mlInference.ts](src/app/audio/mlInference.ts): ONNX Runtime Web 추론
- [src/app/audio/mlSchema.ts](src/app/audio/mlSchema.ts): TS-side schema mirror
- [src/app/audio/diagnosisPostProcessing.ts](src/app/audio/diagnosisPostProcessing.ts): 파생 진단 생성
- [src/app/audio/ruleBasedAnalysis.ts](src/app/audio/ruleBasedAnalysis.ts): rule-based issue fallback
- [src/app/audio/openSourceAudioTagging.ts](src/app/audio/openSourceAudioTagging.ts): YAMNet fallback tagging
- [src/app/audio/stemSeparationClient.ts](src/app/audio/stemSeparationClient.ts): local stem sidecar 연동
- [src/app/audio/sourceAwareEq.ts](src/app/audio/sourceAwareEq.ts): source-aware EQ recommendation
- [src/app/types.ts](src/app/types.ts): 사용자-facing / ML-facing 타입 정의

### Python / ML

- [ml/label_schema.py](ml/label_schema.py): authoritative ML schema
- [ml/preprocessing.py](ml/preprocessing.py): 오디오 로딩, resample, log-mel, spectral stats
- [ml/dataset.py](ml/dataset.py): manifest 빌드, weak label 생성, split 관리
- [ml/model.py](ml/model.py): shared encoder + issue/source heads
- [ml/metrics.py](ml/metrics.py): per-label metric, threshold tuning
- [ml/postprocessing.py](ml/postprocessing.py): Python-side hierarchical inference output / derived diagnosis
- [ml/train.py](ml/train.py): end-to-end training loop
- [ml/export_to_onnx.py](ml/export_to_onnx.py): ONNX export + verification + metadata export
- [ml/lightweight_audio_model.py](ml/lightweight_audio_model.py): compatibility export shim
- [ml/stem_separation_service.py](ml/stem_separation_service.py): local stem separation sidecar

### 테스트

- [ml/tests/test_lightweight_audio_model.py](ml/tests/test_lightweight_audio_model.py)
- [ml/tests/test_export_to_onnx.py](ml/tests/test_export_to_onnx.py)
- [ml/tests/test_training_pipeline.py](ml/tests/test_training_pipeline.py)
- [ml/tests/test_postprocessing.py](ml/tests/test_postprocessing.py)

## 실행 방법

### 1. 프론트엔드 개발 서버

```bash
npm install
npm run dev
```

### 2. 프론트엔드 프로덕션 빌드 확인

```bash
npm run build
```

### 3. ML 테스트 환경 준비

```bash
bash ml/setup_test_env.sh
```

### 4. ML 테스트 실행

```bash
PYTHONPATH=. ./.venv-ml/bin/python -m unittest discover -s ml/tests -p 'test_*.py' -v
```

또는

```bash
bash ml/run_ml_tests.sh
```

### 5. local stem service 준비 및 실행

설치:

```bash
bash ml/setup_stem_service.sh
```

실행:

```bash
bash ml/run_stem_service.sh
```

health check:

```bash
curl http://127.0.0.1:8765/health
```

프론트에서 다른 주소를 쓰려면:

```bash
VITE_STEM_SERVICE_URL=http://127.0.0.1:8765
```

### 6. 모델 학습

기본 명령 예시:

```bash
python3 ml/train.py \
  --openmic-root /path/to/openmic \
  --slakh-root /path/to/slakh \
  --musan-root /path/to/musan \
  --fsd50k-root /path/to/fsd50k \
  --rebuild-manifest \
  --epochs 6 \
  --batch-size 16 \
  --checkpoint-dir ml/checkpoints
```

빠른 실험 예시:

```bash
python3 ml/train.py \
  --openmic-root /path/to/openmic \
  --slakh-root /path/to/slakh \
  --musan-root /path/to/musan \
  --fsd50k-root /path/to/fsd50k \
  --rebuild-manifest \
  --clips-per-file 1 \
  --max-files-per-dataset 50 \
  --epochs 2 \
  --batch-size 8 \
  --device cpu
```

학습 후 ONNX까지 바로 export:

```bash
python3 ml/train.py \
  --openmic-root /path/to/openmic \
  --slakh-root /path/to/slakh \
  --musan-root /path/to/musan \
  --fsd50k-root /path/to/fsd50k \
  --rebuild-manifest \
  --epochs 6 \
  --export-onnx \
  --onnx-output ml/checkpoints/lightweight_audio_model.onnx
```

### 7. 단독 ONNX export

```bash
python3 ml/export_to_onnx.py \
  --checkpoint ml/checkpoints/best_sound_issue_model.pt \
  --output ml/checkpoints/lightweight_audio_model.onnx \
  --time-steps 128 \
  --verify
```

## 학습 결과물 위치

기본 경로 기준:

- manifest: `ml/artifacts/public_dataset_manifest.jsonl`
- best checkpoint: `ml/checkpoints/best_sound_issue_model.pt`
- last checkpoint: `ml/checkpoints/last_sound_issue_model.pt`
- thresholds: `ml/checkpoints/label_thresholds.json`
- training history: `ml/checkpoints/training_history.json`
- ONNX: `ml/checkpoints/lightweight_audio_model.onnx`
- ONNX metadata: `ml/checkpoints/lightweight_audio_model.metadata.json`

브라우저 런타임 기본 모델 경로:

- [public/models/lightweight_audio_model.onnx](public/models/lightweight_audio_model.onnx)
- [public/models/lightweight_audio_model.onnx.data](public/models/lightweight_audio_model.onnx.data)
- [public/models/yamnet.tflite](public/models/yamnet.tflite)

## 검증 상태

현재 저장소 기준으로 다음 검증이 통과한 상태입니다.

- Python unit tests
- end-to-end toy public dataset training test
- ONNX export verification with `onnxruntime`
- frontend production build

대표 검증 포인트:

- variable-length spectrogram 입력 지원
- `(batch, 9)` issue output shape 검증
- `(batch, 5)` source output shape 검증
- PyTorch vs ONNX numerical closeness 검증
- manifest에 hierarchical targets와 `track_group_id` 포함 여부 검증

## 디버깅 포인트

브라우저 콘솔 로그:

- `[audio-features]`
- `[audio-ml]`
- `[audio-rules]`
- `[audio-stems]`
- `[audio-tags]`

해석 가이드:

- `[audio-ml]` 에서 schema와 model load 상태를 확인 가능
- `[audio-stems]` 가 있으면 local stem service 경로 사용 중
- `[audio-tags]` 만 있으면 YAMNet fallback 중심
- source-aware EQ 결과가 이상하면 `detectedSources`, `stemMetrics`, `ml_output.sources` 를 같이 확인

## 현재 한계와 주의사항

- issue labels는 여전히 weak supervision 기반이며, human-reviewed gold label이 아님
- source labels도 dataset metadata, CSV, filename, stem evidence에 일부 의존
- source-specific diagnoses는 v1에서 learned head가 아니라 post-processing derived output
- 프론트는 현재 ONNX metadata JSON을 런타임에 로드하지 않고 TS-side schema mirror를 사용
- `imbalance` 는 product-level diagnostic problem으로 남아 있으며, trainable head로 들어가 있지 않음
- current checked-in browser ONNX artifact는 pipeline validation 성격이 포함될 수 있으므로, 실제 품질 향상을 원하면 재학습한 checkpoint로 재배포해야 함

## 다음 추천 개선

1. reviewed label ingestion 경로 추가
2. frontend runtime이 ONNX metadata JSON을 직접 로드하도록 개선
3. Slakh/OpenMIC source supervision을 더 정확히 파싱
4. held-out test split과 정식 test report 분리
5. derived diagnosis를 artifact-driven threshold로 더 세밀하게 조정
6. 필요한 시점에 optional cause head를 추가하되, reviewed cause label이 확보된 뒤 진행
