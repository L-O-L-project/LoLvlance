# HANDOVER

## 한 줄 상태

현재 LoLvlance는 "브라우저 실시간 오디오 분석 UI + local stem separation fallback 구조 + hierarchical multi-head ML 학습/ONNX export 파이프라인"까지 연결된 상태입니다. 이전 flat 3-label prototype 문서는 더 이상 기준이 아니며, 현재 authoritative ML schema는 `2.0.0` 입니다.

## 이번 기준선에서 가장 중요한 변화

이전 문서와 비교해 달라진 핵심은 아래 네 가지입니다.

1. training pipeline이 실제로 존재합니다.
2. 모델은 더 이상 3-output flat classifier가 아닙니다.
3. source-specific diagnosis는 direct train target이 아니라 post-processing derived output입니다.
4. inference contract가 `issue_probs` + `source_probs` 구조로 확장되었습니다.

즉, 지금부터 이 프로젝트를 이어받는 사람은 "브라우저 추론만 있는 데모"가 아니라 "재학습 가능한 Python ML 시스템 + 브라우저 추론 제품"으로 이해하면 됩니다.

## 현재 실제로 동작하는 것

### 브라우저 런타임

- 마이크 권한 요청 및 상태 표시
- native sample rate 캡처
- 16kHz 분석 버퍼 유지
- 약 3초 rolling buffer 유지
- log-mel spectrogram / RMS feature extraction
- ONNX Runtime Web 기반 경량 ML 추론
- rule-based issue fallback
- local stem separation sidecar 연동
- MediaPipe YAMNet fallback source tagging
- source-aware EQ recommendation 생성
- 문제 카드 / source 카드 / stem metric / EQ UI 렌더링

### Python / ML

- public dataset 기반 manifest 생성
- weak hierarchical label 생성
- masked multi-head training
- per-label metric 계산
- threshold tuning
- checkpoint 저장
- ONNX export
- `onnxruntime` 검증
- post-processing derived diagnosis 로직

### 검증

- Python unit tests 통과
- toy public dataset 기반 end-to-end training test 통과
- frontend build 통과

## 현재 아키텍처를 한 문장으로 설명하면

공용 데이터셋으로 weak hierarchical supervision을 만들고, shared encoder 위에 issue head와 source head를 학습한 뒤, 브라우저에서는 ONNX output을 받아 source-specific diagnosis를 파생하고 stem/rule 결과와 함께 UI에 보여주는 구조입니다.

## 절대 헷갈리면 안 되는 taxonomy

### Head A: 직접 학습

- `muddy`
- `harsh`
- `buried`
- `boomy`
- `thin`
- `boxy`
- `nasal`
- `sibilant`
- `dull`

### Head B: 별도 학습

- `vocal`
- `guitar`
- `bass`
- `drums`
- `keys`

### v1에서 직접 학습하지 않음

- `vocal_buried`
- `guitar_harsh`
- `bass_muddy`
- `drums_overpower`
- `keys_masking`

### product-layer but not train head

- `imbalance`

`imbalance` 는 현재 rule/product 계층에 남아 있고, trainable issue head에는 포함되지 않습니다.

## authoritative schema 위치

### Python source of truth

- [ml/label_schema.py](/Users/kimhajun/Downloads/LoLvlance/ml/label_schema.py:1)

여기서 관리하는 것:

- `SCHEMA_VERSION = "2.0.0"`
- issue labels
- primary UI issues
- source labels
- derived diagnosis labels
- cause metadata labels
- default thresholds
- fallback EQ
- issue-to-cause mapping
- issue-to-source-affinity mapping

### frontend mirror

- [src/app/audio/mlSchema.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/mlSchema.ts:1)

현재 프론트는 ONNX companion metadata JSON을 런타임에 읽지 않고, 이 TS mirror를 사용합니다. Python schema를 바꾸면 이 파일도 같이 맞춰야 합니다.

## 현재 runtime flow

핵심 entrypoint는 [src/app/hooks/useMonitoring.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/hooks/useMonitoring.ts:1) 입니다.

실행 순서:

1. UI에서 분석 시작
2. 마이크 입력 확보
3. native sample rate 원본 버퍼와 16kHz 분석 버퍼를 동시에 유지
4. [src/app/audio/featureExtraction.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/featureExtraction.ts:1) 에서 log-mel / RMS 계산
5. [src/app/audio/mlInference.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/mlInference.ts:1) 로 ONNX 추론
6. `issue_probs` / `source_probs` 를 `ml_output` 으로 구조화
7. [src/app/audio/diagnosisPostProcessing.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/diagnosisPostProcessing.ts:1) 에서 파생 진단 생성
8. local stem service가 켜져 있으면 [src/app/audio/stemSeparationClient.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/stemSeparationClient.ts:1) 결과 병합
9. stem 결과가 부족하면 [src/app/audio/openSourceAudioTagging.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/openSourceAudioTagging.ts:1) fallback 사용
10. [src/app/audio/sourceAwareEq.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/sourceAwareEq.ts:1) 에서 source-aware EQ 생성
11. [src/app/components/ResultCards.tsx](/Users/kimhajun/Downloads/LoLvlance/src/app/components/ResultCards.tsx:1) 에서 사용자에게 표시

## 현재 training flow

핵심 entrypoint는 [ml/train.py](/Users/kimhajun/Downloads/LoLvlance/ml/train.py:1) 입니다.

실행 순서:

1. dataset root 수신
2. manifest 빌드 또는 기존 manifest 로드
3. split별 dataset/dataloader 구성
4. shared encoder + multi-head model 생성
5. issue/source pos weight 계산
6. masked BCEWithLogitsLoss로 epoch 학습
7. validation output 수집
8. per-label metric 계산
9. best epoch 선택
10. validation set 기반 threshold tuning
11. checkpoint / threshold JSON / history 저장
12. 필요 시 ONNX export

## dataset / label generation 현실

핵심 파일은 [ml/dataset.py](/Users/kimhajun/Downloads/LoLvlance/ml/dataset.py:1) 입니다.

현재 지원 public datasets:

- OpenMIC-2018
- Slakh2100
- MUSAN
- FSD50K

manifest 설계 원칙:

- 무조건 flat label vector 하나로 만들지 않음
- `issue_targets` 와 `source_targets` 를 분리
- label quality를 explicit하게 기록
- unavailable source supervision은 `mask = 0`
- `track_group_id` 기준으로 leakage를 줄이는 split 유지

manifest entry에서 특히 봐야 할 필드:

- `schema_version`
- `track_group_id`
- `issue_targets.values`
- `issue_targets.mask`
- `issue_targets.quality`
- `source_targets.values`
- `source_targets.mask`
- `source_targets.quality`
- `metadata.issue_reasons`
- `metadata.source_evidence`
- `metadata.source_support`

중요한 현실적 제한:

- source label은 일부 샘플에서만 충분히 지원됩니다.
- 따라서 source supervision coverage는 sparse할 수 있습니다.
- 이건 버그가 아니라 설계 의도입니다.

## 모델 계약

핵심 파일:

- [ml/model.py](/Users/kimhajun/Downloads/LoLvlance/ml/model.py:1)
- [ml/lightweight_audio_model.py](/Users/kimhajun/Downloads/LoLvlance/ml/lightweight_audio_model.py:1)

현재 모델 계약:

- 입력: `log_mel_spectrogram`
- shape: `(batch, time_steps, 64)`
- shared embedding size: `192`
- issue output count: `9`
- source output count: `5`

forward output:

- `issue_logits`
- `issue_probs`
- `source_logits`
- `source_probs`
- `embedding`
- `problem_probs`

`problem_probs` 는 legacy alias라서 프론트/테스트 일부가 기존 이름을 계속 써도 동작합니다.

## ONNX 계약

핵심 파일:

- [ml/export_to_onnx.py](/Users/kimhajun/Downloads/LoLvlance/ml/export_to_onnx.py:1)

현재 export output names:

- `issue_probs`
- `source_probs`

입력:

- `log_mel_spectrogram`

shape:

- input: `(batch, time_steps, 64)`
- output issue: `(batch, 9)`
- output source: `(batch, 5)`

export 시 함께 생성되는 것:

- `*.metadata.json`

여기에는 schema version, labels, thresholds, fallback EQ mapping 등이 들어갑니다.

중요:

- 현재 `public/models/lightweight_audio_model.onnx` 는 external-data ONNX 형식이라 `.onnx.data` 파일도 같이 필요합니다.
- 예전 문서의 "onnx.data만 있으면 된다" 식 설명은 잘못입니다.
- 올바른 설명은 "현재 커밋된 artifact는 `.onnx` 와 `.onnx.data` 를 함께 유지해야 하고, 새 export는 metadata JSON도 추가로 생성한다" 입니다.

## 프론트 inference contract

[src/app/audio/mlInference.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/mlInference.ts:1) 가 실제 브라우저 추론 계약을 구현합니다.

중요 포인트:

- 새 ONNX outputs `issue_probs`, `source_probs` 우선 사용
- legacy outputs `problem_probs`, `instrument_probs` fallback 허용
- silence RMS 이하에서는 빈 결과 반환
- `AnalysisResult.issues` 는 여전히 `muddy / harsh / buried` 만 primary UI issue로 노출
- 전체 hierarchical 결과는 `AnalysisResult.ml_output` 으로 따로 보존

즉, 기존 UI와 호환성을 유지하면서 더 풍부한 구조를 추가한 상태입니다.

## 파생 진단 규칙

Python:

- [ml/postprocessing.py](/Users/kimhajun/Downloads/LoLvlance/ml/postprocessing.py:1)

Frontend:

- [src/app/audio/diagnosisPostProcessing.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/diagnosisPostProcessing.ts:1)

현재 구현된 규칙:

- `vocal_buried = buried + vocal + presence 부족 보너스`
- `guitar_harsh = harsh + guitar + sibilant/presence peak 보너스`
- `bass_muddy = muddy + bass + boomy 보너스`
- `drums_overpower = harsh|boomy + drums + thin 보너스`
- `keys_masking = buried|boxy + keys + nasal 보너스`

이 값들은 direct training target이 아닙니다. derived thresholds를 넘고 이유가 2개 이상 잡힐 때만 결과에 포함됩니다.

## 가장 먼저 읽을 파일 순서

새 엔지니어가 빨리 적응하려면 아래 순서가 가장 효율적입니다.

1. [README.md](/Users/kimhajun/Downloads/LoLvlance/README.md:1)
2. [src/app/hooks/useMonitoring.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/hooks/useMonitoring.ts:1)
3. [src/app/audio/mlInference.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/mlInference.ts:1)
4. [src/app/types.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/types.ts:1)
5. [ml/label_schema.py](/Users/kimhajun/Downloads/LoLvlance/ml/label_schema.py:1)
6. [ml/dataset.py](/Users/kimhajun/Downloads/LoLvlance/ml/dataset.py:1)
7. [ml/model.py](/Users/kimhajun/Downloads/LoLvlance/ml/model.py:1)
8. [ml/train.py](/Users/kimhajun/Downloads/LoLvlance/ml/train.py:1)
9. [ml/export_to_onnx.py](/Users/kimhajun/Downloads/LoLvlance/ml/export_to_onnx.py:1)
10. [ml/postprocessing.py](/Users/kimhajun/Downloads/LoLvlance/ml/postprocessing.py:1)

## 자주 쓰는 실행 명령

### frontend

```bash
npm install
npm run dev
```

### build

```bash
npm run build
```

### ML 테스트 환경

```bash
bash ml/setup_test_env.sh
```

### ML 테스트

```bash
PYTHONPATH=. ./.venv-ml/bin/python -m unittest discover -s ml/tests -p 'test_*.py' -v
```

### stem service setup / run

```bash
bash ml/setup_stem_service.sh
bash ml/run_stem_service.sh
```

### stem service health

```bash
curl http://127.0.0.1:8765/health
```

### training

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

### training + ONNX export

```bash
python3 ml/train.py \
  --openmic-root /path/to/openmic \
  --slakh-root /path/to/slakh \
  --musan-root /path/to/musan \
  --fsd50k-root /path/to/fsd50k \
  --rebuild-manifest \
  --export-onnx \
  --onnx-output ml/checkpoints/lightweight_audio_model.onnx
```

### standalone ONNX export

```bash
python3 ml/export_to_onnx.py \
  --checkpoint ml/checkpoints/best_sound_issue_model.pt \
  --output ml/checkpoints/lightweight_audio_model.onnx \
  --time-steps 128 \
  --verify
```

## 결과물 경로

학습 결과물 기본 경로:

- `ml/artifacts/public_dataset_manifest.jsonl`
- `ml/checkpoints/best_sound_issue_model.pt`
- `ml/checkpoints/last_sound_issue_model.pt`
- `ml/checkpoints/label_thresholds.json`
- `ml/checkpoints/training_history.json`
- `ml/checkpoints/lightweight_audio_model.onnx`
- `ml/checkpoints/lightweight_audio_model.metadata.json`

브라우저 런타임 모델:

- [public/models/lightweight_audio_model.onnx](/Users/kimhajun/Downloads/LoLvlance/public/models/lightweight_audio_model.onnx:1)
- [public/models/lightweight_audio_model.onnx.data](/Users/kimhajun/Downloads/LoLvlance/public/models/lightweight_audio_model.onnx.data:1)
- [public/models/yamnet.tflite](/Users/kimhajun/Downloads/LoLvlance/public/models/yamnet.tflite:1)

## 디버깅 가이드

### 브라우저 로그 키

- `[audio-features]`
- `[audio-ml]`
- `[audio-rules]`
- `[audio-stems]`
- `[audio-tags]`

### 해석 팁

- `[audio-ml]` 에 schema/version/model readiness가 찍히는지 먼저 확인
- `[audio-stems]` 가 없고 `[audio-tags]` 만 있으면 sidecar가 죽어 있거나 연결 실패
- source-aware EQ가 비어 있으면 `detectedSources` 와 `stemMetrics` 를 먼저 확인
- UI에는 문제가 없는데 derived diagnosis가 없으면 thresholds 때문에 걸러졌을 가능성이 큼

### training 쪽에서 먼저 볼 파일

- threshold tuning 이상: [ml/metrics.py](/Users/kimhajun/Downloads/LoLvlance/ml/metrics.py:1)
- source supervision coverage 부족: [ml/dataset.py](/Users/kimhajun/Downloads/LoLvlance/ml/dataset.py:1)
- ONNX mismatch: [ml/export_to_onnx.py](/Users/kimhajun/Downloads/LoLvlance/ml/export_to_onnx.py:1)

## 현재 알려진 한계

1. issue labels는 여전히 weak labels 중심입니다.
2. source labels는 dataset metadata/stem evidence 가용성에 따라 supervision coverage가 달라집니다.
3. 파생 진단은 heuristic post-processing 입니다.
4. 프론트는 metadata JSON을 artifact-driven으로 읽지 않습니다.
5. `imbalance` 는 rule/product 레이어에 남아 있습니다.
6. committed browser model artifact가 실제 운영용 품질을 보장하지는 않습니다.

## 지금 시점의 우선순위

가장 추천하는 다음 단계는 아래 순서입니다.

1. reviewed labels를 일부라도 도입해 weak labels를 보정
2. train/export된 metadata JSON을 프론트 런타임에서 직접 읽도록 개선
3. public dataset source parsing을 더 정교화
4. held-out test split을 별도로 만들어 정식 benchmark 분리
5. derived diagnosis rule을 threshold calibration과 함께 재정리

## 실무적으로 꼭 기억할 점

- source-specific diagnosis를 primary classes로 다시 평탄화하지 말 것
- Python schema와 TS schema mirror가 어긋나면 런타임 결과 해석이 틀어짐
- 새로운 label을 추가할 때는 반드시 issue/source/derived 중 어느 계층인지 먼저 결정할 것
- current frontend compatibility를 위해 `AnalysisResult.issues` 의 primary subset 동작을 함부로 깨지 말 것
- checked-in browser model을 교체할 때는 `.onnx`, `.onnx.data`, 필요 시 metadata JSON까지 artifact set으로 관리할 것
