# ML README

이 문서는 LoLvlance의 ML 관련 영역만 별도로 설명합니다.

- 현재 런타임 모델 상태와 학습 결과
- 모델 및 export 아키텍처
- degradation 기반 학습 경로
- 평가와 CI gating
- 입력 정합성 테스트 범위
- 모델 히스토리 참조

영문 버전: `ML_README.md`

## 바로가기

[![모델 아키텍처](https://img.shields.io/badge/모델%20아키텍처-2563EB?style=for-the-badge&logo=pytorch&logoColor=white)](#model-architecture)
[![학습 파이프라인](https://img.shields.io/badge/학습%20파이프라인-7C3AED?style=for-the-badge&logo=python&logoColor=white)](#training-pipeline)
[![평가](https://img.shields.io/badge/평가%20시스템-0891B2?style=for-the-badge&logo=target&logoColor=white)](#evaluation-system)
[![입력 정합성](https://img.shields.io/badge/입력%20정합성-DC2626?style=for-the-badge&logo=pytest&logoColor=white)](#input-integrity)
[![메인 README](https://img.shields.io/badge/메인-README-111827?style=for-the-badge&logo=gitbook&logoColor=white)](README.ko.md)
[![Handover](https://img.shields.io/badge/Handover-111827?style=for-the-badge&logo=files&logoColor=white)](HANDOVER.ko.md)

## 런타임 메모

- 브라우저 런타임은 rolling `3.0`초 윈도를 분석합니다.
- 모니터링 패스는 `src/app/hooks/useMonitoring.ts`에서 `1.0`초 stride로 실행됩니다.
- 활성 런타임 모델은 `v0.1-real-data` — MUSAN + OpenMIC-2018으로 학습된 첫 번째 실데이터 체크포인트입니다.
- 후속 학습 실행(`v0.2-fsd50k-extended`, FSD50K 추가)이 현재 진행 중입니다.
- 제품 포지셔닝은 continuous monitoring과 live session analysis를 유지해야 하며, 즉시 반응형 AI로 설명하면 안 됩니다.

<a id="model-architecture"></a>
## 1. 모델 아키텍처

### 활성 브라우저 런타임 계약

브라우저 ONNX 계약은 현재 다음 값을 유지합니다.

- `issue_probs`
- `source_probs`
- `eq_freq`
- `eq_gain_db`

이것은 프론트엔드 호환성을 위한 계약입니다.

### Python 모델 코드

현재 Python 모델 구현은 `ml/model.py`에 있으며 다음 클래스를 포함합니다.

- `AudioIntelligenceNet`
- `ProductionAudioIntelligenceNet`
- `LightweightAudioAnalysisNet`

### Encoder 변형

- `student`
  브라우저 배포를 고려한 lightweight CNN encoder
- `teacher`
  더 강한 오프라인 학습 및 distillation을 위한 transformer 스타일 spectrogram encoder

### 현재 learned head

현재 모델 코드는 다음을 지원합니다.

- source classification head
- source-conditioned issue classification head
- learned multi-band EQ head

learned EQ head는 다음을 예측합니다.

```text
eq_params: (batch, N_bands, 2)
```

각 밴드는 다음을 포함합니다.

- frequency in Hz
- gain in dB

코드 기본 band count:

```text
N_bands = 5
```

### 중요한 구분

Python 학습 모델은 learned multi-band EQ를 지원하지만, 브라우저용 ONNX export는 현재 호환성 요약 출력만 노출합니다.

- `eq_freq`
- `eq_gain_db`

즉, 내부 모델이 아직 single-band라는 뜻이 아니라, **런타임 계약이 아직 요약형**이라는 뜻입니다.

### 레이블 스키마

Issue 레이블:

```text
[muddy, harsh, buried, boomy, thin, boxy, nasal, sibilant, dull]
```

Source 레이블:

```text
[vocal, guitar, bass, drums, keys]
```

스키마 source-of-truth:

- Python: `ml/label_schema.py`
- 프론트엔드 mirror: `src/app/audio/mlSchema.ts`

<a id="training-pipeline"></a>
## 2. 학습 파이프라인

### 학습 히스토리

모든 학습된 모델의 전체 기록은 `ml/model_history.md`를 참고하세요.

요약:

| 버전 | 상태 | 클립 수 | Macro Issue F1 | Macro Source F1 |
|---|---|---|---|---|
| `v0.0-pipeline-check` | 보관됨 | Synthetic | N/A | N/A |
| `v0.1-real-data` | **활성** | 65,738 | 0.531 | 0.612 |
| `v0.2-fsd50k-extended` | 학습 중 | ~85K 예상 | 미정 | 미정 |

### 현재 학습 방향

ML 스택은 이제 순수 규칙 투영 경로가 아니라, real-audio-plus-degradation 경로를 중심으로 구성되어 있습니다.

핵심 파일:

- `ml/train.py`
- `ml/degradation.py`
- `ml/losses.py`
- `ml/model.py`

### 실제 학습 데이터 확보

`ml/download_datasets.py`를 사용해 지원하는 공개 데이터셋을 다운로드합니다.

```bash
# 사용 가능한 데이터셋 목록 확인
python ml/download_datasets.py --list

# MUSAN 다운로드 (~11 GB, 공개 도메인, 권장 시작점)
python ml/download_datasets.py --datasets musan --output-root data/datasets

# 전부 다운로드 (MUSAN + FSD50K + OpenMIC)
python ml/download_datasets.py --datasets musan fsd50k openmic --output-root data/datasets
```

이후 학습:

```bash
python -m ml.train \
  --audio-root data/datasets/musan \
  --musan-root data/datasets/musan \
  --rebuild-manifest --epochs 20 --export-onnx
```

`--audio-root`는 WAV/FLAC 파일이 있는 임의 폴더를 받습니다. 파일명에 `vocal`, `guitar`, `drums` 등이 포함되면 `infer_source_targets_from_path`를 통해 자동으로 source 레이블을 부여합니다.

### 사용자 피드백 수집 및 반영

배포 후 사용자는 브라우저에서 per-analysis 피드백을 내보낼 수 있습니다. 이를 학습에 반영하는 방법:

```bash
python ml/ingest_feedback.py \
    --feedback lolvlance_feedback_2024-01-15.jsonl \
    --output ml/artifacts/feedback_manifest.jsonl

# 병합 후 재학습
cat ml/artifacts/public_dataset_manifest.jsonl \
    ml/artifacts/feedback_manifest.jsonl \
    > ml/artifacts/merged_manifest.jsonl

python -m ml.train \
    --manifest-path ml/artifacts/merged_manifest.jsonl \
    --epochs 10 --export-onnx
```

`verdict=correct` 항목은 weak soft label로, `verdict=wrong` + 사용자 선택 레이블 항목은 reviewed hard label로 변환됩니다.

### 코드 기준 데이터 전략

현재 저장소는 degradation 기반 데이터셋 경로를 지원합니다.

- `--audio-root` 또는 공개 데이터셋 루트로 real source audio manifest 구성
- 학습 중 `RealAudioDegradationDataset`을 통해 실시간 controlled degradation 적용
- degraded audio로 issue/source head 학습
- inverse EQ target으로 EQ head 학습

### 구현된 degradation 종류

현재 degradation 파이프라인은 다음을 포함합니다.

- EQ distortion
- compression
- synthetic reverb
- low-pass / high-pass filter error

생성된 recipe는 다음을 저장합니다.

- inverse EQ band
- issue label
- optional compression metadata
- optional reverb metadata
- optional filter-error metadata

### 학습 loss

`ml/losses.py`의 현재 loss:

- issue prediction용 focal loss
- head mode에 따라 BCE 또는 softmax source loss
- EQ regression용 Smooth L1 loss
- student 학습용 KL 기반 distillation loss

### Distillation 경로

현재 코드는 teacher-student 학습을 지원합니다.

- `ml/train.py`에 teacher checkpoint 전달 가능
- student는 issue logits를 distill 가능
- student는 source logits를 distill 가능
- student는 normalized EQ output도 distill 가능

### 현재 현실 점검

학습 코드는 더 강력한 multi-band EQ 경로와 distillation을 지원합니다. 그러나 브라우저 ONNX 계약은 아직 single-band 호환성 요약 출력입니다.

활성 브라우저 모델(`v0.1-real-data`)은 첫 번째 실데이터 체크포인트입니다. guitar, drums, vocal에서 의미 있는 성능을 냅니다. boxy, nasal, thin은 학습 데이터 부족으로 성능이 약합니다.

<a id="evaluation-system"></a>
## 3. 평가 시스템

### Golden 평가

golden 샘플은 다음 경로에 있습니다.

```text
eval/goldens/
```

Evaluator:

```text
ml/eval/evaluate.py
```

수행 내용:

- golden sample metadata 로드
- 모델 추론 실행
- issue/source 레이블의 precision, recall, F1 계산
- confusion summary 출력
- 저장된 baseline과 비교

### Baseline 추적

baseline 파일:

```text
ml/eval/baseline.json
```

사용 목적:

- 이전 체크포인트와 비교
- per-label regression 감지
- CI gating

### CI 게이트

워크플로:

```text
.github/workflows/eval.yml
```

현재 임계값:

- `min_f1 = 0.65`

### 중요한 한계

현재 golden set은 아직 작습니다. 회귀 감지에는 유용하지만, 강한 모델 품질 주장을 뒷받침할 정도는 아닙니다.

<a id="input-integrity"></a>
## 4. 입력 정합성

저장소에는 이제 전처리 일관성과 데이터셋 hygiene를 위한 명시적 테스트가 있습니다.

### 테스트

- `ml/tests/test_waveform_parity.py`
- `ml/tests/test_feature_parity.py`
- `ml/tests/test_resampler.py`
- `ml/tests/test_manifest_leakage.py`

### 검증 범위

이 테스트는 다음을 확인합니다.

- Python 전처리와 browser-equivalent 전처리의 waveform parity
- 최종 모델 입력 feature parity
- 일반적인 브라우저 캡처 샘플레이트에 대한 resampler 동작
- manifest 내 train/validation leakage

### 왜 필요한가

모델 품질은 아키텍처를 바꾸지 않았더라도 입력 불일치 때문에 무너질 수 있습니다.

이 테스트의 목적은 그 failure mode를 눈에 보이게 만드는 것입니다.

<a id="current-status"></a>
## 5. 현재 상태

### 프로덕션에 가까운 부분

- 브라우저 ML loading 경로 end-to-end 동작
- ONNX export 경로 동작
- `v0.1-real-data`가 실제 공개 오디오로 학습되어 기본값으로 배포됨
- 소스 탐지 동작: guitar F1=0.82, drums AUROC=0.93, vocal AUROC=0.92
- 모니터링 파이프라인은 의도적인 blind gap 없는 rolling window 사용
- golden 평가와 integrity 테스트 존재, CI에서 게이트됨
- 실제 데이터셋 다운로드 파이프라인 존재 (`ml/download_datasets.py`)
- 브라우저 피드백 수집 및 수집 파이프라인 존재 (`FeedbackWidget` → `ml/ingest_feedback.py`)

### 아직 개선이 필요한 부분

- 일부 이슈 레이블이 약함: `boxy` F1=0.0, `nasal`/`thin`/`sibilant` AUROC 약 0.56–0.60
- CI golden set이 3개 샘플뿐 — 실제 프로덕션 벤치마크가 아님
- `v0.2-fsd50k-extended` 학습 진행 중 — 희소 레이블 coverage 개선 목표
- 피드백 수집은 수동 (내보내기 → 스크립트 → 재학습), 아직 자동화되지 않음
- 브라우저 EQ 출력이 아직 single-band 호환성 계약

### 올바른 해석

현재 저장소가 증명하는 것은 다음입니다.

- 파이프라인이 실제 오디오에서 실행된다
- 소스 탐지가 실제 녹음에서 일반화된다
- 이슈 탐지가 일반적인 레이블(muddy, harsh, dull)에서 의미 있다
- 회귀가 CI로 잡아낼 수 있다

하지만 아직 증명하지 못한 것은 다음입니다.

- 9개 이슈 레이블 전체에 걸친 프로덕션 정확도
- boxy, nasal, thin의 신뢰할 수 있는 예측 (학습 데이터 부족)
- 브라우저에서의 프로덕션급 EQ intelligence (single-band만)
