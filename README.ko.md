# LoLvlance

LoLvlance는 교회 사운드팀, 예배팀 자원봉사자, 소규모 라이브 운영자를 위한 브라우저 기반 오디오 모니터링 시스템입니다.

현재 제품 목표는 단순합니다.

**라이브 세션 중 연속 모니터링 + 문제 가이드 + 첫 번째 EQ 조정 제안**

LoLvlance는 브라우저에서 마이크 오디오를 받아 최근 몇 초의 오디오를 롤링 방식으로 분석하고, 몇 초마다 가이드를 갱신해 사용자가 다음에 무엇을 확인해야 하는지 빠르게 판단하도록 돕습니다. 이 시스템은 권위적인 판정 도구가 아니라 보조 도구입니다.

## 바로가기

[![프로젝트 개요](https://img.shields.io/badge/프로젝트%20개요-2563EB?style=for-the-badge&logo=bookstack&logoColor=white)](#project-overview)
[![시스템 아키텍처](https://img.shields.io/badge/시스템%20아키텍처-7C3AED?style=for-the-badge&logo=appveyor&logoColor=white)](#system-architecture)
[![모델 버전 관리](https://img.shields.io/badge/모델%20버전%20관리-DC2626?style=for-the-badge&logo=onnx&logoColor=white)](#model-versioning)
[![모니터링](https://img.shields.io/badge/모니터링%20시스템-0891B2?style=for-the-badge&logo=webaudio&logoColor=white)](#monitoring-system)
[![평가](https://img.shields.io/badge/평가%20시스템-0F766E?style=for-the-badge&logo=githubactions&logoColor=white)](#evaluation-system)
[![테스트](https://img.shields.io/badge/테스트%20인프라-D97706?style=for-the-badge&logo=pytest&logoColor=white)](#test-infrastructure)
[![개발 워크플로](https://img.shields.io/badge/개발%20워크플로-16A34A?style=for-the-badge&logo=vite&logoColor=white)](#development-workflow)
[![로드맵](https://img.shields.io/badge/로드맵-1D4ED8?style=for-the-badge&logo=roadmapdotsh&logoColor=white)](#roadmap)

[![English README](https://img.shields.io/badge/English-README-111827?style=for-the-badge&logo=readme&logoColor=white)](README.md)
[![개발자 인수인계](https://img.shields.io/badge/개발자-Handover-111827?style=for-the-badge&logo=gitbook&logoColor=white)](HANDOVER.ko.md)
[![ML 문서](https://img.shields.io/badge/ML-README-111827?style=for-the-badge&logo=pytorch&logoColor=white)](ML_README.ko.md)

## 문서 언어 버전

- 영어:
  - `README.md`
  - `HANDOVER.md`
  - `ML_README.md`
- 한국어:
  - `README.ko.md`
  - `HANDOVER.ko.md`
  - `ML_README.ko.md`

<a id="project-overview"></a>
## 1. 프로젝트 개요

LoLvlance는 세션 중 다음 세 가지 질문에 답하도록 설계되어 있습니다.

1. 지금 무엇이 이상한가?
2. 어느 소스가 가장 관련 있어 보이는가?
3. 사용자가 먼저 시도해야 할 EQ 조정은 무엇인가?

대상 사용자는 전업 믹싱 엔지니어가 아닙니다. 특히 교회 사운드팀, 소규모 라이브 팀처럼 다른 역할도 함께 수행하면서 빠른 판단이 필요한 운영자를 위한 제품입니다.

### 제품 포지셔닝

- LoLvlance는 **연속 모니터링**이며 즉시 반응형 분석이 아닙니다.
- 시스템은 롤링 오디오 윈도를 분석하고 몇 초마다 가이드를 갱신합니다.
- 결과는 **보조 가이드**로 해석해야 하며, 확정 진단으로 받아들이면 안 됩니다.

### 현재 제품이 하는 일

- 브라우저에서 마이크 오디오를 수집
- 롤링 분석 버퍼 유지
- 로컬에서 오디오 특징 추출
- 활성화되어 있으면 로컬 ONNX 모델 추론 실행
- ML 출력과 규칙 기반 진단, 소스 탐지, EQ 가이드를 병합

### 현재 제품이 주장하지 않는 것

- 아직 프로덕션급 AI 진단 시스템이 아님
- 아직 브라우저 경로에서 완전한 learned EQ 시스템이 아님
- 1초 미만 반응을 보장하지 않음

<a id="system-architecture"></a>
## 2. 시스템 아키텍처

### 전체 흐름

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

### 런타임 구성 요소

- `src/app/hooks/useMonitoring.ts`
  마이크 캡처, 롤링 버퍼, 추론 스케줄링, 스무딩, stem 서비스 연동, UI 업데이트를 총괄합니다.
- `src/app/audio/audioUtils.ts`
  버퍼 유틸리티, 리샘플링 헬퍼, 버퍼 스냅샷 생성, 관련 오디오 수학 로직을 담당합니다.
- `src/app/audio/featureExtraction.ts`
  버퍼된 오디오에서 프론트엔드 특징을 추출합니다.
- `src/app/audio/mlInference.ts`
  브라우저 ONNX 로딩, 출력 파싱, 모델 버전 로깅, ML 실패 시 fallback 동작을 담당합니다.
- `src/app/audio/diagnosisPostProcessing.ts`
  `vocal_buried` 같은 파생 진단을 생성합니다.
- `src/app/audio/sourceAwareEq.ts`
  사람이 읽을 수 있는 EQ 가이드를 생성합니다.
- `src/app/audio/ruleBasedAnalysis.ts`
  ML이 비활성화되거나 실패했을 때 규칙 기반 문제 분석을 수행합니다.

### 역할 분리

- **ML**
  log-mel 입력으로부터 issue 확률과 source 확률을 추정합니다.
- **규칙과 후처리**
  silence gating, 예측 해석, source evidence 병합, 출력 안정화, 사용자용 가이드 생성을 담당합니다.

<a id="model-versioning"></a>
## 3. 모델 버전 관리

LoLvlance에는 이제 명시적인 모델 격리 레이어가 있습니다.

### 현재 런타임 기본값

- 활성 기본 모델 버전: `v0.0-pipeline-check`
- 설정 파일: `src/app/config/modelRuntime.ts`
- 기본 브라우저 모델 경로: `public/models/lightweight_audio_model.onnx`
- 예약된 프로덕션 경로: `public/models/lightweight_audio_model.production.onnx`

### 현재 모델 상태

활성 브라우저 모델은 의도적으로 다음과 같이 표시됩니다.

```text
v0.0-pipeline-check
```

이 의미는 다음과 같습니다.

- 파이프라인 검증용 아티팩트임
- 프로덕션 준비 상태가 아님
- 알려진 편향이 있는 synthetic 데이터로 학습됨
- 실제 음성이나 음악에 대해 부정확할 수 있음

### 킬 스위치와 라우팅

프론트엔드 런타임은 다음을 지원합니다.

- `MODEL_VERSION`
- `ENABLE_MODEL`

동작 방식:

- `ENABLE_MODEL=false`이면 브라우저 ML을 건너뛰고 fallback 분석만 사용
- `MODEL_VERSION === "v0.0-pipeline-check"`이면 실험용 모델 경로 사용
- 그 외에는 미래의 프로덕션 모델 아티팩트로 라우팅 가능

### UI 상태 표시

현재 UI는 실험용 모델이 활성화되어 있을 때 명시적으로 경고합니다.

- `Experimental Mode`
- `Not production-ready`
- `Results may be inaccurate`

<a id="ml-system"></a>
## 4. ML 시스템

현재 ML 시스템은 두 층위로 이해해야 합니다.

### 활성 브라우저 런타임

브라우저는 현재 **실험용 체크포인트**를 제공하며, 호환성 ONNX 계약을 사용합니다.

- `issue_probs`
- `source_probs`
- `eq_freq`
- `eq_gain_db`

이 브라우저 경로는 안전을 위해 격리되어 있으며, 프로덕션 AI로 취급하면 안 됩니다.

### Python 학습 경로

Python 쪽 ML 시스템은 초기 2-head CNN 프로토타입을 넘어섰습니다.

현재 코드가 지원하는 내용:

- `ml/model.py`의 teacher/student 모델 변형
- teacher 경로용 transformer 스타일 spectrogram encoder
- student 경로용 경량 CNN encoder
- source-conditioned issue prediction
- `eq_params`를 출력하는 learned multi-band EQ head
- `ml/degradation.py`의 self-supervised degradation 학습 경로
- `ml/losses.py`의 distillation-ready loss

### 중요한 구분

학습 코드는 활성 브라우저 모델보다 앞서 있습니다.

현재 시점에서:

- **코드베이스**는 더 강한 learning-based 경로를 지원함
- **활성 브라우저 체크포인트**는 여전히 `v0.0-pipeline-check`임
- 따라서 **실제 제품 서사**는 보수적으로 유지되어야 함

<a id="audio-pipeline"></a>
## 5. 오디오 파이프라인

### 브라우저 오디오 경로

브라우저 모니터링 경로는 이제 고정된 윈도 반복 방식이 아니라 연속 캡처를 사용합니다.

주요 구성:

- 가능하면 `AudioWorklet` 기반 마이크 캡처
- 필요 시 `ScriptProcessorNode` fallback
- source/stem 경로용 native sample rate 롤링 버퍼
- ML 특징 추출용 `16 kHz` 리샘플 버퍼

### 전처리 계약

현재 전처리 계약은 다음과 같습니다.

- sample rate: `16_000`
- clip window: `3.0`초
- window size: `25 ms`
- hop size: `10 ms`
- FFT size: `512`
- mel bins: `64`

### 입력 정합성 메모

브라우저 전처리와 Python 전처리가 계속 일치하는지 검증하는 테스트 스위트가 추가되었습니다.

이 스위트는 다음을 다룹니다.

- waveform parity
- feature parity
- resampler consistency
- manifest leakage safety

이 테스트가 존재한다고 해서 전처리 일치가 영구적으로 보장되는 것은 아닙니다. 이 테스트는 릴리스 안전장치이며 항상 통과 상태를 유지해야 합니다.

<a id="monitoring-system"></a>
## 6. 모니터링 시스템

모니터링 시스템은 더 이상 예전의 "3초 수집, 대기, 반복" 패턴을 사용하지 않습니다.

### 현재 모니터링 동작

- 분석 윈도: `3.0`초
- 추론 stride: `1.0`초
- smoothing: EMA with `alpha = 0.3`
- 분석 시작 전 최소 버퍼: `750 ms`

### 무엇이 바뀌었는가

런타임은 이제 **롤링 버퍼**를 유지하고 가장 최근 `3.0`초를 고정 stride로 다시 분석합니다.

즉:

- 분석 사이에 의도적인 blind gap이 없음
- 각 패스가 가장 최근 오디오를 커버함
- UI는 끊어진 청크가 아니라 롤링 방식으로 갱신됨

### 왜 중요한가

기존의 고정 윈도와 cadence 불일치는 분석되지 않는 오디오 구간을 만들었습니다.

현재 설계는 다음을 분리함으로써 이를 해결합니다.

- **연속 캡처**
- **간격 기반 추론**

이 방식은 오디오 캡처 경로를 막지 않으면서도 커버리지를 연속적으로 유지합니다.

### 출력 안정화

출력 깜빡임을 줄이기 위해 런타임은 다음에 EMA smoothing을 적용합니다.

- issue confidence
- detected source confidence
- source EQ recommendation
- raw ML-derived diagnosis score

### 예외 상황 처리

모니터링 런타임은 다음을 명시적으로 처리합니다.

- 워밍업 중 버퍼 부족
- silence short-circuit
- 마이크 권한 실패
- 마이크 중단
- ML 비활성화 모드
- ML 추론 실패 시 규칙 기반 fallback

<a id="evaluation-system"></a>
## 7. 평가 시스템

LoLvlance에는 이제 기본적인 golden set 평가와 CI 게이트가 포함됩니다.

### Golden 데이터셋

golden 샘플은 다음 경로에 있습니다.

```text
eval/goldens/
```

각 샘플 디렉터리는 다음을 포함합니다.

- 오디오 파일 1개
- `metadata.json` 1개

예시 메타데이터 필드:

- `file`
- `expected_source`
- `expected_issue`
- `severity`

### 평가 스크립트

메인 evaluator:

```text
ml/eval/evaluate.py
```

수행 내용:

- golden 샘플 로드
- 모델 추론 실행
- issue/source precision, recall, F1 계산
- confusion summary 출력
- 저장된 baseline과 비교

### Baseline 추적

baseline 파일:

```text
ml/eval/baseline.json
```

저장 내용:

- 이전 overall 성능
- per-label F1 값
- critical label
- regression tolerance

### CI 게이트

워크플로:

```text
.github/workflows/eval.yml
```

현재 게이트:

- `ml/**`, `eval/**`, 워크플로 자체 변경 시 실행
- `ml/eval/evaluate.py` 사용
- `--min-f1 0.65` 강제
- overall F1이 임계값 아래로 떨어지거나 baseline 대비 회귀가 허용 범위를 넘으면 실패

### 중요한 한계

현재 golden set은 아직 작습니다. 회귀 감지에는 유용하지만, 아직 프로덕션 벤치마크로 보기에는 부족합니다.

<a id="test-infrastructure"></a>
## 8. 테스트 인프라

저장소에는 입력 정합성과 ML export 동작을 위한 전용 테스트가 추가되었습니다.

### 입력 정합성 테스트

- `ml/tests/test_waveform_parity.py`
  Python 전처리와 browser-equivalent 전처리의 waveform-level parity를 검증합니다.
- `ml/tests/test_feature_parity.py`
  end-to-end 모델 입력 feature parity를 검증합니다.
- `ml/tests/test_resampler.py`
  `48 kHz`, `44.1 kHz` 등 일반적인 브라우저 캡처 샘플레이트에 대한 리샘플링 동작을 검증합니다.
- `ml/tests/test_manifest_leakage.py`
  split leakage, 중복 파일, track-group overlap, 약한 naming collision을 탐지합니다.

### 기존 ML 파이프라인 테스트

- `ml/tests/test_export_to_onnx.py`
- `ml/tests/test_training_pipeline.py`
- `ml/tests/test_legacy_onnx_adapter.py`

### 이 테스트가 의미하는 것

이 테스트들은 조용히 발생하는 신뢰성 문제를 잡기 위한 것입니다.

- preprocessing drift
- resampling drift
- export contract breakage
- dataset split contamination

미래의 프로덕션 모델 릴리스 기준에 포함되어야 합니다.

<a id="product-positioning"></a>
## 9. 제품 포지셔닝

LoLvlance는 현재 다음과 같이 설명되어야 합니다.

- continuous monitoring
- live session analysis
- updated every few seconds
- 비전문 운영자를 위한 assistive guidance

다음과 같이 설명하면 안 됩니다.

- 실시간 진단
- 즉시 반응형 오디오 인텔리전스
- 프로덕션급 AI 사운드 엔지니어

권장 제품 해석:

> LoLvlance는 세션을 계속 모니터링하고, 롤링 방식으로 가이드를 갱신하며, 사용자가 다음에 무엇을 확인해야 할지 제안합니다.

<a id="development-workflow"></a>
## 10. 개발 워크플로

### 프론트엔드

```bash
npm install
npm run dev
```

빌드 확인:

```bash
npm run build
```

### Python 환경

PowerShell 예시:

```powershell
python -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install --upgrade pip
.\.venv-ml\Scripts\python.exe -m pip install -r ml\requirements-test.txt
```

### Golden 평가 실행

```powershell
.\.venv-ml\Scripts\python.exe ml\eval\evaluate.py `
  --goldens-dir eval\goldens `
  --model-path ml\checkpoints\lightweight_audio_model.onnx `
  --thresholds-path ml\checkpoints\label_thresholds.json `
  --baseline-path ml\eval\baseline.json `
  --min-f1 0.65
```

### ML 테스트 실행

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv-ml\Scripts\python.exe -m unittest discover -s ml/tests -p 'test_*.py' -v
```

### 선택 사항: Stem Service

```powershell
.\.venv-ml\Scripts\python.exe -m pip install -r ml\requirements-stem-service.txt
.\.venv-ml\Scripts\python.exe ml\stem_separation_service.py
```

### 공개 데이터셋 다운로드

```bash
# 사용 가능한 데이터셋 목록 확인
python ml/download_datasets.py --list

# MUSAN 다운로드 (~11 GB, 가장 간단한 시작점)
python ml/download_datasets.py --datasets musan --output-root data/datasets

# 지원하는 데이터셋 전부 다운로드
python ml/download_datasets.py --datasets musan fsd50k openmic --output-root data/datasets
```

다운로드가 완료되면 바로 실행할 수 있는 `ml/train.py` 명령어가 출력됩니다.

### 실제 오디오로 학습

```bash
python -m ml.train \
  --audio-root data/datasets/musan \
  --musan-root data/datasets/musan \
  --rebuild-manifest --epochs 20 --export-onnx
```

### 사용자 피드백 수집 및 재학습

```bash
# 1. 사용자가 브라우저에서 "내보내기" 클릭 → lolvlance_feedback_*.jsonl 저장

# 2. 학습 manifest로 변환
python ml/ingest_feedback.py \
    --feedback lolvlance_feedback_2024-01-15.jsonl \
    --output ml/artifacts/feedback_manifest.jsonl

# 3. 기존 manifest와 합쳐서 재학습
cat ml/artifacts/public_dataset_manifest.jsonl \
    ml/artifacts/feedback_manifest.jsonl \
    > ml/artifacts/merged_manifest.jsonl

python -m ml.train \
    --manifest-path ml/artifacts/merged_manifest.jsonl \
    --epochs 10 --export-onnx
```

### 모델 승격 절차

1. `ml/checkpoints/` 아래에 후보 체크포인트 학습 또는 export
2. golden 평가와 관련 테스트 실행
3. 선택한 ONNX 아티팩트를 `public/models/`로 복사
4. 필요 시 `MODEL_VERSION` / 런타임 설정 갱신
5. 브라우저 동작을 수동 검증한 뒤에만 사용자 노출 모델로 간주

<a id="limitations"></a>
## 11. 한계

### 모델 한계

- 활성 브라우저 모델은 `v0.0-pipeline-check`
- synthetic 데이터 기반이며 프로덕션 준비 상태가 아님
- source prediction은 데이터 편향에 특히 취약함
- 브라우저 런타임은 아직 full multi-band browser schema가 아니라 호환성 EQ 출력 계약을 사용함

### 데이터 한계

- 브라우저에 `FeedbackWidget`을 통한 per-analysis 사용자 피드백 수집 기능이 추가됨 (맞음/틀린 레이블 선택, JSONL 내보내기) — 수집 메커니즘은 갖춰졌으나 아직 자동화된 닫힌 루프는 아님
- 현재 golden 데이터셋은 완전한 벤치마크로 쓰기엔 너무 작음
- synthetic fallback 데이터가 저장소에 남아 있어 품질 신호로 오해될 수 있음
- real 데이터 학습을 위해선 공개 데이터셋을 별도로 다운로드해야 함

### 추론 한계

- 브라우저 추론 품질은 로컬 전처리 일관성에 의존함
- optional source enrichment는 별도 로컬 서비스 또는 브라우저 fallback에 의존함
- 아직 서버 추론 경로가 없음
- silence 및 저레벨 입력에서는 의도적으로 파이프라인 일부가 short-circuit 됨

### 제품 한계

- LoLvlance는 사용자가 무엇을 먼저 시도할지 돕는 도구이며, 진단 정확성을 보장하지 않음
- 현재 시스템을 완전히 신뢰 가능한 AI 믹스 어시스턴트로 마케팅하면 안 됨

<a id="roadmap"></a>
## 12. 로드맵

### 즉시

- 실제 모니터링 동작에 맞게 제품 문구 유지
- real-data 체크포인트가 나올 때까지 모델 격리 유지
- 현재 starter set을 넘어 golden 데이터셋 확장
- parity 및 leakage 테스트를 계속 green 상태로 유지

### 단기

- `ml/download_datasets.py` + `ml/train.py --audio-root`로 real-source 오디오 학습 시작
- 피드백 루프 주기적 실행: 브라우저 피드백 내보내기 → `ml/ingest_feedback.py` → 학습 manifest 병합
- learned EQ 출력을 내부 학습 경로에서 더 실제 런타임 계약 쪽으로 끌어오기
- 입력 정합성과 브라우저 회귀 검사를 CI에서 더 확장
- 실제 평가 오디오 기준 threshold calibration 개선

### 장기

- `v0.0-pipeline-check`에서 진짜 프로덕션 모델로 전환
- 브라우저 전용 추론이 충분한지 재평가
- 더 큰 모델이 필요하면 서버 추론 경로 추가
- 피드백 수집 루프에 human review 단계 추가

<a id="handover-notes"></a>
## 13. 인수인계 메모

프로젝트를 이어받는다면 여기부터 보는 것이 좋습니다.

- `HANDOVER.ko.md`
  운영과 아키텍처 관점의 기술 인수인계 문서
- `src/app/hooks/useMonitoring.ts`
  브라우저 런타임 동작의 핵심 파일
- `src/app/config/modelRuntime.ts`
  모델 격리 및 버전 라우팅
- `ml/eval/evaluate.py`
  회귀 차단 게이트 로직
- `ml/tests/`
  전처리 및 데이터 정합성 검사용 테스트

가장 중요하게 유지해야 할 구분은 다음입니다.

- **파이프라인이 동작한다**
- **모델이 신뢰할 수 있다**

현재는 첫 번째만 참이고, 두 번째는 아직 아닙니다.

## 관련 문서

- `HANDOVER.ko.md`
  개발자용 기술 handover와 운영 가이드
- `ML_README.ko.md`
  ML 구조, 학습, 평가, export 관련 문서
- `HANDOVER.md`
  영문 개발자 handover
- `ML_README.md`
  영문 ML 문서
