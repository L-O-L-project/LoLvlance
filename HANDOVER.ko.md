# LoLvlance Handover

이 문서는 현재 LoLvlance 코드베이스를 위한 기술 인수인계 문서입니다.

다음 엔지니어가 무엇이 실제로 구현되어 있는지, 무엇이 안전을 위해 격리되어 있는지, 그리고 무엇을 아직 프로덕션 준비 상태로 보면 안 되는지 빠르게 이해하도록 돕는 것이 목적입니다.

짧게 요약하면 다음과 같습니다.

- 브라우저 모니터링 경로는 end-to-end로 동작합니다.
- 활성 브라우저 모델은 `v0.1-real-data` — MUSAN + OpenMIC-2018 (65K 클립)으로 학습된 첫 번째 실데이터 체크포인트입니다.
- 런타임은 blind gap이 없는 rolling-window 모니터링을 사용합니다.
- golden 평가와 input-integrity 테스트 인프라가 존재하며 CI에서 게이트됩니다.
- 파이프라인은 신뢰할 수 있으며, 모델은 실제 오디오에서 기능하지만 모든 레이블에서 완벽하지는 않습니다.

영문 버전: `HANDOVER.md`

## 바로가기

[![프로젝트 개요](https://img.shields.io/badge/프로젝트%20개요-2563EB?style=for-the-badge&logo=bookstack&logoColor=white)](#project-overview)
[![시스템 아키텍처](https://img.shields.io/badge/시스템%20아키텍처-7C3AED?style=for-the-badge&logo=appveyor&logoColor=white)](#system-architecture)
[![ML 시스템](https://img.shields.io/badge/ML%20시스템-DC2626?style=for-the-badge&logo=pytorch&logoColor=white)](#ml-system)
[![모니터링](https://img.shields.io/badge/모니터링%20시스템-0891B2?style=for-the-badge&logo=webaudio&logoColor=white)](#monitoring-system)
[![평가](https://img.shields.io/badge/평가%20시스템-0F766E?style=for-the-badge&logo=githubactions&logoColor=white)](#evaluation-system)
[![테스트](https://img.shields.io/badge/테스트%20인프라-D97706?style=for-the-badge&logo=pytest&logoColor=white)](#test-infrastructure)
[![개발 워크플로](https://img.shields.io/badge/개발%20워크플로-16A34A?style=for-the-badge&logo=git&logoColor=white)](#development-workflow)
[![로드맵](https://img.shields.io/badge/로드맵-1D4ED8?style=for-the-badge&logo=roadmapdotsh&logoColor=white)](#roadmap)

<a id="project-overview"></a>
## 1. 프로젝트 개요

LoLvlance는 완전한 자율 믹스 시스템이 아니라, 라이브 세션 운영자에게 가이드를 제공하는 브라우저 우선 오디오 모니터링 제품입니다.

주요 사용 시나리오:

- 교회 사운드팀
- 예배팀 자원봉사자
- 소규모 라이브 운영
- 여러 역할을 동시에 수행하면서 빠른 제안이 필요한 운영자

핵심 제품 질문은 여전히 다음 세 가지입니다.

1. 지금 무엇이 잘못 들리는가?
2. 어느 소스가 가장 관련 있어 보이는가?
3. 사용자가 먼저 시도해야 할 EQ 조정은 무엇인가?

### 제품 포지셔닝

현재 제품 언어는 다음을 유지해야 합니다.

- continuous monitoring
- live session analysis
- updated every few seconds
- assistive guidance

다음 표현은 피해야 합니다.

- 실시간
- 즉시 반응하는 AI 엔지니어
- 프로덕션급 진단 언어

<a id="system-architecture"></a>
## 2. 시스템 아키텍처

### 런타임 흐름

```text
Mic Input
  -> Browser Capture Node
  -> Rolling Buffers
  -> Feature Extraction
  -> ONNX Inference (optional)
  -> Source Enrichment
  -> Rule Fallback / Post-Processing
  -> EMA Smoothing
  -> UI
```

### 주요 런타임 파일

- `src/app/hooks/useMonitoring.ts`
  마이크 캡처, 롤링 버퍼, 추론 스케줄링, 스무딩, 최종 결과 업데이트의 중심 파일입니다.
- `src/app/audio/audioUtils.ts`
  리샘플링, circular buffer 헬퍼, RMS/peak 계산, buffered snapshot 생성을 담당합니다.
- `src/app/audio/featureExtraction.ts`
  브라우저 추론용 feature extraction을 담당합니다.
- `src/app/audio/mlInference.ts`
  브라우저 ONNX 런타임 설정, 출력 파싱, 모델 버전 로깅, fallback 동작을 담당합니다.
- `src/app/audio/diagnosisPostProcessing.ts`
  파생 진단 로직입니다.
- `src/app/audio/sourceAwareEq.ts`
  사람이 읽을 수 있는 EQ 가이드입니다.
- `src/app/audio/ruleBasedAnalysis.ts`
  ML이 비활성화되거나 실패할 때 동작하는 fallback rule engine입니다.
- `src/app/config/modelRuntime.ts`
  모델 버전 라우팅, 실험/프로덕션 분리, kill switch의 기준 파일입니다.

### 현재 런타임 책임

- 마이크 캡처는 연속적으로 진행됩니다.
- 추론은 캡처와 분리된 스케줄로 수행됩니다.
- source enrichment는 로컬 stem 서비스 또는 브라우저 fallback tagging으로 보강될 수 있습니다.
- 최종 UI 출력은 표시 전에 스무딩됩니다.

<a id="ml-system"></a>
## 3. ML 시스템

현재는 두 개의 ML 층위를 구분해서 이해해야 합니다.

### 활성 브라우저 모델

현재 브라우저 모델은 다음과 같습니다.

```text
v0.1-real-data
```

이 모델은:

- 실제 공개 오디오(MUSAN + OpenMIC-2018, 65,738 클립)로 학습됨
- 브라우저에 배포된 첫 번째 실데이터 체크포인트
- CI golden 평가 게이트를 통과하여 검증됨
- 의미 있는 예측 생성 확인: guitar F1=0.82, drums AUROC=0.93, vocal AUROC=0.92, muddy AUROC=0.87
- 알려진 약점: `boxy` F1=0.0, `nasal`/`thin`/`sibilant` AUROC 약 0.56–0.60

전체 per-label 지표와 아티팩트 경로는 `ml/model_history.md`를 참고하세요.

### 현재 브라우저 ONNX 계약

프론트엔드는 현재 다음 값을 읽습니다.

- `issue_probs`
- `source_probs`
- `eq_freq`
- `eq_gain_db`

이것이 현재 브라우저 호환성 계약입니다.

### Python 학습 경로

Python ML 스택은 초기 프로토타입보다 앞서 있으며 다음을 지원합니다.

- `ml/model.py`의 teacher/student 변형
- teacher 경로용 frozen 또는 trainable transformer 스타일 spectrogram encoder
- 브라우저 배포를 고려한 lightweight CNN student 경로
- source-conditioned issue prediction
- `eq_params`를 출력하는 learned multi-band EQ head
- `ml/degradation.py`의 self-supervised degradation 기반 학습
- `ml/losses.py`의 distillation-ready loss

### 운영상 중요한 구분

코드베이스는 더 진보된 learning-based 모델을 지원할 수 있지만, 현재 사용자 노출 체크포인트는 아직 프로덕션 수준의 주장을 정당화하지 못합니다.

이 구분은 제품 문구, 데모, 인수인계 대화에서 항상 명시되어야 합니다.

<a id="model-versioning"></a>
## 4. 모델 버전 관리

브라우저에는 이제 모델 격리 레이어가 있습니다.

### 런타임 기본값

- 활성 기본 버전: `v0.1-real-data` (PRODUCTION)
- 보관된 실험 버전: `v0.0-pipeline-check`
- 활성 프로덕션 모델 경로: `public/models/lightweight_audio_model.production.onnx`
- 보관된 실험 모델 경로: `public/models/lightweight_audio_model.onnx`

기준 파일:

- `src/app/config/modelRuntime.ts`

### 모델 히스토리

3개의 모델 세대가 학습되었습니다. 전체 내용은 `ml/model_history.md`를 참고하세요.

| 버전 | 상태 | 데이터 | 비고 |
|---|---|---|---|
| `v0.0-pipeline-check` | 보관됨 | Synthetic | 파이프라인 검증용만 |
| `v0.1-real-data` | **활성** | MUSAN + OpenMIC (65K) | 현재 기본값 |
| `v0.2-fsd50k-extended` | 학습 중 | + FSD50K partial | ETA: 미정 |

### 런타임 제어값

- `MODEL_VERSION`
- `ENABLE_MODEL`

동작 방식:

- `ENABLE_MODEL=false`이면 ML 추론을 완전히 비활성화
- `MODEL_VERSION === "v0.0-pipeline-check"`이면 보관된 실험용 아티팩트로 라우팅
- `MODEL_VERSION === "v0.1-real-data"` (기본값)이면 프로덕션 아티팩트 경로로 라우팅
- 다른 비실험용 버전도 프로덕션 아티팩트 경로로 라우팅됨

### 사용자 노출 안전장치

실험용 안전 배너(`Experimental Mode`, `Not production-ready`)는 `MODEL_VERSION === "v0.0-pipeline-check"`일 때만 표시됩니다. 기본 `v0.1-real-data`를 사용할 때는 배너가 표시되지 않습니다.

<a id="audio-pipeline"></a>
## 5. 오디오 파이프라인

### 브라우저 전처리 계약

현재 계약:

- sample rate: `16_000`
- clip window: `3.0`초
- window size: `25 ms`
- hop size: `10 ms`
- FFT size: `512`
- mel bins: `64`

### 버퍼 구조

런타임은 두 개의 롤링 버퍼를 유지합니다.

- stem/source 경로용 native sample rate 버퍼
- ML feature extraction용 `16 kHz` 리샘플 버퍼

### 왜 중요한가

이 분리는 모든 downstream consumer가 동일한 리샘플링 경로를 강제 공유하지 않으면서도, ML 입력 계약은 유지하도록 하기 위함입니다.

### 현재 위험 지점

전처리 일관성은 여전히 실제 위험 지점입니다. 저장소에 이를 위한 명시적 테스트가 추가된 이유도 이 failure mode가 조용히 발생하기 쉽기 때문입니다.

<a id="monitoring-system"></a>
## 6. 모니터링 시스템

현재 모니터링 시스템은 고정 윈도 배치 루프가 아니라 rolling-window 파이프라인입니다.

### 현재 런타임 값

- monitoring window: `3.0`초
- monitoring stride: `1.0`초
- EMA alpha: `0.3`
- minimum warm-up buffer: `750 ms`

### 변경 내용

예전 동작:

- 고정 길이 오디오를 수집
- 분석
- 대기
- 반복

현재 동작:

- 연속적으로 캡처
- circular buffer 유지
- 가장 최근 `3.0`초를 `1.0`초마다 다시 분석

### 결과

- 분석 사이에 의도적인 blind gap이 없음
- 업데이트가 더 부드러움
- 세션 커버리지가 더 좋아짐

### 구현 메모

- 가능하면 `AudioWorklet` 사용
- 필요 시 `ScriptProcessorNode` fallback 유지
- 캡처와 추론 스케줄링을 분리
- 추론이 끝나기 전에 다음 stride가 오면 pending pass를 큐잉

### 안정화 레이어

`useMonitoring.ts`는 다음에 EMA smoothing을 적용합니다.

- problem confidence
- detected source confidence
- source-specific EQ recommendation
- raw ML-derived diagnosis score

### 이미 처리되는 예외 상황

- 초기 버퍼 부족
- silence short-circuit
- 마이크 권한 오류
- 마이크 중단
- 모델 비활성화 모드
- ML 추론 실패 시 rule fallback

<a id="evaluation-system"></a>
## 7. 평가 시스템

LoLvlance에는 이제 CI 게이트가 포함된 실제 평가 경로가 있습니다.

### Golden Set

위치:

```text
eval/goldens/
```

각 샘플은 다음을 포함합니다.

- `.wav` 파일
- `metadata.json`

메타데이터는 다음을 포함합니다.

- `file`
- `expected_source`
- `expected_issue`
- `severity`

### Evaluator

메인 스크립트:

```text
ml/eval/evaluate.py
```

수행 내용:

- golden 샘플 로드
- ONNX 모델 실행
- issue/source precision, recall, F1 계산
- confusion detail 출력
- threshold 및 baseline 기준 검사

### Baseline 추적

baseline 파일:

```text
ml/eval/baseline.json
```

현재 baseline은 다음을 저장합니다.

- overall F1
- per-label F1
- critical label
- regression tolerance

### CI 게이트

워크플로:

```text
.github/workflows/eval.yml
```

현재 게이트 동작:

- `ml/**`, `eval/**`, 워크플로 변경 시 트리거
- Python 3.11 사용
- `ml/requirements-test.txt` 설치
- golden evaluator 실행
- `--min-f1 0.65` 강제

### 중요한 한계

현재 golden set은 아직 작습니다. 회귀 감지에는 유용하지만, 완전한 real-world 벤치마크는 아닙니다.

<a id="test-infrastructure"></a>
## 8. 테스트 인프라

### 입력 정합성 테스트

- `ml/tests/test_waveform_parity.py`
  Python 전처리와 browser-equivalent 전처리 간 waveform-level parity 검증
- `ml/tests/test_feature_parity.py`
  모델 입력 feature에 대한 end-to-end parity 검증
- `ml/tests/test_resampler.py`
  일반적인 브라우저 캡처 샘플레이트에 대한 리샘플링 동작 및 출력 안전성 검증
- `ml/tests/test_manifest_leakage.py`
  중복 파일, group overlap, naming collision 기준 train/validation leakage 검증

### 기존 ML 파이프라인 테스트

- `ml/tests/test_export_to_onnx.py`
- `ml/tests/test_training_pipeline.py`
- `ml/tests/test_legacy_onnx_adapter.py`

### 해석

이 테스트들은 모델 품질을 조용히 손상시킬 수 있는 failure mode를 잡기 위해 존재합니다.

- preprocessing drift
- resampler drift
- dataset leakage
- ONNX export/schema breakage

테스트가 존재한다고 해서 문제가 영구적으로 해결된 것은 아닙니다. 이 테스트의 목적은 회귀를 계속 보이게 만드는 것입니다.

<a id="limitations"></a>
## 9. 한계

### 모델 한계

- 활성 브라우저 모델은 `v0.1-real-data` — 실데이터 학습이지만 모든 레이블에서 완벽하지 않음
- `boxy` F1=0.0 (val 샘플 23개뿐); `nasal`, `thin`, `sibilant` AUROC 거의 chance 수준(0.56–0.60)
- guitar 임계값이 낮음(0.25) — 데이터 불균형으로 비기타 오디오에서 과예측 가능
- CI golden set이 3개 샘플뿐 — 임계값이 이 작은 세트에 맞게 보수적으로 설정됨
- 브라우저 런타임은 여전히 single-band 호환성 EQ 출력 계약을 사용함

### 데이터 한계

- 브라우저 측 사용자 피드백 수집 기능이 추가됨 (`FeedbackWidget` + `feedbackStore.ts` + `ml/ingest_feedback.py`) — 단, 수동 내보내기 및 재학습이 필요하며 아직 자동화되지 않음
- human-reviewed production dataset이 아직 없음
- real 데이터 학습을 위해선 `ml/download_datasets.py`로 공개 데이터셋을 별도 다운로드해야 함
- golden 평가 세트가 아직 너무 작음

### 추론 한계

- 브라우저 전용 추론은 기기 의존적임
- 전처리 일관성은 릴리스 리스크로 남아 있음
- 로컬 sidecar 서비스가 source enrichment 품질에 영향을 줌
- silence handling은 의도적으로 일부 출력 경로를 억제함

### 제품 한계

- 시스템을 권위적인 진단 엔진으로 판매하면 안 됨
- 현재 가이드 품질은 강한 AI 성능 주장을 정당화할 수준이 아님

<a id="product-positioning"></a>
## 10. 제품 포지셔닝

현재 제품 언어는 실제 런타임과 계속 맞아야 합니다.

- continuous monitoring
- live session analysis
- updated every few seconds
- guidance over authority

데모 설명 권장 문장:

> LoLvlance는 계속 오디오를 듣고, 최근 몇 초를 롤링 방식으로 검토하며, 사용자가 다음에 무엇을 확인해야 할지 제안합니다.

다음과 같이 포지셔닝하면 안 됩니다.

- 실시간 AI 진단
- 즉시 반응하는 자동 EQ
- 프로덕션급 오디오 인텔리전스

<a id="development-workflow"></a>
## 11. 개발 워크플로

### 프론트엔드

```bash
npm install
npm run dev
npm run build
```

### Python 환경

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

### 공개 데이터셋 다운로드 및 실데이터 학습

```bash
# 공개 데이터 다운로드 (MUSAN이 가장 간단한 시작점)
python ml/download_datasets.py --datasets musan --output-root data/datasets

# 실제 오디오로 학습 (--audio-root는 WAV/FLAC 파일이 있는 임의 폴더 허용)
python -m ml.train \
  --audio-root data/datasets/musan \
  --rebuild-manifest --epochs 20 --export-onnx

# 학습 후 브라우저 모델 업데이트
cp ml/checkpoints/lightweight_audio_model.onnx public/models/
```

### 사용자 피드백을 학습에 반영

```bash
# 1. 사용자가 브라우저에서 피드백 내보내기 → lolvlance_feedback_*.jsonl
# 2. manifest로 변환
python ml/ingest_feedback.py \
    --feedback lolvlance_feedback_2024-01-15.jsonl \
    --output ml/artifacts/feedback_manifest.jsonl
# 3. 기존 manifest와 병합 후 재학습
cat ml/artifacts/public_dataset_manifest.jsonl \
    ml/artifacts/feedback_manifest.jsonl \
    > ml/artifacts/merged_manifest.jsonl
python -m ml.train \
    --manifest-path ml/artifacts/merged_manifest.jsonl \
    --epochs 10 --export-onnx
```

### 모델 승격 체크리스트

1. `ml/checkpoints/` 아래에 후보 체크포인트를 학습 또는 export
2. 평가와 테스트 실행
3. 선택한 ONNX 파일을 `public/models/`로 복사
4. `MODEL_VERSION` 라우팅 확인
5. 브라우저 동작을 수동 검증
6. 그 이후에만 사용자 노출 아티팩트로 간주

<a id="roadmap"></a>
## 12. 로드맵

### 즉시

- `v0.2-fsd50k-extended` 학습(PID 58050, 로그: `ml/train_fsd50k.log`) 완료 후 평가
- golden 평가 세트를 3개 샘플 이상으로 확장
- input-integrity 테스트를 릴리스 프로세스 일부로 유지

### 단기

- `v0.2-fsd50k-extended`를 baseline 대비 평가하고, `boxy`/`nasal`/`thin`이 개선되면 승격
- 피드백 수집 사이클 실행: 브라우저 내보내기 → `ml/ingest_feedback.py` → manifest 병합 → 재학습
- 더 큰 golden 평가 세트로 임계값 calibration 개선
- learned EQ 출력을 런타임 계약 쪽으로 더 승격

### 장기

- single-band 브라우저 EQ 계약에서 multi-band EQ 출력으로 전환
- 브라우저 전용 제약이 문제면 서버 추론 검토
- 피드백 수집 사이클에 human review 단계 추가

<a id="handover-notes"></a>
## 13. 인수인계 메모

### 어디서부터 볼 것인가

- `README.ko.md`
  제품 관점의 개요와 현재 런타임 요약
- `src/app/hooks/useMonitoring.ts`
  가장 중요한 런타임 파일
- `src/app/config/modelRuntime.ts`
  모델 격리와 라우팅
- `ml/eval/evaluate.py`
  회귀 차단 로직
- `ml/tests/`
  input-integrity 및 export/training 회귀 테스트

### 다음 엔지니어를 위한 실무 규칙

- 새 모델 체크포인트를 승격할 때는 반드시 `ml/eval/evaluate.py`를 실행하고 CI 통과를 확인할 것
- 실험 배너 코드를 제거하지 말 것 — `MODEL_VERSION`이 `v0.0-pipeline-check`로 설정되면 자동으로 활성화됨
- 테스트를 돌리기 전 전처리 parity를 가정하지 말 것
- public ONNX 아티팩트를 평가 재실행 및 `ml/model_history.md` 업데이트 없이 교체하지 말 것

### 가장 중요한 사고방식

다음 두 문장은 서로 다릅니다.

- **파이프라인이 동작한다**
- **모델이 신뢰할 수 있다**

2026년 4월 15일 기준으로 `v0.1-real-data`에 대해 두 문장 모두 참입니다. 이 모델은 실제 공개 오디오로 학습되어 CI 평가를 통과했으며 현재 기본값입니다. 모든 레이블에서 완벽하지는 않지만 — `boxy`, `nasal`, `thin`은 추가 학습 데이터가 필요합니다 — 더 이상 synthetic 데이터 전용이 아닙니다.
