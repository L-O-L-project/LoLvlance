# LoLvlance

LoLvlance는 한 가지 제품 목표에 집중하는 브라우저 기반 오디오 모니터링 시스템입니다.

**라이브 세션을 위한 연속 모니터링형 사운드 문제 탐지 + EQ 가이드**

현재 구현은 브라우저에서 마이크 오디오를 수집하고, 로컬에서 오디오 특징을 추출한 뒤, 브라우저 내 ONNX 모델을 실행하고, 그 결과를 결정론적 규칙과 결합하여 문제 라벨, 소스 맥락, EQ 제안을 생성합니다.

현재 브라우저 흐름에서는 약 `3`초 길이의 롤링 윈도를 분석하고, 약 `4`초 간격으로 가이드를 갱신합니다. 따라서 즉각 반응형 도구라기보다, 예배팀과 소규모 라이브 운영을 위한 연속 모니터링 도구로 이해하는 것이 맞습니다.

## 바로가기

[![프로젝트 개요](https://img.shields.io/badge/%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%20%EA%B0%9C%EC%9A%94-2563EB?style=for-the-badge&logo=bookstack&logoColor=white)](#project-overview)
[![시스템 아키텍처](https://img.shields.io/badge/%EC%8B%9C%EC%8A%A4%ED%85%9C%20%EC%95%84%ED%82%A4%ED%85%8D%EC%B2%98-7C3AED?style=for-the-badge&logo=appveyor&logoColor=white)](#system-architecture)
[![현재 ML 상태](https://img.shields.io/badge/%ED%98%84%EC%9E%AC%20ML%20%EC%83%81%ED%83%9C-DC2626?style=for-the-badge&logo=pytorch&logoColor=white)](#current-ml-status)
[![모델 출력 스키마](https://img.shields.io/badge/%EB%AA%A8%EB%8D%B8%20%EC%B6%9C%EB%A0%A5%20%EC%8A%A4%ED%82%A4%EB%A7%88-0891B2?style=for-the-badge&logo=onnx&logoColor=white)](#model-output-schema)

[![폴더 구조](https://img.shields.io/badge/%ED%8F%B4%EB%8D%94%20%EA%B5%AC%EC%A1%B0-0F766E?style=for-the-badge&logo=files&logoColor=white)](#folder-structure)
[![실행 방법](https://img.shields.io/badge/%EC%8B%A4%ED%96%89%20%EB%B0%A9%EB%B2%95-16A34A?style=for-the-badge&logo=vite&logoColor=white)](#how-to-run)
[![알려진 한계](https://img.shields.io/badge/%EC%95%8C%EB%A0%A4%EC%A7%84%20%ED%95%9C%EA%B3%84-D97706?style=for-the-badge&logo=target&logoColor=white)](#known-limitations)
[![다음 단계](https://img.shields.io/badge/%EB%8B%A4%EC%9D%8C%20%EB%8B%A8%EA%B3%84-1D4ED8?style=for-the-badge&logo=roadmapdotsh&logoColor=white)](#next-steps)

[![English README](https://img.shields.io/badge/English-README-111827?style=for-the-badge&logo=readme&logoColor=white)](README.md)
[![개발자 인수인계](https://img.shields.io/badge/%EA%B0%9C%EB%B0%9C%EC%9E%90-Handover-111827?style=for-the-badge&logo=gitbook&logoColor=white)](HANDOVER.ko.md)
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

LoLvlance는 라이브 세션 중 다음 세 가지 질문을 점검하도록 설계되어 있습니다.

1. 지금 신호에서 무엇이 잘못 들리는가?
2. 어떤 소스가 가장 관련 있을 가능성이 높은가?
3. 사용자가 가장 먼저 시도해야 할 EQ 조정은 무엇인가?

현재 이 책임은 ML과 규칙 기반 로직으로 나뉘어 있습니다.

- **ML**
  issue 확률과 source 확률을 예측합니다.
- **규칙**
  source-specific diagnosis를 파생하고, 예측 결과를 EQ 제안으로 변환하며, ML이 사용할 수 없을 때 fallback 경로를 제공합니다.

현재 사용자 경험은 권위적인 판정보다 보조적 가이드에 가깝습니다. LoLvlance는 정답을 선언하기보다, 사용자가 다음에 무엇을 점검해야 할지 빠르게 좁혀 주는 역할을 합니다.

이 시스템은 브라우저 우선 구조입니다. 클라우드 추론 경로는 없고, 아직 프로덕션용 데이터 수집 파이프라인도 없습니다.

<a id="system-architecture"></a>
## 2. 시스템 아키텍처

### 전체 흐름

```text
Mic Input
  -> Browser Rolling Buffer
  -> Feature Extraction
  -> ONNX Inference
  -> Diagnosis Post-Processing
  -> Rule / EQ Layer
  -> UI
```

### 런타임 구성 요소별 역할

- `src/app/hooks/useMonitoring.ts`
  마이크 캡처, 롤링 버퍼, ML 추론 호출, stem service 호출, fallback source tagging, 최종 UI 결과 조립을 총괄합니다.
- `src/app/audio/featureExtraction.ts`
  버퍼링된 오디오를 log-mel spectrogram 특징과 RMS 및 관련 메타데이터로 변환합니다.
- `src/app/audio/mlInference.ts`
  `public/models/lightweight_audio_model.onnx`를 `onnxruntime-web`으로 로드하고, 새 스키마를 직접 읽어 프론트엔드 ML 출력 형태로 변환합니다.
- `src/app/audio/diagnosisPostProcessing.ts`
  issue와 source 확률로부터 `vocal_buried` 같은 파생 diagnosis를 생성합니다.
- `src/app/audio/sourceAwareEq.ts`
  감지된 source와 선택적 stem metric을 이용해 source-aware EQ 추천을 생성합니다.
- `src/app/audio/ruleBasedAnalysis.ts`
  브라우저 ML이 실패했을 때 기본 issue 탐지를 수행하는 fallback rule engine입니다.

### ML 역할과 Rule 역할

- **ML 역할**
  log-mel spectrogram 입력으로부터 `issue_probs`와 `source_probs`를 예측합니다.
- **Rule 역할**
  silence gating, 상위 diagnosis 파생, EQ 가이드 생성, source detection 병합, ML 장애 시 graceful fallback을 담당합니다.

### 현재 모니터링 주기

- 브라우저 분석 윈도: 약 `3.0`초
- 모니터링 업데이트 주기: 약 `4`초
- 제품 해석: 즉시 반응형 진단이 아니라 연속 모니터링과 라이브 세션 분석

<a id="current-ml-status"></a>
## 3. 현재 ML 상태

이 섹션은 현재 상태를 평가할 때 매우 중요하므로 의도적으로 명확하게 적습니다.

- 현재 체크포인트는 **합성 데이터 기반 synthetic fallback dataset**으로 학습되었습니다.
- synthetic dataset은 `ml/generate_synthetic_public_datasets.py`로 생성되었습니다.
- 현재 manifest인 `ml/artifacts/public_dataset_manifest.jsonl`에는 총 **44개 clip**이 있습니다.
  - `22` train
  - `22` validation
- 현재 체크포인트의 목적은 **파이프라인 검증**이지, 프로덕션 수준 분류 성능을 의미하지 않습니다.
- 브라우저 추론은 end-to-end로 동작하지만, 실제 음성/음악에 대한 의미적 정확도는 아직 약합니다.
- 특히 source prediction은 synthetic dataset 편향의 영향을 크게 받습니다.

한 줄로 요약하면, 인프라는 실제로 동작하지만 현재 체크포인트는 아직 프로덕션급 모델이 아닙니다.

<a id="model-output-schema"></a>
## 4. 모델 출력 스키마

프론트엔드는 이제 ONNX에서 아래 스키마를 직접 기대합니다.

| 출력 | Shape | 의미 |
| --- | --- | --- |
| `issue_probs` | `(1, 9)` | multi-label issue 확률 |
| `source_probs` | `(1, 5)` | multi-label source 확률 |
| `eq_freq` | `(1, 1)` | 정규화된 EQ 목표 주파수 |
| `eq_gain_db` | `(1, 1)` | 제안 EQ gain(dB) |

### 라벨 순서

Issue index 순서:

```text
[muddy, harsh, buried, boomy, thin, boxy, nasal, sibilant, dull]
```

Source index 순서:

```text
[vocal, guitar, bass, drums, keys]
```

### Raw ONNX 출력 예시

```json
{
  "issue_probs": [[0.5728, 0.3190, 0.4393, 0.5886, 0.4168, 0.5056, 0.4896, 0.3736, 0.6350]],
  "source_probs": [[0.5122, 0.5328, 0.6024, 0.5892, 0.5437]],
  "eq_freq": [[0.5346]],
  "eq_gain_db": [[-0.9317]]
}
```

### EQ에 대한 중요한 설명

`eq_freq`와 `eq_gain_db`는 ONNX 계약의 일부이지만, **전용 EQ head가 학습한 결과가 아닙니다**.

현재 이 값들은 `ml/onnx_schema_adapter.py`에 있는 결정론적 projection layer가 아래 정보를 사용해 생성합니다.

- issue probabilities
- source probabilities
- 규칙으로 정의된 issue/source EQ 매핑

즉, 브라우저 계약은 4출력을 유지하지만, 실제 trainable model은 아직 2-head 구조입니다.

<a id="folder-structure"></a>
## 5. 폴더 구조

### 핵심 디렉터리

- `ml/`
  Python 학습, export, 전처리, 평가, synthetic data 생성, 선택적 로컬 sidecar service를 포함합니다.
- `src/app/audio/`
  브라우저 측 feature extraction, ONNX inference, rule analysis, diagnosis post-processing, source-aware EQ, source tagging 로직을 포함합니다.
- `src/app/hooks/`
  런타임 orchestration 레이어입니다. `useMonitoring.ts`가 마이크 캡처와 추론의 진입점입니다.
- `public/models/`
  브라우저가 직접 서빙하는 모델 자산입니다. `public/models/lightweight_audio_model.onnx`가 현재 활성 모델입니다.
- `ml/checkpoints/`
  학습 산출물과 export 결과를 저장합니다. 현재 `model.pt`, `config.json`, `thresholds.json`, `training_history.json`, `lightweight_audio_model.onnx` 등이 있습니다.
- `ml/artifacts/`
  파생 manifest와 생성 데이터셋을 저장합니다. 현재 synthetic manifest는 `ml/artifacts/public_dataset_manifest.jsonl`에 있습니다.

### 현재 아티팩트 관련 메모

- `ml/checkpoints/lightweight_audio_model.onnx`
  Python 학습 파이프라인이 export한 최신 ONNX 아티팩트입니다.
- `public/models/lightweight_audio_model.onnx`
  프론트엔드가 실제로 사용하는 브라우저 런타임 모델입니다.
- `public/models/lightweight_audio_model.onnx.data`
  이전 external-data export에서 남아 있는 파일입니다. 현재 런타임 모델은 standalone `.onnx` 파일입니다.

<a id="how-to-run"></a>
## 6. 실행 방법

### 프론트엔드 실행

권장 전제 조건:

- Node.js 18+
- npm

설치 및 실행:

```bash
npm install
npm run dev
```

브라우저에서 접속:

```text
http://localhost:3000
```

그 다음:

1. 마이크 권한을 허용합니다.
2. UI에서 모니터링을 시작합니다.
3. 말하거나, 침묵 상태를 유지하거나, 음악을 재생합니다.
4. UI가 몇 초마다 문제, 감지된 source, EQ 제안을 어떻게 갱신하는지 확인합니다.

### 브라우저 콘솔 로그

- `[audio-ml]`
  모델 준비 상태와 파싱된 ML 출력
- `[audio-ml:raw]`
  통합 검증용 임시 raw tensor 로그
- `[audio-rules]`
  rule-engine 분석 결과
- `[audio-stems]`
  local stem service 동작 상태
- `[audio-tags]`
  fallback source tagging 동작 상태

### 선택 사항: Local Stem Service 실행

앱은 stem service 없이도 동작하지만, source enrichment 품질은 stem service가 있을 때 더 좋습니다.

PowerShell 예시:

```powershell
python -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install --upgrade pip
.\.venv-ml\Scripts\python.exe -m pip install -r ml\requirements-stem-service.txt
.\.venv-ml\Scripts\python.exe ml\stem_separation_service.py
```

### 선택 사항: ML 테스트 실행

PowerShell 예시:

```powershell
python -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install --upgrade pip
.\.venv-ml\Scripts\python.exe -m pip install -r ml\requirements-test.txt
$env:PYTHONPATH = (Get-Location).Path
.\.venv-ml\Scripts\python.exe -m unittest discover -s ml/tests -p 'test_*.py' -v
```

### 프로덕션 빌드 확인

```bash
npm run build
```

<a id="known-limitations"></a>
## 7. 알려진 한계

- 현재 모델은 synthetic data로만 학습되었습니다.
- 현재 브라우저 UX는 약 `3`초 분석 윈도를 사용하고, 약 `4`초 간격으로 가이드를 갱신합니다.
- 실제 환경 기준 validation set이나 benchmark report가 아직 없습니다.
- 데이터 수집이나 사용자 피드백 파이프라인이 없습니다.
- EQ는 supervised target으로 학습된 것이 아니라 결정론적이고 rule-based입니다.
- 현재 trainable model은 두 개의 learned head만 가집니다.
  - issue head
  - source head
- 실제 오디오에 대한 source prediction은 아직 프로덕션 용도로 신뢰할 수준이 아닙니다.
- silence gating이 inference 전에 적용됩니다. 매우 작은 입력은 전체 ML 결과 대신 빈 결과를 반환합니다.
- optional stem service는 별도 의존성입니다. 이것이 실행되지 않으면 브라우저 ML은 정상이어도 UI에 stem-service fallback이 표시될 수 있습니다.
- 현재 제품은 즉시 반응형 진단 도구가 아니라, 연속 모니터링과 라이브 세션 분석 도구로 설명하는 것이 맞습니다.

<a id="next-steps"></a>
## 8. 다음 단계

- synthetic fallback data 대신 실제 public dataset으로 학습하기
- 데이터 수집 및 리뷰 파이프라인 추가
- source label 품질과 커버리지 개선
- held-out real-world evaluation set 및 보고서 구축
- threshold를 synthetic validation이 아니라 실제 데이터로 재보정
- 더 빠른 런타임이 준비되기 전까지는 현재의 몇 초 단위 모니터링 주기에 맞춰 제품 문구를 유지하기
- EQ를 계속 결정론적으로 유지할지, 나중에 learned head로 옮길지 결정
- 브라우저 통합이 안정화되면 임시 raw tensor 로그 제거

## 관련 문서

- `HANDOVER.md`
  개발자용 시스템 handover와 운영 가이드
- `ML_README.md`
  ML 아키텍처, 학습, 평가에 대한 상세 문서
- `HANDOVER.ko.md`
  한국어 개발자 handover 문서
- `ML_README.ko.md`
  한국어 ML 상세 문서
