# LoLvlance — AI 라이브 사운드 보조 앱

**버전**: Mobile v1.2 | **작성**: 2025

---

## 미션

> **공연 중 사운드 문제를, 한 손으로 한 번에 파악하고 즉시 해결한다.**
> 

마이크로 2~4초 듣고 → 무엇이 문제인지 → 어떻게 고쳐야 하는지를 바로 화면에 보여준다.
음원 분리 없이, 패턴 분석만으로 추론한다.

---

## 핵심 문제

공연 중 사운드가 이상하다는 건 느끼지만, 아무도 빠르게 원인을 특정하지 못한다.

| 문제 | 상황 |
| --- | --- |
| 어떤 악기가 원인인지 모름 | 전체 믹스만 들리기 때문에 |
| 볼륨 문제인지 EQ 문제인지 모름 | 판단 기준이 없기 때문에 |
| 판단할 시간이 없음 | 공연은 멈추지 않기 때문에 |
| 전문가가 없음 | 소규모 공연은 엔지니어 없이 운영 |

---

## 제품 컨셉

> **"한 손으로 한 번에."**
> 

버튼 하나 → 2~4초 분석 → 화면에 원인과 조치 표시.
텍스트 최소화, 시각적으로 즉시 이해 가능해야 한다.

### 출력 예시

```
[문제] vocal buried  (0.83)
[원인] guitar (0.71) / keys (0.48)
[조치] 250~350Hz -3dB / 3kHz +2dB
[확신도] 중간
```

---

## 기능별 실현 가능성

설계 전에 각 기능의 현실적인 난이도를 먼저 정의한다.

| 기능 | 가능성 | 판단 근거 |
| --- | --- | --- |
| 마이크 입력 및 스펙트럼 변환 | 높음 | 웹/모바일 모두 표준 기술 |
| 문제 유형 분류 | 중간~높음 | 라벨만 잘 만들면 가장 먼저 동작하는 축 |
| 영향 소스 추정 | 중간 | 혼합음 환경이라 불확실성 큼 |
| EQ 조치 추천 | 중간~낮음 | 가장 어려움. 초기에는 룰 기반이 현실적 |
| 보컬 음역대 분류 | 중간 | male/female은 가능, leader는 음향만으로 불안정 |

---

## 기술 접근

음원 분리(Demucs 등)는 쓰지 않는다.
모바일에서 실시간으로 돌리기에 latency, 배터리, 연산량 모두 부적합하다.

**패턴 기반 추론 + 룰 기반 액션 매퍼**로 설계한다.

### MVP 아키텍처

```
마이크 입력
→ 3초 rolling buffer (0.25초마다 재추론)
→ log-Mel Spectrogram + RMS + Spectral Centroid + Bandwidth
→ 경량 Audio Encoder
→ Head A: 문제 유형  (multi-label sigmoid)
→ Head B: 영향 소스  (multi-label sigmoid)
→ Rule-based Action Mapper  ← EQ 추천은 초기 룰 기반
→ Confidence Smoothing  (최근 3~5개 결과 temporal averaging)
→ 화면 출력
```

**보여주는 속도는 빠르게, 판단은 누적해서 안정화**한다.
단발 1~2초 입력보다 rolling buffer + temporal smoothing이 훨씬 안정적이다.

### 나중 버전 (v2~v3)

데이터가 쌓인 후 아래로 교체한다.

```
BEATs / AST fine-tuning
→ Head C: EQ 추천 학습형 (band class + gain bucket 회귀)
→ Head D: 보컬 음역대 분류 (조건부 실행)
```

---

## 모델 설계 원칙

### 1. Multi-label로 설계한다

라이브 현장은 문제가 동시에 여러 개 발생한다.

```
vocal buried + mix muddy + cymbal harsh
```

softmax 단일 클래스가 아니라 **sigmoid multi-label**로 설계한다.

### 2. 계층형 추론 순서를 지킨다

병렬로 다 뽑으면 오류가 커진다. 순서대로 실행한다.

```
1단계: 문제 유형 탐지   (multi-label)
2단계: 영향 소스 추정   (문제 유형 조건부)
3단계: EQ 조치 생성    (룰 기반 → 이후 학습형)
4단계: 보컬 분류       (vocal confidence 높을 때만)
```

### 3. EQ 추천은 밴드 선택형으로 시작한다

처음부터 `237Hz, -2.7dB` 같은 정밀 수치를 회귀하면 학습이 불안정하다.

초기에는 **band class + gain bucket** 구조로 설계한다.

```python
eq_action = {
  "band": "250-400hz",
  "gain": "-3db",
  "q": "wide"
}
```

데이터가 쌓이면 연속값 회귀로 확장한다.

### 4. Confidence를 반드시 표시한다

틀렸을 때 왜 그렇게 말했는지가 보여야 사용자가 받아들인다.

```
문제: muddy (0.83)
영향 추정: vocal (0.71) / keys (0.48)
추천: 250~350Hz -3dB
확신도: 중간
```

확신도가 낮으면 추천을 표시하되 "참고용"임을 명시한다.

### 5. 룰 기반 Action Mapper (초기)

학습 데이터가 없는 초기에는 룰로 EQ를 매핑한다.

| 문제 + 소스 | 조치 |
| --- | --- |
| muddy + vocal dominant | 250~350Hz 소폭 cut |
| buried + vocal dominant | 2~4kHz 소폭 boost |
| harsh + drums dominant | 5~8kHz 주의 / cut |
| imbalance + bass | 100~200Hz 확인 |

---

## 보컬 분류

### 설계 방향 변경: 사람 분류 → 믹스 역할 분류

`male / female / leader` 분류는 아래 이유로 변경한다.

- `leader`는 역할 속성이지 음향 속성이 아니다. 음향만으로 안정적 정의가 어렵다.
- `male / female`도 음역대 겹침이 많다.
- 제품 입장에서는 "이 보컬이 믹스 안에서 중심인가"가 더 중요하다.

**변경 후 분류 체계**

| 클래스 | 의미 |
| --- | --- |
| vocal present | 보컬 신호 감지됨 |
| dominant vocal | 믹스 내 보컬이 주도적 |
| likely male-range | 저중역 중심 보컬 (100~300Hz) |
| likely female-range | 중고역 중심 보컬 (300~500Hz+) |

### EQ 위치 차이 (핵심 이유)

보컬 음역대에 따라 EQ 처방 위치가 달라지기 때문에 분류가 필요하다.

| 음역대 | 핵심 EQ 영역 |
| --- | --- |
| male-range | 200~500Hz 명료도 / 2kHz 존재감 |
| female-range | 3~5kHz 존재감 / 8kHz air |

### 조건부 실행

보컬이 감지된 경우에만 음역대 분류를 실행한다.

```python
if vocal_presence > threshold:
    vocal_range = classify(male_range / female_range)
else:
    vocal_range = None  # 분류 skip
```

---

## 기능 정의

### 입력

- 스마트폰 내장 마이크 또는 외부 마이크
- 분석 버튼 탭 → 3초 rolling buffer로 자동 수집 및 추론

### 출력 — 4가지

**① 문제 유형** (multi-label)

| 유형 | 의미 |
| --- | --- |
| muddy | 저역이 뭉침, 음이 탁함 |
| harsh | 고역이 거슬림, 귀가 아픔 |
| buried | 특정 악기/보컬이 묻힘 |
| imbalance | 특정 대역이 지나치게 강하거나 약함 |

**② 영향 소스** (multi-label)

vocal / guitar / bass / drums / keys / overall

**③ EQ 조치** (band class + gain bucket)

```
영역          |  조정
250~400Hz    |  -3dB (wide)
3~5kHz       |  +2dB (mid)
```

**④ 보컬 음역대** (vocal 감지 시에만)

likely male-range / likely female-range

---

## 성공 기준

| 항목 | 목표 |
| --- | --- |
| 결과 도출 시간 | 3초 이내 (rolling buffer 포함) |
| 분석 인터랙션 | 1회 탭으로 완료 |
| 결과 이해 시간 | 1초 이내 (시각적 직관) |
| 문제 유형 분류 정확도 | ≥ 80% |
| EQ 방향성 정확도 (boost vs cut) | ≥ 85% |
| 보컬 음역대 분류 정확도 | ≥ 75% (vocal 감지 시) |

---

## 데이터 전략

이 프로젝트의 핵심 병목은 모델이 아니라 **데이터**다.

공개 데이터셋의 한계:

| 데이터셋 | 한계 |
| --- | --- |
| OpenMIC | 악기 인식용. 10초 clip-level 중심. 라이브 PA 환경 아님 |
| Slakh2100 | stem 분리 학습엔 좋지만 실제 라이브 현장과 다름 |
| AST/BEATs 학습 데이터 | 범용 오디오. 라이브 믹스 문제 진단용 정답 없음 |

**자체 데이터셋 구축이 필수**다.

| 항목 | 내용 |
| --- | --- |
| 입력 | 실제 공연/리허설 2~4초 구간 |
| 라벨 1 | 문제 유형 (muddy / harsh / buried / imbalance / normal) |
| 라벨 2 | 영향 소스 (vocal / guitar / bass / drums / keys / overall) |
| 라벨 3 | 추천 조치 (band class + gain bucket) |
| 메타 | 무대 규모, 홀 크기, 스마트폰 기종, 거리, 위치, SPL 수준 |

---

## MVP 단계

### v1 — 2~4주

rule 기반 + 경량 ML. 3가지 케이스만 커버.

- vocal buried
- guitar harsh
- bass muddy

rolling buffer + confidence 표시 포함. 보컬 음역대 분류 미포함.

### v2 — 이후

multi-task 모델 도입. 자체 데이터셋 기반 fine-tuning.

- 문제 유형 4종 + 영향 소스 6종 multi-label
- 보컬 음역대 분류 추가 (조건부 실행)
- EQ 조치 band class + gain bucket 학습형

### v3 — 장기

- BEATs / AST 기반 encoder fine-tuning
- EQ 추천 연속값 회귀 확장
- 믹서 자동 제어 연동

---

## 기술 스택

| 항목 | 선택 | 이유 |
| --- | --- | --- |
| 오디오 입력 | AVFoundation (iOS) / AudioRecord (Android) | 네이티브 저지연 |
| 스펙트럼 변환 | torchaudio / librosa | log-Mel Spectrogram |
| 특징 추출 | log-Mel + RMS + Spectral Centroid + Bandwidth | 경량 특징 조합 |
| 인코더 (MVP) | 경량 커스텀 CNN 또는 MobileNet 계열 | 빠른 추론 |
| 인코더 (v2+) | BEATs 또는 AST fine-tuning | 정확도 향상 |
| 추론 모델 | PyTorch → ONNX 변환 | 모바일 경량화 |
| 런타임 | ONNX Runtime Mobile | 추론 지연 20ms 미만 목표 |
| UI 프레임워크 | React Native 또는 Flutter | 크로스플랫폼 |

---

## 핵심 리스크

| 리스크 | 대응 |
| --- | --- |
| 자체 데이터 부족 | 공연/리허설 녹음 수집 계획 수립. synthetic 데이터로 보완 |
| 혼합음 환경에서 영향 소스 추정 불안정 | confidence 표시 + 낮을 시 "참고용" 명시 |
| 보컬 음역대 오분류 | vocal presence threshold 튜닝. 확신 낮으면 분류 생략 |
| 모바일 실시간성 | rolling buffer + 경량 encoder 구조로 허용 범위 내 처리 |
| 환경 소음 영향 | SNR 체크 후 소음 과다 시 재측정 안내 |

---

## 지금 당장 안 하는 것

| 항목 | 이유 |
| --- | --- |
| 음원 분리 (Demucs 등) | latency, 배터리, 연산량 모두 모바일 부적합 |
| leader 분류 | 음향 속성이 아닌 역할 속성. 음향만으로 정의 불안정 |
| 정밀 주파수 + 정밀 dB 회귀 (초기) | 데이터 없으면 학습 불안정. 밴드 선택형으로 시작 |
| 공개 pretrained만으로 제품 수준 기대 | 도메인 파인튜닝 없이는 라이브 믹스 진단 불가 |
| 수음 마이크 다채널 검증 | v3 이후 |
| 믹서 OSC 자동 제어 | v3 이후 |
| 공연장 공간 분석 | v3 이후 |