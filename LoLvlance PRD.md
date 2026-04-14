# LoLvlance PRD (v2 — Reality-Aligned)

---

## 1. Product Definition

### 제품 정의

LoLvlance는 **라이브 환경에서 지속적으로 오디오를 모니터링하고,  
문제 가능성과 대응 방향을 안내하는 보조 시스템**이다.

> 핵심 개념  
> ❌ “한 번 듣고 바로 정답 제공”  
> ✅ “지속적으로 듣고, 무엇을 확인해야 하는지 안내”

---

## 2. Product Goal

### 목표

라이브 상황에서 사용자가 다음 3가지를 빠르게 판단하도록 돕는다:

1. 지금 뭐가 이상한가?
2. 어떤 소스가 관련되어 있을 가능성이 높은가?
3. 무엇을 먼저 확인해야 하는가?

### 컨셉

"한 손에, 한 번에"

---

## 3. Product Positioning

### 현재 기준 (README 기반 정정)

- continuous monitoring system
- rolling window 기반 분석
- 몇 초 단위로 지속 업데이트
- assistive guidance system (보조 시스템)

### 금지 표현

- real-time diagnosis ❌
- instant AI 판단 ❌
- 자동 해결 ❌
- production-grade AI ❌

### 권장 표현

> LoLvlance는 세션을 지속적으로 모니터링하고  
> 몇 초 단위로 분석을 업데이트하며  
> 사용자가 다음에 무엇을 확인해야 할지 안내한다.

---

## 4. Target User

### 주요 사용자

- 교회 음향팀
- 소규모 공연 운영자
- 비전문 음향 담당자

### 사용자 특징

- 동시에 여러 역할 수행
- 빠른 판단 필요
- 전문 지식 부족

---

## 5. Core UX Flow

### 전체 플로우

```

마이크 입력
→ 지속 캡처 (continuous)
→ rolling buffer 유지
→ 1초마다 분석
→ 결과 smoothing
→ UI 업데이트

```

### 핵심 UX 원칙

- 입력은 지속적 (버튼 기반 X)
- 결과는 누적 기반 (단발 판단 X)
- 출력은 “확정”이 아니라 “가이드”

---

## 6. Functional Specification

### 6.1 입력

- 브라우저 마이크 입력
- 지속 스트리밍 기반 처리
- 초기 최소 버퍼: 750ms

---

### 6.2 분석 구조

- 분석 window: 3초
- stride: 1초
- 방식: rolling buffer

---

### 6.3 출력 (4가지)

#### ① 문제 유형 (multi-label)

- muddy
- harsh
- buried
- imbalance

---

#### ② 영향 소스 (multi-label)

- vocal
- guitar
- bass
- drums
- keys
- overall

---

#### ③ EQ 가이드 (초기)

- rule-based 기반
- human-readable 형태

```

250~350Hz -3dB
3kHz +2dB

```

---

#### ④ Confidence

- 높음 / 중간 / 낮음
- 확신도 기반 UI 표현 필수

---

## 7. System Architecture

### End-to-End Flow

```

Mic Input
→ Audio Capture
→ Rolling Buffer
→ Feature Extraction
→ ONNX Inference (optional)
→ Rule-based fallback
→ EMA smoothing
→ UI

```

---

### 역할 분리

#### ML

- issue_probs
- source_probs
- eq_freq
- eq_gain_db

#### Rule System

- 해석
- 보정
- fallback
- 사용자 가이드 생성

---

## 8. ML Policy (중요)

### 현재 상태

- model version: `v0.0-pipeline-check`
- synthetic data 기반
- production 아님

---

### 정책

- ML 결과는 **참고용**
- 항상 fallback 존재
- UI에 명시:

```

Experimental Mode
Not production-ready
Results may be inaccurate

```

---

## 9. Monitoring System

### 기존 방식 (삭제)

```

3초 수집 → 분석 → 대기 → 반복
→ blind gap 발생

```

---

### 현재 방식 (적용)

```

continuous capture

* rolling buffer
* 1초 stride 분석

```

---

### 장점

- blind gap 없음
- 지속적 커버리지
- 안정적 UX

---

## 10. Output Stabilization

### EMA 적용

- issue confidence
- source confidence
- EQ guidance

```

alpha = 0.3

```

---

### 목적

- flicker 방지
- UX 안정화

---

## 11. Edge Case Handling

- 초기 buffer 부족
- silence
- mic permission 실패
- inference 실패
- ML off 상태

---

## 12. Data Strategy (핵심)

### 문제

- 공개 데이터셋 → 실제 라이브 환경과 mismatch

---

### 필수 전략

자체 데이터 구축

| 항목 | 내용 |
|------|------|
| 입력 | 실제 공연 2~4초 |
| 라벨1 | 문제 유형 |
| 라벨2 | 영향 소스 |
| 라벨3 | EQ 조치 |
| 메타 | 공간, 거리, SPL 등 |

---

### 현재 상태

- feedback export 기능 존재
- closed-loop 미완성

---

## 13. Success Metrics

| 항목 | 목표 |
|------|------|
| 분석 latency | ~1초 업데이트 |
| UX 이해 시간 | 1초 이하 |
| issue 정확도 | ≥ 80% |
| EQ 방향성 | ≥ 85% |
| UX 안정성 | flicker 최소화 |

---

## 14. Limitations

### 모델

- synthetic 기반
- source bias 존재
- 정확도 불안정

---

### 제품

- 정답 제공 시스템 아님
- 추천 시스템

---

### 기술

- browser inference 의존
- preprocessing drift 가능성

---

## 15. Roadmap

### Immediate

- 제품 서사 정합성 유지
- 모델 isolation 유지
- golden dataset 확장

---

### Short-term

- real dataset 학습
- feedback ingestion loop 구축
- threshold tuning

---

### Long-term

- production model 전환
- server inference 검토
- 자동 EQ 확장

---

## 16. Key Principle (가장 중요)

> pipeline works ≠ model is trustworthy

현재 상태:

- 파이프라인: 완성됨 ✅
- 모델: 신뢰 불가 ❌

---

## 17. One-line Summary

> LoLvlance는 라이브 오디오를 지속적으로 모니터링하고  
> 사용자가 다음에 무엇을 확인해야 할지 안내하는 보조 시스템이다.
```