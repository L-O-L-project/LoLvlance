# HANDOVER

## 한 줄 상태

현재 프로젝트는 "브라우저 실시간 오디오 입력 -> feature extraction -> browser ONNX inference + rule-based analysis -> stem separation / fallback source tagging -> source-aware EQ UI"까지 연결된 상태입니다.

## 지금 실제로 되는 것

### Frontend Runtime
- 마이크 권한 요청 및 상태 처리
- native sample rate 오디오 캡처
- 16kHz 분석용 resample
- 약 3초 rolling circular buffer 유지
- mixer-style waveform / analyzer / RMS visualization
- 브라우저 ONNX 추론
- rule-based 문제 감지 fallback
- local stem service 호출
- MediaPipe YAMNet fallback 태깅
- 결과 UI에:
  - 문제 카드
  - 감지된 소스
  - stem service 연결 상태
  - stem별 energy / RMS
  - 악기별 source-aware EQ

### Python / ML Assets
- PyTorch lightweight multi-head model 정의 완료
- ONNX export 스크립트 존재
- ML 테스트 스크립트 존재
- stem separation sidecar 실행 스크립트 존재

## 현재 분석 플로우

1. [App.tsx](/Users/kimhajun/Downloads/LoLvlance/src/app/App.tsx) 에서 분석/모니터링 시작
2. [useMonitoring.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/hooks/useMonitoring.ts) 에서 마이크 확보
3. native sample rate 버퍼와 16kHz 분석 버퍼를 동시에 유지
4. [featureExtraction.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/featureExtraction.ts) 에서 log-mel / RMS 추출
5. [mlInference.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/mlInference.ts) 로 ONNX 추론 시도
6. 동시에 [stemSeparationClient.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/stemSeparationClient.ts) 가 local sidecar 호출
7. stem 결과가 충분하지 않으면 [openSourceAudioTagging.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/openSourceAudioTagging.ts) 로 fallback 태깅
8. [ruleBasedAnalysis.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/ruleBasedAnalysis.ts) 로 issue 분석
9. [sourceAwareEq.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/sourceAwareEq.ts) 에서 instrument-aware EQ recommendation 생성
10. [ResultCards.tsx](/Users/kimhajun/Downloads/LoLvlance/src/app/components/ResultCards.tsx) 와 [EQVisualization.tsx](/Users/kimhajun/Downloads/LoLvlance/src/app/components/EQVisualization.tsx) 에서 표시

## 핵심 파일 우선순위

### 가장 먼저 볼 파일
1. [useMonitoring.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/hooks/useMonitoring.ts)
2. [ResultCards.tsx](/Users/kimhajun/Downloads/LoLvlance/src/app/components/ResultCards.tsx)
3. [mlInference.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/mlInference.ts)
4. [stemSeparationClient.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/stemSeparationClient.ts)
5. [sourceAwareEq.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/sourceAwareEq.ts)

### 분석 관련 파일
- [audioUtils.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/audioUtils.ts)
- [featureExtraction.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/featureExtraction.ts)
- [ruleBasedAnalysis.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/ruleBasedAnalysis.ts)
- [openSourceAudioTagging.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/openSourceAudioTagging.ts)
- [types.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/types.ts)

### Python / ML / Export
- [lightweight_audio_model.py](/Users/kimhajun/Downloads/LoLvlance/ml/lightweight_audio_model.py)
- [export_to_onnx.py](/Users/kimhajun/Downloads/LoLvlance/ml/export_to_onnx.py)
- [stem_separation_service.py](/Users/kimhajun/Downloads/LoLvlance/ml/stem_separation_service.py)
- [tests](/Users/kimhajun/Downloads/LoLvlance/ml/tests)

## 실행 방법

### Frontend
```bash
npm install
npm run dev
```

### Build 검증
```bash
npm run build
```

## Stem Service 운영 메모

기본 URL:

- `http://127.0.0.1:8765`

프론트에서 다른 주소를 쓰려면:

```bash
VITE_STEM_SERVICE_URL=http://127.0.0.1:8765
```

### Setup
```bash
bash ml/setup_stem_service.sh
```

### Run
```bash
bash ml/run_stem_service.sh
```

### Health Check
```bash
curl http://127.0.0.1:8765/health
```

### Stem Service가 켜져 있을 때 기대 동작
- UI에 `Stem Service Connected` 표시
- 결과 카드에 stem별 `Energy` / `RMS` 표시
- stem 결과가 소스 감지와 source-aware EQ 추천에 반영됨

### 꺼져 있을 때 동작
- UI에 `Stem Service Fallback` 표시
- MediaPipe YAMNet fallback 태깅만 사용
- 이 경우 혼합 신호에서 보컬 편향이 강해질 수 있음

## ML 테스트 / ONNX 검증

### 테스트 환경 준비
```bash
bash ml/setup_test_env.sh
```

### 전체 테스트
```bash
bash ml/run_ml_tests.sh
```

포함 검증:
- model forward shape/range
- dummy checkpoint -> ONNX export
- `onnxruntime` inference
- PyTorch vs ONNX 출력 비교

### Python syntax 체크
```bash
python3 -m py_compile ml/lightweight_audio_model.py
python3 -m py_compile ml/export_to_onnx.py
python3 -m py_compile ml/stem_separation_service.py
```

### ONNX export 예시
```bash
python3 ml/export_to_onnx.py \
  --checkpoint path/to/model.ckpt \
  --output ml/lightweight_audio_model.onnx \
  --time-steps 128
```

## 디버그 포인트

브라우저 콘솔 로그:

- `[audio-features]`
- `[audio-ml]`
- `[audio-rules]`
- `[audio-stems]`
- `[audio-tags]`

중요한 해석:

- `[audio-stems]`가 찍히면 local stem service 경로 사용 중
- `[audio-tags]`만 찍히면 브라우저 fallback 중심
- ONNX 모델 로드/실패 여부는 `[audio-ml]` 로그로 확인

## 현재 남아 있는 제약

### 모델 품질
- frontend에 연결된 ONNX 자산은 파이프라인 테스트용일 수 있어, 실제 정확도를 높이려면 학습된 checkpoint로 교체해야 함

### 분석 범위
- 문제 감지는 현재 `muddy / harsh / buried` 중심
- 소스별 EQ는 `vocal / drums / bass / guitar / keys` 기준의 lightweight rule set

### stem separation
- local Python sidecar 필요
- 최초 실행 시 모델 다운로드 및 warm-up 시간이 큼
- 실시간성은 충분히 usable하지만 strict low-latency production 수준으로는 추가 최적화 필요

### training pipeline
- dataset loader 없음
- trainer 없음
- loss 설계 없음
- checkpoint 생성 파이프라인 없음

## 다음 작업 우선순위

1. 학습 데이터 기준 training pipeline 추가
2. 실제 학습 checkpoint로 ONNX 모델 재생성
3. source-specific problem taxonomy 확장
4. stem별 세분화 rule 추가
5. UI 디버그 토글과 모델/engine 선택 토글 추가
6. stem sidecar latency/profile 최적화

## 실무적으로 기억할 점

- source detection 정확도는 stem sidecar가 켜져 있을 때와 아닐 때 차이가 큼
- 문제 카드 소스 표시는 너무 많은 악기가 한 번에 붙지 않도록 제한되어 있음
- native sample rate 버퍼는 stem/fallback 태깅 품질 때문에 유지되고, ML feature는 16kHz 버퍼를 사용함
- `public/models/lightweight_audio_model.onnx` 와 `public/models/lightweight_audio_model.onnx.data`를 같이 유지해야 함
