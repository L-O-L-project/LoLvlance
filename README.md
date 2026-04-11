# SoundFix

실시간 마이크 입력을 받아 오디오 특징을 추출하고, 브라우저 내 ML 추론과 rule-based 분석을 결합해 문제 유형, 감지된 악기/보컬, EQ 가이드를 보여주는 오디오 분석 프로토타입입니다.

현재 구현은 다음까지 연결되어 있습니다.

- 브라우저 마이크 입력 및 권한 처리
- native sample rate 캡처 + 16kHz 분석용 정규화
- 약 3초 rolling circular buffer 유지
- log-mel spectrogram / RMS feature extraction
- ONNX Runtime Web 기반 브라우저 ML 추론
- rule-based 문제 감지 및 EQ 추천
- MediaPipe YAMNet 기반 오픈소스 fallback 태깅
- `audio-separator` + `htdemucs_6s` 기반 local stem separation sidecar
- stem service 연결 상태, stem별 RMS/energy, 악기별 source-aware EQ를 보여주는 UI

원본 Figma 프로젝트:
https://www.figma.com/design/8RDv0AsmeEeS799aTihSO0/Real-Time-Sound-Analysis-App

## 현재 아키텍처

### 1. Browser Audio Capture
- 마이크 권한 요청 및 `granted / denied / prompt / unsupported` 상태 처리
- `AudioWorklet` 우선, 실패 시 `ScriptProcessorNode` fallback
- `AnalyserNode` 기반 실시간 waveform / spectrum 시각화
- native sample rate 원본 버퍼와 16kHz 분석 버퍼를 동시에 유지

### 2. Feature Extraction
- 입력: 약 3초 mono buffer
- target sample rate: `16kHz`
- framing: `25ms`
- hop size: `10ms`
- Hann window
- STFT
- mel filter bank
- epsilon-safe log-mel spectrogram
- RMS 계산

### 3. Analysis Layers
- 1차: 브라우저 ONNX 모델 추론
- 2차: rule-based 문제 감지 fallback
- 소스 감지:
  - 우선 `audio-separator` sidecar의 stem 결과 사용
  - 부족할 경우 MediaPipe YAMNet fallback 결과를 병합
- 소스별 rule-based EQ:
  - `vocal / drums / bass / guitar / keys` 별로 분리된 추천 제공

### 4. UI Output
- mixer-style EQ / waveform visualization
- 문제 카드
- 감지된 소스 카드
- `Stem Service Connected / Fallback` 상태 표시
- stem별 energy share / RMS 카드
- 악기별 EQ recommendation 카드

## 주요 파일

- [App.tsx](/Users/kimhajun/Downloads/LoLvlance/src/app/App.tsx)
  앱 진입점, 분석/모니터링 상태 관리
- [useMonitoring.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/hooks/useMonitoring.ts)
  오디오 캡처, 버퍼 유지, feature extraction, ML/rule/stem/fallback 흐름 통합
- [audioUtils.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/audioUtils.ts)
  resampling, circular buffer, RMS/peak/dbFS 유틸
- [featureExtraction.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/featureExtraction.ts)
  log-mel spectrogram / RMS 추출
- [mlInference.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/mlInference.ts)
  `onnxruntime-web` 기반 브라우저 추론
- [ruleBasedAnalysis.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/ruleBasedAnalysis.ts)
  `muddy / harsh / buried` 분석 및 EQ recommendation
- [openSourceAudioTagging.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/openSourceAudioTagging.ts)
  MediaPipe YAMNet 기반 악기/보컬 fallback 태깅
- [stemSeparationClient.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/stemSeparationClient.ts)
  local stem service 호출 및 응답 파싱
- [sourceAwareEq.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/sourceAwareEq.ts)
  감지된 악기와 stem metric 기반 소스별 EQ 추천
- [ResultCards.tsx](/Users/kimhajun/Downloads/LoLvlance/src/app/components/ResultCards.tsx)
  stem 상태, stem metrics, 소스 EQ, 문제 카드 렌더링
- [EQVisualization.tsx](/Users/kimhajun/Downloads/LoLvlance/src/app/components/EQVisualization.tsx)
  mixer-style live visualization
- [lightweight_audio_model.py](/Users/kimhajun/Downloads/LoLvlance/ml/lightweight_audio_model.py)
  PyTorch lightweight multi-head model 정의
- [export_to_onnx.py](/Users/kimhajun/Downloads/LoLvlance/ml/export_to_onnx.py)
  PyTorch -> ONNX export 스크립트
- [stem_separation_service.py](/Users/kimhajun/Downloads/LoLvlance/ml/stem_separation_service.py)
  local stem separation HTTP sidecar

## 모델 / 자산 위치

- ONNX model: [lightweight_audio_model.onnx](/Users/kimhajun/Downloads/LoLvlance/public/models/lightweight_audio_model.onnx)
- ONNX external data: [lightweight_audio_model.onnx.data](/Users/kimhajun/Downloads/LoLvlance/public/models/lightweight_audio_model.onnx.data)
- YAMNet model: [yamnet.tflite](/Users/kimhajun/Downloads/LoLvlance/public/models/yamnet.tflite)
- MediaPipe wasm: [public/mediapipe/wasm](/Users/kimhajun/Downloads/LoLvlance/public/mediapipe/wasm)

## 실행 방법

### Frontend
```bash
npm install
npm run dev
```

### Production Build
```bash
npm run build
```

## Local Stem Service

정확한 악기/보컬 분리를 원하면 local stem sidecar를 함께 실행하는 것을 권장합니다.

기본 주소:

- `http://127.0.0.1:8765`

필요 시 프론트에서 다음 환경변수로 변경할 수 있습니다.

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

stem service가 살아 있으면 UI 결과 카드에 `Stem Service Connected`가 표시되고, 연결되지 않으면 브라우저 fallback 경로가 사용됩니다.

## ML 테스트

### 테스트 환경 준비
```bash
bash ml/setup_test_env.sh
```

### 전체 테스트 실행
```bash
bash ml/run_ml_tests.sh
```

포함 내용:

- PyTorch model forward shape/range 검증
- dummy checkpoint 생성 후 ONNX export
- `onnxruntime` 로 ONNX inference
- PyTorch vs ONNX 출력 비교

## ONNX Export

학습된 체크포인트가 있을 때:

```bash
python3 ml/export_to_onnx.py \
  --checkpoint path/to/model.ckpt \
  --output ml/lightweight_audio_model.onnx \
  --time-steps 128
```

입력 shape:

- `(1, T, 64)`
- `T`는 dynamic axes로 export

## 현재 실제 동작 플로우

1. 사용자가 `Start Analysis` 또는 `Start Monitoring` 선택
2. 브라우저가 마이크 권한 요청 및 상태 처리
3. 입력 오디오를 native sample rate로 캡처
4. 동시에 16kHz 분석용 버퍼로 resample
5. 약 3초 rolling buffer 유지
6. 현재 buffer에서 log-mel / RMS 추출
7. 브라우저 ONNX 모델 추론 시도
8. 병렬로 local stem service 호출
9. stem 결과가 부족하면 YAMNet fallback 태깅으로 보강
10. rule-based 문제 감지 및 source-aware EQ 구성
11. UI에 문제, 소스, stem metrics, EQ recommendation 표시

## 디버그 로그

브라우저 콘솔에서 아래 로그를 확인할 수 있습니다.

```ts
[audio-features] { logMelSpectrogramShape: [time, melBins], rms: 0.1234 }
[audio-ml] { ... }
[audio-rules] { issues: [...], eq_recommendations: [...] }
[audio-stems] { model: 'htdemucs_6s.yaml', detectedSources: [...], stems: [...] }
[audio-tags] { detectedSources: [...], topCategories: [...] }
```

## 현재 한계

- frontend에 연결된 ONNX 모델은 추론 파이프라인 검증용 자산일 수 있으므로, 실제 품질은 학습된 체크포인트로 교체해야 의미 있게 올라갑니다.
- rule-based 문제 감지는 현재 `muddy / harsh / buried` 중심입니다.
- stem separation은 local Python sidecar가 필요하며, 최초 실행 시 모델 다운로드 때문에 느릴 수 있습니다.
- 브라우저 fallback(YAMNet)은 stem separation이 아니므로 혼합 신호에서 보컬 편향이 생길 수 있습니다.
- training pipeline, dataset loader, trainer는 아직 repo에 없습니다.

## 다음 추천 작업

1. 실제 학습 데이터셋 기준으로 PyTorch training pipeline 추가
2. 학습 완료 checkpoint로 ONNX 자산 교체
3. stem별 세부 rule을 더 늘려 instrument-aware 진단 정밀도 향상
4. source-specific problem taxonomy 확장
5. sidecar latency 최적화 및 batching/queue 전략 검토
