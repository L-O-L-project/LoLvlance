# HANDOVER

## 프로젝트 상태 요약

현재 프로젝트는 "브라우저 실시간 오디오 입력 -> feature extraction -> rule-based 분석 -> UI 표시"까지 동작하는 상태입니다.  
추가로 향후 ML 추론을 위한 PyTorch 모델 정의와 ONNX export 스크립트까지 준비되어 있습니다.

핵심 포인트:
- 마이크 입력 파이프라인은 이미 실시간 동작하도록 정리됨
- 16kHz 기준 약 3초 rolling buffer 유지
- frontend 분석은 현재 ML이 아니라 rule-based logic 사용
- ML 모델은 별도 Python 파일로 정의되어 있으나 아직 frontend inference에 연결되지는 않음

## 지금까지 구현된 항목

### 1. Microphone / Audio Capture
- 마이크 권한 요청 및 상태 처리
- permission denied / unsupported / device not found 대응
- `AudioContext` + `AnalyserNode` 기반 실시간 캡처
- AudioWorklet 우선, 실패 시 ScriptProcessor fallback
- 입력 sample rate를 16kHz로 정규화
- 약 3초 분량 circular buffer 유지

관련 파일:
- [useMonitoring.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/hooks/useMonitoring.ts)
- [audioUtils.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/audioUtils.ts)

### 2. Visualization
- 기존 레이아웃은 유지
- mixer-style analyzer panel로 변경
- waveform / spectrum / RMS meter를 한 패널에 표시

관련 파일:
- [EQVisualization.tsx](/Users/kimhajun/Downloads/LoLvlance/src/app/components/EQVisualization.tsx)

### 3. Feature Extraction
- 고정 길이 buffer를 기준으로 log-mel spectrogram 생성
- framing: 25ms
- hop: 10ms
- Hann window
- FFT / STFT
- mel filter bank
- log scaling with epsilon
- RMS 추출
- console debug log 추가

관련 파일:
- [featureExtraction.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/featureExtraction.ts)
- [types.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/types.ts)

### 4. Rule-Based Audio Issue Detection
- ML 없이 deterministic rule 기반 문제 감지
- 지원 문제:
  - `muddy`
  - `harsh`
  - `buried`
- issue별 EQ recommendation 생성
- 기존 UI가 쓰는 `AnalysisResult.problems` 형태로 매핑
- 동시에 structured output도 유지:
  - `issues`
  - `eq_recommendations`

관련 파일:
- [ruleBasedAnalysis.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/ruleBasedAnalysis.ts)
- [useMonitoring.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/hooks/useMonitoring.ts)

### 5. PyTorch Model
- lightweight CNN encoder
- optional small transformer
- multi-head output
  - problem classification: 4 classes
  - instrument estimation: 5-label sigmoid
  - EQ recommendation: freq + gain

관련 파일:
- [lightweight_audio_model.py](/Users/kimhajun/Downloads/LoLvlance/ml/lightweight_audio_model.py)

### 6. ONNX Export
- trained checkpoint 로드
- ONNX-friendly wrapper 사용
- `(1, T, 64)` 입력
- `T` dynamic axes 처리
- `onnxruntime-web` 호환 고려

관련 파일:
- [export_to_onnx.py](/Users/kimhajun/Downloads/LoLvlance/ml/export_to_onnx.py)

## 현재 실제 동작 플로우

1. 사용자가 분석 시작 또는 모니터링 시작
2. `useMonitoring`에서 마이크 스트림 확보
3. 입력을 16kHz로 정규화
4. 3초 rolling buffer 갱신
5. 분석 시 현재 buffer로 feature extraction 수행
6. log-mel + RMS 기반 rule 분석 수행
7. UI에 `problems`와 EQ action 문자열 표시
8. console에는 feature shape / RMS / rule 결과 출력

## 아직 안 된 것

### ML 관련
- training script 없음
- dataset loader 없음
- loss 정의 없음
- checkpoint 없음
- frontend에서 ONNX 모델 추론 없음
- ONNX export 실사용 검증은 checkpoint 부재로 미완료

### 분석 로직
- 현재 rule-based 감지는 3개 문제만 다룸
- `normal` class는 별도 label로 저장하지 않고 "no issues"로 처리
- instrument estimation은 ML 모델 구조에만 있고 frontend 분석에는 아직 미반영

## 다음 작업 우선순위 제안

1. Python training pipeline 추가
2. 실제 dataset schema 정리
3. 학습 완료 checkpoint 생성
4. `export_to_onnx.py`로 ONNX export 검증
5. frontend에서 `onnxruntime-web` 로드 후 inference 연결
6. rule-based vs ML 결과 비교 모드 추가

## 실행 / 검증 메모

### Frontend
```bash
npm install
npm run dev
```

### Frontend build 확인
```bash
npm run build
```

### Python syntax 확인
```bash
python3 -m py_compile ml/lightweight_audio_model.py
python3 -m py_compile ml/export_to_onnx.py
```

### ONNX export 예시
```bash
python3 ml/export_to_onnx.py \
  --checkpoint path/to/model.ckpt \
  --output ml/lightweight_audio_model.onnx \
  --time-steps 128
```

## 주의사항

- 현재 frontend 오디오 분석은 전부 TypeScript 쪽 rule engine을 사용합니다.
- Python 모델 파일은 준비만 된 상태이며, runtime dependency로 연결되어 있지 않습니다.
- README에도 적었지만 Python dependency manager는 아직 없습니다.
- checkpoint format은 raw `state_dict`, `state_dict` wrapper, `model_state_dict` wrapper를 지원합니다.

## 작업자에게 바로 유용한 파일 우선순위

1. [useMonitoring.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/hooks/useMonitoring.ts)
2. [featureExtraction.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/featureExtraction.ts)
3. [ruleBasedAnalysis.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/ruleBasedAnalysis.ts)
4. [EQVisualization.tsx](/Users/kimhajun/Downloads/LoLvlance/src/app/components/EQVisualization.tsx)
5. [lightweight_audio_model.py](/Users/kimhajun/Downloads/LoLvlance/ml/lightweight_audio_model.py)
6. [export_to_onnx.py](/Users/kimhajun/Downloads/LoLvlance/ml/export_to_onnx.py)
