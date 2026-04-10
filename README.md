# SoundFix

실시간 오디오 입력을 받아 특징을 추출하고, rule-based 분석 결과와 EQ 추천을 보여주는 프론트엔드 프로토타입입니다.  
현재 브라우저 마이크 입력, 16kHz 정규화, 3초 롤링 버퍼, log-mel spectrogram 추출, RMS 계산, rule-based 문제 감지, PyTorch 모델 정의, ONNX export 스크립트까지 구현되어 있습니다.

원본 Figma 프로젝트:
https://www.figma.com/design/8RDv0AsmeEeS799aTihSO0/Real-Time-Sound-Analysis-App

## 현재 구현 범위

### Frontend / Audio Pipeline
- 마이크 권한 요청 및 상태 처리
- 브라우저별 대응을 고려한 실시간 오디오 캡처
- 입력 오디오 16kHz 정규화
- 약 3초 길이의 rolling circular buffer 유지
- 실시간 mixer-style waveform / analyzer UI
- 분석 시 console에 feature 추출 결과 출력

### Feature Extraction
- 입력: 약 3초, 16kHz mono buffer
- framing: 25ms
- hop length: 10ms
- Hann window
- FFT / STFT
- mel filter bank
- log-mel spectrogram 생성
- RMS 계산

### Rule-Based Analysis
- `muddy`: 200Hz ~ 500Hz 에너지 비율 기반
- `harsh`: 2kHz ~ 6kHz 및 3kHz ~ 4kHz presence peak 기반
- `buried`: 1kHz ~ 3kHz presence 부족 및 band contrast 기반
- 각 문제에 대해 deterministic EQ recommendation 생성

### ML / Export 준비 상태
- PyTorch lightweight multi-head model 정의 완료
- ONNX export 스크립트 작성 완료
- 아직 학습된 체크포인트와 웹 추론 연결은 미완료

## 주요 파일

- [src/app/hooks/useMonitoring.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/hooks/useMonitoring.ts)
  실시간 마이크 입력, analyser, rolling buffer, feature extraction 호출, rule-based 분석 연결
- [src/app/audio/audioUtils.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/audioUtils.ts)
  resampling, circular buffer, RMS/peak 계산
- [src/app/audio/featureExtraction.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/featureExtraction.ts)
  log-mel spectrogram / RMS 추출
- [src/app/audio/ruleBasedAnalysis.ts](/Users/kimhajun/Downloads/LoLvlance/src/app/audio/ruleBasedAnalysis.ts)
  rule-based issue detection 및 EQ recommendation
- [src/app/components/EQVisualization.tsx](/Users/kimhajun/Downloads/LoLvlance/src/app/components/EQVisualization.tsx)
  mixer-style waveform / analyzer / RMS meter UI
- [ml/lightweight_audio_model.py](/Users/kimhajun/Downloads/LoLvlance/ml/lightweight_audio_model.py)
  PyTorch multi-head audio analysis model
- [ml/export_to_onnx.py](/Users/kimhajun/Downloads/LoLvlance/ml/export_to_onnx.py)
  trained checkpoint를 ONNX로 export하는 스크립트

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

## Python / ML 관련

이 저장소에는 Python dependency manager가 아직 정리되어 있지 않습니다.  
ONNX export를 사용하려면 최소한 아래 패키지가 필요합니다.

```bash
pip install torch onnx
```

선택적으로 export 결과를 검증하려면:

```bash
pip install onnxruntime
```

## ONNX Export 예시

학습된 체크포인트가 있다고 가정할 때:

```bash
python3 ml/export_to_onnx.py \
  --checkpoint path/to/model.ckpt \
  --output ml/lightweight_audio_model.onnx \
  --time-steps 128
```

스크립트는 `(1, T, 64)` 형태의 example input을 사용하고, `T`는 dynamic axes로 export됩니다.

## 현재 출력 형태

### Feature Extraction Console Log
```ts
[audio-features] {
  logMelSpectrogramShape: [time_steps, mel_bins],
  rms: 0.1234
}
```

### Rule-Based Analysis Console Log
```ts
[audio-rules] {
  issues: ["muddy", "harsh"],
  eq_recommendations: [
    { freq_range: "200-400Hz", gain: "-3dB", reason: "reduce muddiness in the low-mid range" }
  ]
}
```

## 현재 한계

- 실제 ML inference는 아직 프론트엔드에 연결되지 않았습니다.
- rule-based 분석은 현재 3가지 문제만 다룹니다: `muddy`, `harsh`, `buried`
- ONNX export 스크립트는 학습된 checkpoint가 있어야 실제 export가 가능합니다.
- Python training pipeline, dataset loader, loss, trainer는 아직 없습니다.

## 다음 추천 작업

1. PyTorch training pipeline 추가
2. 학습된 checkpoint 생성
3. ONNX export 실제 검증
4. `onnxruntime-web` 기반 프론트 추론 연결
5. rule-based 결과와 ML 결과를 함께 비교하는 디버그 모드 추가
