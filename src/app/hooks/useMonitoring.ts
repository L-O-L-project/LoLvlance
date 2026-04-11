import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  AnalysisResult,
  AnalysisEngine,
  BufferedAudioSnapshot,
  ExtractedAudioFeatures,
  MicrophoneErrorCode,
  MicrophonePermissionState
} from '../types';
import {
  appendToCircularBuffer,
  createBufferedAudioSnapshot,
  readCircularBuffer,
  ROLLING_BUFFER_SIZE,
  ROLLING_BUFFER_SECONDS,
  TARGET_SAMPLE_RATE,
  resampleMonoBuffer
} from '../audio/audioUtils';
import { extractAudioFeatures, logExtractedAudioFeatures } from '../audio/featureExtraction';
import {
  analyzeWithMlInference,
  MODEL_MEL_BIN_COUNT,
  warmUpMlInference
} from '../audio/mlInference';
import {
  detectOpenSourceAudioSources,
  warmUpOpenSourceAudioTagger
} from '../audio/openSourceAudioTagging';
import {
  detectStemSeparatedSources,
  warmUpStemSeparationService
} from '../audio/stemSeparationClient';
import { buildSourceAwareEqRecommendations } from '../audio/sourceAwareEq';
import {
  analyzeRuleBasedAudioIssues,
  logRuleBasedAnalysis,
  ruleAnalysisToDiagnosticProblems
} from '../audio/ruleBasedAnalysis';

const MONITORING_INTERVAL_MS = 4000;
// -38 dBFS ≈ 0.012 linear — expressed in dBFS for clarity
const SILENCE_THRESHOLD_DBFS = -38;
const CAPTURE_WORKLET_NAME = 'soundfix-microphone-capture';
const CAPTURE_WORKLET_SOURCE = `
class SoundFixMicrophoneCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunkSize = 2048;
    this.buffer = new Float32Array(this.chunkSize);
    this.offset = 0;
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const channel = input && input[0];
    const output = outputs[0];

    if (output && output[0]) {
      output[0].fill(0);
    }

    if (channel && channel.length) {
      let cursor = 0;

      while (cursor < channel.length) {
        const remaining = this.chunkSize - this.offset;
        const copyCount = Math.min(remaining, channel.length - cursor);

        this.buffer.set(channel.subarray(cursor, cursor + copyCount), this.offset);
        this.offset += copyCount;
        cursor += copyCount;

        if (this.offset === this.chunkSize) {
          this.port.postMessage(
            { type: 'samples', samples: this.buffer.buffer },
            [this.buffer.buffer]
          );
          this.buffer = new Float32Array(this.chunkSize);
          this.offset = 0;
        }
      }
    }

    return true;
  }
}

registerProcessor('${CAPTURE_WORKLET_NAME}', SoundFixMicrophoneCaptureProcessor);
`;

type CaptureNode = AudioWorkletNode | ScriptProcessorNode;

type AudioContextConstructor = typeof AudioContext;

function normalizePermissionState(state: PermissionState['state']): MicrophonePermissionState {
  if (state === 'granted' || state === 'denied') {
    return state;
  }

  return 'prompt';
}

function getAudioContextConstructor(): AudioContextConstructor | null {
  return window.AudioContext
    ?? (window as Window & { webkitAudioContext?: AudioContextConstructor }).webkitAudioContext
    ?? null;
}

function buildAnalysisResult(
  snapshot: BufferedAudioSnapshot,
  features: ExtractedAudioFeatures,
  engine: AnalysisEngine = 'rule-based'
): AnalysisResult {
  if (
    snapshot.samples.length < TARGET_SAMPLE_RATE / 2 ||
    snapshot.dbRms < SILENCE_THRESHOLD_DBFS
  ) {
    logRuleBasedAnalysis({
      issues: [],
      eq_recommendations: [],
      detections: []
    });

    return {
      problems: [],
      issues: [],
      eq_recommendations: [],
      engine,
      timestamp: Date.now()
    };
  }

  const ruleBasedAnalysis = analyzeRuleBasedAudioIssues(features);
  logRuleBasedAnalysis(ruleBasedAnalysis);

  const problems = ruleAnalysisToDiagnosticProblems(ruleBasedAnalysis);

  // Crest factor below ~3 (≈9.5 dB) indicates heavy limiting/compression.
  // Flag it as an imbalance problem so the user knows the signal is over-compressed.
  if (snapshot.crestFactor < 3) {
    problems.push({
      type: 'imbalance',
      confidence: Math.min(0.9, 0.6 + (3 - snapshot.crestFactor) * 0.15),
      sources: ['overall'],
      details: ['transient_overload'],
      actions: [
        `Signal is heavily compressed (crest factor ${snapshot.crestFactor.toFixed(1)}×). Consider reducing limiting or gain staging.`
      ]
    });
  }

  return {
    problems,
    issues: ruleBasedAnalysis.issues,
    eq_recommendations: ruleBasedAnalysis.eq_recommendations,
    engine,
    timestamp: Date.now()
  };
}

function mergeDetectedSources(result: AnalysisResult, detectedSources: AnalysisResult['detectedSources']) {
  if (!detectedSources || detectedSources.length === 0) {
    return result;
  }

  const detectedSourceNames = detectedSources.map((entry) => entry.source);

  return {
    ...result,
    detectedSources,
    problems: result.problems.map((problem) => {
      const existingSources = problem.sources.filter((source) => source !== 'overall');
      const mergedSources = existingSources.length === 0
        ? detectedSourceNames.slice(0, 3)
        : uniqueInOrder([
            ...existingSources,
            ...detectedSourceNames.filter((source) => !existingSources.includes(source)).slice(0, 1)
          ]);

      return {
        ...problem,
        sources: mergedSources.length > 0 ? mergedSources : problem.sources
      };
    })
  };
}

function mergeInstrumentDetections(
  stemDetectedSources: NonNullable<AnalysisResult['detectedSources']>,
  fallbackDetectedSources: NonNullable<AnalysisResult['detectedSources']>,
  stemMetrics: NonNullable<AnalysisResult['stemMetrics']>
) {
  const merged = new Map<
    NonNullable<AnalysisResult['detectedSources']>[number]['source'],
    NonNullable<AnalysisResult['detectedSources']>[number]
  >();

  const stemMetricBySource = new Map(
    stemMetrics
      .filter((stem) => stem.source)
      .map((stem) => [stem.source!, stem])
  );

  stemDetectedSources.forEach((entry) => {
    const stemMetric = stemMetricBySource.get(entry.source);
    const weightedConfidence = clamp(
      entry.confidence * 0.72 + (stemMetric?.energyRatio ?? 0) * 0.28,
      0,
      1
    );

    merged.set(entry.source, {
      source: entry.source,
      confidence: Number(weightedConfidence.toFixed(2)),
      labels: uniqueInOrder(entry.labels)
    });
  });

  fallbackDetectedSources.forEach((entry) => {
    const existing = merged.get(entry.source);

    if (!existing) {
      const boostThreshold = stemDetectedSources.length > 0 ? 0.14 : 0.08;

      if (entry.confidence >= boostThreshold) {
        merged.set(entry.source, {
          ...entry,
          confidence: Number(entry.confidence.toFixed(2)),
          labels: uniqueInOrder(entry.labels)
        });
      }

      return;
    }

    const blendedConfidence = clamp(
      existing.confidence * 0.78 + entry.confidence * 0.32,
      0,
      1
    );

    merged.set(entry.source, {
      source: entry.source,
      confidence: Number(Math.max(existing.confidence, blendedConfidence).toFixed(2)),
      labels: uniqueInOrder([...existing.labels, ...entry.labels])
    });
  });

  stemMetrics.forEach((stem) => {
    if (!stem.source || merged.has(stem.source)) {
      return;
    }

    if (stem.energyRatio < 0.035 && stem.rms < 0.0035) {
      return;
    }

    merged.set(stem.source, {
      source: stem.source,
      confidence: Number(
        clamp(stem.energyRatio * 0.8 + stem.rms * 3.5, 0.1, 0.72).toFixed(2)
      ),
      labels: [stem.stem]
    });
  });

  return [...merged.values()]
    .sort((left, right) => right.confidence - left.confidence)
    .slice(0, 5);
}

function enrichAnalysisResult(
  result: AnalysisResult,
  {
    detectedSources,
    stemConnected,
    stemModel,
    stemMetrics
  }: {
    detectedSources: NonNullable<AnalysisResult['detectedSources']>;
    stemConnected: boolean;
    stemModel?: string;
    stemMetrics: NonNullable<AnalysisResult['stemMetrics']>;
  }
): AnalysisResult {
  const mergedResult = mergeDetectedSources(result, detectedSources);

  return {
    ...mergedResult,
    stemMetrics,
    stemService: {
      connected: stemConnected,
      provider: stemConnected ? 'stem-service' : 'browser-fallback',
      model: stemModel
    },
    sourceEqRecommendations: buildSourceAwareEqRecommendations({
      detectedSources,
      issues: mergedResult.issues ?? [],
      stemMetrics
    })
  };
}

export function useMonitoring(onUpdate: (result: AnalysisResult) => void) {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [latestFeatures, setLatestFeatures] = useState<ExtractedAudioFeatures | null>(null);
  const [permissionState, setPermissionState] = useState<MicrophonePermissionState>('prompt');
  const [permissionError, setPermissionError] = useState<MicrophoneErrorCode>(null);
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);

  const intervalRef = useRef<number | null>(null);
  const setupPromiseRef = useRef<Promise<boolean> | null>(null);
  const permissionStatusRef = useRef<PermissionStatus | null>(null);
  const workletModuleUrlRef = useRef<string | null>(null);
  const analysisInFlightRef = useRef(false);
  const captureSessionIdRef = useRef(0);

  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const outputGainRef = useRef<GainNode | null>(null);
  const captureNodeRef = useRef<CaptureNode | null>(null);

  const rollingBufferRef = useRef(new Float32Array(ROLLING_BUFFER_SIZE));
  const rollingBufferWriteIndexRef = useRef(0);
  const rollingBufferLengthRef = useRef(0);
  const nativeRollingBufferRef = useRef(new Float32Array(48_000 * ROLLING_BUFFER_SECONDS));
  const nativeRollingBufferWriteIndexRef = useRef(0);
  const nativeRollingBufferLengthRef = useRef(0);
  const nativeSampleRateRef = useRef(48_000);

  const clearMonitoringInterval = useCallback(() => {
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const handleIncomingSamples = useCallback((samples: Float32Array, sourceSampleRate: number) => {
    const nativeTargetLength = Math.max(1, Math.round(sourceSampleRate * ROLLING_BUFFER_SECONDS));

    if (
      nativeRollingBufferRef.current.length !== nativeTargetLength
      || nativeSampleRateRef.current !== sourceSampleRate
    ) {
      nativeRollingBufferRef.current = new Float32Array(nativeTargetLength);
      nativeRollingBufferWriteIndexRef.current = 0;
      nativeRollingBufferLengthRef.current = 0;
      nativeSampleRateRef.current = sourceSampleRate;
    }

    const nativeState = appendToCircularBuffer(
      nativeRollingBufferRef.current,
      samples,
      nativeRollingBufferWriteIndexRef.current,
      nativeRollingBufferLengthRef.current
    );

    nativeRollingBufferWriteIndexRef.current = nativeState.writeIndex;
    nativeRollingBufferLengthRef.current = nativeState.filledLength;

    const normalizedSamples = resampleMonoBuffer(samples, sourceSampleRate, TARGET_SAMPLE_RATE);
    const nextState = appendToCircularBuffer(
      rollingBufferRef.current,
      normalizedSamples,
      rollingBufferWriteIndexRef.current,
      rollingBufferLengthRef.current
    );

    rollingBufferWriteIndexRef.current = nextState.writeIndex;
    rollingBufferLengthRef.current = nextState.filledLength;
  }, []);

  const getBufferedAudio = useCallback((): BufferedAudioSnapshot => {
    const samples = readCircularBuffer(
      rollingBufferRef.current,
      rollingBufferWriteIndexRef.current,
      rollingBufferLengthRef.current
    );

    return createBufferedAudioSnapshot(samples);
  }, []);

  const getNativeBufferedAudio = useCallback((): BufferedAudioSnapshot => {
    const samples = readCircularBuffer(
      nativeRollingBufferRef.current,
      nativeRollingBufferWriteIndexRef.current,
      nativeRollingBufferLengthRef.current
    );

    return createBufferedAudioSnapshot(samples, nativeSampleRateRef.current);
  }, []);

  const extractCurrentFeatures = useCallback((snapshot?: BufferedAudioSnapshot) => {
    const bufferedSnapshot = snapshot ?? getBufferedAudio();
    const extractedFeatures = extractAudioFeatures(bufferedSnapshot, {
      melBinCount: MODEL_MEL_BIN_COUNT
    });

    setLatestFeatures(extractedFeatures);
    logExtractedAudioFeatures(extractedFeatures);

    return extractedFeatures;
  }, [getBufferedAudio]);

  const analyseCurrentBuffer = useCallback(async (): Promise<AnalysisResult> => {
    const snapshot = getBufferedAudio();
    const nativeSnapshot = getNativeBufferedAudio();
    const extractedFeatures = extractCurrentFeatures(snapshot);
    const [mlResult, stemAnalysis] = await Promise.all([
      analyzeWithMlInference(extractedFeatures),
      detectStemSeparatedSources(nativeSnapshot)
    ]);
    const shouldRunFallbackTagger = !stemAnalysis.connected || stemAnalysis.detectedSources.length < 3;
    const fallbackDetectedSources = shouldRunFallbackTagger
      ? await detectOpenSourceAudioSources(nativeSnapshot)
      : [];
    const detectedSources = mergeInstrumentDetections(
      stemAnalysis.detectedSources,
      fallbackDetectedSources,
      stemAnalysis.stems
    );

    if (mlResult) {
      return enrichAnalysisResult(mlResult, {
        detectedSources,
        stemConnected: stemAnalysis.connected,
        stemModel: stemAnalysis.model,
        stemMetrics: stemAnalysis.stems
      });
    }

    return enrichAnalysisResult(
      buildAnalysisResult(snapshot, extractedFeatures, 'rule-based-fallback'),
      {
        detectedSources,
        stemConnected: stemAnalysis.connected,
        stemModel: stemAnalysis.model,
        stemMetrics: stemAnalysis.stems
      }
    );
  }, [extractCurrentFeatures, getBufferedAudio, getNativeBufferedAudio]);

  const runMonitoringPass = useCallback(async () => {
    if (analysisInFlightRef.current) {
      return false;
    }

    const activeSessionId = captureSessionIdRef.current;
    analysisInFlightRef.current = true;

    try {
      const nextResult = await analyseCurrentBuffer();

      if (activeSessionId !== captureSessionIdRef.current) {
        return false;
      }

      onUpdate(nextResult);
      return true;
    } finally {
      analysisInFlightRef.current = false;
    }
  }, [analyseCurrentBuffer, onUpdate]);

  const stopCapture = useCallback(() => {
    captureSessionIdRef.current += 1;
    setIsCapturing(false);
    setAnalyserNode(null);

    if (captureNodeRef.current) {
      if ('port' in captureNodeRef.current) {
        captureNodeRef.current.port.onmessage = null;
      } else {
        captureNodeRef.current.onaudioprocess = null;
      }

      captureNodeRef.current.disconnect();
      captureNodeRef.current = null;
    }

    analyserRef.current?.disconnect();
    analyserRef.current = null;

    mediaSourceRef.current?.disconnect();
    mediaSourceRef.current = null;

    outputGainRef.current?.disconnect();
    outputGainRef.current = null;

    mediaStreamRef.current?.getTracks().forEach(track => track.stop());
    mediaStreamRef.current = null;

    const audioContext = audioContextRef.current;
    audioContextRef.current = null;

    if (audioContext && audioContext.state !== 'closed') {
      void audioContext.close().catch(() => undefined);
    }
  }, []);

  const createScriptProcessorCaptureNode = useCallback((context: AudioContext) => {
    const scriptProcessor = context.createScriptProcessor(2048, 1, 1);

    scriptProcessor.onaudioprocess = (event) => {
      const inputChannel = event.inputBuffer.getChannelData(0);
      handleIncomingSamples(inputChannel, context.sampleRate);
      event.outputBuffer.getChannelData(0).fill(0);
    };

    return scriptProcessor;
  }, [handleIncomingSamples]);

  const getWorkletModuleUrl = useCallback(() => {
    if (!workletModuleUrlRef.current) {
      workletModuleUrlRef.current = URL.createObjectURL(
        new Blob([CAPTURE_WORKLET_SOURCE], { type: 'application/javascript' })
      );
    }

    return workletModuleUrlRef.current;
  }, []);

  const createAudioWorkletCaptureNode = useCallback(async (context: AudioContext) => {
    await context.audioWorklet.addModule(getWorkletModuleUrl());

    const workletNode = new AudioWorkletNode(context, CAPTURE_WORKLET_NAME, {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1]
    });

    workletNode.port.onmessage = (event) => {
      const message = event.data as { type?: string; samples?: ArrayBuffer };

      if (message.type !== 'samples' || !message.samples) {
        return;
      }

      handleIncomingSamples(new Float32Array(message.samples), context.sampleRate);
    };

    return workletNode;
  }, [getWorkletModuleUrl, handleIncomingSamples]);

  const ensureMicrophoneReady = useCallback(async () => {
    const AudioContextClass = getAudioContextConstructor();

    if (!navigator.mediaDevices?.getUserMedia || !AudioContextClass) {
      setPermissionState('unsupported');
      setPermissionError('unsupported');
      return false;
    }

    if (setupPromiseRef.current) {
      return setupPromiseRef.current;
    }

    if (audioContextRef.current && mediaStreamRef.current && analyserRef.current) {
      await audioContextRef.current.resume().catch(() => undefined);
      setPermissionState('granted');
      setPermissionError(null);
      setIsCapturing(true);
      setAnalyserNode(analyserRef.current);
      return true;
    }

    const setupPromise = (async () => {
      stopCapture();

      const clearedBuffer = rollingBufferRef.current;
      clearedBuffer.fill(0);
      rollingBufferWriteIndexRef.current = 0;
      rollingBufferLengthRef.current = 0;
      nativeRollingBufferRef.current.fill(0);
      nativeRollingBufferWriteIndexRef.current = 0;
      nativeRollingBufferLengthRef.current = 0;

      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: { ideal: 1 },
            // Do NOT constrain sampleRate here — let the browser capture at its
            // native rate (44.1 / 48 kHz) so the AnalyserNode has full frequency
            // coverage up to 20 kHz.  Resampling to TARGET_SAMPLE_RATE happens
            // later in handleIncomingSamples before ML feature extraction.
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false
          },
          video: false
        });

        let audioContext: AudioContext;

        try {
          // 'balanced' is better than 'interactive' for analysis: we don't need
          // the lowest possible latency, so the browser can use larger buffers
          // which reduces CPU overhead and jitter.
          audioContext = new AudioContextClass({ latencyHint: 'balanced' });
        } catch {
          audioContext = new AudioContextClass();
        }

        await audioContext.resume().catch(() => undefined);

        const mediaSource = audioContext.createMediaStreamSource(mediaStream);
        const analyser = audioContext.createAnalyser();
        // 4096 bins → ~11 Hz/bin at 48 kHz; much better low-freq resolution.
        analyser.fftSize = 4096;
        // Disable built-in smoothing entirely — EQVisualization applies its own
        // asymmetric attack/release ballistics which is more accurate.
        analyser.smoothingTimeConstant = 0.0;
        analyser.minDecibels = -90;
        // -3 dBFS headroom: was -12, which clipped any signal above -12 dBFS
        // and made loud mixes look artificially flat at the top of the spectrum.
        analyser.maxDecibels = -3;

        const outputGain = audioContext.createGain();
        outputGain.gain.value = 0;
        outputGain.connect(audioContext.destination);

        mediaSource.connect(analyser);
        analyser.connect(outputGain);

        let captureNode: CaptureNode;

        if (audioContext.audioWorklet) {
          try {
            captureNode = await createAudioWorkletCaptureNode(audioContext);
          } catch {
            captureNode = createScriptProcessorCaptureNode(audioContext);
          }
        } else {
          captureNode = createScriptProcessorCaptureNode(audioContext);
        }

        mediaSource.connect(captureNode);
        captureNode.connect(outputGain);

        audioContextRef.current = audioContext;
        mediaStreamRef.current = mediaStream;
        mediaSourceRef.current = mediaSource;
        analyserRef.current = analyser;
        outputGainRef.current = outputGain;
        captureNodeRef.current = captureNode;

        setPermissionState('granted');
        setPermissionError(null);
        setIsCapturing(true);
        setAnalyserNode(analyser);

        return true;
      } catch (error) {
        stopCapture();

        const browserError = error as DOMException;

        if (browserError.name === 'NotAllowedError' || browserError.name === 'SecurityError') {
          setPermissionState('denied');
          setPermissionError('denied');
          return false;
        }

        if (browserError.name === 'NotFoundError' || browserError.name === 'DevicesNotFoundError') {
          setPermissionState('prompt');
          setPermissionError('not_found');
          return false;
        }

        setPermissionState('prompt');
        setPermissionError('unknown');
        return false;
      } finally {
        setupPromiseRef.current = null;
      }
    })();

    setupPromiseRef.current = setupPromise;
    return setupPromise;
  }, [createAudioWorkletCaptureNode, createScriptProcessorCaptureNode, stopCapture]);

  const startMonitoring = useCallback(async () => {
    const isReady = await ensureMicrophoneReady();

    if (!isReady) {
      return false;
    }

    clearMonitoringInterval();
    setIsMonitoring(true);
    await runMonitoringPass();

    intervalRef.current = window.setInterval(() => {
      void runMonitoringPass();
    }, MONITORING_INTERVAL_MS);

    return true;
  }, [clearMonitoringInterval, ensureMicrophoneReady, runMonitoringPass]);

  const stopMonitoring = useCallback(() => {
    clearMonitoringInterval();
    setIsMonitoring(false);
    stopCapture();
  }, [clearMonitoringInterval, stopCapture]);

  useEffect(() => {
    void warmUpMlInference();
    void warmUpStemSeparationService();
    void warmUpOpenSourceAudioTagger();
  }, []);

  useEffect(() => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setPermissionState('unsupported');
      setPermissionError('unsupported');
      return;
    }

    if (!navigator.permissions?.query) {
      return;
    }

    let isDisposed = false;

    void navigator.permissions
      .query({ name: 'microphone' as PermissionName })
      .then((status) => {
        if (isDisposed) {
          return;
        }

        permissionStatusRef.current = status;
        setPermissionState(normalizePermissionState(status.state));
        setPermissionError(status.state === 'denied' ? 'denied' : null);

        status.onchange = () => {
          setPermissionState(normalizePermissionState(status.state));

          if (status.state === 'granted') {
            setPermissionError(null);
          } else if (status.state === 'denied') {
            setPermissionError('denied');
          }
        };
      })
      .catch(() => undefined);

    return () => {
      isDisposed = true;

      if (permissionStatusRef.current) {
        permissionStatusRef.current.onchange = null;
        permissionStatusRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    return () => {
      clearMonitoringInterval();
      stopCapture();

      if (workletModuleUrlRef.current) {
        URL.revokeObjectURL(workletModuleUrlRef.current);
        workletModuleUrlRef.current = null;
      }
    };
  }, [clearMonitoringInterval, stopCapture]);

  return {
    analyserNode,
    analyseCurrentBuffer,
    extractCurrentFeatures,
    getBufferedAudio,
    isCapturing,
    isMonitoring,
    latestFeatures,
    permissionError,
    permissionState,
    requestMicrophoneAccess: ensureMicrophoneReady,
    startMonitoring,
    stopCapture,
    stopMonitoring,
    targetSampleRate: TARGET_SAMPLE_RATE
  };
}

function uniqueInOrder<T>(values: T[]) {
  return values.filter((value, index) => values.indexOf(value) === index);
}
