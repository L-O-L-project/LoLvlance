import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  AnalysisResult,
  BufferedAudioSnapshot,
  MicrophoneErrorCode,
  MicrophonePermissionState
} from '../types';
import { generateMockProblems } from '../data/diagnosticData';
import {
  appendToCircularBuffer,
  createBufferedAudioSnapshot,
  readCircularBuffer,
  ROLLING_BUFFER_SIZE,
  TARGET_SAMPLE_RATE,
  resampleMonoBuffer
} from '../audio/audioUtils';

const MONITORING_INTERVAL_MS = 4000;
const SILENCE_THRESHOLD = 0.012;
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

function buildAnalysisResult(snapshot: BufferedAudioSnapshot): AnalysisResult {
  if (
    snapshot.samples.length < TARGET_SAMPLE_RATE / 2 ||
    snapshot.rms < SILENCE_THRESHOLD ||
    snapshot.peak < SILENCE_THRESHOLD * 4
  ) {
    return {
      problems: [],
      timestamp: Date.now()
    };
  }

  const problemCount = snapshot.peak > 0.85
    ? 3
    : snapshot.rms > 0.08 || snapshot.peak > 0.55
      ? 2
      : 1;

  return {
    problems: generateMockProblems(problemCount),
    timestamp: Date.now()
  };
}

export function useMonitoring(onUpdate: (result: AnalysisResult) => void) {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [permissionState, setPermissionState] = useState<MicrophonePermissionState>('prompt');
  const [permissionError, setPermissionError] = useState<MicrophoneErrorCode>(null);
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);

  const intervalRef = useRef<number | null>(null);
  const setupPromiseRef = useRef<Promise<boolean> | null>(null);
  const permissionStatusRef = useRef<PermissionStatus | null>(null);
  const workletModuleUrlRef = useRef<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const outputGainRef = useRef<GainNode | null>(null);
  const captureNodeRef = useRef<CaptureNode | null>(null);

  const rollingBufferRef = useRef(new Float32Array(ROLLING_BUFFER_SIZE));
  const rollingBufferWriteIndexRef = useRef(0);
  const rollingBufferLengthRef = useRef(0);

  const clearMonitoringInterval = useCallback(() => {
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const handleIncomingSamples = useCallback((samples: Float32Array, sourceSampleRate: number) => {
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

  const analyseCurrentBuffer = useCallback((): AnalysisResult => {
    return buildAnalysisResult(getBufferedAudio());
  }, [getBufferedAudio]);

  const stopCapture = useCallback(() => {
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

      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: { ideal: 1 },
            sampleRate: { ideal: TARGET_SAMPLE_RATE },
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false
          },
          video: false
        });

        let audioContext: AudioContext;

        try {
          audioContext = new AudioContextClass({ latencyHint: 'interactive' });
        } catch {
          audioContext = new AudioContextClass();
        }

        await audioContext.resume().catch(() => undefined);

        const mediaSource = audioContext.createMediaStreamSource(mediaStream);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        analyser.smoothingTimeConstant = 0.82;
        analyser.minDecibels = -90;
        analyser.maxDecibels = -12;

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
    onUpdate(analyseCurrentBuffer());

    intervalRef.current = window.setInterval(() => {
      onUpdate(analyseCurrentBuffer());
    }, MONITORING_INTERVAL_MS);

    return true;
  }, [analyseCurrentBuffer, clearMonitoringInterval, ensureMicrophoneReady, onUpdate]);

  const stopMonitoring = useCallback(() => {
    clearMonitoringInterval();
    setIsMonitoring(false);
    stopCapture();
  }, [clearMonitoringInterval, stopCapture]);

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
    getBufferedAudio,
    isCapturing,
    isMonitoring,
    permissionError,
    permissionState,
    requestMicrophoneAccess: ensureMicrophoneReady,
    startMonitoring,
    stopCapture,
    stopMonitoring,
    targetSampleRate: TARGET_SAMPLE_RATE
  };
}
