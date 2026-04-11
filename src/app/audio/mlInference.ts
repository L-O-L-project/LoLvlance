import type {
  AnalysisResult,
  DiagnosticProblem,
  EQRecommendation,
  ExtractedAudioFeatures,
  Instrument,
  ProblemType
} from '../types';

const MODEL_MEL_BIN_COUNT = 64;
const MODEL_SILENCE_RMS_THRESHOLD = 0.012;
const MODEL_MIN_HZ = 80;
const MODEL_MAX_HZ = 8000;
const INSTRUMENT_THRESHOLD = 0.45;

const PROBLEM_LABELS = ['muddy', 'harsh', 'buried', 'normal'] as const;
const INSTRUMENT_LABELS = ['vocal', 'guitar', 'bass', 'drums', 'keys'] as const;

type MlProblemLabel = (typeof PROBLEM_LABELS)[number];
type OrtModule = typeof import('onnxruntime-web');

interface MlInferenceSnapshot {
  problemType: MlProblemLabel;
  problemConfidence: number;
  problemProbabilities: number[];
  instrumentProbabilities: number[];
  instruments: Instrument[];
  eqFrequencyNormalized: number;
  eqFrequencyHz: number;
  eqGainDb: number;
  eqRecommendation: EQRecommendation | null;
}

let ortModulePromise: Promise<OrtModule> | null = null;
let sessionPromise: Promise<import('onnxruntime-web').InferenceSession> | null = null;
let hasLoggedModelReady = false;

const modelProblemProfiles: Record<
  Exclude<MlProblemLabel, 'normal'>,
  {
    details: DiagnosticProblem['details'];
    fallbackReason: string;
  }
> = {
  muddy: {
    details: ['low_mid_overlap', 'low_frequency_buildup'],
    fallbackReason: 'model suggests reducing low-mid buildup'
  },
  harsh: {
    details: ['high_frequency_spike', 'guitar_presence_peak'],
    fallbackReason: 'model suggests taming aggressive upper-mid energy'
  },
  buried: {
    details: ['lack_of_presence', 'mid_range_masking'],
    fallbackReason: 'model suggests restoring presence and clarity'
  }
};

export { MODEL_MEL_BIN_COUNT };

export async function warmUpMlInference() {
  try {
    await getInferenceSession();

    if (!hasLoggedModelReady) {
      hasLoggedModelReady = true;
      console.info('[audio-ml]', {
        status: 'ready',
        model: getModelUrl()
      });
    }

    return true;
  } catch (error) {
    console.warn('[audio-ml] Model warm-up skipped.', error);
    return false;
  }
}

export async function analyzeWithMlInference(
  features: ExtractedAudioFeatures
): Promise<AnalysisResult | null> {
  if (features.melBinCount !== MODEL_MEL_BIN_COUNT) {
    console.warn('[audio-ml] Skipping inference because the mel-bin count does not match the model.', {
      expectedMelBins: MODEL_MEL_BIN_COUNT,
      receivedMelBins: features.melBinCount
    });
    return null;
  }

  if (features.rms < MODEL_SILENCE_RMS_THRESHOLD) {
    return {
      problems: [],
      issues: [],
      eq_recommendations: [],
      engine: 'ml',
      timestamp: Date.now()
    };
  }

  try {
    const [timeSteps, melBins] = features.logMelSpectrogramShape;
    const ort = await getOrtModule();
    const session = await getInferenceSession();
    const feeds = {
      log_mel_spectrogram: new ort.Tensor('float32', features.logMelSpectrogram, [1, timeSteps, melBins])
    };
    const outputs = await session.run(feeds);
    const snapshot = parseInferenceOutputs(outputs);

    logMlInference(snapshot);
    return mlInferenceToAnalysisResult(snapshot);
  } catch (error) {
    console.warn('[audio-ml] Inference failed. Falling back to the rule-based engine.', error);
    return null;
  }
}

async function getOrtModule() {
  if (!ortModulePromise) {
    ortModulePromise = import('onnxruntime-web');
  }

  return ortModulePromise;
}

async function getInferenceSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await getOrtModule();

      return ort.InferenceSession.create(getModelUrl(), {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
    })().catch((error) => {
      sessionPromise = null;
      throw error;
    });
  }

  return sessionPromise;
}

function getModelUrl() {
  return new URL(`${import.meta.env.BASE_URL}models/lightweight_audio_model.onnx`, window.location.origin).toString();
}

function parseInferenceOutputs(
  outputs: Awaited<ReturnType<import('onnxruntime-web').InferenceSession['run']>>
): MlInferenceSnapshot {
  const problemProbabilities = readTensorData(outputs.problem_probs, 'problem_probs');
  const instrumentProbabilities = readTensorData(outputs.instrument_probs, 'instrument_probs');
  const eqFrequencyNormalized = clamp(readScalar(outputs.eq_freq, 'eq_freq'), 0, 1);
  const eqGainDb = clamp(readScalar(outputs.eq_gain_db, 'eq_gain_db'), -6, 6);
  const problemIndex = argMax(problemProbabilities);
  const problemType = PROBLEM_LABELS[problemIndex] ?? 'normal';
  const problemConfidence = problemProbabilities[problemIndex] ?? 0;
  const instruments = selectInstruments(instrumentProbabilities);
  const eqFrequencyHz = normalizedToFrequency(eqFrequencyNormalized);

  return {
    problemType,
    problemConfidence,
    problemProbabilities,
    instrumentProbabilities,
    instruments,
    eqFrequencyNormalized,
    eqFrequencyHz,
    eqGainDb,
    eqRecommendation:
      problemType === 'normal'
        ? null
        : {
            freq_range: formatEqRange(eqFrequencyHz),
            gain: formatGain(eqGainDb),
            reason: buildRecommendationReason(problemType, eqGainDb)
          }
  };
}

function mlInferenceToAnalysisResult(snapshot: MlInferenceSnapshot): AnalysisResult {
  if (snapshot.problemType === 'normal' || !snapshot.eqRecommendation) {
    return {
      problems: [],
      issues: [],
      eq_recommendations: [],
      engine: 'ml',
      timestamp: Date.now()
    };
  }

  const profile = modelProblemProfiles[snapshot.problemType];
  const sources = snapshot.instruments.length > 0 ? snapshot.instruments : ['overall'];
  const actions = [
    `${snapshot.eqRecommendation.freq_range} ${snapshot.eqRecommendation.gain}`,
    snapshot.eqRecommendation.reason
  ];

  return {
    problems: [
      {
        type: snapshot.problemType as ProblemType,
        confidence: Number(snapshot.problemConfidence.toFixed(2)),
        sources,
        details: profile.details,
        actions
      }
    ],
    issues: [snapshot.problemType],
    eq_recommendations: [snapshot.eqRecommendation],
    engine: 'ml',
    timestamp: Date.now()
  };
}

function buildRecommendationReason(problemType: Exclude<MlProblemLabel, 'normal'>, gainDb: number) {
  const profile = modelProblemProfiles[problemType];
  const roundedGain = Math.abs(gainDb);

  if (roundedGain < 0.5) {
    return `${profile.fallbackReason} with a subtle adjustment`;
  }

  return profile.fallbackReason;
}

function selectInstruments(instrumentProbabilities: number[]) {
  const predicted = INSTRUMENT_LABELS
    .map((label, index) => ({
      label,
      probability: instrumentProbabilities[index] ?? 0
    }))
    .filter((entry) => entry.probability >= INSTRUMENT_THRESHOLD)
    .map((entry) => entry.label as Instrument);

  if (predicted.length > 0) {
    return predicted;
  }

  const strongestIndex = argMax(instrumentProbabilities);
  const strongestProbability = instrumentProbabilities[strongestIndex] ?? 0;

  if (strongestProbability > 0) {
    return [INSTRUMENT_LABELS[strongestIndex] as Instrument];
  }

  return [];
}

function readTensorData(value: unknown, outputName: string) {
  if (!value || typeof value !== 'object' || !('data' in value)) {
    throw new Error(`Missing tensor output: ${outputName}`);
  }

  const data = (value as { data: ArrayLike<number> }).data;
  return Array.from(data, (entry) => Number(entry));
}

function readScalar(value: unknown, outputName: string) {
  const values = readTensorData(value, outputName);
  return values[0] ?? 0;
}

function normalizedToFrequency(value: number) {
  const clamped = clamp(value, 0, 1);
  return MODEL_MIN_HZ * ((MODEL_MAX_HZ / MODEL_MIN_HZ) ** clamped);
}

function formatEqRange(centerFrequencyHz: number) {
  const lower = roundFrequency(centerFrequencyHz / Math.SQRT2);
  const upper = roundFrequency(centerFrequencyHz * Math.SQRT2);
  return `${lower}-${upper}Hz`;
}

function roundFrequency(value: number) {
  if (value < 200) {
    return Math.max(20, Math.round(value / 5) * 5);
  }

  if (value < 1000) {
    return Math.round(value / 10) * 10;
  }

  return Math.round(value / 50) * 50;
}

function formatGain(gainDb: number) {
  const rounded = Math.round(gainDb * 10) / 10;
  return `${rounded > 0 ? '+' : ''}${rounded.toFixed(Math.abs(rounded % 1) < 0.05 ? 0 : 1)}dB`;
}

function argMax(values: number[]) {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < values.length; index += 1) {
    if (values[index] > bestValue) {
      bestValue = values[index];
      bestIndex = index;
    }
  }

  return bestIndex;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function logMlInference(snapshot: MlInferenceSnapshot) {
  console.info('[audio-ml]', {
    problem: {
      label: snapshot.problemType,
      confidence: Number(snapshot.problemConfidence.toFixed(4)),
      probabilities: snapshot.problemProbabilities.map((value) => Number(value.toFixed(4)))
    },
    instruments: Object.fromEntries(
      INSTRUMENT_LABELS.map((label, index) => [
        label,
        Number((snapshot.instrumentProbabilities[index] ?? 0).toFixed(4))
      ])
    ),
    eq: {
      freqNormalized: Number(snapshot.eqFrequencyNormalized.toFixed(4)),
      freqHz: Math.round(snapshot.eqFrequencyHz),
      gainDb: Number(snapshot.eqGainDb.toFixed(3))
    }
  });
}
