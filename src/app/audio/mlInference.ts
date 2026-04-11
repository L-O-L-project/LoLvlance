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
const MULTI_LABEL_THRESHOLD = 0.5;

const PROBLEM_LABELS = ['muddy', 'harsh', 'buried'] as const;
const LEGACY_PROBLEM_LABELS = ['muddy', 'harsh', 'buried', 'normal'] as const;
const INSTRUMENT_LABELS = ['vocal', 'guitar', 'bass', 'drums', 'keys'] as const;

type MlProblemLabel = (typeof PROBLEM_LABELS)[number];
type OrtModule = typeof import('onnxruntime-web');

interface PredictedProblem {
  type: MlProblemLabel;
  confidence: number;
  eqFrequencyHz: number;
  eqGainDb: number;
  eqRecommendation: EQRecommendation;
}

interface MlInferenceSnapshot {
  problemProbabilities: number[];
  instrumentProbabilities: number[];
  instruments: Instrument[];
  predictedProblems: PredictedProblem[];
  usesLegacyHead: boolean;
  legacyNormalProbability: number;
}

let ortModulePromise: Promise<OrtModule> | null = null;
let sessionPromise: Promise<import('onnxruntime-web').InferenceSession> | null = null;
let hasLoggedModelReady = false;

const modelProblemProfiles: Record<
  MlProblemLabel,
  {
    details: DiagnosticProblem['details'];
    fallbackReason: string;
    fallbackEq: {
      frequencyHz: number;
      gainDb: number;
    };
  }
> = {
  muddy: {
    details: ['low_mid_overlap', 'low_frequency_buildup'],
    fallbackReason: 'model suggests reducing low-mid buildup',
    fallbackEq: {
      frequencyHz: 315,
      gainDb: -3
    }
  },
  harsh: {
    details: ['high_frequency_spike', 'guitar_presence_peak'],
    fallbackReason: 'model suggests taming aggressive upper-mid energy',
    fallbackEq: {
      frequencyHz: 5600,
      gainDb: -2.5
    }
  },
  buried: {
    details: ['lack_of_presence', 'mid_range_masking'],
    fallbackReason: 'model suggests restoring presence and clarity',
    fallbackEq: {
      frequencyHz: 3000,
      gainDb: 2.5
    }
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
  const outputMap = outputs as Record<string, unknown>;
  const rawProblemProbabilities = readTensorData(outputMap.problem_probs, 'problem_probs');
  const usesLegacyHead = rawProblemProbabilities.length >= LEGACY_PROBLEM_LABELS.length;
  const legacyNormalProbability = usesLegacyHead ? rawProblemProbabilities[3] ?? 0 : 0;
  const problemProbabilities = rawProblemProbabilities.slice(0, PROBLEM_LABELS.length);
  const instrumentProbabilities = readOptionalTensorData(outputMap.instrument_probs, INSTRUMENT_LABELS.length);
  const eqFrequencyNormalized = readOptionalScalar(outputMap.eq_freq);
  const eqGainDb = readOptionalScalar(outputMap.eq_gain_db);
  const instruments = selectInstruments(instrumentProbabilities);
  const predictedProblems = usesLegacyHead
    ? deriveLegacyProblems(problemProbabilities, legacyNormalProbability, eqFrequencyNormalized, eqGainDb)
    : deriveMultiLabelProblems(problemProbabilities);

  return {
    problemProbabilities,
    instrumentProbabilities,
    instruments,
    predictedProblems,
    usesLegacyHead,
    legacyNormalProbability
  };
}

function deriveLegacyProblems(
  problemProbabilities: number[],
  legacyNormalProbability: number,
  eqFrequencyNormalized: number | null,
  eqGainDb: number | null
) {
  const paddedProbabilities = [...problemProbabilities, legacyNormalProbability];
  const topIndex = argMax(paddedProbabilities);

  if (topIndex >= PROBLEM_LABELS.length) {
    return [];
  }

  const problemType = PROBLEM_LABELS[topIndex];
  const profile = modelProblemProfiles[problemType];
  const frequencyHz = eqFrequencyNormalized === null
    ? profile.fallbackEq.frequencyHz
    : normalizedToFrequency(eqFrequencyNormalized);
  const gainDb = eqGainDb === null ? profile.fallbackEq.gainDb : eqGainDb;

  return [
    buildPredictedProblem(problemType, problemProbabilities[topIndex] ?? 0, frequencyHz, gainDb)
  ];
}

function deriveMultiLabelProblems(problemProbabilities: number[]) {
  return PROBLEM_LABELS
    .map((problemType, index) => ({
      problemType,
      confidence: problemProbabilities[index] ?? 0
    }))
    .filter((entry) => entry.confidence >= MULTI_LABEL_THRESHOLD)
    .sort((left, right) => right.confidence - left.confidence)
    .map((entry) => {
      const profile = modelProblemProfiles[entry.problemType];

      return buildPredictedProblem(
        entry.problemType,
        entry.confidence,
        profile.fallbackEq.frequencyHz,
        profile.fallbackEq.gainDb
      );
    });
}

function buildPredictedProblem(
  problemType: MlProblemLabel,
  confidence: number,
  frequencyHz: number,
  gainDb: number
): PredictedProblem {
  return {
    type: problemType,
    confidence,
    eqFrequencyHz: frequencyHz,
    eqGainDb: gainDb,
    eqRecommendation: {
      freq_range: formatEqRange(frequencyHz),
      gain: formatGain(gainDb),
      reason: buildRecommendationReason(problemType, gainDb)
    }
  };
}

function mlInferenceToAnalysisResult(snapshot: MlInferenceSnapshot): AnalysisResult {
  if (snapshot.predictedProblems.length === 0) {
    return {
      problems: [],
      issues: [],
      eq_recommendations: [],
      engine: 'ml',
      timestamp: Date.now()
    };
  }

  const sources = snapshot.instruments.length > 0 ? snapshot.instruments : ['overall'];
  const problems = snapshot.predictedProblems.map((prediction) => {
    const profile = modelProblemProfiles[prediction.type];

    return {
      type: prediction.type as ProblemType,
      confidence: Number(prediction.confidence.toFixed(2)),
      sources,
      details: profile.details,
      actions: [
        `${prediction.eqRecommendation.freq_range} ${prediction.eqRecommendation.gain}`,
        prediction.eqRecommendation.reason
      ]
    };
  });

  return {
    problems,
    issues: problems.map((problem) => problem.type as ProblemType) as AnalysisResult['issues'],
    eq_recommendations: snapshot.predictedProblems.map((problem) => problem.eqRecommendation),
    engine: 'ml',
    timestamp: Date.now()
  };
}

function buildRecommendationReason(problemType: MlProblemLabel, gainDb: number) {
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

function readOptionalTensorData(value: unknown, expectedLength: number) {
  if (!value || typeof value !== 'object' || !('data' in value)) {
    return Array.from({ length: expectedLength }, () => 0);
  }

  const data = (value as { data: ArrayLike<number> }).data;
  return Array.from(data, (entry) => Number(entry));
}

function readOptionalScalar(value: unknown) {
  if (!value || typeof value !== 'object' || !('data' in value)) {
    return null;
  }

  const data = (value as { data: ArrayLike<number> }).data;
  const firstValue = Array.from(data, (entry) => Number(entry))[0];
  return Number.isFinite(firstValue) ? firstValue : null;
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
    problemProbabilities: Object.fromEntries(
      PROBLEM_LABELS.map((label, index) => [
        label,
        Number((snapshot.problemProbabilities[index] ?? 0).toFixed(4))
      ])
    ),
    predictedProblems: snapshot.predictedProblems.map((problem) => ({
      label: problem.type,
      confidence: Number(problem.confidence.toFixed(4)),
      eqFrequencyHz: Math.round(problem.eqFrequencyHz),
      eqGainDb: Number(problem.eqGainDb.toFixed(3))
    })),
    instruments: Object.fromEntries(
      INSTRUMENT_LABELS.map((label, index) => [
        label,
        Number((snapshot.instrumentProbabilities[index] ?? 0).toFixed(4))
      ])
    ),
    legacyHead: snapshot.usesLegacyHead,
    legacyNormalProbability: Number(snapshot.legacyNormalProbability.toFixed(4))
  });
}
