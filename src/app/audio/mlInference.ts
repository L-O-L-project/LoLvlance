import type {
  AnalysisResult,
  DetectedAudioSource,
  DiagnosticProblem,
  EQRecommendation,
  ExtractedAudioFeatures,
  ProblemType,
  RuleBasedIssue,
  SourceLabel,
  TrainableIssueLabel
} from '../types';
import { problemCauseMap } from '../data/diagnosticData';
import { buildMlInferenceOutput } from './diagnosisPostProcessing';
import {
  ISSUE_DEFAULT_THRESHOLDS,
  ISSUE_LABELS,
  ISSUE_PROFILES,
  ML_SCHEMA_VERSION,
  PRIMARY_UI_ISSUES,
  SOURCE_DEFAULT_THRESHOLDS,
  SOURCE_LABELS
} from './mlSchema';
import {
  ENABLE_MODEL,
  getConfiguredModelUrl,
  getModelRuntimeSnapshot,
  MODEL_VERSION
} from '../config/modelRuntime';
import ortWasmJsepModuleUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url';
import ortWasmJsepBinaryUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url';

const MODEL_MEL_BIN_COUNT = 64;
const MODEL_SILENCE_RMS_THRESHOLD = 0.012;
const MODEL_MIN_HZ = 80;
const MODEL_MAX_HZ = 8000;
const MIN_SOURCE_DISPLAY_SCORE = 0.25;

type OrtModule = typeof import('onnxruntime-web');

interface ParsedModelOutputs {
  issueScores: Record<TrainableIssueLabel, number>;
  sourceScores: Record<SourceLabel, number>;
  modelEqFrequencyNormalized: number;
  modelEqGainDb: number;
}

let ortModulePromise: Promise<OrtModule> | null = null;
let sessionPromise: Promise<import('onnxruntime-web').InferenceSession> | null = null;
let hasLoggedModelReady = false;

export { MODEL_MEL_BIN_COUNT };

export async function warmUpMlInference() {
  if (!ENABLE_MODEL) {
    console.info('[audio-ml]', createModelLogMetadata({
      status: 'disabled',
      reason: 'enableModel=false'
    }));
    return false;
  }

  try {
    await getInferenceSession();

    if (!hasLoggedModelReady) {
      hasLoggedModelReady = true;
      console.info('[audio-ml]', createModelLogMetadata({
        status: 'ready',
        model: getModelUrl(),
        schema: ML_SCHEMA_VERSION
      }));
    }

    return true;
  } catch (error) {
    console.warn('[audio-ml] Model warm-up skipped.', createModelLogMetadata(), error);
    return false;
  }
}

export async function analyzeWithMlInference(
  features: ExtractedAudioFeatures
): Promise<AnalysisResult | null> {
  if (!ENABLE_MODEL) {
    console.info('[audio-ml]', createModelLogMetadata({
      status: 'skipped',
      reason: 'enableModel=false'
    }));
    return null;
  }

  if (features.melBinCount !== MODEL_MEL_BIN_COUNT) {
    console.warn('[audio-ml] Skipping inference because the mel-bin count does not match the model.', {
      ...createModelLogMetadata(),
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
    const outputs = await session.run({
      log_mel_spectrogram: new ort.Tensor('float32', features.logMelSpectrogram, [1, timeSteps, melBins])
    });
    const parsedOutputs = parseModelOutputs(outputs);
    logRawModelOutputs(parsedOutputs);
    const mlOutput = buildMlInferenceOutput(parsedOutputs.issueScores, parsedOutputs.sourceScores);
    const detectedSources = buildDetectedSources(parsedOutputs.sourceScores);
    const result = buildAnalysisResult({
      mlOutput,
      detectedSources,
      modelEqFrequencyNormalized: parsedOutputs.modelEqFrequencyNormalized,
      modelEqGainDb: parsedOutputs.modelEqGainDb
    });

    logMlInference(result.ml_output);
    return result;
  } catch (error) {
    console.warn('[audio-ml] Inference failed. Falling back to the rule-based engine.', createModelLogMetadata(), error);
    return null;
  }
}

async function getOrtModule() {
  if (!ortModulePromise) {
    ortModulePromise = import('onnxruntime-web').then((ort) => {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = {
        mjs: ortWasmJsepModuleUrl,
        wasm: ortWasmJsepBinaryUrl
      };
      return ort;
    });
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
  return getConfiguredModelUrl(import.meta.env.BASE_URL, window.location.origin, MODEL_VERSION);
}

function parseModelOutputs(
  outputs: Awaited<ReturnType<import('onnxruntime-web').InferenceSession['run']>>
): ParsedModelOutputs {
  const outputMap = outputs as Record<string, unknown>;
  const rawIssueScores = readTensorData(outputMap.issue_probs, 'issue_probs', ISSUE_LABELS.length);
  const rawSourceScores = readTensorData(outputMap.source_probs, 'source_probs', SOURCE_LABELS.length);

  const issueScores = Object.fromEntries(
    ISSUE_LABELS.map((label, index) => [label, clamp(rawIssueScores[index] ?? 0, 0, 1)])
  ) as Record<TrainableIssueLabel, number>;
  const sourceScores = Object.fromEntries(
    SOURCE_LABELS.map((label, index) => [label, clamp(rawSourceScores[index] ?? 0, 0, 1)])
  ) as Record<SourceLabel, number>;

  return {
    issueScores,
    sourceScores,
    modelEqFrequencyNormalized: clamp(readScalar(outputMap.eq_freq, 'eq_freq'), 0, 1),
    modelEqGainDb: clamp(readScalar(outputMap.eq_gain_db, 'eq_gain_db'), -6, 6)
  };
}

function buildDetectedSources(sourceScores: Record<SourceLabel, number>) {
  const detected = SOURCE_LABELS
    .map((source) => ({
      source,
      confidence: sourceScores[source],
      labels: [`model:${source}`]
    }))
    .filter((entry) => entry.confidence >= SOURCE_DEFAULT_THRESHOLDS[entry.source] || entry.confidence >= MIN_SOURCE_DISPLAY_SCORE)
    .sort((left, right) => right.confidence - left.confidence)
    .map((entry) => ({
      source: entry.source,
      confidence: Number(entry.confidence.toFixed(2)),
      labels: entry.labels
    })) as DetectedAudioSource[];

  if (detected.length > 0) {
    return detected;
  }

  const strongestSource = SOURCE_LABELS
    .map((source) => ({ source, score: sourceScores[source] }))
    .sort((left, right) => right.score - left.score)[0];

  if (!strongestSource || strongestSource.score <= 0) {
    return [];
  }

  return [{
    source: strongestSource.source,
    confidence: Number(strongestSource.score.toFixed(2)),
    labels: [`model:${strongestSource.source}`]
  }];
}

function buildAnalysisResult({
  mlOutput,
  detectedSources,
  modelEqFrequencyNormalized,
  modelEqGainDb
}: {
  mlOutput: ReturnType<typeof buildMlInferenceOutput>;
  detectedSources: DetectedAudioSource[];
  modelEqFrequencyNormalized: number;
  modelEqGainDb: number;
}): AnalysisResult {
  const issueProblems = ISSUE_LABELS
    .filter((label) => (mlOutput.issues[label] ?? 0) >= ISSUE_DEFAULT_THRESHOLDS[label])
    .sort((left, right) => (mlOutput.issues[right] ?? 0) - (mlOutput.issues[left] ?? 0))
    .map((label, index) => buildIssueProblem({
      label,
      score: mlOutput.issues[label] ?? 0,
      detectedSources,
      useModelEq: index === 0,
      modelEqFrequencyNormalized,
      modelEqGainDb
    }));
  const derivedProblems = Object.entries(mlOutput.derived_diagnoses)
    .map(([label, diagnosis]) => buildDerivedProblem(label as ProblemType, diagnosis!))
    .filter((problem): problem is DiagnosticProblem => problem !== null);
  const eqRecommendations = issueProblems.map((problem) => problem.actions[0] ? buildRecommendationFromProblem(problem) : null)
    .filter((recommendation): recommendation is EQRecommendation => recommendation !== null);
  const primaryIssues = issueProblems
    .map((problem) => problem.type)
    .filter((type): type is RuleBasedIssue => PRIMARY_UI_ISSUES.includes(type as RuleBasedIssue));

  return {
    problems: [...issueProblems, ...derivedProblems],
    issues: primaryIssues,
    eq_recommendations: eqRecommendations,
    detectedSources,
    ml_output: mlOutput,
    engine: 'ml',
    timestamp: Date.now()
  };
}

function buildIssueProblem({
  label,
  score,
  detectedSources,
  useModelEq,
  modelEqFrequencyNormalized,
  modelEqGainDb
}: {
  label: TrainableIssueLabel;
  score: number;
  detectedSources: DetectedAudioSource[];
  useModelEq: boolean;
  modelEqFrequencyNormalized: number;
  modelEqGainDb: number;
}): DiagnosticProblem {
  const profile = ISSUE_PROFILES[label];
  const frequencyHz = useModelEq
    ? normalizedToFrequency(modelEqFrequencyNormalized)
    : profile.fallbackEq.frequencyHz;
  const gainDb = useModelEq
    ? modelEqGainDb
    : profile.fallbackEq.gainDb;
  const recommendation = formatRecommendation(frequencyHz, gainDb, profile.fallbackEq.reason);
  const sources = detectedSources.length > 0 ? detectedSources.map((entry) => entry.source) : ['overall'];

  return {
    type: label,
    confidence: Number(score.toFixed(2)),
    sources,
    details: profile.details,
    actions: [`${recommendation.freq_range} ${recommendation.gain}`, recommendation.reason]
  };
}

function buildDerivedProblem(label: ProblemType, diagnosis: NonNullable<ReturnType<typeof buildMlInferenceOutput>['derived_diagnoses'][keyof ReturnType<typeof buildMlInferenceOutput>['derived_diagnoses']]>) {
  if (!diagnosis) {
    return null;
  }

  const source = label.split('_')[0] as SourceLabel;

  return {
    type: label,
    confidence: Number(diagnosis.score.toFixed(2)),
    sources: [source],
    details: problemCauseMap[label] ?? [],
    actions: diagnosis.explanation ? [diagnosis.explanation] : diagnosis.reasons
  };
}

function buildRecommendationFromProblem(problem: DiagnosticProblem): EQRecommendation | null {
  const action = problem.actions[0];
  if (!action) {
    return null;
  }

  const [freq_range, gain] = action.split(' ');
  return {
    freq_range,
    gain,
    reason: problem.actions[1] ?? 'model-driven recommendation'
  };
}

function formatRecommendation(centerFrequencyHz: number, gainDb: number, reason: string): EQRecommendation {
  return {
    freq_range: formatEqRange(centerFrequencyHz),
    gain: formatGain(gainDb),
    reason
  };
}

function readTensorData(value: unknown, outputName: string, expectedLength?: number) {
  if (!value || typeof value !== 'object' || !('data' in value)) {
    throw new Error(`Missing tensor output: ${outputName}`);
  }

  const data = (value as { data: ArrayLike<number> }).data;
  const values = Array.from(data, (entry) => Number(entry));

  if (expectedLength !== undefined && values.length !== expectedLength) {
    throw new Error(`Unexpected tensor length for ${outputName}: expected ${expectedLength}, received ${values.length}`);
  }

  return values;
}

function readScalar(value: unknown, outputName: string) {
  if (!value || typeof value !== 'object' || !('data' in value)) {
    throw new Error(`Missing scalar output: ${outputName}`);
  }

  const data = (value as { data: ArrayLike<number> }).data;
  const scalar = Array.from(data, (entry) => Number(entry))[0];

  if (!Number.isFinite(scalar)) {
    throw new Error(`Non-finite scalar output: ${outputName}`);
  }

  return scalar;
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

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function logMlInference(output: AnalysisResult['ml_output']) {
  console.info('[audio-ml]', createModelLogMetadata({
    schema_version: output?.schema_version ?? ML_SCHEMA_VERSION,
    issues: output?.issues,
    sources: output?.sources,
    derived_diagnoses: output?.derived_diagnoses
  }));
}

function logRawModelOutputs({
  issueScores,
  sourceScores,
  modelEqFrequencyNormalized,
  modelEqGainDb
}: ParsedModelOutputs) {
  console.info('[audio-ml:raw]', createModelLogMetadata({
    issue_probs: ISSUE_LABELS.map((label) => Number(issueScores[label].toFixed(4))),
    source_probs: SOURCE_LABELS.map((label) => Number(sourceScores[label].toFixed(4))),
    eq_freq: Number(modelEqFrequencyNormalized.toFixed(4)),
    eq_gain_db: Number(modelEqGainDb.toFixed(4))
  }));
}

function createModelLogMetadata(extra: Record<string, unknown> = {}) {
  return {
    ...getModelRuntimeSnapshot(MODEL_VERSION),
    inferenceTimestamp: new Date().toISOString(),
    ...extra
  };
}
