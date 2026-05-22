import type {
  AnalysisResult,
  ExtractedAudioFeatures,
  SourceLabel,
  TrainableIssueLabel
} from '../types';
import {
  ISSUE_LABELS,
  ML_SCHEMA_VERSION,
  SOURCE_LABELS
} from './mlSchema';
import {
  HIGH_CONFIDENCE_THRESHOLD,
  LOW_CONFIDENCE_THRESHOLD,
  MEDIUM_CONFIDENCE_THRESHOLD,
  MIN_USABLE_CONFIDENCE
} from './mlThresholds';
import { buildMlAnalysisResult, normalizedToFrequency } from './mlPresentation';
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
const MODEL_INPUT_NAME = 'log_mel_spectrogram';

type OrtModule = typeof import('onnxruntime-web');

interface ParsedModelOutputs {
  issueScores: Record<TrainableIssueLabel, number>;
  sourceScores: Record<SourceLabel, number>;
  modelEqFrequencyNormalized: number;
  modelEqGainDb: number;
}

interface MlInferenceFailure {
  code: 'model_load_failed' | 'onnx_runtime_failed' | 'invalid_model_output';
  message: string;
}

interface MlInferenceResponse {
  result: AnalysisResult | null;
  failure?: MlInferenceFailure;
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
): Promise<MlInferenceResponse> {
  if (!ENABLE_MODEL) {
    console.info('[audio-ml]', createModelLogMetadata({
      status: 'skipped',
      reason: 'enableModel=false'
    }));
    return { result: null };
  }

  if (features.melBinCount !== MODEL_MEL_BIN_COUNT) {
    console.warn('[audio-ml] Skipping inference because the mel-bin count does not match the model.', {
      ...createModelLogMetadata(),
      expectedMelBins: MODEL_MEL_BIN_COUNT,
      receivedMelBins: features.melBinCount
    });
    return {
      result: null,
      failure: {
        code: 'invalid_model_output',
        message: `Feature shape mismatch: expected ${MODEL_MEL_BIN_COUNT} mel bins, received ${features.melBinCount}.`
      }
    };
  }

  if (features.rms < MODEL_SILENCE_RMS_THRESHOLD) {
    return { result: {
      problems: [],
      issues: [],
      eq_recommendations: [],
      engine: 'ml',
      timestamp: Date.now()
    } };
  }

  try {
    const [timeSteps, melBins] = features.logMelSpectrogramShape;
    validateFeatureTensor(features, timeSteps, melBins);
    const ort = await getOrtModule();
    const session = await getInferenceSession();
    validateModelInputContract(session);
    const outputs = await session.run({
      [MODEL_INPUT_NAME]: new ort.Tensor('float32', features.logMelSpectrogram, [1, timeSteps, melBins])
    });
    const parsedOutputs = parseModelOutputs(outputs);
    logRawModelOutputs(parsedOutputs);
    const result = buildMlAnalysisResult({
      issueScores: parsedOutputs.issueScores,
      sourceScores: parsedOutputs.sourceScores,
      confidenceTier: getConfidenceTier(Math.max(
        ...Object.values(parsedOutputs.issueScores),
        ...Object.values(parsedOutputs.sourceScores)
      )),
      eq: {
        frequencyHz: normalizedToFrequency(parsedOutputs.modelEqFrequencyNormalized),
        gainDb: parsedOutputs.modelEqGainDb
      }
    });

    logMlInference(result.ml_output);
    return { result };
  } catch (error) {
    const failure = classifyMlInferenceFailure(error);
    console.warn('[audio-ml] Inference failed. Falling back to the rule-based engine.', createModelLogMetadata({ failure }), error);
    return {
      result: null,
      failure
    };
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
  const rawIssueScores = readTensorData(outputMap.issue_probs, 'issue_probs', {
    expectedLength: ISSUE_LABELS.length,
    expectedDims: [1, ISSUE_LABELS.length],
    expectedType: 'float32'
  });
  const rawSourceScores = readTensorData(outputMap.source_probs, 'source_probs', {
    expectedLength: SOURCE_LABELS.length,
    expectedDims: [1, SOURCE_LABELS.length],
    expectedType: 'float32'
  });

  const issueScores = Object.fromEntries(
    ISSUE_LABELS.map((label, index) => [label, normalizeProbability(rawIssueScores[index] ?? 0, `issue_probs[${index}]`)])
  ) as Record<TrainableIssueLabel, number>;
  const sourceScores = Object.fromEntries(
    SOURCE_LABELS.map((label, index) => [label, normalizeProbability(rawSourceScores[index] ?? 0, `source_probs[${index}]`)])
  ) as Record<SourceLabel, number>;

  return {
    issueScores,
    sourceScores,
    modelEqFrequencyNormalized: clamp(readScalar(outputMap.eq_freq, 'eq_freq', [1, 1]), 0, 1),
    modelEqGainDb: clamp(readScalar(outputMap.eq_gain_db, 'eq_gain_db', [1, 1]), -6, 6)
  };
}

function validateFeatureTensor(features: ExtractedAudioFeatures, timeSteps: number, melBins: number) {
  if (timeSteps <= 0 || melBins !== MODEL_MEL_BIN_COUNT) {
    throw new Error(`Invalid feature shape: [${timeSteps}, ${melBins}]`);
  }

  if (features.logMelSpectrogram.length !== timeSteps * melBins) {
    throw new Error(`Feature tensor length mismatch: expected ${timeSteps * melBins}, received ${features.logMelSpectrogram.length}`);
  }

  for (let index = 0; index < features.logMelSpectrogram.length; index += 1) {
    if (!Number.isFinite(features.logMelSpectrogram[index])) {
      throw new Error(`Non-finite feature value at index ${index}`);
    }
  }
}

function validateModelInputContract(session: import('onnxruntime-web').InferenceSession) {
  if (!session.inputNames.includes(MODEL_INPUT_NAME)) {
    throw new Error(`Missing model input: ${MODEL_INPUT_NAME}`);
  }
}

function readTensorData(
  value: unknown,
  outputName: string,
  options: {
    expectedLength: number;
    expectedDims: number[];
    expectedType: string;
  }
) {
  if (!value || typeof value !== 'object' || !('data' in value)) {
    throw new Error(`Missing tensor output: ${outputName}`);
  }

  const tensor = value as { data: ArrayLike<number>; dims?: readonly number[]; type?: string };

  if (tensor.type && tensor.type !== options.expectedType) {
    throw new Error(`Unexpected tensor type for ${outputName}: expected ${options.expectedType}, received ${tensor.type}`);
  }

  if (tensor.dims && !dimsMatch(tensor.dims, options.expectedDims, options.expectedLength)) {
    throw new Error(`Unexpected tensor dims for ${outputName}: expected [${options.expectedDims.join(',')}], received [${tensor.dims.join(',')}]`);
  }

  const data = tensor.data;
  const values = Array.from(data, (entry) => Number(entry));

  if (values.length !== options.expectedLength) {
    throw new Error(`Unexpected tensor length for ${outputName}: expected ${options.expectedLength}, received ${values.length}`);
  }

  if (values.some((entry) => !Number.isFinite(entry))) {
    throw new Error(`Non-finite tensor output: ${outputName}`);
  }

  return values;
}

function classifyMlInferenceFailure(error: unknown): MlInferenceFailure {
  const message = error instanceof Error ? error.message : String(error);
  const normalized = message.toLowerCase();

  if (
    normalized.includes('unexpected tensor')
    || normalized.includes('missing tensor')
    || normalized.includes('missing scalar')
    || normalized.includes('non-finite')
  ) {
    return {
      code: 'invalid_model_output',
      message
    };
  }

  if (
    normalized.includes('wasm')
    || normalized.includes('ort-wasm')
    || normalized.includes('webassembly')
  ) {
    return {
      code: 'onnx_runtime_failed',
      message
    };
  }

  return {
    code: 'model_load_failed',
    message
  };
}

function readScalar(value: unknown, outputName: string, expectedDims: number[]) {
  if (!value || typeof value !== 'object' || !('data' in value)) {
    throw new Error(`Missing scalar output: ${outputName}`);
  }

  const tensor = value as { data: ArrayLike<number>; dims?: readonly number[]; type?: string };

  if (tensor.type && tensor.type !== 'float32') {
    throw new Error(`Unexpected scalar type for ${outputName}: expected float32, received ${tensor.type}`);
  }

  if (tensor.dims && !dimsMatch(tensor.dims, expectedDims, 1)) {
    throw new Error(`Unexpected scalar dims for ${outputName}: expected [${expectedDims.join(',')}], received [${tensor.dims.join(',')}]`);
  }

  const data = tensor.data;
  const scalar = Array.from(data, (entry) => Number(entry))[0];

  if (!Number.isFinite(scalar)) {
    throw new Error(`Non-finite scalar output: ${outputName}`);
  }

  return scalar;
}

function dimsMatch(actual: readonly number[], expected: readonly number[], expectedLength: number) {
  const actualProduct = actual.reduce((total, dim) => total * dim, 1);

  if (actualProduct !== expectedLength) {
    return false;
  }

  if (actual.length === expected.length && actual.every((dim, index) => dim === expected[index])) {
    return true;
  }

  return actual.length === 1 && actual[0] === expectedLength;
}

function normalizeProbability(value: number, label: string) {
  if (!Number.isFinite(value)) {
    throw new Error(`Non-finite probability: ${label}`);
  }

  if (value < -0.001 || value > 1.001) {
    throw new Error(`Probability outside expected range: ${label}=${value}`);
  }

  return clamp(value, 0, 1);
}

function getConfidenceTier(score: number) {
  if (score >= HIGH_CONFIDENCE_THRESHOLD) {
    return 'high';
  }

  if (score >= MEDIUM_CONFIDENCE_THRESHOLD) {
    return 'medium';
  }

  if (score >= LOW_CONFIDENCE_THRESHOLD || score >= MIN_USABLE_CONFIDENCE) {
    return 'low';
  }

  return 'invalid';
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
