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
    const result = buildMlAnalysisResult({
      issueScores: parsedOutputs.issueScores,
      sourceScores: parsedOutputs.sourceScores,
      eq: {
        frequencyHz: normalizedToFrequency(parsedOutputs.modelEqFrequencyNormalized),
        gainDb: parsedOutputs.modelEqGainDb
      }
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
