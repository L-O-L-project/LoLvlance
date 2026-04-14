const DEFAULT_EXPERIMENTAL_MODEL_VERSION = 'v0.0-pipeline-check';
const DEFAULT_EXPERIMENTAL_MODEL_PATH = 'models/lightweight_audio_model.onnx';
const DEFAULT_PRODUCTION_MODEL_PATH = 'models/lightweight_audio_model.production.onnx';

function normalizeEnvValue(value: string | undefined) {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function parseBooleanEnv(value: string | undefined, fallback: boolean) {
  const normalized = normalizeEnvValue(value)?.toLowerCase();

  if (normalized === 'true') {
    return true;
  }

  if (normalized === 'false') {
    return false;
  }

  return fallback;
}

export const EXPERIMENTAL_MODEL_VERSION = DEFAULT_EXPERIMENTAL_MODEL_VERSION;
export const MODEL_VERSION =
  normalizeEnvValue(import.meta.env.VITE_MODEL_VERSION as string | undefined)
  ?? EXPERIMENTAL_MODEL_VERSION;
export const ENABLE_MODEL = parseBooleanEnv(
  import.meta.env.VITE_ENABLE_MODEL as string | undefined,
  true
);

const EXPERIMENTAL_MODEL_PATH =
  normalizeEnvValue(import.meta.env.VITE_EXPERIMENTAL_MODEL_PATH as string | undefined)
  ?? DEFAULT_EXPERIMENTAL_MODEL_PATH;
const PRODUCTION_MODEL_PATH =
  normalizeEnvValue(import.meta.env.VITE_PRODUCTION_MODEL_PATH as string | undefined)
  ?? DEFAULT_PRODUCTION_MODEL_PATH;

export function isExperimentalModelVersion(modelVersion = MODEL_VERSION) {
  return modelVersion === EXPERIMENTAL_MODEL_VERSION;
}

export function getConfiguredModelPath(modelVersion = MODEL_VERSION) {
  return isExperimentalModelVersion(modelVersion)
    ? EXPERIMENTAL_MODEL_PATH
    : PRODUCTION_MODEL_PATH;
}

export function getConfiguredModelUrl(
  baseUrl: string,
  origin: string,
  modelVersion = MODEL_VERSION
) {
  return new URL(`${baseUrl}${getConfiguredModelPath(modelVersion)}`, origin).toString();
}

export function getModelRuntimeSnapshot(modelVersion = MODEL_VERSION) {
  return {
    enabled: ENABLE_MODEL,
    modelVersion,
    experimental: isExperimentalModelVersion(modelVersion),
    modelPath: getConfiguredModelPath(modelVersion)
  };
}
