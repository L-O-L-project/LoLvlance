import type {
  BufferedAudioSnapshot,
  DetectedAudioSource,
  Instrument
} from '../types';

const SOURCE_DETECTION_MIN_RMS = 0.012;
const SOURCE_DETECTION_MIN_DURATION_MS = 900;
const SOURCE_DETECTION_SCORE_THRESHOLD = 0.08;
const SOURCE_DETECTION_DISPLAY_THRESHOLD = 0.12;
const SOURCE_DETECTION_FALLBACK_THRESHOLD = 0.06;

type TasksAudioModule = typeof import('@mediapipe/tasks-audio');

interface SourceAccumulator {
  totalScore: number;
  maxScore: number;
  labels: Set<string>;
}

let tasksAudioModulePromise: Promise<TasksAudioModule> | null = null;
let audioClassifierPromise: Promise<import('@mediapipe/tasks-audio').AudioClassifier> | null = null;
let hasLoggedTaggerReady = false;

const allowedCategoryNames = [
  'Speech',
  'Singing',
  'Choir',
  'Humming',
  'Rapping',
  'Vocal music',
  'A capella',
  'Guitar',
  'Electric guitar',
  'Bass guitar',
  'Acoustic guitar',
  'Steel guitar, slide guitar',
  'Strum',
  'Keyboard (musical)',
  'Piano',
  'Electric piano',
  'Organ',
  'Electronic organ',
  'Hammond organ',
  'Synthesizer',
  'Sampler',
  'Harpsichord',
  'Drum kit',
  'Drum machine',
  'Drum',
  'Snare drum',
  'Bass drum',
  'Cymbal',
  'Hi-hat',
  'Tambourine',
  'Maraca',
  'Double bass'
] as const;

const labelToSourceMap: Record<string, Instrument> = {
  Speech: 'vocal',
  Singing: 'vocal',
  Choir: 'vocal',
  Humming: 'vocal',
  Rapping: 'vocal',
  'Vocal music': 'vocal',
  'A capella': 'vocal',
  Guitar: 'guitar',
  'Electric guitar': 'guitar',
  'Acoustic guitar': 'guitar',
  'Steel guitar, slide guitar': 'guitar',
  Strum: 'guitar',
  'Bass guitar': 'bass',
  'Double bass': 'bass',
  'Keyboard (musical)': 'keys',
  Piano: 'keys',
  'Electric piano': 'keys',
  Organ: 'keys',
  'Electronic organ': 'keys',
  'Hammond organ': 'keys',
  Synthesizer: 'keys',
  Sampler: 'keys',
  Harpsichord: 'keys',
  'Drum kit': 'drums',
  'Drum machine': 'drums',
  Drum: 'drums',
  'Snare drum': 'drums',
  'Bass drum': 'drums',
  Cymbal: 'drums',
  'Hi-hat': 'drums',
  Tambourine: 'drums',
  Maraca: 'drums'
};

export async function warmUpOpenSourceAudioTagger() {
  try {
    await getAudioClassifier();

    if (!hasLoggedTaggerReady) {
      hasLoggedTaggerReady = true;
      console.info('[audio-tags]', {
        status: 'ready',
        model: getModelUrl()
      });
    }

    return true;
  } catch (error) {
    console.warn('[audio-tags] Open-source tagger warm-up skipped.', error);
    return false;
  }
}

export async function detectOpenSourceAudioSources(snapshot: BufferedAudioSnapshot) {
  if (
    snapshot.rms < SOURCE_DETECTION_MIN_RMS
    || snapshot.durationMs < SOURCE_DETECTION_MIN_DURATION_MS
  ) {
    return [];
  }

  try {
    const classifier = await getAudioClassifier();
    const results = classifier.classify(snapshot.samples, snapshot.sampleRate);
    const detectedSources = aggregateClassificationResults(results);

    logDetectedSources(detectedSources);
    return detectedSources;
  } catch (error) {
    console.warn('[audio-tags] Open-source tagging failed.', error);
    return [];
  }
}

async function getTasksAudioModule() {
  if (!tasksAudioModulePromise) {
    tasksAudioModulePromise = import('@mediapipe/tasks-audio');
  }

  return tasksAudioModulePromise;
}

async function getAudioClassifier() {
  if (!audioClassifierPromise) {
    audioClassifierPromise = (async () => {
      const { AudioClassifier, FilesetResolver } = await getTasksAudioModule();
      const audioFileset = await FilesetResolver.forAudioTasks(getWasmBaseUrl());

      return AudioClassifier.createFromOptions(audioFileset, {
        baseOptions: {
          modelAssetPath: getModelUrl()
        },
        displayNamesLocale: 'en',
        maxResults: 12,
        scoreThreshold: SOURCE_DETECTION_SCORE_THRESHOLD,
        categoryAllowlist: [...allowedCategoryNames]
      });
    })().catch((error) => {
      audioClassifierPromise = null;
      throw error;
    });
  }

  return audioClassifierPromise;
}

function aggregateClassificationResults(
  results: Array<{
    classifications: Array<{
      categories: Array<{
        score: number;
        categoryName?: string;
        displayName?: string;
      }>;
    }>;
  }>
) {
  const perSource = new Map<Instrument, SourceAccumulator>();

  results.forEach((result) => {
    result.classifications.forEach((classification) => {
      classification.categories.forEach((category) => {
        const label = category.categoryName || category.displayName;

        if (!label) {
          return;
        }

        const source = labelToSourceMap[label];

        if (!source) {
          return;
        }

        const accumulator = perSource.get(source) ?? {
          totalScore: 0,
          maxScore: 0,
          labels: new Set<string>()
        };

        accumulator.totalScore += category.score;
        accumulator.maxScore = Math.max(accumulator.maxScore, category.score);
        accumulator.labels.add(label);

        perSource.set(source, accumulator);
      });
    });
  });

  const aggregatedSources = [...perSource.entries()]
    .map(([source, accumulator]) => ({
      source,
      confidence: Number(
        clamp(accumulator.maxScore * 0.7 + accumulator.totalScore * 0.3, 0, 1).toFixed(2)
      ),
      labels: [...accumulator.labels].sort()
    }))
    .sort((left, right) => right.confidence - left.confidence);

  const visibleSources = aggregatedSources.filter(
    (entry) => entry.confidence >= SOURCE_DETECTION_DISPLAY_THRESHOLD
  );

  if (visibleSources.length > 0) {
    return visibleSources;
  }

  const fallbackSource = aggregatedSources[0];
  return fallbackSource && fallbackSource.confidence >= SOURCE_DETECTION_FALLBACK_THRESHOLD
    ? [fallbackSource]
    : [];
}

function getWasmBaseUrl() {
  return new URL(`${import.meta.env.BASE_URL}mediapipe/wasm`, window.location.origin).toString();
}

function getModelUrl() {
  return new URL(`${import.meta.env.BASE_URL}models/yamnet.tflite`, window.location.origin).toString();
}

function logDetectedSources(detectedSources: DetectedAudioSource[]) {
  console.info('[audio-tags]', {
    detectedSources: detectedSources.map((entry) => ({
      source: entry.source,
      confidence: Number(entry.confidence.toFixed(3)),
      labels: entry.labels
    }))
  });
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
