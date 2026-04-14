import type {
  AnalysisResult,
  DetectedAudioSource,
  DerivedDiagnosisResult,
  DerivedDiagnosisType,
  DiagnosticProblem,
  EQRecommendation,
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
  PRIMARY_UI_ISSUES,
  SOURCE_DEFAULT_THRESHOLDS,
  SOURCE_LABELS
} from './mlSchema';

const MODEL_MIN_HZ = 80;
const MODEL_MAX_HZ = 8000;
const MIN_SOURCE_DISPLAY_SCORE = 0.25;
const DERIVED_DIAGNOSIS_PREREQUISITES: Partial<Record<DerivedDiagnosisType, TrainableIssueLabel[]>> = {
  vocal_buried: ['buried'],
  guitar_harsh: ['harsh'],
  bass_muddy: ['muddy'],
  drums_overpower: ['harsh', 'boomy'],
  keys_masking: ['buried', 'boxy']
};

export interface MlPresentationEq {
  frequencyHz: number;
  gainDb: number;
}

export interface BuildMlAnalysisResultParams {
  issueScores: Record<TrainableIssueLabel, number>;
  sourceScores: Record<SourceLabel, number>;
  eq: MlPresentationEq;
  activeIssues?: readonly TrainableIssueLabel[];
  displayedSource?: SourceLabel | null;
  detectedSources?: DetectedAudioSource[];
}

export function buildMlAnalysisResult({
  issueScores,
  sourceScores,
  eq,
  activeIssues,
  displayedSource,
  detectedSources
}: BuildMlAnalysisResultParams): AnalysisResult {
  const mlOutput = buildMlInferenceOutput(issueScores, sourceScores, {
    frequency_hz: eq.frequencyHz,
    frequency_normalized: frequencyToNormalized(eq.frequencyHz),
    gain_db: eq.gainDb
  });
  const problemIssueScores = activeIssues
    ? ISSUE_LABELS.reduce((record, label) => {
        record[label] = activeIssues.includes(label) ? (issueScores[label] ?? 0) : 0;
        return record;
      }, {} as Record<TrainableIssueLabel, number>)
    : issueScores;
  const effectiveDetectedSources = detectedSources ?? buildDetectedSourcesFromScores(sourceScores, {
    dominantSource: displayedSource
  });
  const issueLabels = (
    activeIssues
      ? [...activeIssues]
      : ISSUE_LABELS.filter((label) => (problemIssueScores[label] ?? 0) >= ISSUE_DEFAULT_THRESHOLDS[label])
  ).sort((left, right) => (problemIssueScores[right] ?? 0) - (problemIssueScores[left] ?? 0));

  const issueProblems = issueLabels.map((label, index) => buildIssueProblem({
    label,
    score: problemIssueScores[label] ?? 0,
    detectedSources: effectiveDetectedSources,
    useModelEq: index === 0,
    modelEqFrequencyHz: eq.frequencyHz,
    modelEqGainDb: eq.gainDb
  }));
  const derivedProblems = Object.entries(mlOutput.derived_diagnoses)
    .filter(([label]) => shouldDisplayDerivedProblem(label as DerivedDiagnosisType, activeIssues, displayedSource))
    .map(([label, diagnosis]) => buildDerivedProblem(label as ProblemType, diagnosis!))
    .filter((problem): problem is DiagnosticProblem => problem !== null);
  const eqRecommendations = issueProblems
    .map((problem) => problem.actions[0] ? buildRecommendationFromProblem(problem) : null)
    .filter((recommendation): recommendation is EQRecommendation => recommendation !== null);
  const primaryIssues = issueProblems
    .map((problem) => problem.type)
    .filter((type): type is RuleBasedIssue => PRIMARY_UI_ISSUES.includes(type as RuleBasedIssue));

  return {
    problems: [...issueProblems, ...derivedProblems],
    issues: primaryIssues,
    eq_recommendations: eqRecommendations,
    detectedSources: effectiveDetectedSources,
    ml_output: mlOutput,
    engine: 'ml',
    timestamp: Date.now()
  };
}

export function buildDetectedSourcesFromScores(
  sourceScores: Record<SourceLabel, number>,
  options: {
    dominantSource?: SourceLabel | null;
  } = {}
) {
  const strongestSource = SOURCE_LABELS
    .map((source) => ({ source, score: sourceScores[source] ?? 0 }))
    .sort((left, right) => right.score - left.score)[0];
  const dominantSource = options.dominantSource ?? null;
  const candidateMap = new Map(
    SOURCE_LABELS
      .map((source) => ({
        source,
        score: sourceScores[source] ?? 0
      }))
      .filter((entry) => (
        entry.score >= SOURCE_DEFAULT_THRESHOLDS[entry.source]
        || entry.score >= MIN_SOURCE_DISPLAY_SCORE
      ))
      .map((entry) => [entry.source, entry.score])
  );

  if (dominantSource && (sourceScores[dominantSource] ?? 0) > 0) {
    candidateMap.set(dominantSource, sourceScores[dominantSource] ?? 0);
  }

  if (candidateMap.size === 0) {
    if (!strongestSource || strongestSource.score <= 0) {
      return [];
    }

    return [{
      source: strongestSource.source,
      confidence: Number(strongestSource.score.toFixed(2)),
      labels: [`model:${strongestSource.source}`]
    }] satisfies DetectedAudioSource[];
  }

  return [...candidateMap.entries()]
    .map(([source, confidence]) => ({
      source,
      confidence,
      labels: [`model:${source}`]
    }))
    .sort((left, right) => {
      if (dominantSource && left.source === dominantSource && right.source !== dominantSource) {
        return -1;
      }

      if (dominantSource && right.source === dominantSource && left.source !== dominantSource) {
        return 1;
      }

      return right.confidence - left.confidence;
    })
    .map((entry) => ({
      source: entry.source,
      confidence: Number(entry.confidence.toFixed(2)),
      labels: entry.labels
    })) as DetectedAudioSource[];
}

export function normalizedToFrequency(value: number) {
  const clamped = clamp(value, 0, 1);
  return MODEL_MIN_HZ * ((MODEL_MAX_HZ / MODEL_MIN_HZ) ** clamped);
}

export function frequencyToNormalized(value: number) {
  const clamped = clamp(value, MODEL_MIN_HZ, MODEL_MAX_HZ);
  return Math.log(clamped / MODEL_MIN_HZ) / Math.log(MODEL_MAX_HZ / MODEL_MIN_HZ);
}

function buildIssueProblem({
  label,
  score,
  detectedSources,
  useModelEq,
  modelEqFrequencyHz,
  modelEqGainDb
}: {
  label: TrainableIssueLabel;
  score: number;
  detectedSources: DetectedAudioSource[];
  useModelEq: boolean;
  modelEqFrequencyHz: number;
  modelEqGainDb: number;
}): DiagnosticProblem {
  const profile = ISSUE_PROFILES[label];
  const frequencyHz = useModelEq
    ? modelEqFrequencyHz
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

function buildDerivedProblem(
  label: ProblemType,
  diagnosis: DerivedDiagnosisResult
) {
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

function shouldDisplayDerivedProblem(
  label: DerivedDiagnosisType,
  activeIssues: readonly TrainableIssueLabel[] | undefined,
  displayedSource: SourceLabel | null | undefined
) {
  if (displayedSource) {
    const derivedSource = label.split('_')[0] as SourceLabel;

    if (derivedSource !== displayedSource) {
      return false;
    }
  }

  if (!activeIssues) {
    return true;
  }

  const prerequisites = DERIVED_DIAGNOSIS_PREREQUISITES[label] ?? [];

  return prerequisites.some((issue) => activeIssues.includes(issue));
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
