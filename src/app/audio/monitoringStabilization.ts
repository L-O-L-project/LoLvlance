import type {
  AnalysisResult,
  MonitoringEqDecision,
  MonitoringIssueDecision,
  MonitoringSourceDecision,
  MonitoringStabilizationDebug,
  SourceLabel,
  TrainableIssueLabel
} from '../types';
import {
  EQ_EMA_ALPHA,
  EQ_MIN_DISPLAY_FREQ_DELTA_HZ,
  EQ_MIN_DISPLAY_GAIN_DELTA_DB,
  ISSUE_EMA_ALPHA,
  ISSUE_OFF_THRESHOLD,
  ISSUE_ON_THRESHOLD,
  SOURCE_EMA_ALPHA,
  SOURCE_SWITCH_MARGIN
} from '../config/monitoringStabilization';
import { buildSourceAwareEqRecommendations } from './sourceAwareEq';
import { buildDetectedSourcesFromScores, buildMlAnalysisResult } from './mlPresentation';
import { ISSUE_LABELS, SOURCE_LABELS } from './mlSchema';

export interface MonitoringMlRuntimeState {
  hasSmoothedFrame: boolean;
  rawIssueProbs: Record<TrainableIssueLabel, number>;
  rawSourceProbs: Record<SourceLabel, number>;
  rawEqFreqHz: number | null;
  rawEqGainDb: number | null;
  smoothedIssueProbs: Record<TrainableIssueLabel, number>;
  smoothedSourceProbs: Record<SourceLabel, number>;
  smoothedEqFreqHz: number | null;
  smoothedEqGainDb: number | null;
  displayedIssueState: Record<TrainableIssueLabel, boolean>;
  displayedSourceState: SourceLabel | null;
  displayedEqFreqHz: number | null;
  displayedEqGainDb: number | null;
}

interface MonitoringMlPayload {
  issueProbs: Record<TrainableIssueLabel, number>;
  sourceProbs: Record<SourceLabel, number>;
  eqFreqHz: number;
  eqGainDb: number;
}

export function createInitialMonitoringMlRuntimeState(): MonitoringMlRuntimeState {
  return {
    hasSmoothedFrame: false,
    rawIssueProbs: createIssueProbabilityRecord(),
    rawSourceProbs: createSourceProbabilityRecord(),
    rawEqFreqHz: null,
    rawEqGainDb: null,
    smoothedIssueProbs: createIssueProbabilityRecord(),
    smoothedSourceProbs: createSourceProbabilityRecord(),
    smoothedEqFreqHz: null,
    smoothedEqGainDb: null,
    displayedIssueState: createIssueStateRecord(),
    displayedSourceState: null,
    displayedEqFreqHz: null,
    displayedEqGainDb: null
  };
}

export function extractMonitoringMlPayload(result: AnalysisResult): MonitoringMlPayload | null {
  const modelEq = result.ml_output?.metadata.model_eq;

  if (!result.ml_output || !modelEq) {
    return null;
  }

  return {
    issueProbs: ISSUE_LABELS.reduce((record, label) => {
      record[label] = clamp(result.ml_output?.issues[label] ?? 0, 0, 1);
      return record;
    }, createIssueProbabilityRecord()),
    sourceProbs: SOURCE_LABELS.reduce((record, label) => {
      record[label] = clamp(result.ml_output?.sources[label] ?? 0, 0, 1);
      return record;
    }, createSourceProbabilityRecord()),
    eqFreqHz: Math.max(20, modelEq.frequency_hz),
    eqGainDb: clamp(modelEq.gain_db, -6, 6)
  };
}

export function smoothScalarEMA(previous: number | null, next: number, alpha: number) {
  if (previous === null || !Number.isFinite(previous)) {
    return next;
  }

  const normalizedAlpha = clamp(alpha, 0.01, 1);
  return (normalizedAlpha * next) + ((1 - normalizedAlpha) * previous);
}

export function smoothVectorEMA<T extends string>(
  labels: readonly T[],
  previous: Record<T, number>,
  next: Record<T, number>,
  alpha: number,
  initialized: boolean
) {
  if (!initialized) {
    return labels.reduce((record, label) => {
      record[label] = clamp(next[label] ?? 0, 0, 1);
      return record;
    }, {} as Record<T, number>);
  }

  const normalizedAlpha = clamp(alpha, 0.01, 1);
  return labels.reduce((record, label) => {
    record[label] = clamp(
      (normalizedAlpha * (next[label] ?? 0)) + ((1 - normalizedAlpha) * (previous[label] ?? 0)),
      0,
      1
    );
    return record;
  }, {} as Record<T, number>);
}

export function advanceMonitoringMlRuntimeState({
  currentState,
  rawResult,
  now,
  commitDisplayed
}: {
  currentState: MonitoringMlRuntimeState;
  rawResult: AnalysisResult;
  now: number;
  commitDisplayed: boolean;
}) {
  const payload = extractMonitoringMlPayload(rawResult);

  if (!payload) {
    return {
      nextState: currentState,
      displayedResult: undefined,
      debugSnapshot: undefined
    };
  }

  const nextState: MonitoringMlRuntimeState = {
    ...currentState,
    hasSmoothedFrame: true,
    rawIssueProbs: payload.issueProbs,
    rawSourceProbs: payload.sourceProbs,
    rawEqFreqHz: payload.eqFreqHz,
    rawEqGainDb: payload.eqGainDb,
    smoothedIssueProbs: smoothVectorEMA(
      ISSUE_LABELS,
      currentState.smoothedIssueProbs,
      payload.issueProbs,
      ISSUE_EMA_ALPHA,
      currentState.hasSmoothedFrame
    ),
    smoothedSourceProbs: smoothVectorEMA(
      SOURCE_LABELS,
      currentState.smoothedSourceProbs,
      payload.sourceProbs,
      SOURCE_EMA_ALPHA,
      currentState.hasSmoothedFrame
    ),
    smoothedEqFreqHz: smoothScalarEMA(currentState.smoothedEqFreqHz, payload.eqFreqHz, EQ_EMA_ALPHA),
    smoothedEqGainDb: smoothScalarEMA(currentState.smoothedEqGainDb, payload.eqGainDb, EQ_EMA_ALPHA)
  };

  const issueUpdate = commitDisplayed
    ? applyIssueHysteresis(currentState.displayedIssueState, nextState.smoothedIssueProbs)
    : null;
  const sourceDecision = commitDisplayed
    ? selectStableDominantSource(currentState.displayedSourceState, nextState.smoothedSourceProbs)
    : null;
  const freqDecision = commitDisplayed
    ? applyEqDisplayThreshold(currentState.displayedEqFreqHz, nextState.smoothedEqFreqHz, EQ_MIN_DISPLAY_FREQ_DELTA_HZ)
    : null;
  const gainDecision = commitDisplayed
    ? applyEqDisplayThreshold(currentState.displayedEqGainDb, nextState.smoothedEqGainDb, EQ_MIN_DISPLAY_GAIN_DELTA_DB)
    : null;

  if (issueUpdate) {
    nextState.displayedIssueState = issueUpdate.nextState;
  }

  if (sourceDecision) {
    nextState.displayedSourceState = sourceDecision.nextSource;
  }

  if (freqDecision) {
    nextState.displayedEqFreqHz = freqDecision.displayedValue;
  }

  if (gainDecision) {
    nextState.displayedEqGainDb = gainDecision.displayedValue;
  }

  const debugSnapshot: MonitoringStabilizationDebug = {
    inferenceTimestamp: now,
    displayCommitted: commitDisplayed,
    rawIssueProbs: roundScoreRecord(nextState.rawIssueProbs),
    rawSourceProbs: roundScoreRecord(nextState.rawSourceProbs),
    rawEqFreqHz: roundNullable(nextState.rawEqFreqHz),
    rawEqGainDb: roundNullable(nextState.rawEqGainDb, 1),
    smoothedIssueProbs: roundScoreRecord(nextState.smoothedIssueProbs),
    smoothedSourceProbs: roundScoreRecord(nextState.smoothedSourceProbs),
    smoothedEqFreqHz: roundNullable(nextState.smoothedEqFreqHz),
    smoothedEqGainDb: roundNullable(nextState.smoothedEqGainDb, 1),
    displayedIssueState: { ...nextState.displayedIssueState },
    displayedSourceState: nextState.displayedSourceState,
    displayedEqFreqHz: roundNullable(nextState.displayedEqFreqHz),
    displayedEqGainDb: roundNullable(nextState.displayedEqGainDb, 1),
    issueDecisions: issueUpdate?.decisions ?? [],
    sourceDecision,
    eqDecisions: {
      freq: freqDecision ?? createMissingEqDecision(EQ_MIN_DISPLAY_FREQ_DELTA_HZ),
      gain: gainDecision ?? createMissingEqDecision(EQ_MIN_DISPLAY_GAIN_DELTA_DB)
    }
  };

  if (!commitDisplayed) {
    return {
      nextState,
      displayedResult: undefined,
      debugSnapshot
    };
  }

  const displayedDetectedSources = (rawResult.detectedSources?.length ?? 0) > 0
    ? rawResult.detectedSources!
    : buildDetectedSourcesFromScores(nextState.smoothedSourceProbs, {
        dominantSource: nextState.displayedSourceState
      });
  const displayedAnalysis = buildMlAnalysisResult({
    issueScores: nextState.smoothedIssueProbs,
    sourceScores: nextState.smoothedSourceProbs,
    eq: {
      frequencyHz: nextState.displayedEqFreqHz ?? nextState.smoothedEqFreqHz ?? payload.eqFreqHz,
      gainDb: nextState.displayedEqGainDb ?? nextState.smoothedEqGainDb ?? payload.eqGainDb
    },
    activeIssues: ISSUE_LABELS.filter((label) => nextState.displayedIssueState[label]),
    displayedSource: nextState.displayedSourceState,
    detectedSources: displayedDetectedSources
  });
  const displayedResult: AnalysisResult = {
    ...rawResult,
    problems: displayedAnalysis.problems,
    issues: displayedAnalysis.issues,
    eq_recommendations: displayedAnalysis.eq_recommendations,
    detectedSources: displayedDetectedSources,
    sourceEqRecommendations: buildSourceAwareEqRecommendations({
      detectedSources: displayedDetectedSources,
      issues: displayedAnalysis.issues ?? [],
      stemMetrics: rawResult.stemMetrics ?? []
    }),
    ml_output: displayedAnalysis.ml_output,
    monitoringStabilization: debugSnapshot,
    timestamp: now
  };

  return {
    nextState,
    displayedResult,
    debugSnapshot
  };
}

function applyIssueHysteresis(
  previousState: Record<TrainableIssueLabel, boolean>,
  smoothedProbabilities: Record<TrainableIssueLabel, number>
) {
  const nextState = { ...previousState };
  const decisions: MonitoringIssueDecision[] = [];

  ISSUE_LABELS.forEach((label) => {
    const smoothedProbability = clamp(smoothedProbabilities[label] ?? 0, 0, 1);
    const wasActive = previousState[label];
    let isActive = wasActive;
    let reason: MonitoringIssueDecision['reason'] = 'hold';

    if (!wasActive && smoothedProbability >= ISSUE_ON_THRESHOLD) {
      isActive = true;
      reason = 'switch_on';
    } else if (wasActive && smoothedProbability <= ISSUE_OFF_THRESHOLD) {
      isActive = false;
      reason = 'switch_off';
    }

    nextState[label] = isActive;
    decisions.push({
      label,
      previousState: wasActive,
      nextState: isActive,
      smoothedProbability: Number(smoothedProbability.toFixed(4)),
      reason
    });
  });

  return {
    nextState,
    decisions
  };
}

function selectStableDominantSource(
  previousSource: SourceLabel | null,
  smoothedProbabilities: Record<SourceLabel, number>
): MonitoringSourceDecision {
  const rankedSources = SOURCE_LABELS
    .map((source) => ({
      source,
      probability: clamp(smoothedProbabilities[source] ?? 0, 0, 1)
    }))
    .sort((left, right) => right.probability - left.probability);
  const candidate = rankedSources[0];
  const candidateSource = candidate?.source ?? null;
  const candidateProbability = candidate ? Number(candidate.probability.toFixed(4)) : null;

  if (!candidateSource || candidateProbability === null || candidateProbability <= 0) {
    return {
      previousSource,
      candidateSource: null,
      nextSource: previousSource,
      previousProbability: previousSource ? Number((smoothedProbabilities[previousSource] ?? 0).toFixed(4)) : null,
      candidateProbability: null,
      margin: SOURCE_SWITCH_MARGIN,
      reason: 'empty'
    };
  }

  if (!previousSource) {
    return {
      previousSource: null,
      candidateSource,
      nextSource: candidateSource,
      previousProbability: null,
      candidateProbability,
      margin: SOURCE_SWITCH_MARGIN,
      reason: 'initial'
    };
  }

  const previousProbability = Number((smoothedProbabilities[previousSource] ?? 0).toFixed(4));

  if (candidateSource === previousSource) {
    return {
      previousSource,
      candidateSource,
      nextSource: previousSource,
      previousProbability,
      candidateProbability,
      margin: SOURCE_SWITCH_MARGIN,
      reason: 'retain'
    };
  }

  if ((candidateProbability - previousProbability) >= SOURCE_SWITCH_MARGIN) {
    return {
      previousSource,
      candidateSource,
      nextSource: candidateSource,
      previousProbability,
      candidateProbability,
      margin: SOURCE_SWITCH_MARGIN,
      reason: 'switch'
    };
  }

  return {
    previousSource,
    candidateSource,
    nextSource: previousSource,
    previousProbability,
    candidateProbability,
    margin: SOURCE_SWITCH_MARGIN,
    reason: 'retain'
  };
}

function applyEqDisplayThreshold(
  previousValue: number | null,
  smoothedValue: number | null,
  threshold: number
): MonitoringEqDecision {
  if (smoothedValue === null || !Number.isFinite(smoothedValue)) {
    return createMissingEqDecision(threshold);
  }

  const roundedSmoothedValue = roundForDisplay(smoothedValue, threshold);

  if (previousValue === null || !Number.isFinite(previousValue)) {
    return {
      previousValue: null,
      smoothedValue: roundedSmoothedValue,
      displayedValue: roundedSmoothedValue,
      delta: null,
      threshold,
      reason: 'initial'
    };
  }

  const delta = Math.abs(smoothedValue - previousValue);

  if (delta >= threshold) {
    return {
      previousValue: roundNullable(previousValue, threshold < 1 ? 1 : 0),
      smoothedValue: roundedSmoothedValue,
      displayedValue: roundedSmoothedValue,
      delta: Number(delta.toFixed(threshold < 1 ? 2 : 1)),
      threshold,
      reason: 'updated'
    };
  }

  return {
    previousValue: roundNullable(previousValue, threshold < 1 ? 1 : 0),
    smoothedValue: roundedSmoothedValue,
    displayedValue: roundNullable(previousValue, threshold < 1 ? 1 : 0),
    delta: Number(delta.toFixed(threshold < 1 ? 2 : 1)),
    threshold,
    reason: 'suppressed'
  };
}

function createMissingEqDecision(threshold: number): MonitoringEqDecision {
  return {
    previousValue: null,
    smoothedValue: null,
    displayedValue: null,
    delta: null,
    threshold,
    reason: 'missing'
  };
}

function createIssueProbabilityRecord() {
  return ISSUE_LABELS.reduce((record, label) => {
    record[label] = 0;
    return record;
  }, {} as Record<TrainableIssueLabel, number>);
}

function createSourceProbabilityRecord() {
  return SOURCE_LABELS.reduce((record, label) => {
    record[label] = 0;
    return record;
  }, {} as Record<SourceLabel, number>);
}

function createIssueStateRecord() {
  return ISSUE_LABELS.reduce((record, label) => {
    record[label] = false;
    return record;
  }, {} as Record<TrainableIssueLabel, boolean>);
}

function roundScoreRecord<T extends string>(record: Record<T, number>) {
  return Object.fromEntries(
    Object.entries(record).map(([label, value]) => [label, Number(value.toFixed(4))])
  ) as Record<T, number>;
}

function roundNullable(value: number | null, decimals = 0) {
  if (value === null || !Number.isFinite(value)) {
    return null;
  }

  return Number(value.toFixed(decimals));
}

function roundForDisplay(value: number, threshold: number) {
  return threshold < 1
    ? Number(value.toFixed(1))
    : Math.round(value);
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
