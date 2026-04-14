import type {
  AnalysisResult,
  DetectedAudioSource,
  MonitoringEqDecision,
  MonitoringInterpretation,
  MonitoringIssueDecision,
  MonitoringIssueState,
  MonitoringSourceDecision,
  MonitoringStabilizationDebug,
  SourceLabel,
  TrainableIssueLabel
} from '../types';
import {
  EQ_EMA_ALPHA,
  EQ_MIN_DISPLAY_FREQ_DELTA_HZ,
  EQ_MIN_DISPLAY_GAIN_DELTA_DB,
  EQ_SUDDEN_SHIFT_FREQ_DELTA_HZ,
  EQ_SUDDEN_SHIFT_GAIN_DELTA_DB,
  ISSUE_EMA_ALPHA,
  ISSUE_OFF_THRESHOLD,
  ISSUE_ON_THRESHOLD,
  ISSUE_PERSISTENT_SLOW_FORCE_HOLD_THRESHOLD,
  ISSUE_PERSISTENT_SLOW_FORCE_ON_THRESHOLD,
  ISSUE_PERSISTENT_SLOW_HOLD_THRESHOLD,
  ISSUE_PERSISTENT_SLOW_ON_THRESHOLD,
  ISSUE_TRANSIENT_SLOW_MAX_HOLD_THRESHOLD,
  ISSUE_TRANSIENT_SLOW_MAX_ON_THRESHOLD,
  SOURCE_EMA_ALPHA,
  SOURCE_SWITCH_MARGIN,
  SOURCE_SWITCH_SUSTAINED_FRAMES
} from '../config/monitoringStabilization';
import { buildSourceAwareEqRecommendations } from './sourceAwareEq';
import { buildDetectedSourcesFromScores, buildMlAnalysisResult } from './mlPresentation';
import {
  appendMonitoringTimeBufferFrame,
  buildSlowSignalFromTimeBuffer,
  createMonitoringTimeBuffer,
  getRecentDominantSourceFrameCount,
  serializeMonitoringTimeBuffer,
  type MonitoringTimeBuffer
} from './monitoringTimeBuffer';
import { ISSUE_LABELS, SOURCE_LABELS } from './mlSchema';

export interface MonitoringMlRuntimeState {
  hasSmoothedFrame: boolean;
  timeBuffer: MonitoringTimeBuffer;
  rawIssueProbs: Record<TrainableIssueLabel, number>;
  rawSourceProbs: Record<SourceLabel, number>;
  rawEqFreqHz: number | null;
  rawEqGainDb: number | null;
  smoothedIssueProbs: Record<TrainableIssueLabel, number>;
  smoothedSourceProbs: Record<SourceLabel, number>;
  smoothedEqFreqHz: number | null;
  smoothedEqGainDb: number | null;
  slowIssueProbs: Record<TrainableIssueLabel, number>;
  slowSourceProbs: Record<SourceLabel, number>;
  slowEqFreqHz: number | null;
  slowEqGainDb: number | null;
  displayedIssueStates: Record<TrainableIssueLabel, MonitoringIssueState>;
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
    timeBuffer: createMonitoringTimeBuffer(),
    rawIssueProbs: createIssueProbabilityRecord(),
    rawSourceProbs: createSourceProbabilityRecord(),
    rawEqFreqHz: null,
    rawEqGainDb: null,
    smoothedIssueProbs: createIssueProbabilityRecord(),
    smoothedSourceProbs: createSourceProbabilityRecord(),
    smoothedEqFreqHz: null,
    smoothedEqGainDb: null,
    slowIssueProbs: createIssueProbabilityRecord(),
    slowSourceProbs: createSourceProbabilityRecord(),
    slowEqFreqHz: null,
    slowEqGainDb: null,
    displayedIssueStates: createIssueStateRecord(),
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
  commitDisplayed,
  captureDebugSnapshot = false
}: {
  currentState: MonitoringMlRuntimeState;
  rawResult: AnalysisResult;
  now: number;
  commitDisplayed: boolean;
  captureDebugSnapshot?: boolean;
}) {
  const payload = extractMonitoringMlPayload(rawResult);

  if (!payload) {
    return {
      nextState: currentState,
      displayedResult: undefined,
      debugSnapshot: undefined
    };
  }

  const fastIssueProbs = smoothVectorEMA(
    ISSUE_LABELS,
    currentState.smoothedIssueProbs,
    payload.issueProbs,
    ISSUE_EMA_ALPHA,
    currentState.hasSmoothedFrame
  );
  const fastSourceProbs = smoothVectorEMA(
    SOURCE_LABELS,
    currentState.smoothedSourceProbs,
    payload.sourceProbs,
    SOURCE_EMA_ALPHA,
    currentState.hasSmoothedFrame
  );
  const fastEqFreqHz = smoothScalarEMA(currentState.smoothedEqFreqHz, payload.eqFreqHz, EQ_EMA_ALPHA);
  const fastEqGainDb = smoothScalarEMA(currentState.smoothedEqGainDb, payload.eqGainDb, EQ_EMA_ALPHA);
  const nextState: MonitoringMlRuntimeState = {
    ...currentState,
    hasSmoothedFrame: true,
    rawIssueProbs: payload.issueProbs,
    rawSourceProbs: payload.sourceProbs,
    rawEqFreqHz: payload.eqFreqHz,
    rawEqGainDb: payload.eqGainDb,
    smoothedIssueProbs: fastIssueProbs,
    smoothedSourceProbs: fastSourceProbs,
    smoothedEqFreqHz: fastEqFreqHz,
    smoothedEqGainDb: fastEqGainDb
  };

  if (fastEqFreqHz !== null && fastEqGainDb !== null) {
    appendMonitoringTimeBufferFrame(nextState.timeBuffer, {
      timestamp: now,
      fastIssueProbs,
      fastSourceProbs,
      fastEqFreqHz,
      fastEqGainDb
    });
  }

  const slowSignal = buildSlowSignalFromTimeBuffer(nextState.timeBuffer);
  nextState.slowIssueProbs = slowSignal.issueProbs;
  nextState.slowSourceProbs = slowSignal.sourceProbs;
  nextState.slowEqFreqHz = slowSignal.eqFreqHz;
  nextState.slowEqGainDb = slowSignal.eqGainDb;

  const issueUpdate = commitDisplayed
    ? applyIssueStateDecisions(
        currentState.displayedIssueStates,
        fastIssueProbs,
        nextState.slowIssueProbs
      )
    : null;
  const sourceDecision = commitDisplayed
    ? selectStableDominantSource(
        currentState.displayedSourceState,
        nextState.slowSourceProbs,
        nextState.timeBuffer
      )
    : null;
  const freqDecision = commitDisplayed
    ? applyEqDisplayThreshold(
        currentState.displayedEqFreqHz,
        fastEqFreqHz,
        nextState.slowEqFreqHz,
        EQ_MIN_DISPLAY_FREQ_DELTA_HZ,
        EQ_SUDDEN_SHIFT_FREQ_DELTA_HZ
      )
    : null;
  const gainDecision = commitDisplayed
    ? applyEqDisplayThreshold(
        currentState.displayedEqGainDb,
        fastEqGainDb,
        nextState.slowEqGainDb,
        EQ_MIN_DISPLAY_GAIN_DELTA_DB,
        EQ_SUDDEN_SHIFT_GAIN_DELTA_DB
      )
    : null;

  if (issueUpdate) {
    nextState.displayedIssueStates = issueUpdate.nextStates;
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

  const interpretation = buildMonitoringInterpretation(nextState, {
    suddenEqShift: Boolean(freqDecision?.suddenShift || gainDecision?.suddenShift)
  });
  const debugSnapshot: MonitoringStabilizationDebug | undefined = captureDebugSnapshot
    ? {
        inferenceTimestamp: now,
        displayCommitted: commitDisplayed,
        bufferSize: nextState.timeBuffer.size,
        bufferCapacity: nextState.timeBuffer.capacity,
        bufferFrames: serializeMonitoringTimeBuffer(nextState.timeBuffer),
        rawIssueProbs: roundScoreRecord(nextState.rawIssueProbs),
        rawSourceProbs: roundScoreRecord(nextState.rawSourceProbs),
        rawEqFreqHz: roundNullable(nextState.rawEqFreqHz),
        rawEqGainDb: roundNullable(nextState.rawEqGainDb, 1),
        fastIssueProbs: roundScoreRecord(nextState.smoothedIssueProbs),
        fastSourceProbs: roundScoreRecord(nextState.smoothedSourceProbs),
        fastEqFreqHz: roundNullable(nextState.smoothedEqFreqHz),
        fastEqGainDb: roundNullable(nextState.smoothedEqGainDb, 1),
        slowIssueProbs: roundScoreRecord(nextState.slowIssueProbs),
        slowSourceProbs: roundScoreRecord(nextState.slowSourceProbs),
        slowEqFreqHz: roundNullable(nextState.slowEqFreqHz),
        slowEqGainDb: roundNullable(nextState.slowEqGainDb, 1),
        displayedIssueStates: { ...nextState.displayedIssueStates },
        displayedSourceState: nextState.displayedSourceState,
        displayedEqFreqHz: roundNullable(nextState.displayedEqFreqHz),
        displayedEqGainDb: roundNullable(nextState.displayedEqGainDb, 1),
        issueDecisions: issueUpdate?.decisions ?? [],
        sourceDecision,
        eqDecisions: {
          freq: freqDecision ?? createMissingEqDecision(EQ_MIN_DISPLAY_FREQ_DELTA_HZ),
          gain: gainDecision ?? createMissingEqDecision(EQ_MIN_DISPLAY_GAIN_DELTA_DB)
        }
      }
    : undefined;

  if (!commitDisplayed) {
    return {
      nextState,
      displayedResult: undefined,
      debugSnapshot
    };
  }

  const displayedDetectedSources = buildDisplayedDetectedSources(
    rawResult.detectedSources,
    nextState.slowSourceProbs,
    nextState.displayedSourceState
  );
  const displayedAnalysis = buildMlAnalysisResult({
    issueScores: nextState.slowIssueProbs,
    sourceScores: nextState.slowSourceProbs,
    eq: {
      frequencyHz: nextState.displayedEqFreqHz ?? nextState.slowEqFreqHz ?? fastEqFreqHz ?? payload.eqFreqHz,
      gainDb: nextState.displayedEqGainDb ?? nextState.slowEqGainDb ?? fastEqGainDb ?? payload.eqGainDb
    },
    activeIssues: ISSUE_LABELS.filter((label) => nextState.displayedIssueStates[label] === 'persistent'),
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
    monitoringInterpretation: interpretation,
    monitoringStabilization: debugSnapshot,
    timestamp: now
  };

  return {
    nextState,
    displayedResult,
    debugSnapshot
  };
}

function applyIssueStateDecisions(
  previousStates: Record<TrainableIssueLabel, MonitoringIssueState>,
  fastProbabilities: Record<TrainableIssueLabel, number>,
  slowProbabilities: Record<TrainableIssueLabel, number>
) {
  const nextStates = { ...previousStates };
  const decisions: MonitoringIssueDecision[] = [];

  ISSUE_LABELS.forEach((label) => {
    const previousState = previousStates[label];
    const fastProbability = clamp(fastProbabilities[label] ?? 0, 0, 1);
    const slowProbability = clamp(slowProbabilities[label] ?? 0, 0, 1);
    const persistentOn = slowProbability >= ISSUE_PERSISTENT_SLOW_FORCE_ON_THRESHOLD
      || (fastProbability >= ISSUE_ON_THRESHOLD && slowProbability >= ISSUE_PERSISTENT_SLOW_ON_THRESHOLD);
    const persistentHold = slowProbability >= ISSUE_PERSISTENT_SLOW_FORCE_HOLD_THRESHOLD
      || (fastProbability >= ISSUE_OFF_THRESHOLD && slowProbability >= ISSUE_PERSISTENT_SLOW_HOLD_THRESHOLD);
    const transientOn = fastProbability >= ISSUE_ON_THRESHOLD
      && slowProbability <= ISSUE_TRANSIENT_SLOW_MAX_ON_THRESHOLD;
    const transientHold = fastProbability >= ISSUE_OFF_THRESHOLD
      && slowProbability <= ISSUE_TRANSIENT_SLOW_MAX_HOLD_THRESHOLD;
    let nextState: MonitoringIssueState = 'off';
    let reason: MonitoringIssueDecision['reason'] = 'off';

    if (previousState === 'persistent') {
      if (persistentHold) {
        nextState = 'persistent';
        reason = slowProbability >= ISSUE_PERSISTENT_SLOW_FORCE_HOLD_THRESHOLD
          ? 'persistent_from_slow'
          : 'persistent_hold';
      } else if (transientOn) {
        nextState = 'transient';
        reason = 'transient_on';
      }
    } else if (persistentOn) {
      nextState = 'persistent';
      reason = slowProbability >= ISSUE_PERSISTENT_SLOW_FORCE_ON_THRESHOLD
        ? 'persistent_from_slow'
        : 'persistent_on';
    } else if (previousState === 'transient' ? transientHold : transientOn) {
      nextState = 'transient';
      reason = previousState === 'transient' ? 'transient_hold' : 'transient_on';
    }

    nextStates[label] = nextState;
    decisions.push({
      label,
      previousState,
      nextState,
      fastProbability: Number(fastProbability.toFixed(4)),
      slowProbability: Number(slowProbability.toFixed(4)),
      reason
    });
  });

  return {
    nextStates,
    decisions
  };
}

function selectStableDominantSource(
  previousSource: SourceLabel | null,
  slowProbabilities: Record<SourceLabel, number>,
  timeBuffer: MonitoringTimeBuffer
): MonitoringSourceDecision {
  const rankedSources = SOURCE_LABELS
    .map((source) => ({
      source,
      probability: clamp(slowProbabilities[source] ?? 0, 0, 1)
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
      previousProbability: previousSource ? Number((slowProbabilities[previousSource] ?? 0).toFixed(4)) : null,
      candidateProbability: null,
      margin: SOURCE_SWITCH_MARGIN,
      sustainedFrames: 0,
      requiredFrames: SOURCE_SWITCH_SUSTAINED_FRAMES,
      reason: 'empty'
    };
  }

  const sustainedFrames = getRecentDominantSourceFrameCount(
    timeBuffer,
    candidateSource,
    SOURCE_SWITCH_SUSTAINED_FRAMES
  );

  if (!previousSource) {
    return {
      previousSource: null,
      candidateSource,
      nextSource: candidateSource,
      previousProbability: null,
      candidateProbability,
      margin: SOURCE_SWITCH_MARGIN,
      sustainedFrames,
      requiredFrames: SOURCE_SWITCH_SUSTAINED_FRAMES,
      reason: 'initial'
    };
  }

  const previousProbability = Number((slowProbabilities[previousSource] ?? 0).toFixed(4));

  if (candidateSource === previousSource) {
    return {
      previousSource,
      candidateSource,
      nextSource: previousSource,
      previousProbability,
      candidateProbability,
      margin: SOURCE_SWITCH_MARGIN,
      sustainedFrames,
      requiredFrames: SOURCE_SWITCH_SUSTAINED_FRAMES,
      reason: 'retain'
    };
  }

  if ((candidateProbability - previousProbability) >= SOURCE_SWITCH_MARGIN) {
    if (sustainedFrames >= SOURCE_SWITCH_SUSTAINED_FRAMES) {
      return {
        previousSource,
        candidateSource,
        nextSource: candidateSource,
        previousProbability,
        candidateProbability,
        margin: SOURCE_SWITCH_MARGIN,
        sustainedFrames,
        requiredFrames: SOURCE_SWITCH_SUSTAINED_FRAMES,
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
      sustainedFrames,
      requiredFrames: SOURCE_SWITCH_SUSTAINED_FRAMES,
      reason: 'pending'
    };
  }

  return {
    previousSource,
    candidateSource,
    nextSource: previousSource,
    previousProbability,
    candidateProbability,
    margin: SOURCE_SWITCH_MARGIN,
    sustainedFrames,
    requiredFrames: SOURCE_SWITCH_SUSTAINED_FRAMES,
    reason: 'retain'
  };
}

function applyEqDisplayThreshold(
  previousValue: number | null,
  fastValue: number | null,
  slowValue: number | null,
  threshold: number,
  suddenShiftThreshold: number
): MonitoringEqDecision {
  if (slowValue === null || !Number.isFinite(slowValue)) {
    return createMissingEqDecision(threshold);
  }

  const roundedSlowValue = roundForDisplay(slowValue, threshold);
  const roundedFastValue = roundNullable(fastValue, threshold < 1 ? 1 : 0);
  const suddenShift = fastValue !== null
    && Number.isFinite(fastValue)
    && Math.abs((fastValue ?? 0) - slowValue) >= suddenShiftThreshold;

  if (previousValue === null || !Number.isFinite(previousValue)) {
    return {
      previousValue: null,
      fastValue: roundedFastValue,
      slowValue: roundedSlowValue,
      displayedValue: roundedSlowValue,
      delta: null,
      threshold,
      suddenShift,
      reason: 'initial'
    };
  }

  const delta = Math.abs(slowValue - previousValue);

  if (delta >= threshold) {
    return {
      previousValue: roundNullable(previousValue, threshold < 1 ? 1 : 0),
      fastValue: roundedFastValue,
      slowValue: roundedSlowValue,
      displayedValue: roundedSlowValue,
      delta: Number(delta.toFixed(threshold < 1 ? 2 : 1)),
      threshold,
      suddenShift,
      reason: 'updated'
    };
  }

  return {
    previousValue: roundNullable(previousValue, threshold < 1 ? 1 : 0),
    fastValue: roundedFastValue,
    slowValue: roundedSlowValue,
    displayedValue: roundNullable(previousValue, threshold < 1 ? 1 : 0),
    delta: Number(delta.toFixed(threshold < 1 ? 2 : 1)),
    threshold,
    suddenShift,
    reason: 'suppressed'
  };
}

function buildMonitoringInterpretation(
  state: MonitoringMlRuntimeState,
  options: {
    suddenEqShift: boolean;
  }
): MonitoringInterpretation {
  const transientIssues = ISSUE_LABELS.filter((label) => state.displayedIssueStates[label] === 'transient');
  const persistentIssues = ISSUE_LABELS.filter((label) => state.displayedIssueStates[label] === 'persistent');

  return {
    fastIssueProbs: roundScoreRecord(state.smoothedIssueProbs),
    slowIssueProbs: roundScoreRecord(state.slowIssueProbs),
    fastSourceProbs: roundScoreRecord(state.smoothedSourceProbs),
    slowSourceProbs: roundScoreRecord(state.slowSourceProbs),
    fastEqFreqHz: roundNullable(state.smoothedEqFreqHz),
    slowEqFreqHz: roundNullable(state.slowEqFreqHz),
    fastEqGainDb: roundNullable(state.smoothedEqGainDb, 1),
    slowEqGainDb: roundNullable(state.slowEqGainDb, 1),
    issueStates: { ...state.displayedIssueStates },
    transientIssues,
    persistentIssues,
    displayedSourceState: state.displayedSourceState,
    suddenEqShift: options.suddenEqShift
  };
}

function buildDisplayedDetectedSources(
  detectedSources: AnalysisResult['detectedSources'],
  slowSourceProbabilities: Record<SourceLabel, number>,
  displayedSource: SourceLabel | null
) {
  if (!detectedSources || detectedSources.length === 0) {
    return buildDetectedSourcesFromScores(slowSourceProbabilities, {
      dominantSource: displayedSource
    });
  }

  const sourceMap = new Map(
    detectedSources.map((entry) => [entry.source, {
      ...entry,
      labels: uniqueInOrder(entry.labels)
    }])
  );

  if (displayedSource && (slowSourceProbabilities[displayedSource] ?? 0) > 0) {
    const existing = sourceMap.get(displayedSource);
    sourceMap.set(displayedSource, {
      source: displayedSource,
      confidence: Number(Math.max(existing?.confidence ?? 0, slowSourceProbabilities[displayedSource] ?? 0).toFixed(2)),
      labels: uniqueInOrder([
        `stable:${displayedSource}`,
        `model:${displayedSource}`,
        ...(existing?.labels ?? [])
      ])
    } satisfies DetectedAudioSource);
  }

  return [...sourceMap.values()]
    .sort((left, right) => {
      if (displayedSource && left.source === displayedSource && right.source !== displayedSource) {
        return -1;
      }

      if (displayedSource && right.source === displayedSource && left.source !== displayedSource) {
        return 1;
      }

      return right.confidence - left.confidence;
    })
    .slice(0, 5);
}

function createMissingEqDecision(threshold: number): MonitoringEqDecision {
  return {
    previousValue: null,
    fastValue: null,
    slowValue: null,
    displayedValue: null,
    delta: null,
    threshold,
    suddenShift: false,
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
    record[label] = 'off';
    return record;
  }, {} as Record<TrainableIssueLabel, MonitoringIssueState>);
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

function uniqueInOrder<T>(values: T[]) {
  return values.filter((value, index) => values.indexOf(value) === index);
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
