import type {
  MonitoringHistoryFrameSnapshot,
  SourceLabel,
  TrainableIssueLabel
} from '../types';
import { SLOW_SIGNAL_BUFFER_SECONDS, MONITORING_INFERENCE_STRIDE_MS } from '../config/monitoringStabilization';
import { ISSUE_LABELS, SOURCE_LABELS } from './mlSchema';

export interface MonitoringTimeBufferFrame {
  timestamp: number;
  fastIssueProbs: Record<TrainableIssueLabel, number>;
  fastSourceProbs: Record<SourceLabel, number>;
  fastEqFreqHz: number;
  fastEqGainDb: number;
}

export interface MonitoringTimeBufferSlowSignal {
  issueProbs: Record<TrainableIssueLabel, number>;
  sourceProbs: Record<SourceLabel, number>;
  eqFreqHz: number | null;
  eqGainDb: number | null;
}

export interface MonitoringTimeBuffer {
  capacity: number;
  size: number;
  nextIndex: number;
  frames: Array<MonitoringTimeBufferFrame | null>;
  issueSums: Record<TrainableIssueLabel, number>;
  sourceSums: Record<SourceLabel, number>;
}

export function createMonitoringTimeBuffer(
  capacity = Math.max(10, Math.round(SLOW_SIGNAL_BUFFER_SECONDS * 1000 / MONITORING_INFERENCE_STRIDE_MS))
): MonitoringTimeBuffer {
  return {
    capacity,
    size: 0,
    nextIndex: 0,
    frames: Array.from({ length: capacity }, () => null),
    issueSums: ISSUE_LABELS.reduce((record, label) => {
      record[label] = 0;
      return record;
    }, {} as Record<TrainableIssueLabel, number>),
    sourceSums: SOURCE_LABELS.reduce((record, label) => {
      record[label] = 0;
      return record;
    }, {} as Record<SourceLabel, number>)
  };
}

export function appendMonitoringTimeBufferFrame(
  buffer: MonitoringTimeBuffer,
  frame: MonitoringTimeBufferFrame
) {
  const evictedFrame = buffer.frames[buffer.nextIndex];

  if (evictedFrame) {
    ISSUE_LABELS.forEach((label) => {
      buffer.issueSums[label] -= evictedFrame.fastIssueProbs[label] ?? 0;
    });
    SOURCE_LABELS.forEach((label) => {
      buffer.sourceSums[label] -= evictedFrame.fastSourceProbs[label] ?? 0;
    });
  }

  buffer.frames[buffer.nextIndex] = frame;

  ISSUE_LABELS.forEach((label) => {
    buffer.issueSums[label] += frame.fastIssueProbs[label] ?? 0;
  });
  SOURCE_LABELS.forEach((label) => {
    buffer.sourceSums[label] += frame.fastSourceProbs[label] ?? 0;
  });

  if (buffer.size < buffer.capacity) {
    buffer.size += 1;
  }

  buffer.nextIndex = (buffer.nextIndex + 1) % buffer.capacity;
}

export function buildSlowSignalFromTimeBuffer(buffer: MonitoringTimeBuffer): MonitoringTimeBufferSlowSignal {
  if (buffer.size === 0) {
    return {
      issueProbs: ISSUE_LABELS.reduce((record, label) => {
        record[label] = 0;
        return record;
      }, {} as Record<TrainableIssueLabel, number>),
      sourceProbs: SOURCE_LABELS.reduce((record, label) => {
        record[label] = 0;
        return record;
      }, {} as Record<SourceLabel, number>),
      eqFreqHz: null,
      eqGainDb: null
    };
  }

  const denominator = buffer.size;
  const orderedFrames = getMonitoringTimeBufferFrames(buffer);

  return {
    issueProbs: ISSUE_LABELS.reduce((record, label) => {
      record[label] = clamp(buffer.issueSums[label] / denominator, 0, 1);
      return record;
    }, {} as Record<TrainableIssueLabel, number>),
    sourceProbs: SOURCE_LABELS.reduce((record, label) => {
      record[label] = clamp(buffer.sourceSums[label] / denominator, 0, 1);
      return record;
    }, {} as Record<SourceLabel, number>),
    eqFreqHz: median(orderedFrames.map((frame) => frame.fastEqFreqHz)),
    eqGainDb: median(orderedFrames.map((frame) => frame.fastEqGainDb))
  };
}

export function getMonitoringTimeBufferFrames(buffer: MonitoringTimeBuffer) {
  if (buffer.size === 0) {
    return [] as MonitoringTimeBufferFrame[];
  }

  const orderedFrames: MonitoringTimeBufferFrame[] = [];
  const startIndex = buffer.size === buffer.capacity ? buffer.nextIndex : 0;

  for (let offset = 0; offset < buffer.size; offset += 1) {
    const frame = buffer.frames[(startIndex + offset) % buffer.capacity];

    if (frame) {
      orderedFrames.push(frame);
    }
  }

  return orderedFrames;
}

export function getRecentDominantSourceFrameCount(
  buffer: MonitoringTimeBuffer,
  candidateSource: SourceLabel,
  maxFrames: number
) {
  const orderedFrames = getMonitoringTimeBufferFrames(buffer);
  let count = 0;

  for (let index = orderedFrames.length - 1; index >= 0 && count < maxFrames; index -= 1) {
    const frame = orderedFrames[index];
    const dominantSource = SOURCE_LABELS
      .map((source) => ({
        source,
        probability: frame.fastSourceProbs[source] ?? 0
      }))
      .sort((left, right) => right.probability - left.probability)[0]?.source ?? null;

    if (dominantSource !== candidateSource) {
      break;
    }

    count += 1;
  }

  return count;
}

export function serializeMonitoringTimeBuffer(
  buffer: MonitoringTimeBuffer
): MonitoringHistoryFrameSnapshot[] {
  return getMonitoringTimeBufferFrames(buffer).map((frame) => ({
    timestamp: frame.timestamp,
    fastIssueProbs: roundRecord(frame.fastIssueProbs),
    fastSourceProbs: roundRecord(frame.fastSourceProbs),
    fastEqFreqHz: Math.round(frame.fastEqFreqHz),
    fastEqGainDb: Number(frame.fastEqGainDb.toFixed(1))
  }));
}

function median(values: number[]) {
  if (values.length === 0) {
    return null;
  }

  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1] + sorted[middle]) / 2;
  }

  return sorted[middle];
}

function roundRecord<T extends string>(record: Record<T, number>) {
  return Object.fromEntries(
    Object.entries(record).map(([label, value]) => [label, Number(value.toFixed(4))])
  ) as Record<T, number>;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
