import type { BufferedAudioSnapshot } from '../types';
import {
  CLIPPING_PEAK_THRESHOLD,
  CLIPPING_SAMPLE_RATIO_THRESHOLD,
  MIN_MODEL_AUDIO_DURATION_MS,
  SILENCE_DBFS_THRESHOLD,
  SILENCE_RMS_THRESHOLD,
  VALID_SAMPLE_RATE_MAX,
  VALID_SAMPLE_RATE_MIN
} from './mlThresholds';

export type AudioQualityIssue =
  | 'empty_audio'
  | 'too_short'
  | 'silent'
  | 'clipped'
  | 'invalid_sample_rate'
  | 'non_finite_samples';

export interface AudioQualityReport {
  usableForMl: boolean;
  usableForFallback: boolean;
  issues: AudioQualityIssue[];
  durationMs: number;
  rms: number;
  dbRms: number;
  peak: number;
  clippingRatio: number;
  sampleRate: number;
}

export function analyzeAudioQuality(snapshot: BufferedAudioSnapshot): AudioQualityReport {
  const issues: AudioQualityIssue[] = [];
  const clippingRatio = calculateClippingRatio(snapshot.samples);

  if (snapshot.samples.length === 0) {
    issues.push('empty_audio');
  }

  if (!Number.isFinite(snapshot.sampleRate) || snapshot.sampleRate < VALID_SAMPLE_RATE_MIN || snapshot.sampleRate > VALID_SAMPLE_RATE_MAX) {
    issues.push('invalid_sample_rate');
  }

  if (snapshot.samples.some((sample) => !Number.isFinite(sample))) {
    issues.push('non_finite_samples');
  }

  if (snapshot.durationMs < MIN_MODEL_AUDIO_DURATION_MS) {
    issues.push('too_short');
  }

  if (snapshot.rms < SILENCE_RMS_THRESHOLD || snapshot.dbRms < SILENCE_DBFS_THRESHOLD) {
    issues.push('silent');
  }

  if (snapshot.peak >= CLIPPING_PEAK_THRESHOLD || clippingRatio >= CLIPPING_SAMPLE_RATIO_THRESHOLD) {
    issues.push('clipped');
  }

  const blockingIssues: AudioQualityIssue[] = [
    'empty_audio',
    'too_short',
    'silent',
    'invalid_sample_rate',
    'non_finite_samples'
  ];

  return {
    usableForMl: !issues.some((issue) => blockingIssues.includes(issue)),
    usableForFallback: !issues.some((issue) => issue === 'empty_audio' || issue === 'invalid_sample_rate' || issue === 'non_finite_samples'),
    issues,
    durationMs: snapshot.durationMs,
    rms: snapshot.rms,
    dbRms: snapshot.dbRms,
    peak: snapshot.peak,
    clippingRatio,
    sampleRate: snapshot.sampleRate
  };
}

export function sanitizeAudioSamples(samples: Float32Array) {
  const sanitized = new Float32Array(samples.length);

  for (let index = 0; index < samples.length; index += 1) {
    const sample = samples[index];
    sanitized[index] = Number.isFinite(sample)
      ? Math.max(-1, Math.min(1, sample))
      : 0;
  }

  return sanitized;
}

function calculateClippingRatio(samples: Float32Array) {
  if (samples.length === 0) {
    return 0;
  }

  let clippedCount = 0;

  for (let index = 0; index < samples.length; index += 1) {
    if (Math.abs(samples[index]) >= CLIPPING_PEAK_THRESHOLD) {
      clippedCount += 1;
    }
  }

  return clippedCount / samples.length;
}
