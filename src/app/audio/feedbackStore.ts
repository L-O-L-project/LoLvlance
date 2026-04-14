/**
 * feedbackStore.ts
 *
 * Stores per-analysis user feedback in localStorage and provides
 * an export function that produces JSONL for ML retraining.
 *
 * Captured per entry:
 *   - timestamp, session_id
 *   - analysis result (problems, engine, ml_output)
 *   - audio features snapshot (no raw audio)
 *   - verdict: 'correct' | 'wrong'
 *   - corrected_labels: optional override selected by user
 */

import type { AnalysisResult, ExtractedAudioFeatures } from '../types';

const STORAGE_KEY = 'lolvlance_feedback_v1';
const SESSION_KEY = 'lolvlance_session_id';
const MAX_ENTRIES = 2000;

export type FeedbackVerdict = 'correct' | 'wrong';

export interface FeedbackEntry {
  feedback_schema_version: '1.0';
  entry_id: string;
  session_id: string;
  timestamp_ms: number;
  verdict: FeedbackVerdict;
  corrected_labels: string[];          // user-selected actual issue labels (empty if correct)
  analysis: {
    engine: string;
    problems: Array<{ type: string; confidence: number; sources: string[] }>;
    ml_issues: Record<string, number>;
    ml_sources: Record<string, number>;
  };
  audio_features: {
    rms?: number;
    spectrogram_shape?: [number, number];
  };
}

// ---------------------------------------------------------------------------
// Session ID
// ---------------------------------------------------------------------------

function getOrCreateSessionId(): string {
  let id = sessionStorage.getItem(SESSION_KEY);
  if (!id) {
    id = `s_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
    sessionStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

// ---------------------------------------------------------------------------
// Serialise analysis result into a compact, stable shape
// ---------------------------------------------------------------------------

function summariseResult(result: AnalysisResult): FeedbackEntry['analysis'] {
  return {
    engine: result.engine ?? 'unknown',
    problems: result.problems.map((p) => ({
      type: p.type,
      confidence: Math.round(p.confidence * 1000) / 1000,
      sources: p.sources,
    })),
    ml_issues: result.ml_output
      ? Object.fromEntries(
          Object.entries(result.ml_output.issues).map(([k, v]) => [k, Math.round((v ?? 0) * 1000) / 1000])
        )
      : {},
    ml_sources: result.ml_output
      ? Object.fromEntries(
          Object.entries(result.ml_output.sources).map(([k, v]) => [k, Math.round((v ?? 0) * 1000) / 1000])
        )
      : {},
  };
}

function summariseFeatures(features: ExtractedAudioFeatures | undefined): FeedbackEntry['audio_features'] {
  if (!features) return {};
  return {
    rms: Math.round(features.rms * 10000) / 10000,
    spectrogram_shape: features.logMelSpectrogramShape,
  };
}

// ---------------------------------------------------------------------------
// localStorage read / write
// ---------------------------------------------------------------------------

function readEntries(): FeedbackEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return raw
      .split('\n')
      .filter(Boolean)
      .map((line) => JSON.parse(line) as FeedbackEntry);
  } catch {
    return [];
  }
}

function appendEntry(entry: FeedbackEntry): void {
  try {
    const existing = readEntries();
    // evict oldest if we hit the cap
    const trimmed = existing.length >= MAX_ENTRIES ? existing.slice(-(MAX_ENTRIES - 1)) : existing;
    trimmed.push(entry);
    localStorage.setItem(STORAGE_KEY, trimmed.map((e) => JSON.stringify(e)).join('\n'));
  } catch {
    // localStorage full or unavailable — silently ignore
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function saveFeedback({
  result,
  features,
  verdict,
  correctedLabels = [],
}: {
  result: AnalysisResult;
  features?: ExtractedAudioFeatures;
  verdict: FeedbackVerdict;
  correctedLabels?: string[];
}): FeedbackEntry {
  const entry: FeedbackEntry = {
    feedback_schema_version: '1.0',
    entry_id: `fb_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
    session_id: getOrCreateSessionId(),
    timestamp_ms: Date.now(),
    verdict,
    corrected_labels: correctedLabels,
    analysis: summariseResult(result),
    audio_features: summariseFeatures(features),
  };
  appendEntry(entry);
  return entry;
}

export function getFeedbackCount(): number {
  return readEntries().length;
}

export function exportFeedbackAsJsonl(): string {
  return readEntries()
    .map((e) => JSON.stringify(e))
    .join('\n');
}

export function downloadFeedback(): void {
  const jsonl = exportFeedbackAsJsonl();
  if (!jsonl.trim()) return;
  const blob = new Blob([jsonl], { type: 'application/x-ndjson' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `lolvlance_feedback_${new Date().toISOString().slice(0, 10)}.jsonl`;
  a.click();
  URL.revokeObjectURL(url);
}

export function clearFeedback(): void {
  localStorage.removeItem(STORAGE_KEY);
}
