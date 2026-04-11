import type { BufferedAudioSnapshot, DetectedAudioSource } from '../types';

const STEM_SERVICE_URL = (import.meta.env.VITE_STEM_SERVICE_URL as string | undefined)
  ?? 'http://127.0.0.1:8765';
const STEM_SERVICE_MIN_DURATION_MS = 900;
const STEM_SERVICE_MIN_RMS = 0.012;
const HEALTH_TIMEOUT_MS = 1200;
const ANALYZE_TIMEOUT_MS = 20000;
const RETRY_BACKOFF_MS = 15000;

interface StemServiceResponse {
  detectedSources?: DetectedAudioSource[];
  model?: string;
  stems?: Array<{
    stem: string;
    source?: string | null;
    confidence: number;
    rms: number;
    peak: number;
  }>;
}

let lastHealthCheckAt = 0;
let lastKnownAvailability = false;
let retryAfterTimestamp = 0;

export async function warmUpStemSeparationService() {
  return checkStemServiceHealth(true);
}

export async function detectStemSeparatedSources(snapshot: BufferedAudioSnapshot) {
  if (
    snapshot.rms < STEM_SERVICE_MIN_RMS
    || snapshot.durationMs < STEM_SERVICE_MIN_DURATION_MS
    || Date.now() < retryAfterTimestamp
  ) {
    return [];
  }

  const isAvailable = await checkStemServiceHealth();

  if (!isAvailable) {
    return [];
  }

  try {
    const wavBuffer = encodeMonoPcm16Wav(snapshot.samples, snapshot.sampleRate);
    const response = await fetchWithTimeout(`${STEM_SERVICE_URL}/analyze-stems`, {
      method: 'POST',
      headers: {
        'Content-Type': 'audio/wav'
      },
      body: wavBuffer
    }, ANALYZE_TIMEOUT_MS);

    if (!response.ok) {
      throw new Error(`Stem service returned ${response.status}`);
    }

    const payload = await response.json() as StemServiceResponse;
    const detectedSources = Array.isArray(payload.detectedSources)
      ? payload.detectedSources.filter(isDetectedSource)
      : [];

    console.info('[audio-stems]', {
      model: payload.model ?? 'unknown',
      detectedSources,
      stems: payload.stems ?? []
    });

    return detectedSources;
  } catch (error) {
    retryAfterTimestamp = Date.now() + RETRY_BACKOFF_MS;
    lastKnownAvailability = false;
    console.warn('[audio-stems] Local stem service request failed.', error);
    return [];
  }
}

async function checkStemServiceHealth(force = false) {
  const now = Date.now();

  if (!force && now - lastHealthCheckAt < RETRY_BACKOFF_MS) {
    return lastKnownAvailability;
  }

  lastHealthCheckAt = now;

  try {
    const response = await fetchWithTimeout(`${STEM_SERVICE_URL}/health`, {
      method: 'GET'
    }, HEALTH_TIMEOUT_MS);

    lastKnownAvailability = response.ok;

    if (response.ok) {
      retryAfterTimestamp = 0;
    }

    return response.ok;
  } catch {
    lastKnownAvailability = false;
    return false;
  }
}

async function fetchWithTimeout(input: RequestInfo | URL, init: RequestInit, timeoutMs: number) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(input, {
      ...init,
      signal: controller.signal
    });
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function encodeMonoPcm16Wav(samples: Float32Array, sampleRate: number) {
  const clampedSampleRate = Math.max(1, Math.round(sampleRate));
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, clampedSampleRate, true);
  view.setUint32(28, clampedSampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;

  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    const scaled = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    view.setInt16(offset, Math.round(scaled), true);
    offset += 2;
  }

  return buffer;
}

function writeAscii(view: DataView, offset: number, value: string) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function isDetectedSource(value: unknown): value is DetectedAudioSource {
  return Boolean(
    value
    && typeof value === 'object'
    && 'source' in value
    && 'confidence' in value
    && 'labels' in value
  );
}
