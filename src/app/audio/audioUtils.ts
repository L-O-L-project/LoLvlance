import type { BufferedAudioSnapshot } from '../types';

export const TARGET_SAMPLE_RATE = 16_000;
export const ROLLING_BUFFER_SECONDS = 3;
export const ROLLING_BUFFER_SIZE = TARGET_SAMPLE_RATE * ROLLING_BUFFER_SECONDS;

export function resampleMonoBuffer(
  input: Float32Array,
  sourceSampleRate: number,
  targetSampleRate = TARGET_SAMPLE_RATE
): Float32Array {
  if (input.length === 0 || sourceSampleRate <= 0) {
    return new Float32Array(0);
  }

  if (sourceSampleRate === targetSampleRate) {
    return input.slice();
  }

  if (sourceSampleRate < targetSampleRate) {
    const ratio = sourceSampleRate / targetSampleRate;
    const outputLength = Math.max(1, Math.round(input.length / ratio));
    const output = new Float32Array(outputLength);

    for (let index = 0; index < outputLength; index += 1) {
      const position = index * ratio;
      const baseIndex = Math.floor(position);
      const nextIndex = Math.min(baseIndex + 1, input.length - 1);
      const fraction = position - baseIndex;

      output[index] = input[baseIndex] + (input[nextIndex] - input[baseIndex]) * fraction;
    }

    return output;
  }

  const ratio = sourceSampleRate / targetSampleRate;
  const outputLength = Math.max(1, Math.round(input.length / ratio));
  const output = new Float32Array(outputLength);
  let inputOffset = 0;

  for (let index = 0; index < outputLength; index += 1) {
    const nextOffset = Math.min(input.length, Math.round((index + 1) * ratio));
    let sum = 0;
    let count = 0;

    for (let cursor = inputOffset; cursor < nextOffset; cursor += 1) {
      sum += input[cursor];
      count += 1;
    }

    output[index] = count > 0
      ? sum / count
      : input[Math.min(inputOffset, input.length - 1)];

    inputOffset = nextOffset;
  }

  return output;
}

export function appendToCircularBuffer(
  target: Float32Array,
  input: Float32Array,
  writeIndex: number,
  filledLength: number
): { writeIndex: number; filledLength: number } {
  let nextWriteIndex = writeIndex;
  let nextFilledLength = filledLength;

  for (let index = 0; index < input.length; index += 1) {
    target[nextWriteIndex] = input[index];
    nextWriteIndex = (nextWriteIndex + 1) % target.length;
    nextFilledLength = Math.min(target.length, nextFilledLength + 1);
  }

  return {
    writeIndex: nextWriteIndex,
    filledLength: nextFilledLength
  };
}

export function readCircularBuffer(
  source: Float32Array,
  writeIndex: number,
  filledLength: number
): Float32Array {
  if (filledLength === 0) {
    return new Float32Array(0);
  }

  const output = new Float32Array(filledLength);
  const startIndex = (writeIndex - filledLength + source.length) % source.length;

  if (startIndex + filledLength <= source.length) {
    output.set(source.subarray(startIndex, startIndex + filledLength));
    return output;
  }

  const firstChunk = source.length - startIndex;
  output.set(source.subarray(startIndex), 0);
  output.set(source.subarray(0, filledLength - firstChunk), firstChunk);

  return output;
}

export function calculateRms(samples: Float32Array): number {
  if (samples.length === 0) {
    return 0;
  }

  let sum = 0;

  for (let index = 0; index < samples.length; index += 1) {
    const value = samples[index];
    sum += value * value;
  }

  return Math.sqrt(sum / samples.length);
}

export function calculatePeak(samples: Float32Array): number {
  let peak = 0;

  for (let index = 0; index < samples.length; index += 1) {
    peak = Math.max(peak, Math.abs(samples[index]));
  }

  return peak;
}

export function createBufferedAudioSnapshot(samples: Float32Array): BufferedAudioSnapshot {
  const durationMs = (samples.length / TARGET_SAMPLE_RATE) * 1000;

  return {
    samples,
    sampleRate: TARGET_SAMPLE_RATE,
    durationMs,
    rms: calculateRms(samples),
    peak: calculatePeak(samples)
  };
}
