import { ROLLING_BUFFER_SECONDS } from './audioUtils';
import type { BufferedAudioSnapshot, ExtractedAudioFeatures } from '../types';

const DEFAULT_WINDOW_MS = 25;
const DEFAULT_HOP_MS = 10;
const DEFAULT_FFT_SIZE = 512;
const DEFAULT_MEL_BIN_COUNT = 40;
const LOG_EPSILON = 1e-6;
const MIN_MEL_FREQUENCY = 20;

const hannWindowCache = new Map<number, Float32Array>();
const melFilterBankCache = new Map<string, Float32Array>();

interface FeatureExtractionConfig {
  fftSize?: number;
  melBinCount?: number;
  hopMs?: number;
  windowMs?: number;
}

export function extractAudioFeatures(
  snapshot: BufferedAudioSnapshot,
  config: FeatureExtractionConfig = {}
): ExtractedAudioFeatures {
  const sampleRate = snapshot.sampleRate;
  const frameSize = Math.max(1, Math.round((config.windowMs ?? DEFAULT_WINDOW_MS) * sampleRate / 1000));
  const hopSize = Math.max(1, Math.round((config.hopMs ?? DEFAULT_HOP_MS) * sampleRate / 1000));
  const fftSize = Math.max(nextPowerOfTwo(frameSize), config.fftSize ?? DEFAULT_FFT_SIZE);
  const melBinCount = config.melBinCount ?? DEFAULT_MEL_BIN_COUNT;
  const fixedSampleCount = Math.max(
    frameSize,
    Math.round(sampleRate * ROLLING_BUFFER_SECONDS)
  );
  const fixedSamples = normalizeSnapshotToFixedLength(snapshot.samples, fixedSampleCount);
  const frameCount = Math.floor((fixedSamples.length - frameSize) / hopSize) + 1;

  const hannWindow = getHannWindow(frameSize);
  const melFilterBank = getMelFilterBank(sampleRate, fftSize, melBinCount);
  const spectrumBinCount = fftSize / 2 + 1;
  const logMelSpectrogram = new Float32Array(frameCount * melBinCount);

  const fftInput = new Float32Array(fftSize);
  const realBuffer = new Float32Array(fftSize);
  const imaginaryBuffer = new Float32Array(fftSize);
  const powerSpectrum = new Float32Array(spectrumBinCount);

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const frameOffset = frameIndex * hopSize;
    fftInput.fill(0);

    // Window each frame before projecting it into the frequency domain.
    for (let sampleIndex = 0; sampleIndex < frameSize; sampleIndex += 1) {
      fftInput[sampleIndex] = fixedSamples[frameOffset + sampleIndex] * hannWindow[sampleIndex];
    }

    computePowerSpectrum(fftInput, realBuffer, imaginaryBuffer, powerSpectrum);

    // Collapse the linear spectrum into a compact mel representation.
    for (let melIndex = 0; melIndex < melBinCount; melIndex += 1) {
      const filterOffset = melIndex * spectrumBinCount;
      let melEnergy = 0;

      for (let binIndex = 0; binIndex < spectrumBinCount; binIndex += 1) {
        melEnergy += powerSpectrum[binIndex] * melFilterBank[filterOffset + binIndex];
      }

      logMelSpectrogram[frameIndex * melBinCount + melIndex] = Math.log(Math.max(LOG_EPSILON, melEnergy));
    }
  }

  return {
    logMelSpectrogram,
    logMelSpectrogramShape: [frameCount, melBinCount],
    rms: clamp(snapshot.rms, 0, 1),
    sampleRate,
    fftSize,
    frameSize,
    hopSize,
    melBinCount
  };
}

export function logExtractedAudioFeatures(features: ExtractedAudioFeatures) {
  console.info('[audio-features]', {
    logMelSpectrogramShape: features.logMelSpectrogramShape,
    rms: Number(features.rms.toFixed(4))
  });
}

function normalizeSnapshotToFixedLength(samples: Float32Array, targetLength: number) {
  if (samples.length === targetLength) {
    return samples.slice();
  }

  if (samples.length > targetLength) {
    return samples.slice(samples.length - targetLength);
  }

  const normalized = new Float32Array(targetLength);
  normalized.set(samples, targetLength - samples.length);
  return normalized;
}

function getHannWindow(length: number) {
  const cachedWindow = hannWindowCache.get(length);

  if (cachedWindow) {
    return cachedWindow;
  }

  const window = new Float32Array(length);

  for (let index = 0; index < length; index += 1) {
    window[index] = 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / Math.max(1, length - 1));
  }

  hannWindowCache.set(length, window);
  return window;
}

function getMelFilterBank(sampleRate: number, fftSize: number, melBinCount: number) {
  const cacheKey = `${sampleRate}:${fftSize}:${melBinCount}`;
  const cachedFilterBank = melFilterBankCache.get(cacheKey);

  if (cachedFilterBank) {
    return cachedFilterBank;
  }

  const spectrumBinCount = fftSize / 2 + 1;
  const maxFrequency = sampleRate / 2;
  const melMin = hzToMel(MIN_MEL_FREQUENCY);
  const melMax = hzToMel(maxFrequency);
  const melPoints = new Float32Array(melBinCount + 2);
  const hzPoints = new Float32Array(melBinCount + 2);
  const fftBins = new Uint16Array(melBinCount + 2);

  for (let index = 0; index < melPoints.length; index += 1) {
    melPoints[index] = melMin + ((melMax - melMin) * index) / (melBinCount + 1);
    hzPoints[index] = melToHz(melPoints[index]);
    fftBins[index] = Math.min(
      spectrumBinCount - 1,
      Math.floor(((fftSize + 1) * hzPoints[index]) / sampleRate)
    );
  }

  const filterBank = new Float32Array(melBinCount * spectrumBinCount);

  for (let melIndex = 0; melIndex < melBinCount; melIndex += 1) {
    const startBin = fftBins[melIndex];
    const centerBin = Math.max(startBin + 1, fftBins[melIndex + 1]);
    const endBin = Math.max(centerBin + 1, fftBins[melIndex + 2]);
    const filterOffset = melIndex * spectrumBinCount;

    for (let binIndex = startBin; binIndex < centerBin; binIndex += 1) {
      const denominator = Math.max(1, centerBin - startBin);
      filterBank[filterOffset + binIndex] = (binIndex - startBin) / denominator;
    }

    for (let binIndex = centerBin; binIndex < endBin && binIndex < spectrumBinCount; binIndex += 1) {
      const denominator = Math.max(1, endBin - centerBin);
      filterBank[filterOffset + binIndex] = (endBin - binIndex) / denominator;
    }
  }

  melFilterBankCache.set(cacheKey, filterBank);
  return filterBank;
}

function computePowerSpectrum(
  fftInput: Float32Array,
  realBuffer: Float32Array,
  imaginaryBuffer: Float32Array,
  powerSpectrum: Float32Array
) {
  realBuffer.set(fftInput);
  imaginaryBuffer.fill(0);
  performInPlaceFft(realBuffer, imaginaryBuffer);

  for (let binIndex = 0; binIndex < powerSpectrum.length; binIndex += 1) {
    const real = realBuffer[binIndex];
    const imaginary = imaginaryBuffer[binIndex];
    powerSpectrum[binIndex] = (real * real + imaginary * imaginary) / fftInput.length;
  }
}

function performInPlaceFft(real: Float32Array, imaginary: Float32Array) {
  const size = real.length;
  let swapIndex = 0;

  for (let index = 1; index < size; index += 1) {
    let bit = size >> 1;

    while (swapIndex & bit) {
      swapIndex ^= bit;
      bit >>= 1;
    }

    swapIndex ^= bit;

    if (index < swapIndex) {
      const tempReal = real[index];
      const tempImaginary = imaginary[index];
      real[index] = real[swapIndex];
      imaginary[index] = imaginary[swapIndex];
      real[swapIndex] = tempReal;
      imaginary[swapIndex] = tempImaginary;
    }
  }

  for (let blockSize = 2; blockSize <= size; blockSize <<= 1) {
    const halfBlockSize = blockSize >> 1;
    const phaseStep = -2 * Math.PI / blockSize;

    for (let blockStart = 0; blockStart < size; blockStart += blockSize) {
      for (let pairIndex = 0; pairIndex < halfBlockSize; pairIndex += 1) {
        const angle = phaseStep * pairIndex;
        const twiddleReal = Math.cos(angle);
        const twiddleImaginary = Math.sin(angle);
        const evenIndex = blockStart + pairIndex;
        const oddIndex = evenIndex + halfBlockSize;

        const oddReal = real[oddIndex] * twiddleReal - imaginary[oddIndex] * twiddleImaginary;
        const oddImaginary = real[oddIndex] * twiddleImaginary + imaginary[oddIndex] * twiddleReal;
        const evenReal = real[evenIndex];
        const evenImaginary = imaginary[evenIndex];

        real[evenIndex] = evenReal + oddReal;
        imaginary[evenIndex] = evenImaginary + oddImaginary;
        real[oddIndex] = evenReal - oddReal;
        imaginary[oddIndex] = evenImaginary - oddImaginary;
      }
    }
  }
}

function nextPowerOfTwo(value: number) {
  let power = 1;

  while (power < value) {
    power <<= 1;
  }

  return power;
}

function hzToMel(frequency: number) {
  return 2595 * Math.log10(1 + frequency / 700);
}

function melToHz(mel: number) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
