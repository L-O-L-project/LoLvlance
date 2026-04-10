import type {
  DetailedCause,
  DiagnosticProblem,
  EQRecommendation,
  ExtractedAudioFeatures,
  Instrument,
  RuleBasedAnalysis,
  RuleBasedIssue
} from '../types';

const LOG_EPSILON = 1e-6;
const MIN_MEL_FREQUENCY = 20;
const SILENCE_RMS_THRESHOLD = 0.012;

const LOW_MID_BAND: FrequencyRange = { minHz: 200, maxHz: 500 };
const HARSH_BAND: FrequencyRange = { minHz: 2000, maxHz: 6000 };
const PRESENCE_PEAK_BAND: FrequencyRange = { minHz: 3000, maxHz: 4000 };
const PRESENCE_BAND: FrequencyRange = { minHz: 1000, maxHz: 3000 };
const LOW_REFERENCE_BAND: FrequencyRange = { minHz: 200, maxHz: 1000 };
const HIGH_REFERENCE_BAND: FrequencyRange = { minHz: 3000, maxHz: 6000 };

const MUDDY_THRESHOLD = 0.39;
const HARSH_THRESHOLD = 0.31;
const HARSH_PEAK_THRESHOLD = 0.1;
const BURIED_THRESHOLD = 0.2;
const BURIED_CONTRAST_THRESHOLD = 0.78;

interface FrequencyRange {
  minHz: number;
  maxHz: number;
}

interface BandMetrics {
  totalEnergy: number;
  lowMidRatio: number;
  highRatio: number;
  presencePeakRatio: number;
  presenceRatio: number;
  presenceContrast: number;
}

interface RuleProfile {
  details: DetailedCause[];
  sources: Instrument[];
  getRecommendation: (confidence: number) => EQRecommendation;
}

const ruleProfiles: Record<RuleBasedIssue, RuleProfile> = {
  muddy: {
    details: ['low_mid_overlap', 'low_frequency_buildup'],
    sources: ['overall', 'bass', 'keys'],
    getRecommendation: (confidence) => ({
      freq_range: '200-400Hz',
      gain: formatGain(-interpolateGain(confidence, 2, 5)),
      reason: 'reduce muddiness in the low-mid range'
    })
  },
  harsh: {
    details: ['high_frequency_spike', 'guitar_presence_peak'],
    sources: ['overall', 'vocal', 'guitar'],
    getRecommendation: (confidence) => ({
      freq_range: '3000-5000Hz',
      gain: formatGain(-interpolateGain(confidence, 2, 6)),
      reason: 'tame harsh presence peaks'
    })
  },
  buried: {
    details: ['lack_of_presence', 'mid_range_masking'],
    sources: ['overall', 'vocal', 'guitar'],
    getRecommendation: (confidence) => ({
      freq_range: '1000-3000Hz',
      gain: formatGain(interpolateGain(confidence, 2, 5)),
      reason: 'restore presence and intelligibility'
    })
  }
};

export function analyzeRuleBasedAudioIssues(features: ExtractedAudioFeatures): RuleBasedAnalysis {
  if (features.rms < SILENCE_RMS_THRESHOLD) {
    return {
      issues: [],
      eq_recommendations: [],
      detections: []
    };
  }

  const averageMelEnergies = getAverageMelEnergies(features);
  const melBinCenters = getMelBinCenterFrequencies(features.sampleRate, features.melBinCount);
  const metrics = calculateBandMetrics(averageMelEnergies, melBinCenters);
  const detections = [
    detectMuddy(metrics),
    detectHarsh(metrics),
    detectBuried(metrics)
  ]
    .filter((detection): detection is NonNullable<typeof detection> => detection !== null)
    .sort((left, right) => right.confidence - left.confidence);

  return {
    issues: detections.map((detection) => detection.issue),
    eq_recommendations: detections.map((detection) => detection.eq_recommendation),
    detections
  };
}

export function ruleAnalysisToDiagnosticProblems(analysis: RuleBasedAnalysis): DiagnosticProblem[] {
  return analysis.detections.map((detection) => {
    const profile = ruleProfiles[detection.issue];
    const eqAction = `${detection.eq_recommendation.freq_range} ${detection.eq_recommendation.gain}`;

    return {
      type: detection.issue,
      confidence: detection.confidence,
      sources: profile.sources,
      details: profile.details,
      actions: [eqAction, detection.eq_recommendation.reason]
    };
  });
}

export function logRuleBasedAnalysis(analysis: RuleBasedAnalysis) {
  console.info('[audio-rules]', {
    issues: analysis.issues,
    eq_recommendations: analysis.eq_recommendations
  });
}

export function calculateBandEnergy(
  melEnergies: Float32Array,
  melBinCenters: Float32Array,
  range: FrequencyRange
) {
  let energy = 0;
  let matchedBins = 0;

  for (let index = 0; index < melEnergies.length; index += 1) {
    const frequency = melBinCenters[index];

    if (frequency >= range.minHz && frequency <= range.maxHz) {
      energy += melEnergies[index];
      matchedBins += 1;
    }
  }

  if (matchedBins > 0) {
    return energy;
  }

  let nearestIndex = 0;
  let nearestDistance = Number.POSITIVE_INFINITY;
  const midpoint = (range.minHz + range.maxHz) / 2;

  for (let index = 0; index < melBinCenters.length; index += 1) {
    const distance = Math.abs(melBinCenters[index] - midpoint);

    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestIndex = index;
    }
  }

  return melEnergies[nearestIndex];
}

function detectMuddy(metrics: BandMetrics) {
  if (metrics.lowMidRatio <= MUDDY_THRESHOLD) {
    return null;
  }

  return createDetection('muddy', metrics.lowMidRatio, MUDDY_THRESHOLD);
}

function detectHarsh(metrics: BandMetrics) {
  const harshRatio = Math.max(metrics.highRatio, metrics.presencePeakRatio * 1.35);

  if (harshRatio <= HARSH_THRESHOLD || metrics.presencePeakRatio <= HARSH_PEAK_THRESHOLD) {
    return null;
  }

  return createDetection('harsh', harshRatio, HARSH_THRESHOLD);
}

function detectBuried(metrics: BandMetrics) {
  const buriedScore = Math.max(
    BURIED_THRESHOLD - metrics.presenceRatio,
    BURIED_CONTRAST_THRESHOLD - metrics.presenceContrast
  );

  if (metrics.presenceRatio >= BURIED_THRESHOLD || metrics.presenceContrast >= BURIED_CONTRAST_THRESHOLD) {
    return null;
  }

  const ratio = Math.max(0, buriedScore + BURIED_THRESHOLD);
  return createDetection('buried', ratio, BURIED_THRESHOLD);
}

function createDetection(issue: RuleBasedIssue, ratio: number, threshold: number) {
  const confidence = clamp(0.55 + normalizeSeverity(ratio, threshold) * 0.4, 0.55, 0.95);
  const recommendation = ruleProfiles[issue].getRecommendation(confidence);

  return {
    issue,
    confidence: Number(confidence.toFixed(2)),
    ratio: Number(ratio.toFixed(4)),
    threshold,
    eq_recommendation: recommendation
  };
}

function getAverageMelEnergies(features: ExtractedAudioFeatures) {
  const [frameCount, melBinCount] = features.logMelSpectrogramShape;
  const averageEnergies = new Float32Array(melBinCount);

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const frameOffset = frameIndex * melBinCount;

    for (let melIndex = 0; melIndex < melBinCount; melIndex += 1) {
      averageEnergies[melIndex] += Math.exp(features.logMelSpectrogram[frameOffset + melIndex]);
    }
  }

  for (let melIndex = 0; melIndex < melBinCount; melIndex += 1) {
    averageEnergies[melIndex] /= Math.max(1, frameCount);
  }

  return averageEnergies;
}

function getMelBinCenterFrequencies(sampleRate: number, melBinCount: number) {
  const maxFrequency = sampleRate / 2;
  const melMin = hzToMel(MIN_MEL_FREQUENCY);
  const melMax = hzToMel(maxFrequency);
  const centers = new Float32Array(melBinCount);

  for (let index = 0; index < melBinCount; index += 1) {
    const normalizedIndex = (index + 1) / (melBinCount + 1);
    const centerMel = melMin + (melMax - melMin) * normalizedIndex;
    centers[index] = melToHz(centerMel);
  }

  return centers;
}

function calculateBandMetrics(melEnergies: Float32Array, melBinCenters: Float32Array): BandMetrics {
  const totalEnergy = Math.max(LOG_EPSILON, sum(melEnergies));
  const lowMidEnergy = calculateBandEnergy(melEnergies, melBinCenters, LOW_MID_BAND);
  const highEnergy = calculateBandEnergy(melEnergies, melBinCenters, HARSH_BAND);
  const presencePeakEnergy = calculateBandEnergy(melEnergies, melBinCenters, PRESENCE_PEAK_BAND);
  const presenceEnergy = calculateBandEnergy(melEnergies, melBinCenters, PRESENCE_BAND);
  const lowReferenceEnergy = calculateBandEnergy(melEnergies, melBinCenters, LOW_REFERENCE_BAND);
  const highReferenceEnergy = calculateBandEnergy(melEnergies, melBinCenters, HIGH_REFERENCE_BAND);
  const referenceEnergy = Math.max(LOG_EPSILON, (lowReferenceEnergy + highReferenceEnergy) * 0.5);

  return {
    totalEnergy,
    lowMidRatio: lowMidEnergy / totalEnergy,
    highRatio: highEnergy / totalEnergy,
    presencePeakRatio: presencePeakEnergy / totalEnergy,
    presenceRatio: presenceEnergy / totalEnergy,
    presenceContrast: presenceEnergy / referenceEnergy
  };
}

function normalizeSeverity(ratio: number, threshold: number) {
  const maxExpectedRatio = threshold + 0.2;
  return clamp((ratio - threshold) / Math.max(LOG_EPSILON, maxExpectedRatio - threshold), 0, 1);
}

function interpolateGain(confidence: number, minGain: number, maxGain: number) {
  const normalizedConfidence = clamp((confidence - 0.55) / 0.4, 0, 1);
  return Math.round((minGain + (maxGain - minGain) * normalizedConfidence) * 10) / 10;
}

function formatGain(gain: number) {
  return `${gain > 0 ? '+' : ''}${gain.toFixed(gain % 1 === 0 ? 0 : 1)}dB`;
}

function hzToMel(frequency: number) {
  return 2595 * Math.log10(1 + frequency / 700);
}

function melToHz(mel: number) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function sum(values: Float32Array) {
  let total = 0;

  for (let index = 0; index < values.length; index += 1) {
    total += values[index];
  }

  return total;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
