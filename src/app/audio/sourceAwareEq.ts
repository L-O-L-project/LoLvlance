import type {
  DetectedAudioSource,
  RuleBasedIssue,
  SourceEqRecommendation,
  StemMetric
} from '../types';

interface SourceEqContext {
  detectedSources: DetectedAudioSource[];
  issues: RuleBasedIssue[];
  stemMetrics: StemMetric[];
}

interface RecommendationTemplate {
  freq_range: string;
  gain: number;
  reason: string;
}

export function buildSourceAwareEqRecommendations({
  detectedSources,
  issues,
  stemMetrics
}: SourceEqContext): SourceEqRecommendation[] {
  const stemMap = new Map(
    stemMetrics
      .filter((stem) => stem.source)
      .map((stem) => [stem.source!, stem])
  );

  return detectedSources
    .map((entry) => {
      const stemMetric = stemMap.get(entry.source);
      const recommendation = selectRecommendation(entry.source, issues, stemMetric, entry.confidence);

      if (!recommendation) {
        return null;
      }

      const confidence = clamp(
        Math.max(entry.confidence, stemMetric?.energyRatio ?? 0) * 0.65 + recommendation.confidenceBias,
        0.4,
        0.95
      );

      return {
        source: entry.source,
        freq_range: recommendation.template.freq_range,
        gain: formatGain(recommendation.template.gain),
        reason: recommendation.template.reason,
        confidence: Number(confidence.toFixed(2))
      };
    })
    .filter((recommendation): recommendation is SourceEqRecommendation => recommendation !== null);
}

function selectRecommendation(
  source: DetectedAudioSource['source'],
  issues: RuleBasedIssue[],
  stemMetric: StemMetric | undefined,
  sourceConfidence: number
) {
  const energyRatio = stemMetric?.energyRatio ?? sourceConfidence;

  switch (source) {
    case 'vocal':
      return pickByIssue(issues, {
        muddy: {
          template: { freq_range: '180-320Hz', gain: -3.5, reason: 'clean up vocal mud and proximity buildup' },
          confidenceBias: 0.18
        },
        harsh: {
          template: { freq_range: '3000-5000Hz', gain: -3.0, reason: 'smooth vocal presence and sibilant edge' },
          confidenceBias: 0.2
        },
        buried: {
          template: { freq_range: '1500-3000Hz', gain: 3.0, reason: 'bring the vocal forward in the mix' },
          confidenceBias: 0.22
        }
      }) ?? (energyRatio < 0.12
        ? {
            template: { freq_range: '1500-2500Hz', gain: 2.0, reason: 'add vocal presence when the part feels recessed' },
            confidenceBias: 0.12
          }
        : null);
    case 'bass':
      return pickByIssue(issues, {
        muddy: {
          template: { freq_range: '180-320Hz', gain: -3.0, reason: 'tighten bass low-mid bloom' },
          confidenceBias: 0.2
        }
      }) ?? (energyRatio > 0.34
        ? {
            template: { freq_range: '60-90Hz', gain: -2.0, reason: 'rebalance dominant low-end weight' },
            confidenceBias: 0.14
          }
        : energyRatio < 0.08
          ? {
            template: { freq_range: '60-90Hz', gain: 2.5, reason: 'restore low-end foundation if bass feels light' },
            confidenceBias: 0.08
          }
          : null);
    case 'drums':
      return pickByIssue(issues, {
        harsh: {
          template: { freq_range: '4500-8000Hz', gain: -3.0, reason: 'tame cymbal splash and snare edge' },
          confidenceBias: 0.18
        }
      }) ?? (energyRatio > 0.35
        ? {
            template: { freq_range: '220-380Hz', gain: -2.0, reason: 'reduce kit masking in the low mids' },
            confidenceBias: 0.12
          }
        : energyRatio < 0.08
          ? {
            template: { freq_range: '2500-4000Hz', gain: 2.0, reason: 'add attack and definition to the drum stem' },
            confidenceBias: 0.06
          }
          : null);
    case 'guitar':
      return pickByIssue(issues, {
        muddy: {
          template: { freq_range: '200-350Hz', gain: -2.5, reason: 'clear boxy guitar low mids' },
          confidenceBias: 0.16
        },
        harsh: {
          template: { freq_range: '2500-4500Hz', gain: -3.0, reason: 'soften guitar bite and pick attack' },
          confidenceBias: 0.2
        },
        buried: {
          template: { freq_range: '1500-2800Hz', gain: 2.5, reason: 'help the guitar speak through the arrangement' },
          confidenceBias: 0.16
        }
      });
    case 'keys':
      return pickByIssue(issues, {
        muddy: {
          template: { freq_range: '250-450Hz', gain: -2.5, reason: 'reduce keyboard masking and low-mid fog' },
          confidenceBias: 0.18
        },
        buried: {
          template: { freq_range: '1000-2500Hz', gain: 2.5, reason: 'add clarity and melodic definition to the keys' },
          confidenceBias: 0.16
        }
      }) ?? (energyRatio > 0.24
        ? {
            template: { freq_range: '250-500Hz', gain: -2.0, reason: 'make room if keys are covering the mix center' },
            confidenceBias: 0.1
          }
        : null);
    default:
      return null;
  }
}

function pickByIssue(
  issues: RuleBasedIssue[],
  templates: Partial<Record<RuleBasedIssue, { template: RecommendationTemplate; confidenceBias: number }>>
) {
  for (const issue of issues) {
    const recommendation = templates[issue];

    if (recommendation) {
      return recommendation;
    }
  }

  return null;
}

function formatGain(gain: number) {
  const rounded = Math.round(gain * 10) / 10;
  return `${rounded > 0 ? '+' : ''}${rounded.toFixed(Math.abs(rounded % 1) < 0.05 ? 0 : 1)}dB`;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
