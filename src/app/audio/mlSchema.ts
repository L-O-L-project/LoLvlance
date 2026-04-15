import type {
  DetailedCause,
  DerivedDiagnosisType,
  SourceLabel,
  TrainableIssueLabel
} from '../types';

export const ML_SCHEMA_VERSION = '2.0.0';

export const ISSUE_LABELS = [
  'muddy',
  'harsh',
  'buried',
  'boomy',
  'thin',
  'boxy',
  'nasal',
  'sibilant',
  'dull'
] as const satisfies readonly TrainableIssueLabel[];

export const PRIMARY_UI_ISSUES = ['muddy', 'harsh', 'buried'] as const;

export const SOURCE_LABELS = ['vocal', 'guitar', 'bass', 'drums', 'keys'] as const satisfies readonly SourceLabel[];

export const DERIVED_DIAGNOSIS_LABELS = [
  'vocal_buried',
  'guitar_harsh',
  'bass_muddy',
  'drums_overpower',
  'keys_masking'
] as const satisfies readonly DerivedDiagnosisType[];

export const ISSUE_DEFAULT_THRESHOLDS: Record<TrainableIssueLabel, number> = {
  muddy: 0.4,
  harsh: 0.35,
  buried: 0.55,
  boomy: 0.3,
  thin: 0.35,
  boxy: 0.2,
  nasal: 0.45,
  sibilant: 0.48,
  dull: 0.35
};

export const SOURCE_DEFAULT_THRESHOLDS: Record<SourceLabel, number> = {
  vocal: 0.8,
  guitar: 0.25,
  bass: 0.8,
  drums: 0.75,
  keys: 0.8
};

export const DERIVED_DEFAULT_THRESHOLDS: Record<DerivedDiagnosisType, number> = {
  vocal_buried: 0.55,
  guitar_harsh: 0.55,
  bass_muddy: 0.55,
  drums_overpower: 0.58,
  keys_masking: 0.56
};

export const ISSUE_PROFILES: Record<
  TrainableIssueLabel,
  {
    details: DetailedCause[];
    fallbackEq: {
      frequencyHz: number;
      gainDb: number;
      reason: string;
    };
  }
> = {
  muddy: {
    details: ['low_frequency_buildup', 'low_mid_overlap', 'overlapping_sources'],
    fallbackEq: {
      frequencyHz: 315,
      gainDb: -3,
      reason: 'reduce low-mid buildup'
    }
  },
  harsh: {
    details: ['high_frequency_spike', 'guitar_presence_peak', 'sibilance'],
    fallbackEq: {
      frequencyHz: 5600,
      gainDb: -2.5,
      reason: 'tame upper-mid harshness'
    }
  },
  buried: {
    details: ['lack_of_presence', 'mid_range_masking', 'competing_sources'],
    fallbackEq: {
      frequencyHz: 3000,
      gainDb: 2.5,
      reason: 'restore presence and intelligibility'
    }
  },
  boomy: {
    details: ['low_frequency_buildup', 'boomy_resonance', 'room_resonance'],
    fallbackEq: {
      frequencyHz: 120,
      gainDb: -3,
      reason: 'tighten low-end resonance'
    }
  },
  thin: {
    details: ['missing_low_end', 'frequency_gap'],
    fallbackEq: {
      frequencyHz: 110,
      gainDb: 2.5,
      reason: 'restore low-end weight'
    }
  },
  boxy: {
    details: ['boxy_resonance', 'low_mid_overlap'],
    fallbackEq: {
      frequencyHz: 650,
      gainDb: -2.5,
      reason: 'reduce boxy midrange resonance'
    }
  },
  nasal: {
    details: ['nasal_peak', 'mid_range_masking'],
    fallbackEq: {
      frequencyHz: 1100,
      gainDb: -2,
      reason: 'soften nasal midrange focus'
    }
  },
  sibilant: {
    details: ['sibilance', 'high_frequency_spike'],
    fallbackEq: {
      frequencyHz: 6500,
      gainDb: -2.5,
      reason: 'control sibilant edge'
    }
  },
  dull: {
    details: ['missing_high_end', 'frequency_gap'],
    fallbackEq: {
      frequencyHz: 6500,
      gainDb: 2,
      reason: 'restore brightness and air'
    }
  }
};
