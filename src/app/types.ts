export type Instrument = 'vocal' | 'guitar' | 'bass' | 'drums' | 'keys' | 'overall';

// LEVEL 1 - USER FACING PROBLEMS
export type ProblemType = 
  // Primary
  | 'muddy'
  | 'harsh'
  | 'buried'
  | 'imbalance'
  // Secondary (expandable)
  | 'boomy'
  | 'thin'
  | 'boxy'
  | 'nasal'
  | 'sibilant'
  | 'dull'
  // Source-specific
  | 'vocal_buried'
  | 'guitar_harsh'
  | 'bass_muddy'
  | 'drums_overpower'
  | 'keys_masking';

// LEVEL 3 - DETAILED CAUSES
export type DetailedCause = 
  // Muddy
  | 'low_frequency_buildup'
  | 'low_mid_overlap'
  | 'boxy_resonance'
  | 'room_resonance'
  | 'overlapping_sources'
  // Harsh
  | 'high_frequency_spike'
  | 'sibilance'
  | 'cymbal_dominance'
  | 'guitar_presence_peak'
  // Buried
  | 'mid_range_masking'
  | 'lack_of_presence'
  | 'competing_sources'
  | 'level_too_low'
  // Imbalance
  | 'level_imbalance'
  | 'tonal_imbalance'
  | 'bass_overpower'
  | 'drums_dominance'
  | 'missing_low_end'
  | 'missing_high_end'
  // Secondary
  | 'boomy_resonance'
  | 'nasal_peak'
  | 'transient_overload'
  | 'frequency_gap';

// LEVEL 4 - ACTION TYPES
export type ActionType =
  // EQ Actions
  | 'cut_low_mid'
  | 'boost_presence'
  | 'cut_harsh'
  | 'control_sibilance'
  | 'boost_high'
  | 'cut_boomy'
  | 'cut_low_frequency'
  | 'boost_low_end'
  | 'reduce_nasal'
  // Mixing Actions
  | 'reduce_competing_sources'
  | 'increase_target_level'
  | 'rebalance_levels'
  | 'adjust_frequency_overlap'
  | 'isolate_problem_source';

export interface DiagnosticProblem {
  type: ProblemType;
  confidence: number; // 0-1
  sources: Instrument[];
  details: DetailedCause[];
  actions: string[];
}

export type RuleBasedIssue = 'muddy' | 'harsh' | 'buried';

export interface EQRecommendation {
  freq_range: string;
  gain: string;
  reason: string;
}

export interface RuleDetection {
  issue: RuleBasedIssue;
  confidence: number;
  ratio: number;
  threshold: number;
  eq_recommendation: EQRecommendation;
}

export interface RuleBasedAnalysis {
  issues: RuleBasedIssue[];
  eq_recommendations: EQRecommendation[];
  detections: RuleDetection[];
}

export interface AnalysisResult {
  problems: DiagnosticProblem[];
  issues?: RuleBasedIssue[];
  eq_recommendations?: EQRecommendation[];
  timestamp?: number;
}

export type MicrophonePermissionState = 'prompt' | 'granted' | 'denied' | 'unsupported';

export type MicrophoneErrorCode = 'denied' | 'not_found' | 'unsupported' | 'unknown' | null;

export interface BufferedAudioSnapshot {
  samples: Float32Array;
  sampleRate: number;
  durationMs: number;
  rms: number;
  peak: number;
}

export interface ExtractedAudioFeatures {
  logMelSpectrogram: Float32Array;
  logMelSpectrogramShape: [number, number];
  rms: number;
  sampleRate: number;
  fftSize: number;
  frameSize: number;
  hopSize: number;
  melBinCount: number;
}
