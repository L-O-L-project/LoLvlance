export type Instrument = 'vocal' | 'guitar' | 'bass' | 'drums' | 'keys' | 'overall';

export type TrainableIssueLabel =
  | 'muddy'
  | 'harsh'
  | 'buried'
  | 'boomy'
  | 'thin'
  | 'boxy'
  | 'nasal'
  | 'sibilant'
  | 'dull';

export type DerivedDiagnosisType =
  | 'vocal_buried'
  | 'guitar_harsh'
  | 'bass_muddy'
  | 'drums_overpower'
  | 'keys_masking';

export type SourceLabel = Exclude<Instrument, 'overall'>;

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
  | DerivedDiagnosisType;

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
export type LabelQuality = 'weak' | 'reviewed' | 'derived' | 'unavailable';

export interface EQRecommendation {
  freq_range: string;
  gain: string;
  reason: string;
}

export interface DetectedAudioSource {
  source: Instrument;
  confidence: number;
  labels: string[];
}

export interface StemMetric {
  stem: string;
  source?: Instrument | null;
  confidence: number;
  rms: number;
  peak: number;
  energy: number;
  energyRatio: number;
  sampleRate: number;
  frames: number;
}

export interface StemServiceStatus {
  connected: boolean;
  provider: 'stem-service' | 'browser-fallback';
  model?: string;
}

export interface DerivedDiagnosisResult {
  score: number;
  reasons: string[];
  explanation?: string;
}

export interface MlEqRuntimeOutput {
  frequency_normalized: number;
  frequency_hz: number;
  gain_db: number;
}

export interface MlInferenceOutput {
  schema_version: string;
  issues: Partial<Record<TrainableIssueLabel, number>>;
  sources: Partial<Record<SourceLabel, number>>;
  derived_diagnoses: Partial<Record<DerivedDiagnosisType, DerivedDiagnosisResult>>;
  metadata: {
    thresholds_used: Record<string, number>;
    label_quality: Partial<Record<TrainableIssueLabel | SourceLabel | DerivedDiagnosisType, LabelQuality>>;
    model_eq?: MlEqRuntimeOutput;
  };
}

export type MonitoringIssueState = 'off' | 'transient' | 'persistent';

export interface MonitoringInterpretation {
  fastIssueProbs: Partial<Record<TrainableIssueLabel, number>>;
  slowIssueProbs: Partial<Record<TrainableIssueLabel, number>>;
  fastSourceProbs: Partial<Record<SourceLabel, number>>;
  slowSourceProbs: Partial<Record<SourceLabel, number>>;
  fastEqFreqHz: number | null;
  slowEqFreqHz: number | null;
  fastEqGainDb: number | null;
  slowEqGainDb: number | null;
  issueStates: Record<TrainableIssueLabel, MonitoringIssueState>;
  transientIssues: TrainableIssueLabel[];
  persistentIssues: TrainableIssueLabel[];
  displayedSourceState: SourceLabel | null;
  suddenEqShift: boolean;
}

export interface MonitoringIssueDecision {
  label: TrainableIssueLabel;
  previousState: MonitoringIssueState;
  nextState: MonitoringIssueState;
  fastProbability: number;
  slowProbability: number;
  reason: 'persistent_on' | 'persistent_hold' | 'persistent_from_slow' | 'transient_on' | 'transient_hold' | 'off';
}

export interface MonitoringSourceDecision {
  previousSource: SourceLabel | null;
  candidateSource: SourceLabel | null;
  nextSource: SourceLabel | null;
  previousProbability: number | null;
  candidateProbability: number | null;
  margin: number;
  sustainedFrames: number;
  requiredFrames: number;
  reason: 'initial' | 'retain' | 'switch' | 'pending' | 'empty';
}

export interface MonitoringEqDecision {
  previousValue: number | null;
  fastValue: number | null;
  slowValue: number | null;
  displayedValue: number | null;
  delta: number | null;
  threshold: number;
  suddenShift: boolean;
  reason: 'initial' | 'updated' | 'suppressed' | 'missing';
}

export interface MonitoringHistoryFrameSnapshot {
  timestamp: number;
  fastIssueProbs: Partial<Record<TrainableIssueLabel, number>>;
  fastSourceProbs: Partial<Record<SourceLabel, number>>;
  fastEqFreqHz: number | null;
  fastEqGainDb: number | null;
}

export interface MonitoringStabilizationDebug {
  inferenceTimestamp: number;
  displayCommitted: boolean;
  bufferSize: number;
  bufferCapacity: number;
  bufferFrames: MonitoringHistoryFrameSnapshot[];
  rawIssueProbs: Partial<Record<TrainableIssueLabel, number>>;
  rawSourceProbs: Partial<Record<SourceLabel, number>>;
  rawEqFreqHz: number | null;
  rawEqGainDb: number | null;
  fastIssueProbs: Partial<Record<TrainableIssueLabel, number>>;
  fastSourceProbs: Partial<Record<SourceLabel, number>>;
  fastEqFreqHz: number | null;
  fastEqGainDb: number | null;
  slowIssueProbs: Partial<Record<TrainableIssueLabel, number>>;
  slowSourceProbs: Partial<Record<SourceLabel, number>>;
  slowEqFreqHz: number | null;
  slowEqGainDb: number | null;
  displayedIssueStates: Record<TrainableIssueLabel, MonitoringIssueState>;
  displayedSourceState: SourceLabel | null;
  displayedEqFreqHz: number | null;
  displayedEqGainDb: number | null;
  issueDecisions: MonitoringIssueDecision[];
  sourceDecision: MonitoringSourceDecision | null;
  eqDecisions: {
    freq: MonitoringEqDecision;
    gain: MonitoringEqDecision;
  };
}

export interface SourceEqRecommendation {
  source: Instrument;
  freq_range: string;
  gain: string;
  reason: string;
  confidence: number;
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

export type AnalysisEngine = 'ml' | 'rule-based' | 'rule-based-fallback';

export interface AnalysisResult {
  problems: DiagnosticProblem[];
  issues?: RuleBasedIssue[];
  eq_recommendations?: EQRecommendation[];
  detectedSources?: DetectedAudioSource[];
  stemMetrics?: StemMetric[];
  stemService?: StemServiceStatus;
  sourceEqRecommendations?: SourceEqRecommendation[];
  ml_output?: MlInferenceOutput;
  monitoringInterpretation?: MonitoringInterpretation;
  monitoringStabilization?: MonitoringStabilizationDebug;
  engine?: AnalysisEngine;
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
  /** RMS level in dBFS. -Infinity for silence. */
  dbRms: number;
  /** Peak / RMS ratio (linear). Typical music: 4–10. Over-compressed: < 3. */
  crestFactor: number;
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
