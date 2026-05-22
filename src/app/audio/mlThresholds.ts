// MVP-safe thresholds. These should be recalibrated with real validation audio
// before presenting model confidence as production-grade accuracy.
export const HIGH_CONFIDENCE_THRESHOLD = 0.75;
export const MEDIUM_CONFIDENCE_THRESHOLD = 0.45;
export const LOW_CONFIDENCE_THRESHOLD = 0.25;
export const MIN_USABLE_CONFIDENCE = 0.25;

export const SILENCE_RMS_THRESHOLD = 0.012;
export const SILENCE_DBFS_THRESHOLD = -38;
export const MIN_MODEL_AUDIO_DURATION_MS = 900;
export const MIN_RULE_AUDIO_DURATION_MS = 500;
export const CLIPPING_PEAK_THRESHOLD = 0.98;
export const CLIPPING_SAMPLE_RATIO_THRESHOLD = 0.01;
export const VALID_SAMPLE_RATE_MIN = 8_000;
export const VALID_SAMPLE_RATE_MAX = 96_000;

export const FALLBACK_MAX_CONFIDENCE = 0.72;
export const FALLBACK_BASE_CONFIDENCE = 0.45;

export const SOURCE_LIKELY_CONFIDENCE = 0.55;
export const SOURCE_UNCERTAIN_CONFIDENCE = 0.25;
