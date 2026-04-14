export const MONITORING_WINDOW_SECONDS = 3;
export const MONITORING_INFERENCE_STRIDE_MS = 1000;
export const MONITORING_DISPLAY_COMMIT_MS = 2000;
export const MONITORING_MIN_BUFFER_MS = 750;
export const SLOW_SIGNAL_BUFFER_SECONDS = 15;
export const SOURCE_SWITCH_SUSTAINED_FRAMES = 3;

export const ISSUE_EMA_ALPHA = 0.35;
export const SOURCE_EMA_ALPHA = 0.35;
export const EQ_EMA_ALPHA = 0.25;

export const ISSUE_ON_THRESHOLD = 0.6;
export const ISSUE_OFF_THRESHOLD = 0.4;
export const ISSUE_PERSISTENT_SLOW_ON_THRESHOLD = 0.5;
export const ISSUE_PERSISTENT_SLOW_HOLD_THRESHOLD = 0.4;
export const ISSUE_PERSISTENT_SLOW_FORCE_ON_THRESHOLD = 0.6;
export const ISSUE_PERSISTENT_SLOW_FORCE_HOLD_THRESHOLD = 0.5;
export const ISSUE_TRANSIENT_SLOW_MAX_ON_THRESHOLD = 0.3;
export const ISSUE_TRANSIENT_SLOW_MAX_HOLD_THRESHOLD = 0.4;
export const SOURCE_SWITCH_MARGIN = 0.12;
export const EQ_MIN_DISPLAY_FREQ_DELTA_HZ = 150;
export const EQ_MIN_DISPLAY_GAIN_DELTA_DB = 0.8;
export const EQ_SUDDEN_SHIFT_FREQ_DELTA_HZ = 400;
export const EQ_SUDDEN_SHIFT_GAIN_DELTA_DB = 1.5;

export function isMonitoringStabilizationDebugEnabled() {
  if (typeof window === 'undefined') {
    return false;
  }

  const runtimeWindow = window as Window & {
    __LOLVLANCE_MONITORING_DEBUG__?: boolean;
  };

  return import.meta.env.DEV && (
    import.meta.env.VITE_MONITORING_STABILIZATION_DEBUG === 'true'
    || runtimeWindow.__LOLVLANCE_MONITORING_DEBUG__ === true
  );
}
