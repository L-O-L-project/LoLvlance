import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, Loader2, Mic } from 'lucide-react';
import { Header } from './components/Header';
import { ActionButton } from './components/ActionButton';
import { ResultCards } from './components/ResultCards';
import { MicrophoneStatusCard } from './components/MicrophoneStatusCard';
import { type Language, translations } from './translations';
import { useMonitoring } from './hooks/useMonitoring';
import {
  ENABLE_MODEL,
  isExperimentalModelVersion,
  MODEL_VERSION
} from './config/modelRuntime';
import type {
  AnalysisResult,
  MicrophoneErrorCode,
  MicrophonePermissionState
} from './types';
import { EQVisualization } from './components/EQVisualization';

type AppState = 'idle' | 'listening' | 'result' | 'monitoring' | 'error';
type AnalysisPhase = 'idle' | 'requesting_microphone' | 'recording' | 'analyzing' | 'complete' | 'failed';

const ANALYSIS_WINDOW_MS = 3000;
const ANALYSIS_TIMEOUT_MS = 12000;

export default function App() {
  const [appState, setAppState] = useState<AppState>('idle');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [language, setLanguage] = useState<Language>('en');
  const [analysisPhase, setAnalysisPhase] = useState<AnalysisPhase>('idle');
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const analysisTimeoutRef = useRef<number | null>(null);
  const analysisRunIdRef = useRef(0);

  const {
    analyserNode,
    analyseCurrentBuffer,
    latestFeatures,
    permissionError,
    permissionState,
    requestMicrophoneAccess,
    startMonitoring,
    stopCapture,
    stopMonitoring
  } = useMonitoring((newResult) => {
    setResult(newResult);
    setAppState('monitoring');
  });

  const t = translations[language];
  const showExperimentalBanner = ENABLE_MODEL && isExperimentalModelVersion(MODEL_VERSION);
  const showDisabledBanner = !ENABLE_MODEL;

  const clearAnalysisTimer = () => {
    if (analysisTimeoutRef.current !== null) {
      window.clearTimeout(analysisTimeoutRef.current);
      analysisTimeoutRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      clearAnalysisTimer();
    };
  }, []);

  const handleAnalyze = async () => {
    analysisRunIdRef.current += 1;
    clearAnalysisTimer();
    setAnalysisPhase('requesting_microphone');
    setAnalysisError(null);

    const isReady = await requestMicrophoneAccess();

    if (!isReady) {
      setAnalysisPhase('failed');
      return;
    }

    setResult(null);
    setAppState('listening');
    setAnalysisPhase('recording');

    const activeRunId = analysisRunIdRef.current;

    analysisTimeoutRef.current = window.setTimeout(() => {
      void (async () => {
        setAnalysisPhase('analyzing');

        try {
          const nextResult = await withTimeout(
            analyseCurrentBuffer(),
            ANALYSIS_TIMEOUT_MS,
            t.analysisTimeoutMessage
          );

          if (activeRunId !== analysisRunIdRef.current) {
            return;
          }

          if (nextResult.problems.length === 0 && (nextResult.runtimeWarnings?.some((warning) => warning.code === 'empty_analysis_result') ?? false)) {
            setAnalysisError(t.emptyAnalysisMessage);
          }

          setResult(nextResult);
          setAppState('result');
          setAnalysisPhase('complete');
        } catch (error) {
          if (activeRunId !== analysisRunIdRef.current) {
            return;
          }

          setAnalysisError(error instanceof Error ? error.message : t.analysisFailedMessage);
          setAppState('error');
          setAnalysisPhase('failed');
        } finally {
          analysisTimeoutRef.current = null;
          stopCapture();
        }
      })();
    }, ANALYSIS_WINDOW_MS);
  };

  const handleReset = () => {
    analysisRunIdRef.current += 1;
    clearAnalysisTimer();
    stopCapture();
    setAppState('idle');
    setResult(null);
    setAnalysisPhase('idle');
    setAnalysisError(null);
  };

  const handleStartMonitoring = async () => {
    clearAnalysisTimer();

    const didStart = await startMonitoring();

    if (!didStart) {
      return;
    }

    setAppState('monitoring');
    setAnalysisPhase('complete');
  };

  const handleStopMonitoring = () => {
    analysisRunIdRef.current += 1;
    clearAnalysisTimer();
    stopMonitoring();
    setAppState('idle');
    setResult(null);
    setAnalysisPhase('idle');
    setAnalysisError(null);
  };

  const toggleLanguage = () => {
    setLanguage((previousLanguage) => (previousLanguage === 'en' ? 'ko' : 'en'));
  };

  const headerStatus = getHeaderStatus({
    appState,
    permissionError,
    permissionState,
    translations: t
  });

  const shouldShowMicrophoneStatus = permissionError !== null && appState !== 'listening' && appState !== 'monitoring';

  return (
    /*
     * Mobile-only layout — 360–430px width, always full-screen single column.
     * No tablet / desktop variants.
     */
    <div className="min-h-screen bg-[#0a0a0f] text-white">
      <div className="w-full min-h-screen bg-[#0f0f14] flex flex-col overflow-hidden">

        {/* Status-bar safe-area spacer */}
        <div className="h-12 flex-shrink-0" />

        <Header
          state={appState}
          language={language}
          onToggleLanguage={toggleLanguage}
          statusLabel={headerStatus.label}
          statusColorClassName={headerStatus.colorClassName}
          showPulse={headerStatus.showPulse}
        />

        {/* Body — stacked single column */}
        <div className="flex-1 flex flex-col min-h-0">

          {/* EQ Visualization */}
          <div className="flex-shrink-0 px-4 mb-3">
            <EQVisualization
              analyserNode={analyserNode}
              featureRms={appState === 'idle' ? null : latestFeatures?.rms ?? null}
              problems={result?.problems || []}
              isLive={appState === 'monitoring' || appState === 'listening'}
              state={appState}
            />
          </div>

          {/* Scrollable content + sticky CTA */}
          <div className="flex-1 flex flex-col px-4 pb-4 min-h-0">
            <div className="flex-1 min-h-0 overflow-y-auto">
              <div className="pb-3 space-y-3">

                {showDisabledBanner && (
                  <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4">
                    <div className="text-xs font-semibold uppercase tracking-[0.2em] text-amber-200">
                      Sample Analysis
                    </div>
                    <div className="mt-1.5 text-base font-semibold text-white">
                      Model disabled
                    </div>
                    <div className="mt-1 text-sm text-amber-100/80">
                      ML inference is turned off. Preview Result uses fallback analysis only.
                    </div>
                    <div className="mt-2 text-[11px] font-mono text-amber-200/70">
                      {MODEL_VERSION}
                    </div>
                  </div>
                )}

                {showExperimentalBanner && (
                  <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4">
                    <div className="text-xs font-semibold uppercase tracking-[0.2em] text-amber-200">
                      Experimental Mode
                    </div>
                    <div className="mt-1.5 text-base font-semibold text-white">
                      Not production-ready
                    </div>
                    <div className="mt-1 text-sm text-amber-100/80">
                      Results may be inaccurate.
                    </div>
                    <div className="mt-2 text-[11px] font-mono text-amber-200/70">
                      {MODEL_VERSION}
                    </div>
                  </div>
                )}

                {shouldShowMicrophoneStatus && (
                  <MicrophoneStatusCard
                    error={permissionError as Exclude<MicrophoneErrorCode, null>}
                    language={language}
                  />
                )}

                <AnalysisStatusCard
                  phase={analysisPhase}
                  appState={appState}
                  language={language}
                  errorMessage={analysisError}
                />

                {(appState === 'result' || appState === 'monitoring') && result && (
                  <ResultCards
                    result={result}
                    language={language}
                    isLive={appState === 'monitoring'}
                    features={latestFeatures ?? undefined}
                  />
                )}
              </div>
            </div>

            <ActionButton
              state={appState}
              onAnalyze={handleAnalyze}
              onReset={handleReset}
              onStartMonitoring={handleStartMonitoring}
              onStopMonitoring={handleStopMonitoring}
              language={language}
            />
          </div>
        </div>

        {/* Home-indicator bar */}
        <div className="h-8 flex items-center justify-center flex-shrink-0">
          <div className="w-32 h-1 bg-white/20 rounded-full" />
        </div>
      </div>
    </div>
  );
}

function AnalysisStatusCard({
  phase,
  appState,
  language,
  errorMessage
}: {
  phase: AnalysisPhase;
  appState: AppState;
  language: Language;
  errorMessage: string | null;
}) {
  if (phase === 'idle' || phase === 'complete' || appState === 'monitoring') {
    return null;
  }

  const t = translations[language];
  const status = getAnalysisStatusCopy(phase, t, errorMessage);
  const Icon = status.icon;

  return (
    <div className={`rounded-2xl border p-4 ${status.className}`}>
      <div className="flex items-start gap-3">
        <div className="w-11 h-11 rounded-full bg-white/10 flex items-center justify-center flex-shrink-0">
          <Icon className={`w-5 h-5 ${status.spin ? 'animate-spin' : ''}`} />
        </div>
        <div className="min-w-0">
          <div className="text-sm font-semibold text-white">{status.title}</div>
          <div className="mt-1 text-sm text-gray-300 leading-relaxed">{status.message}</div>
          {status.hint && (
            <div className="mt-2 text-xs text-gray-400">{status.hint}</div>
          )}
        </div>
      </div>
    </div>
  );
}

function getAnalysisStatusCopy(
  phase: AnalysisPhase,
  t: typeof translations.en,
  errorMessage: string | null
) {
  if (phase === 'requesting_microphone') {
    return {
      title: t.microphoneWaitingTitle,
      message: t.microphoneWaitingMessage,
      hint: t.microphoneWaitingHint,
      className: 'border-cyan-500/25 bg-cyan-500/10',
      icon: Mic,
      spin: false
    };
  }

  if (phase === 'recording') {
    return {
      title: t.recordingTitle,
      message: t.recordingMessage,
      hint: t.recordingHint,
      className: 'border-purple-500/25 bg-purple-500/10',
      icon: Mic,
      spin: false
    };
  }

  if (phase === 'analyzing') {
    return {
      title: t.modelAnalyzingTitle,
      message: t.modelAnalyzingMessage,
      hint: t.modelAnalyzingHint,
      className: 'border-blue-500/25 bg-blue-500/10',
      icon: Loader2,
      spin: true
    };
  }

  return {
    title: t.analysisFailedTitle,
    message: errorMessage ?? t.analysisFailedMessage,
    hint: t.analysisFailedHint,
    className: 'border-amber-500/30 bg-amber-500/10',
    icon: AlertTriangle,
    spin: false
  };
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  let timeoutId: number | null = null;

  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timeoutId = window.setTimeout(() => {
          reject(new Error(message));
        }, timeoutMs);
      })
    ]);
  } finally {
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId);
    }
  }
}

function getHeaderStatus({
  appState,
  permissionError,
  permissionState,
  translations: t
}: {
  appState: AppState;
  permissionError: MicrophoneErrorCode;
  permissionState: MicrophonePermissionState;
  translations: typeof translations.en;
}) {
  if (appState === 'monitoring') {
    return {
      label: undefined,
      colorClassName: undefined,
      showPulse: true
    };
  }

  if (permissionError === 'denied' || permissionState === 'denied') {
    return {
      label: t.microphoneBlocked,
      colorClassName: 'text-amber-300',
      showPulse: false
    };
  }

  if (permissionError === 'not_found') {
    return {
      label: t.microphoneUnavailable,
      colorClassName: 'text-amber-300',
      showPulse: false
    };
  }

  if (permissionError === 'unsupported' || permissionState === 'unsupported') {
    return {
      label: t.microphoneUnsupported,
      colorClassName: 'text-amber-300',
      showPulse: false
    };
  }

  return {
    label: undefined,
    colorClassName: undefined,
    showPulse: false
  };
}
