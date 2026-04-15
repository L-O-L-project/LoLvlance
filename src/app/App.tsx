import { useEffect, useRef, useState } from 'react';
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

type AppState = 'idle' | 'listening' | 'result' | 'monitoring';

const ANALYSIS_WINDOW_MS = 3000;

export default function App() {
  const [appState, setAppState] = useState<AppState>('idle');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [language, setLanguage] = useState<Language>('en');
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

    const isReady = await requestMicrophoneAccess();

    if (!isReady) {
      return;
    }

    setResult(null);
    setAppState('listening');

    const activeRunId = analysisRunIdRef.current;

    analysisTimeoutRef.current = window.setTimeout(() => {
      void (async () => {
        const nextResult = await analyseCurrentBuffer();

        if (activeRunId !== analysisRunIdRef.current) {
          return;
        }

        setResult(nextResult);
        setAppState('result');
        analysisTimeoutRef.current = null;
        stopCapture();
      })();
    }, ANALYSIS_WINDOW_MS);
  };

  const handleReset = () => {
    analysisRunIdRef.current += 1;
    clearAnalysisTimer();
    stopCapture();
    setAppState('idle');
    setResult(null);
  };

  const handleStartMonitoring = async () => {
    clearAnalysisTimer();

    const didStart = await startMonitoring();

    if (!didStart) {
      return;
    }

    setAppState('monitoring');
  };

  const handleStopMonitoring = () => {
    analysisRunIdRef.current += 1;
    clearAnalysisTimer();
    stopMonitoring();
    setAppState('idle');
    setResult(null);
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
     * Responsive layout strategy:
     *   Mobile  (<768px)  — edge-to-edge single column, phone-chrome visible
     *   Tablet  (768–1023px) — centered floating card, single column, no phone chrome
     *   Desktop (≥1024px) — centered two-column card: EQ viz left, controls right
     */
    <div className="min-h-screen bg-[#0a0a0f] text-white md:flex md:items-center md:justify-center md:p-6 lg:p-10">
      <div className="
        w-full min-h-screen
        md:min-h-0 md:max-w-[680px] md:rounded-3xl md:shadow-2xl md:border md:border-gray-800/50
        lg:max-w-[1200px] lg:rounded-2xl lg:min-h-[720px]
        bg-[#0f0f14] flex flex-col relative overflow-hidden
      ">
        {/* Phone status-bar spacer — mobile only */}
        <div className="h-12 flex-shrink-0 md:hidden" />

        <Header
          state={appState}
          language={language}
          onToggleLanguage={toggleLanguage}
          statusLabel={headerStatus.label}
          statusColorClassName={headerStatus.colorClassName}
          showPulse={headerStatus.showPulse}
        />

        {/* Body: stacked on mobile/tablet, side-by-side on desktop */}
        <div className="flex-1 flex flex-col lg:flex-row min-h-0">

          {/* EQ Visualization — inline on mobile/tablet, left panel on desktop */}
          <div className="
            flex-shrink-0 flex justify-center px-6 mb-4
            md:px-8 md:mb-6
            lg:w-[480px] lg:border-r lg:border-gray-800/40
            lg:flex lg:items-center lg:justify-center
            lg:px-8 lg:py-10 lg:mb-0
          ">
            {/* Width cap on tablet so the viz doesn't stretch to 616px */}
            <div className="w-full md:max-w-[480px] lg:max-w-none">
              <EQVisualization
                analyserNode={analyserNode}
                featureRms={appState === 'idle' ? null : latestFeatures?.rms ?? null}
                problems={result?.problems || []}
                isLive={appState === 'monitoring' || appState === 'listening'}
                state={appState}
              />
            </div>
          </div>

          {/* Content + Actions */}
          <div className="flex-1 flex flex-col px-6 pb-8 md:px-8 lg:px-8 lg:pt-4 lg:pb-8 min-h-0">
            <div className="flex-1 min-h-0 overflow-y-auto">
              <div className="pb-4 space-y-3 md:space-y-4">

                {showDisabledBanner && (
                  <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4 md:p-5">
                    <div className="text-xs font-semibold uppercase tracking-[0.2em] text-amber-200">
                      Sample Analysis
                    </div>
                    <div className="mt-2 text-base md:text-lg font-semibold text-white">
                      Model disabled
                    </div>
                    <div className="mt-1 text-sm text-amber-100/80">
                      ML inference is turned off. Preview Result uses fallback analysis only.
                    </div>
                    <div className="mt-3 text-[11px] font-mono text-amber-200/70">
                      {MODEL_VERSION}
                    </div>
                  </div>
                )}

                {showExperimentalBanner && (
                  <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4 md:p-5">
                    <div className="text-xs font-semibold uppercase tracking-[0.2em] text-amber-200">
                      Experimental Mode
                    </div>
                    <div className="mt-2 text-base md:text-lg font-semibold text-white">
                      Not production-ready
                    </div>
                    <div className="mt-1 text-sm text-amber-100/80">
                      Results may be inaccurate.
                    </div>
                    <div className="mt-3 text-[11px] font-mono text-amber-200/70">
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

        {/* Home-indicator bar — mobile only */}
        <div className="h-8 flex items-center justify-center flex-shrink-0 md:hidden">
          <div className="w-32 h-1 bg-white/20 rounded-full" />
        </div>
      </div>
    </div>
  );
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
