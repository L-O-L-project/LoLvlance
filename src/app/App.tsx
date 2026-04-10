import { useEffect, useRef, useState } from 'react';
import { Header } from './components/Header';
import { ActionButton } from './components/ActionButton';
import { ResultCards } from './components/ResultCards';
import { MicrophoneStatusCard } from './components/MicrophoneStatusCard';
import { type Language, translations } from './translations';
import { useMonitoring } from './hooks/useMonitoring';
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

  const {
    analyserNode,
    analyseCurrentBuffer,
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
    clearAnalysisTimer();

    const isReady = await requestMicrophoneAccess();

    if (!isReady) {
      return;
    }

    setResult(null);
    setAppState('listening');

    analysisTimeoutRef.current = window.setTimeout(() => {
      const nextResult = analyseCurrentBuffer();
      setResult(nextResult);
      setAppState('result');
      analysisTimeoutRef.current = null;
      stopCapture();
    }, ANALYSIS_WINDOW_MS);
  };

  const handleReset = () => {
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
    <div className="min-h-screen bg-[#0a0a0f] text-white flex items-center justify-center p-4">
      <div className="w-full max-w-[390px] h-[844px] bg-[#0f0f14] rounded-[40px] shadow-2xl border border-gray-800/50 flex flex-col relative overflow-hidden">
        <div className="h-12" />

        <Header
          state={appState}
          language={language}
          onToggleLanguage={toggleLanguage}
          statusLabel={headerStatus.label}
          statusColorClassName={headerStatus.colorClassName}
          showPulse={headerStatus.showPulse}
        />

        <div className="flex-1 flex flex-col px-6 pb-8 min-h-0">
          <div className="flex-shrink-0 mb-4 flex justify-center">
            <EQVisualization
              analyserNode={analyserNode}
              problems={result?.problems || []}
              isLive={appState === 'monitoring' || appState === 'listening'}
              state={appState}
            />
          </div>

          <div className="flex-1 min-h-0 overflow-y-auto">
            <div className="pb-4 space-y-4">
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

        <div className="h-8 flex items-center justify-center">
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
