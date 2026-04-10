import { useState } from 'react';
import { Header } from './components/Header';
import { AudioVisualization } from './components/AudioVisualization';
import { ActionButton } from './components/ActionButton';
import { ResultCards } from './components/ResultCards';
import { type Language } from './translations';
import { useMonitoring } from './hooks/useMonitoring';
import type { AnalysisResult } from './types';
import { generateMockProblems } from './data/diagnosticData';
import { EQVisualization } from './components/EQVisualization';

type AppState = 'idle' | 'listening' | 'result' | 'monitoring';

export default function App() {
  const [appState, setAppState] = useState<AppState>('idle');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [language, setLanguage] = useState<Language>('en');
  
  const { isMonitoring, startMonitoring, stopMonitoring } = useMonitoring((newResult) => {
    setResult(newResult);
    setAppState('monitoring');
  });

  const handleAnalyze = () => {
    setAppState('listening');
    
    // Simulate 2-4 seconds of listening and analysis
    setTimeout(() => {
      // Generate 1-2 problems per analysis
      const problemCount = Math.random() < 0.7 ? 1 : 2;
      const mockResult: AnalysisResult = {
        problems: generateMockProblems(problemCount),
        timestamp: Date.now()
      };
      
      setResult(mockResult);
      setAppState('result');
    }, 3000);
  };

  const handleReset = () => {
    setAppState('idle');
    setResult(null);
  };

  const handleStartMonitoring = () => {
    startMonitoring();
  };

  const handleStopMonitoring = () => {
    stopMonitoring();
    setAppState('idle');
    setResult(null);
  };

  const toggleLanguage = () => {
    setLanguage(prev => prev === 'en' ? 'ko' : 'en');
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white flex items-center justify-center p-4">
      {/* Mobile Container */}
      <div className="w-full max-w-[390px] h-[844px] bg-[#0f0f14] rounded-[40px] shadow-2xl border border-gray-800/50 flex flex-col relative overflow-hidden">
        {/* Status Bar Spacer */}
        <div className="h-12" />
        
        {/* Header */}
        <Header state={appState} language={language} onToggleLanguage={toggleLanguage} />
        
        {/* Main Content Area */}
        <div className="flex-1 flex flex-col px-6 pb-8 min-h-0">
          {/* Fixed EQ Visualization Area */}
          <div className="flex-shrink-0 mb-4 flex justify-center">
            <EQVisualization 
              problems={result?.problems || []} 
              isLive={appState === 'monitoring' || appState === 'analyzing'}
              state={appState}
            />
          </div>
          
          {/* Scrollable Result Area */}
          <div className="flex-1 min-h-0 overflow-y-auto">
            <div className="pb-4">
              {/* Show results when available */}
              {(appState === 'result' || appState === 'monitoring') && result && (
                <ResultCards 
                  result={result} 
                  language={language} 
                  isLive={appState === 'monitoring'}
                />
              )}
            </div>
          </div>
          
          {/* Action Button */}
          <ActionButton 
            state={appState} 
            onAnalyze={handleAnalyze}
            onReset={handleReset}
            onStartMonitoring={handleStartMonitoring}
            onStopMonitoring={handleStopMonitoring}
            language={language}
          />
        </div>
        
        {/* Home Indicator */}
        <div className="h-8 flex items-center justify-center">
          <div className="w-32 h-1 bg-white/20 rounded-full" />
        </div>
      </div>
    </div>
  );
}