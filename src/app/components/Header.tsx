import { Languages } from 'lucide-react';
import { type Language, translations } from '../translations';

interface HeaderProps {
  state: 'idle' | 'listening' | 'result' | 'monitoring';
  language: Language;
  onToggleLanguage: () => void;
}

const stateColors = {
  idle: 'text-gray-400',
  listening: 'text-purple-400',
  result: 'text-blue-400',
  monitoring: 'text-green-400'
};

export function Header({ state, language, onToggleLanguage }: HeaderProps) {
  const t = translations[language];
  
  const stateLabels = {
    idle: t.ready,
    listening: t.listening,
    result: t.result,
    monitoring: t.monitoring
  };

  return (
    <div className="px-6 py-4 flex items-center justify-between">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">{t.appName}</h1>
      </div>
      
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          {state === 'monitoring' && (
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          )}
          <div className={`text-sm font-medium ${stateColors[state]}`}>
            {stateLabels[state]}
          </div>
        </div>
        
        <button
          onClick={onToggleLanguage}
          className="w-8 h-8 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors flex items-center justify-center"
          aria-label="Toggle language"
        >
          <Languages className="w-4 h-4 text-gray-400" />
        </button>
      </div>
    </div>
  );
}