import { Languages } from 'lucide-react';
import { type Language, translations } from '../translations';

interface HeaderProps {
  state: 'idle' | 'listening' | 'result' | 'monitoring';
  language: Language;
  onToggleLanguage: () => void;
  statusLabel?: string;
  statusColorClassName?: string;
  showPulse?: boolean;
}

const stateColors = {
  idle: 'text-gray-400',
  listening: 'text-purple-400',
  result: 'text-blue-400',
  monitoring: 'text-green-400'
};

export function Header({
  state,
  language,
  onToggleLanguage,
  statusLabel,
  statusColorClassName,
  showPulse = false
}: HeaderProps) {
  const t = translations[language];
  
  const stateLabels = {
    idle: t.ready,
    listening: t.listening,
    result: t.result,
    monitoring: t.monitoring
  };

  return (
    <div className="px-6 py-4 md:px-8 md:py-5 lg:py-6 flex items-center justify-between">
      <div>
        <h1 className="text-xl lg:text-2xl font-semibold tracking-tight">{t.appName}</h1>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          {(showPulse || state === 'monitoring') && (
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          )}
          <div className={`text-sm font-medium ${statusColorClassName ?? stateColors[state]}`}>
            {statusLabel ?? stateLabels[state]}
          </div>
        </div>

        <button
          onClick={onToggleLanguage}
          className="w-8 h-8 md:w-10 md:h-10 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors flex items-center justify-center"
          aria-label="Toggle language"
        >
          <Languages className="w-4 h-4 text-gray-400" />
        </button>
      </div>
    </div>
  );
}
