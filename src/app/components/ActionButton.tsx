import { motion } from 'motion/react';
import { Mic, RotateCcw, Play, Square } from 'lucide-react';
import { type Language, translations } from '../translations';

interface ActionButtonProps {
  state: 'idle' | 'listening' | 'result' | 'monitoring';
  onAnalyze: () => void;
  onReset: () => void;
  onStartMonitoring: () => void;
  onStopMonitoring: () => void;
  language: Language;
}

export function ActionButton({ state, onAnalyze, onReset, onStartMonitoring, onStopMonitoring, language }: ActionButtonProps) {
  const isListening = state === 'listening';
  const isResult = state === 'result';
  const isMonitoring = state === 'monitoring';
  const isIdle = state === 'idle';
  const t = translations[language];

  if (isMonitoring) {
    return (
      <div className="pt-4 md:pt-6 flex-shrink-0">
        <motion.button
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          onClick={onStopMonitoring}
          className="w-full h-14 md:h-16 rounded-2xl bg-red-600 hover:bg-red-700 active:bg-red-800 transition-colors flex items-center justify-center gap-3 font-semibold text-base md:text-lg shadow-lg shadow-red-500/30"
        >
          <Square className="w-5 h-5 fill-current" />
          {t.stopMonitoring}
        </motion.button>
      </div>
    );
  }

  if (isResult) {
    return (
      <div className="pt-4 md:pt-6 flex-shrink-0 space-y-3">
        <motion.button
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          onClick={onStartMonitoring}
          className="w-full h-14 md:h-16 rounded-2xl bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 active:from-green-700 active:to-emerald-700 transition-all flex items-center justify-center gap-3 font-semibold text-base md:text-lg shadow-lg shadow-green-500/30"
        >
          <Play className="w-5 h-5 fill-current" />
          {t.startMonitoring}
        </motion.button>
        <motion.button
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          onClick={onReset}
          className="w-full h-12 md:h-14 rounded-2xl bg-gray-800 hover:bg-gray-700 active:bg-gray-600 transition-colors flex items-center justify-center gap-3 font-medium"
        >
          <RotateCcw className="w-5 h-5" />
          {t.analyzeAgain}
        </motion.button>
      </div>
    );
  }

  return (
    <div className="pt-4 md:pt-6 flex-shrink-0 space-y-3">
      <motion.button
        onClick={onAnalyze}
        disabled={isListening}
        className={`w-full h-14 md:h-16 rounded-2xl font-semibold text-base md:text-lg flex items-center justify-center gap-3 relative overflow-hidden ${
          isListening
            ? 'bg-purple-600 cursor-not-allowed'
            : 'bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 active:from-purple-700 active:to-blue-700 transition-all shadow-lg shadow-purple-500/30'
        }`}
        whileTap={!isListening ? { scale: 0.98 } : {}}
      >
        {isListening && (
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
            animate={{
              x: ['-100%', '100%']
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        )}
        <Mic className="w-6 h-6" />
        {isListening ? t.analyzing : t.startAnalysis}
      </motion.button>
      
      {isIdle && (
        <motion.button
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          onClick={onStartMonitoring}
          className="w-full h-12 md:h-14 rounded-2xl bg-gray-800 hover:bg-gray-700 active:bg-gray-600 transition-colors flex items-center justify-center gap-3 font-medium"
        >
          <Play className="w-4 h-4" />
          {t.startMonitoring}
        </motion.button>
      )}
    </div>
  );
}