import { motion } from 'motion/react';
import { MicOff } from 'lucide-react';
import type { MicrophoneErrorCode } from '../types';
import { type Language, translations } from '../translations';

interface MicrophoneStatusCardProps {
  error: Exclude<MicrophoneErrorCode, null>;
  language: Language;
}

export function MicrophoneStatusCard({ error, language }: MicrophoneStatusCardProps) {
  const t = translations[language];
  const title = error === 'not_found'
    ? t.microphoneUnavailable
    : error === 'unsupported'
      ? t.microphoneUnsupported
      : t.microphonePermissionTitle;

  const message = error === 'not_found'
    ? t.microphoneUnavailableHelp
    : error === 'unsupported'
      ? t.microphoneUnsupportedHelp
      : error === 'unknown'
        ? t.microphoneUnknownHelp
        : t.microphonePermissionHelp;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full"
    >
      <div className="bg-gradient-to-br from-amber-500/10 to-red-500/10 border border-amber-500/30 rounded-2xl p-4">
        <div className="flex items-start gap-3">
          <div className="w-11 h-11 rounded-full bg-amber-500/15 flex items-center justify-center flex-shrink-0">
            <MicOff className="w-5 h-5 text-amber-300" />
          </div>

          <div className="space-y-1.5">
            <div className="text-sm font-semibold text-amber-200">
              {title}
            </div>
            <div className="text-sm text-gray-300 leading-relaxed">
              {message}
            </div>
            <div className="text-xs text-gray-400">
              {t.microphoneRetryHint}
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
