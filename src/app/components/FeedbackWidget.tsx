/**
 * FeedbackWidget
 *
 * Shown below analysis results. Lets the user mark an analysis as
 * correct (👍) or wrong (👎). On "wrong", they can optionally pick
 * what the actual issue was. All entries are stored in localStorage
 * via feedbackStore and can be exported as JSONL for ML retraining.
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ThumbsUp, ThumbsDown, Download, X } from 'lucide-react';
import type { AnalysisResult, ExtractedAudioFeatures } from '../types';
import { saveFeedback, downloadFeedback, getFeedbackCount } from '../audio/feedbackStore';
import { type Language, translations } from '../translations';

const ISSUE_LABELS = [
  'muddy', 'harsh', 'buried', 'boomy', 'thin', 'boxy', 'nasal', 'sibilant', 'dull',
] as const;

type Phase = 'idle' | 'picking_wrong' | 'done';

interface FeedbackWidgetProps {
  result: AnalysisResult;
  features?: ExtractedAudioFeatures;
  language: Language;
}

export function FeedbackWidget({ result, features, language }: FeedbackWidgetProps) {
  const t = translations[language];
  const [phase, setPhase] = useState<Phase>('idle');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [count, setCount] = useState(getFeedbackCount);

  function handleCorrect() {
    saveFeedback({ result, features, verdict: 'correct' });
    setCount((c) => c + 1);
    setPhase('done');
  }

  function handleWrong() {
    setPhase('picking_wrong');
  }

  function toggleLabel(label: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(label) ? next.delete(label) : next.add(label);
      return next;
    });
  }

  function submitWrong() {
    saveFeedback({
      result,
      features,
      verdict: 'wrong',
      correctedLabels: [...selected],
    });
    setCount((c) => c + 1);
    setPhase('done');
  }

  function dismiss() {
    setPhase('done');
  }

  return (
    <div className="w-full mt-2">
      <AnimatePresence mode="wait">
        {phase === 'idle' && (
          <motion.div
            key="idle"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            className="flex items-center justify-between gap-3 px-4 py-3 bg-gray-800/40 border border-gray-700/50 rounded-xl"
          >
            <span className="text-xs text-gray-400">
              {t.feedbackPrompt}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={handleCorrect}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-green-500/10 border border-green-500/20 text-green-400 hover:bg-green-500/20 transition-colors text-xs font-medium"
                title={t.feedbackCorrect}
              >
                <ThumbsUp className="w-3.5 h-3.5" />
                {t.feedbackCorrect}
              </button>
              <button
                onClick={handleWrong}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 transition-colors text-xs font-medium"
                title={t.feedbackWrong}
              >
                <ThumbsDown className="w-3.5 h-3.5" />
                {t.feedbackWrong}
              </button>
            </div>
          </motion.div>
        )}

        {phase === 'picking_wrong' && (
          <motion.div
            key="picking"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            className="px-4 py-3 bg-gray-800/40 border border-red-500/20 rounded-xl space-y-3"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-300">{t.feedbackWhatActually}</span>
              <button onClick={dismiss} className="text-gray-500 hover:text-gray-300 transition-colors">
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {ISSUE_LABELS.map((label) => {
                const active = selected.has(label);
                return (
                  <button
                    key={label}
                    onClick={() => toggleLabel(label)}
                    className={`px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors ${
                      active
                        ? 'bg-red-500/20 border-red-500/40 text-red-300'
                        : 'bg-gray-700/40 border-gray-600/40 text-gray-400 hover:border-gray-500/60'
                    }`}
                  >
                    {t[label as keyof typeof t] ?? label}
                  </button>
                );
              })}
              <button
                key="none"
                onClick={() => toggleLabel('none')}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors ${
                  selected.has('none')
                    ? 'bg-red-500/20 border-red-500/40 text-red-300'
                    : 'bg-gray-700/40 border-gray-600/40 text-gray-400 hover:border-gray-500/60'
                }`}
              >
                {t.feedbackNoIssue}
              </button>
            </div>
            <button
              onClick={submitWrong}
              className="w-full py-1.5 rounded-lg bg-red-500/15 border border-red-500/30 text-red-300 hover:bg-red-500/25 transition-colors text-xs font-medium"
            >
              {t.feedbackSubmit}
            </button>
          </motion.div>
        )}

        {phase === 'done' && (
          <motion.div
            key="done"
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="flex items-center justify-between gap-3 px-4 py-3 bg-gray-800/30 border border-gray-700/40 rounded-xl"
          >
            <span className="text-xs text-gray-500">
              {t.feedbackThanks} · {count} {t.feedbackCollected}
            </span>
            <button
              onClick={downloadFeedback}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-700/40 border border-gray-600/40 text-gray-400 hover:bg-gray-700/60 transition-colors text-xs"
              title={t.feedbackExport}
            >
              <Download className="w-3.5 h-3.5" />
              {t.feedbackExport}
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
