import { motion } from 'motion/react';
import { AlertCircle, Radio, Sliders, Info, CheckCircle } from 'lucide-react';
import type { AnalysisResult, DiagnosticProblem } from '../types';
import { type Language, translations } from '../translations';

interface ResultCardsProps {
  result: AnalysisResult;
  language: Language;
  isLive?: boolean;
}

export function ResultCards({ result, language, isLive = false }: ResultCardsProps) {
  const t = translations[language];

  // Handle "no issues" case
  if (result.problems.length === 0) {
    return (
      <motion.div
        key="no-issues"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        transition={{ duration: 0.3 }}
        className="w-full"
      >
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-br from-green-600/20 to-emerald-600/20 border border-green-500/30 rounded-2xl p-8 text-center"
        >
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center">
              <CheckCircle className="w-8 h-8 text-green-400" />
            </div>
            <div>
              <div className="text-2xl font-bold mb-2">{t.noIssues}</div>
              <div className="text-sm text-gray-400">{t.soundQualityGood}</div>
            </div>
            {isLive && (
              <div className="flex items-center gap-2 text-xs text-green-400 mt-2">
                <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                {t.liveMonitoring}
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    );
  }

  return (
    <div className="w-full space-y-4">
      {/* Live Indicator */}
      {isLive && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-center gap-2 text-xs text-green-400"
        >
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          {t.liveMonitoring}
        </motion.div>
      )}

      {/* Render all problems */}
      {result.problems.map((problem, index) => (
        <ProblemCard
          key={`${problem.type}-${index}-${result.timestamp}`}
          problem={problem}
          language={language}
          isPrimary={index === 0}
          delay={isLive ? 0 : index * 0.1}
          isLive={isLive}
        />
      ))}
    </div>
  );
}

interface ProblemCardProps {
  problem: DiagnosticProblem;
  language: Language;
  isPrimary: boolean;
  delay: number;
  isLive?: boolean;
}

function ProblemCard({ problem, language, isPrimary, delay, isLive = false }: ProblemCardProps) {
  const t = translations[language];

  // Format confidence as percentage
  const confidencePercent = Math.round(problem.confidence * 100);
  
  // Determine confidence level based on thresholds
  const confidenceLevel = problem.confidence >= 0.75 ? 'high' : problem.confidence >= 0.4 ? 'medium' : 'low';
  
  const confidenceColors = {
    high: 'bg-green-500/10 text-green-400 border-green-500/30',
    medium: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30',
    low: 'bg-red-500/10 text-red-400 border-red-500/30'
  };

  // Animation variants for live updates
  const cardVariants = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 }
  };

  return (
    <motion.div
      layout={isLive}
      variants={cardVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={{ 
        delay,
        layout: { duration: 0.3 }
      }}
      className="w-full space-y-3"
    >
      {/* LEVEL 1: PROBLEM NAME - Most Prominent */}
      <motion.div
        layout={isLive}
        className={`${
          isPrimary
            ? 'bg-gradient-to-br from-purple-600/20 to-blue-600/20 border border-purple-500/30'
            : 'bg-gray-800/50 border border-gray-700/50'
        } rounded-2xl p-6`}
      >
        <div className="flex items-start gap-3">
          <div
            className={`w-10 h-10 rounded-full ${
              isPrimary ? 'bg-purple-500/20' : 'bg-gray-700/50'
            } flex items-center justify-center flex-shrink-0`}
          >
            <AlertCircle className={`w-5 h-5 ${isPrimary ? 'text-purple-400' : 'text-gray-400'}`} />
          </div>
          <div className="flex-1">
            <div className="text-xs text-gray-400 mb-1">
              {isPrimary ? t.problemDetected : t.secondaryProblem}
            </div>
            <motion.div 
              layout={isLive}
              className={`${isPrimary ? 'text-2xl' : 'text-xl'} font-bold`}
            >
              {t[problem.type as keyof typeof t] || problem.type}
            </motion.div>
            
            {/* CONFIDENCE - minimal but visible with animation */}
            <div className="mt-2">
              <motion.span 
                layout={isLive}
                key={`confidence-${confidencePercent}`}
                initial={isLive ? { scale: 1.1, opacity: 0.7 } : false}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.3 }}
                className={`inline-block px-3 py-1 rounded-full text-xs font-semibold border ${confidenceColors[confidenceLevel]}`}
              >
                {confidencePercent}%
              </motion.span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* LEVEL 2: SOURCE TAGS - compact and scannable */}
      {problem.sources.length > 0 && (
        <motion.div 
          layout={isLive}
          className="bg-gray-800/50 border border-gray-700 rounded-xl p-4"
        >
          <div className="flex items-start gap-3">
            <div className="w-7 h-7 rounded-lg bg-blue-500/20 flex items-center justify-center flex-shrink-0">
              <Radio className="w-4 h-4 text-blue-400" />
            </div>
            <div className="flex-1">
              <div className="text-xs text-gray-400 mb-2">{t.source}</div>
              <motion.div layout={isLive} className="flex flex-wrap gap-2">
                {problem.sources.map((source, index) => (
                  <motion.span
                    key={source}
                    layout={isLive}
                    initial={isLive ? { scale: 0.9, opacity: 0 } : false}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: index * 0.05 }}
                    className="px-3 py-1.5 bg-blue-500/10 text-blue-300 rounded-lg text-sm font-medium border border-blue-500/20"
                  >
                    {t[source as keyof typeof t] || source}
                  </motion.span>
                ))}
              </motion.div>
            </div>
          </div>
        </motion.div>
      )}

      {/* LEVEL 3: DETAILED CAUSES - subtle and secondary */}
      {problem.details.length > 0 && (
        <motion.div 
          layout={isLive}
          className="bg-gray-800/30 border border-gray-700/50 rounded-xl p-4"
        >
          <div className="flex items-start gap-3">
            <div className="w-7 h-7 rounded-lg bg-gray-700/50 flex items-center justify-center flex-shrink-0">
              <Info className="w-4 h-4 text-gray-400" />
            </div>
            <div className="flex-1">
              <div className="text-xs text-gray-500 mb-2">{t.details}</div>
              <motion.div layout={isLive} className="space-y-1.5">
                {problem.details.map((detail, index) => (
                  <motion.div
                    key={detail}
                    layout={isLive}
                    initial={isLive ? { x: -10, opacity: 0 } : false}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ delay: index * 0.05 }}
                    className="text-sm text-gray-300"
                  >
                    • {t[detail as keyof typeof t] || detail}
                  </motion.div>
                ))}
              </motion.div>
            </div>
          </div>
        </motion.div>
      )}

      {/* LEVEL 4: ACTION LIST - clear and readable with real-time EQ updates */}
      {problem.actions.length > 0 && (
        <motion.div 
          layout={isLive}
          className="bg-gray-800/50 border border-gray-700 rounded-xl p-4"
        >
          <div className="flex items-start gap-3">
            <div className="w-7 h-7 rounded-lg bg-purple-500/20 flex items-center justify-center flex-shrink-0">
              <Sliders className="w-4 h-4 text-purple-400" />
            </div>
            <div className="flex-1">
              <div className="text-xs text-gray-400 mb-2">{t.actions}</div>
              <motion.div layout={isLive} className="space-y-1.5">
                {problem.actions.map((action, index) => (
                  <motion.div
                    key={`${action}-${index}`}
                    layout={isLive}
                    initial={isLive ? { 
                      scale: 0.95, 
                      opacity: 0,
                      x: -10
                    } : false}
                    animate={{ 
                      scale: 1, 
                      opacity: 1,
                      x: 0
                    }}
                    transition={{ 
                      delay: index * 0.05,
                      type: "spring",
                      stiffness: 300,
                      damping: 25
                    }}
                    className="px-3 py-2 bg-purple-500/10 text-purple-200 rounded-lg font-mono text-sm border border-purple-500/20"
                  >
                    <motion.span
                      key={action}
                      initial={isLive ? { opacity: 0.5 } : false}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.4 }}
                    >
                      {action}
                    </motion.span>
                  </motion.div>
                ))}
              </motion.div>
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}