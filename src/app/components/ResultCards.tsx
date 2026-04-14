import { motion } from 'motion/react';
import { AlertCircle, Radio, Sliders, Info, CheckCircle } from 'lucide-react';
import { linearToDbfs } from '../audio/audioUtils';
import type {
  AnalysisResult,
  DetectedAudioSource,
  DiagnosticProblem,
  ExtractedAudioFeatures,
  SourceEqRecommendation,
  StemMetric
} from '../types';
import { type Language, translations } from '../translations';
import { FeedbackWidget } from './FeedbackWidget';

interface ResultCardsProps {
  result: AnalysisResult;
  language: Language;
  isLive?: boolean;
  features?: ExtractedAudioFeatures;
}

export function ResultCards({ result, language, isLive = false, features }: ResultCardsProps) {
  const t = translations[language];

  // Handle "no issues" case
  if (result.problems.length === 0) {
    return (
      <>
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
            <StemServiceStatusCard result={result} language={language} centered />
            <DetectedSourcesCard
              detectedSources={result.detectedSources}
              language={language}
              centered
            />
            <StemMetricsCard
              stemMetrics={result.stemMetrics}
              language={language}
              centered
            />
            <SourceEqRecommendationsCard
              recommendations={result.sourceEqRecommendations}
              language={language}
            />
            {isLive && (
              <div className="flex items-center gap-2 text-xs text-green-400 mt-2">
                <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                {t.liveMonitoring}
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
      <FeedbackWidget result={result} features={features} language={language} />
    </>
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

      <StemServiceStatusCard result={result} language={language} />

      <DetectedSourcesCard
        detectedSources={result.detectedSources}
        language={language}
      />

      <StemMetricsCard
        stemMetrics={result.stemMetrics}
        language={language}
      />

      <SourceEqRecommendationsCard
        recommendations={result.sourceEqRecommendations}
        language={language}
      />

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

      <FeedbackWidget result={result} features={features} language={language} />
    </div>
  );
}

function StemServiceStatusCard({
  result,
  language,
  centered = false
}: {
  result: AnalysisResult;
  language: Language;
  centered?: boolean;
}) {
  if (!result.stemService) {
    return null;
  }

  const t = translations[language];
  const connected = result.stemService.connected;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`w-full border rounded-xl p-3 ${
        connected
          ? 'bg-emerald-500/10 border-emerald-500/20'
          : 'bg-amber-500/10 border-amber-500/20'
      } ${centered ? 'text-center' : ''}`}
    >
      <div className={`flex items-center gap-2 ${centered ? 'justify-center' : 'justify-between'}`}>
        <div className={`flex items-center gap-2 ${centered ? 'justify-center' : ''}`}>
          <div className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-400' : 'bg-amber-400'}`} />
          <div className={`text-xs font-semibold tracking-wide uppercase ${connected ? 'text-emerald-200' : 'text-amber-200'}`}>
            {connected ? t.stemServiceConnected : t.stemServiceFallback}
          </div>
        </div>
        {!centered && result.stemService.model && (
          <div className="text-[10px] font-mono text-gray-400">
            {result.stemService.model}
          </div>
        )}
      </div>
      {centered && result.stemService.model && (
        <div className="text-[10px] font-mono text-gray-400 mt-1">
          {result.stemService.model}
        </div>
      )}
    </motion.div>
  );
}

function DetectedSourcesCard({
  detectedSources,
  language,
  centered = false
}: {
  detectedSources?: DetectedAudioSource[];
  language: Language;
  centered?: boolean;
}) {
  if (!detectedSources || detectedSources.length === 0) {
    return null;
  }

  const t = translations[language];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`w-full bg-gray-800/50 border border-gray-700 rounded-xl p-4 ${centered ? 'text-center' : ''}`}
    >
      <div className={`text-xs text-gray-400 mb-3 ${centered ? '' : 'text-left'}`}>
        {t.detectedSources}
      </div>
      <div className={`flex flex-wrap gap-2 ${centered ? 'justify-center' : ''}`}>
        {detectedSources.map((entry) => (
          <span
            key={entry.source}
            className="px-3 py-1.5 bg-cyan-500/10 text-cyan-200 rounded-lg text-sm font-medium border border-cyan-500/20"
            title={entry.labels.join(', ')}
          >
            {(t[entry.source as keyof typeof t] || entry.source)} {Math.round(entry.confidence * 100)}%
          </span>
        ))}
      </div>
    </motion.div>
  );
}

function StemMetricsCard({
  stemMetrics,
  language,
  centered = false
}: {
  stemMetrics?: StemMetric[];
  language: Language;
  centered?: boolean;
}) {
  if (!stemMetrics || stemMetrics.length === 0) {
    return null;
  }

  const t = translations[language];
  const visibleStems = stemMetrics
    .filter((stem) => stem.energyRatio >= 0.01 || stem.rms >= 0.002)
    .sort((left, right) => right.energyRatio - left.energyRatio)
    .slice(0, 6);

  if (visibleStems.length === 0) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`w-full bg-gray-800/50 border border-gray-700 rounded-xl p-4 ${centered ? 'text-center' : ''}`}
    >
      <div className={`text-xs text-gray-400 mb-3 ${centered ? '' : 'text-left'}`}>
        {t.stemAnalysis}
      </div>
      <div className="space-y-2">
        {visibleStems.map((stem) => (
          <div key={`${stem.stem}-${stem.source ?? 'other'}`} className="rounded-lg bg-black/20 border border-white/5 p-3">
            <div className="flex items-center justify-between gap-3">
              <div className="text-sm font-medium text-white">
                {stem.source ? (t[stem.source as keyof typeof t] || stem.source) : stem.stem}
              </div>
              <div className="text-[11px] font-mono text-cyan-300">
                {Math.round(stem.energyRatio * 100)}%
              </div>
            </div>
            <div className="mt-2 h-1.5 rounded-full bg-white/5 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-emerald-400"
                style={{ width: `${Math.max(4, Math.min(100, stem.energyRatio * 100))}%` }}
              />
            </div>
            <div className={`mt-2 flex text-[11px] text-gray-400 ${centered ? 'justify-center gap-4' : 'justify-between'}`}>
              <span>{t.energyShare}: {Math.round(stem.energyRatio * 100)}%</span>
              <span>{t.rmsLevel}: {formatDbfs(stem.rms)}</span>
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

function SourceEqRecommendationsCard({
  recommendations,
  language
}: {
  recommendations?: SourceEqRecommendation[];
  language: Language;
}) {
  if (!recommendations || recommendations.length === 0) {
    return null;
  }

  const t = translations[language];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full bg-gray-800/50 border border-gray-700 rounded-xl p-4"
    >
      <div className="text-xs text-gray-400 mb-3">{t.sourceEqByInstrument}</div>
      <div className="space-y-2">
        {recommendations.map((recommendation) => (
          <div
            key={`${recommendation.source}-${recommendation.freq_range}-${recommendation.gain}`}
            className="rounded-lg border border-purple-500/15 bg-purple-500/5 p-3"
          >
            <div className="flex items-center justify-between gap-3">
              <div className="text-sm font-medium text-white">
                {t[recommendation.source as keyof typeof t] || recommendation.source}
              </div>
              <div className="text-[11px] font-mono text-purple-300">
                {Math.round(recommendation.confidence * 100)}%
              </div>
            </div>
            <div className="mt-1 text-sm font-mono text-purple-200">
              {recommendation.freq_range} {recommendation.gain}
            </div>
            <div className="mt-1 text-xs text-gray-400">
              {recommendation.reason}
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

function formatDbfs(rms: number) {
  const dbfs = linearToDbfs(rms);

  if (!Number.isFinite(dbfs)) {
    return '-inf dBFS';
  }

  return `${dbfs.toFixed(1)} dBFS`;
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
