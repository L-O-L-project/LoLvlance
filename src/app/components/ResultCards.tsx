import { motion } from 'motion/react';
import { AlertCircle, Radio, Sliders, Info, CheckCircle, AlertTriangle } from 'lucide-react';
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
import {
  HIGH_CONFIDENCE_THRESHOLD,
  MEDIUM_CONFIDENCE_THRESHOLD
} from '../audio/mlThresholds';

interface ResultCardsProps {
  result: AnalysisResult;
  language: Language;
  isLive?: boolean;
  features?: ExtractedAudioFeatures;
}

export function ResultCards({ result, language, isLive = false, features }: ResultCardsProps) {
  const t = translations[language];
  const transientIssues = isLive ? (result.monitoringInterpretation?.transientIssues ?? []) : [];
  const hasFallbackWarning = result.runtimeWarnings?.some((warning) => warning.recoverable) ?? false;

  // Handle "no issues" case
  if (result.problems.length === 0 && transientIssues.length === 0) {
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
          className="bg-gradient-to-br from-green-600/20 to-emerald-600/20 border border-green-500/30 rounded-2xl p-6 text-center"
        >
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center">
              <CheckCircle className="w-8 h-8 text-green-400" />
            </div>
            <div>
              <div className="text-2xl font-bold mb-2">{t.noIssues}</div>
              <div className="text-sm text-gray-400">{t.soundQualityGood}</div>
            </div>
            <RuntimeWarningsCard result={result} language={language} />
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

      {hasFallbackWarning && (
        <RuntimeWarningsCard result={result} language={language} />
      )}

      <TransientWarningsCard
        transientIssues={transientIssues}
        language={language}
      />

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

function RuntimeWarningsCard({
  result,
  language
}: {
  result: AnalysisResult;
  language: Language;
}) {
  const warnings = result.runtimeWarnings?.filter((warning) => warning.recoverable) ?? [];

  if (warnings.length === 0) {
    return null;
  }

  const t = translations[language];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full rounded-xl border border-amber-500/25 bg-amber-500/10 p-4"
    >
      <div className="flex items-start gap-3">
        <div className="w-8 h-8 rounded-lg bg-amber-500/15 flex items-center justify-center flex-shrink-0">
          <AlertTriangle className="w-4 h-4 text-amber-300" />
        </div>
        <div>
          <div className="text-sm font-semibold text-amber-100">{t.fallbackTitle}</div>
          <div className="mt-1 text-sm text-amber-100/80">{t.fallbackMessage}</div>
        </div>
      </div>
    </motion.div>
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
  if (!result.stemService || !result.stemService.connected) {
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
            className="px-3 py-2 bg-cyan-500/10 text-cyan-200 rounded-lg text-sm font-medium border border-cyan-500/20"
            title={entry.labels.join(', ')}
          >
            {(t[entry.source as keyof typeof t] || entry.source)} {Math.round(entry.confidence * 100)}%
            <span className="ml-1 text-[11px] text-cyan-100/70">
              {getSourceQualityLabel(entry.quality, entry.confidence, t)}
            </span>
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

function TransientWarningsCard({
  transientIssues,
  language
}: {
  transientIssues: string[];
  language: Language;
}) {
  if (transientIssues.length === 0) {
    return null;
  }

  const t = translations[language];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full rounded-xl border border-amber-500/25 bg-amber-500/10 p-4"
    >
      <div className="text-xs text-amber-200/80 mb-3">{t.transientWarnings}</div>
      <div className="flex flex-wrap gap-2">
        {transientIssues.map((issue) => (
          <span
            key={issue}
            className="px-3 py-1.5 rounded-lg border border-amber-400/20 bg-black/20 text-sm font-medium text-amber-100"
          >
            {t[issue as keyof typeof t] || issue}
          </span>
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
  const confidenceLevel = problem.confidence >= HIGH_CONFIDENCE_THRESHOLD ? 'high' : problem.confidence >= MEDIUM_CONFIDENCE_THRESHOLD ? 'medium' : 'low';
  const isLowConfidence = confidenceLevel === 'low';
  const explanation = getProblemExplanation(problem.type, language);
  
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
        } rounded-2xl p-4`}
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
              className={`${isPrimary ? 'text-xl' : 'text-lg'} font-bold`}
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
                {t.confidence}: {confidencePercent}% ({t[confidenceLevel]})
              </motion.span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* RESULT SUMMARY: plain-language interpretation for non-technical users */}
      <motion.div
        layout={isLive}
        className="bg-gray-800/50 border border-gray-700 rounded-xl p-4"
      >
        <div className="space-y-3">
          <div>
            <div className="text-xs text-gray-400 mb-1">{t.resultSummary}</div>
            <div className="text-sm text-gray-200 leading-relaxed">{explanation.what}</div>
          </div>
          <div>
            <div className="text-xs text-gray-400 mb-1">{t.whyItMatters}</div>
            <div className="text-sm text-gray-300 leading-relaxed">{explanation.why}</div>
          </div>
          <div>
            <div className="text-xs text-gray-400 mb-1">{t.nextStep}</div>
            <div className="text-sm text-gray-300 leading-relaxed">{explanation.next}</div>
          </div>
        </div>
      </motion.div>

      {isLowConfidence && (
        <motion.div
          layout={isLive}
          className="rounded-xl border border-amber-500/25 bg-amber-500/10 p-4"
        >
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-300 flex-shrink-0 mt-0.5" />
            <div>
              <div className="text-sm font-semibold text-amber-100">{t.lowConfidenceTitle}</div>
              <div className="mt-1 text-sm text-amber-100/80">{t.lowConfidenceMessage}</div>
            </div>
          </div>
        </motion.div>
      )}

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
                    className="px-3 py-2 bg-blue-500/10 text-blue-300 rounded-lg text-sm font-medium border border-blue-500/20"
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
                    className="px-3 py-2.5 bg-purple-500/10 text-purple-200 rounded-lg font-mono text-sm border border-purple-500/20 min-h-[40px] flex items-center"
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

function getSourceQualityLabel(
  quality: DetectedAudioSource['quality'],
  confidence: number,
  t: typeof translations.en
) {
  const effectiveQuality = quality ?? (confidence >= 0.55 ? 'likely' : confidence >= 0.25 ? 'uncertain' : 'fallback');

  if (effectiveQuality === 'likely') {
    return t.likelySource;
  }

  if (effectiveQuality === 'uncertain') {
    return t.sourceUncertain;
  }

  return t.fallbackSourceEstimate;
}

function getProblemExplanation(type: DiagnosticProblem['type'], language: Language) {
  const isKo = language === 'ko';
  const explanations: Record<DiagnosticProblem['type'], { what: string; why: string; next: string }> = {
    muddy: isKo
      ? { what: '저역과 저중역이 쌓여 소리가 흐릿하게 들릴 수 있습니다.', why: '라이브에서는 보컬과 킥, 베이스의 구분이 어려워지고 전체 음압만 커질 수 있습니다.', next: '베이스/킥/건반의 저중역을 조금 줄이고 문제가 줄어드는지 확인하세요.' }
      : { what: 'Low and low-mid energy may be building up, making the mix feel cloudy.', why: 'In a live mix this can hide vocals, kick, and bass definition while only adding loudness.', next: 'Try reducing low-mids on bass, kick, or keys and check whether the mix clears up.' },
    harsh: isKo
      ? { what: '고역 또는 존재감 대역이 강해 귀에 거칠게 들릴 수 있습니다.', why: '라이브에서는 피로감이 빠르게 생기고 보컬/기타/심벌이 과하게 튈 수 있습니다.', next: '가장 튀는 소스를 찾고 3-8 kHz 부근을 좁게 줄여 보세요.' }
      : { what: 'Upper-mid or high-frequency energy may be too strong.', why: 'Live audiences can hear this as sharpness or fatigue, especially from vocal, guitar, or cymbals.', next: 'Find the loudest bright source and try a narrow reduction around 3-8 kHz.' },
    buried: isKo
      ? { what: '중요한 소스가 다른 악기에 가려 앞으로 나오지 못할 수 있습니다.', why: '라이브에서는 보컬이나 리드 악기의 전달력이 떨어집니다.', next: '대상 소스를 조금 올리거나 경쟁 소스의 1-4 kHz 대역을 줄여 보세요.' }
      : { what: 'An important source may be masked by other instruments.', why: 'In live sound this reduces intelligibility and makes lead parts hard to follow.', next: 'Raise the target slightly or carve 1-4 kHz from competing sources.' },
    imbalance: isKo
      ? { what: '레벨 또는 톤 밸런스가 한쪽으로 치우쳐 있습니다.', why: '관객 위치에 따라 특정 악기만 과하게 들리거나 전체 믹스가 불안정해질 수 있습니다.', next: '드럼, 베이스, 보컬의 상대 레벨을 먼저 맞춘 뒤 EQ를 조정하세요.' }
      : { what: 'The mix may be tilted in level or tone.', why: 'In a venue this can make one source dominate and make the whole mix feel unstable.', next: 'Recheck relative levels for drums, bass, and vocal before making deeper EQ moves.' },
    boomy: isKo
      ? { what: '저역 울림이 과해 둥둥거리게 들릴 수 있습니다.', why: '공간 공진과 겹치면 작은 조정도 크게 들릴 수 있습니다.', next: '80-180 Hz 부근을 조금 줄이고 룸 반응을 확인하세요.' }
      : { what: 'Low-frequency resonance may be making the mix boom.', why: 'Room modes can exaggerate this in live spaces.', next: 'Try a small reduction around 80-180 Hz and listen from the room.' },
    thin: isKo
      ? { what: '저역 또는 바디감이 부족해 얇게 들릴 수 있습니다.', why: '라이브에서는 힘이 부족하고 작은 스피커처럼 느껴질 수 있습니다.', next: '소스 레벨을 확인한 뒤 80-250 Hz를 필요한 만큼만 보강하세요.' }
      : { what: 'The sample may lack low-end weight or body.', why: 'Live mixes can feel weak or small when this happens.', next: 'Check source level first, then add only the needed amount around 80-250 Hz.' },
    boxy: isKo
      ? { what: '중저역 공진 때문에 답답하거나 상자 같은 톤이 날 수 있습니다.', why: '보컬과 기타가 좁고 닫힌 느낌으로 들릴 수 있습니다.', next: '300-600 Hz 부근을 좁게 줄여 보세요.' }
      : { what: 'Low-mid resonance may be creating a boxed-in tone.', why: 'Vocals and guitars can sound closed or small in the room.', next: 'Try a narrow cut around 300-600 Hz.' },
    nasal: isKo
      ? { what: '중역 피크 때문에 콧소리처럼 들릴 수 있습니다.', why: '보컬 명료도는 남아도 자연스러움이 떨어집니다.', next: '800 Hz-1.5 kHz 부근을 조금 줄여 보세요.' }
      : { what: 'A midrange peak may be creating a nasal tone.', why: 'This keeps things audible but less natural.', next: 'Try a small cut around 800 Hz-1.5 kHz.' },
    sibilant: isKo
      ? { what: '치찰음이 과하게 튈 수 있습니다.', why: '라이브에서는 보컬이 날카롭고 피곤하게 들립니다.', next: '보컬 채널의 디에서 또는 5-10 kHz를 확인하세요.' }
      : { what: 'Sibilance may be jumping out too much.', why: 'Live vocals can become sharp and tiring.', next: 'Check vocal de-essing or the 5-10 kHz range.' },
    dull: isKo
      ? { what: '고역 정보가 부족해 답답하게 들릴 수 있습니다.', why: '라이브에서는 선명도와 공간감이 줄어듭니다.', next: '먼저 과한 저중역을 줄이고, 필요하면 8-12 kHz를 조금 더하세요.' }
      : { what: 'The sample may lack top-end clarity.', why: 'Live mixes can lose openness and detail.', next: 'First reduce excess low-mids, then add a small amount around 8-12 kHz if needed.' },
    vocal_buried: isKo
      ? { what: '보컬이 반주에 묻힐 수 있습니다.', why: '가사 전달이 약해져 공연의 중심이 흐려집니다.', next: '보컬 레벨과 2-4 kHz 존재감, 경쟁 악기 레벨을 확인하세요.' }
      : { what: 'The vocal may be covered by the band.', why: 'Lyrics and lead focus can become hard to follow.', next: 'Check vocal level, 2-4 kHz presence, and competing instrument levels.' },
    guitar_harsh: isKo
      ? { what: '기타 존재감 대역이 과하게 튈 수 있습니다.', why: '보컬과 심벌까지 함께 거칠게 느껴질 수 있습니다.', next: '기타의 2-5 kHz를 좁게 줄여 보세요.' }
      : { what: 'Guitar presence may be too aggressive.', why: 'This can make the whole mix feel sharp alongside vocal and cymbals.', next: 'Try a narrow reduction around 2-5 kHz on guitar.' },
    bass_muddy: isKo
      ? { what: '베이스가 저중역을 많이 차지할 수 있습니다.', why: '킥과 보컬의 윤곽이 흐려질 수 있습니다.', next: '베이스의 120-300 Hz를 정리하고 킥과의 역할을 나누세요.' }
      : { what: 'Bass may be occupying too much low-mid space.', why: 'Kick and vocal definition can disappear.', next: 'Clean up 120-300 Hz on bass and separate it from the kick.' },
    drums_overpower: isKo
      ? { what: '드럼이 전체 믹스를 과하게 지배할 수 있습니다.', why: '다른 악기와 보컬이 뒤로 밀립니다.', next: '드럼 버스 레벨과 심벌/스네어 피크를 먼저 확인하세요.' }
      : { what: 'Drums may be dominating the mix.', why: 'Vocals and instruments can get pushed behind the kit.', next: 'Check drum bus level and cymbal/snare peaks first.' },
    keys_masking: isKo
      ? { what: '건반이 중역을 넓게 차지해 다른 소스를 가릴 수 있습니다.', why: '보컬이나 기타가 선명하게 나오기 어렵습니다.', next: '건반 패드의 중역을 줄이거나 스테레오/레벨을 정리하세요.' }
      : { what: 'Keys may be filling too much midrange space.', why: 'Vocals or guitars can struggle to cut through.', next: 'Reduce midrange on pads or rebalance keys width and level.' }
  };

  return explanations[type];
}
