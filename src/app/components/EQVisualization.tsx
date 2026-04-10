import { motion } from 'motion/react';
import { useEffect, useState } from 'react';
import type { DiagnosticProblem } from '../types';

interface EQVisualizationProps {
  problems: DiagnosticProblem[];
  isLive?: boolean;
  state?: 'idle' | 'listening' | 'result' | 'monitoring';
}

interface FrequencyBand {
  freq: number;
  label: string;
  level: number; // 0-1 for bar height
  gain: number; // -12 to +12 dB for EQ adjustment
  hasProblem: boolean;
  problemType?: 'cut' | 'boost';
}

// Frequency bands for visualization (logarithmic scale)
const baseBands = [
  { freq: 31, label: '31' },
  { freq: 63, label: '63' },
  { freq: 125, label: '125' },
  { freq: 250, label: '250' },
  { freq: 500, label: '500' },
  { freq: 1000, label: '1k' },
  { freq: 2000, label: '2k' },
  { freq: 4000, label: '4k' },
  { freq: 8000, label: '8k' },
  { freq: 16000, label: '16k' }
];

export function EQVisualization({ problems, isLive = false, state = 'idle' }: EQVisualizationProps) {
  const [frequencyBands, setFrequencyBands] = useState<FrequencyBand[]>([]);

  useEffect(() => {
    // Generate frequency band data
    const bands = generateFrequencyBands(problems, isLive, state);
    setFrequencyBands(bands);
  }, [problems, isLive, state]);

  // Determine status text and color
  const getStatusInfo = () => {
    if (state === 'monitoring') {
      return { text: 'Real-time Monitoring', color: 'text-green-400', dotColor: 'bg-green-400' };
    }
    if (state === 'listening') {
      return { text: 'Analyzing...', color: 'text-purple-400', dotColor: 'bg-purple-400' };
    }
    if (state === 'result' && problems.length > 0) {
      return { text: 'Analysis Complete', color: 'text-blue-400', dotColor: 'bg-blue-400' };
    }
    if (state === 'result' && problems.length === 0) {
      return { text: 'No Issues Detected', color: 'text-green-400', dotColor: 'bg-green-400' };
    }
    return { text: 'Ready', color: 'text-gray-400', dotColor: 'bg-gray-400' };
  };

  const statusInfo = getStatusInfo();

  return (
    <motion.div 
      layout
      className="w-full bg-gray-900/50 border border-gray-700/50 rounded-xl p-4"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs text-gray-400 font-medium">Frequency Spectrum</div>
        <div className={`flex items-center gap-2 text-xs ${statusInfo.color}`}>
          {(isLive || state === 'listening') && (
            <div className={`w-1.5 h-1.5 rounded-full ${statusInfo.dotColor} animate-pulse`} />
          )}
          {statusInfo.text}
        </div>
      </div>
      
      {/* Mixer-style frequency bars with EQ overlay */}
      <div className="relative w-full h-40 bg-gray-950/50 rounded-lg border border-gray-800/50 overflow-hidden p-3">
        {/* Grid lines for dB reference */}
        <div className="absolute inset-0 flex flex-col justify-between py-3 px-3 pointer-events-none">
          {['+12', '+6', '0', '-6', '-12'].map((db, i) => (
            <div key={i} className="flex items-center">
              <div className="text-[8px] text-gray-600 font-mono w-6">{db}</div>
              <div className="flex-1 h-px bg-gray-800/50" style={{ 
                backgroundColor: db === '0' ? '#374151' : '#1f2937',
                height: db === '0' ? '1.5px' : '0.5px'
              }} />
            </div>
          ))}
        </div>

        {/* Frequency bars */}
        <div className="relative h-full flex items-end justify-between gap-1 px-8">
          {frequencyBands.map((band, index) => (
            <FrequencyBar
              key={`${band.freq}-${index}`}
              band={band}
              index={index}
              isActive={isLive || state === 'listening'}
            />
          ))}
        </div>

        {/* EQ curve overlay */}
        {frequencyBands.length > 0 && (
          <svg className="absolute inset-0 w-full h-full pointer-events-none px-8">
            <defs>
              <linearGradient id="eq-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#8b5cf6" />
                <stop offset="50%" stopColor="#6366f1" />
                <stop offset="100%" stopColor="#3b82f6" />
              </linearGradient>
            </defs>
            
            <motion.path
              key={`curve-${frequencyBands.map(b => b.gain).join('-')}`}
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.8 }}
              transition={{ 
                duration: isLive ? 0.4 : 0.8,
                ease: "easeInOut"
              }}
              d={generateEQCurvePath(frequencyBands)}
              fill="none"
              stroke="url(#eq-gradient)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* EQ adjustment points */}
            {frequencyBands.map((band, i) => {
              if (Math.abs(band.gain) < 0.5) return null; // Only show significant adjustments
              
              const x = ((i + 0.5) / frequencyBands.length) * 100;
              const y = gainToY(band.gain);
              
              return (
                <motion.g key={`point-${i}-${band.freq}`}>
                  <motion.circle
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ 
                      delay: i * 0.05,
                      type: "spring",
                      stiffness: 300,
                      damping: 20
                    }}
                    cx={`${x}%`}
                    cy={`${y}%`}
                    r="4"
                    fill={band.problemType === 'cut' ? '#ef4444' : '#8b5cf6'}
                    stroke="#1f2937"
                    strokeWidth="2"
                  />
                  {/* dB value label */}
                  <text
                    x={`${x}%`}
                    y={`${y}%`}
                    dy="-10"
                    fontSize="8"
                    fill={band.problemType === 'cut' ? '#ef4444' : '#8b5cf6'}
                    textAnchor="middle"
                    className="font-mono font-bold"
                  >
                    {band.gain > 0 ? '+' : ''}{band.gain.toFixed(1)}
                  </text>
                </motion.g>
              );
            })}
          </svg>
        )}
      </div>

      {/* Frequency labels */}
      <div className="flex justify-between mt-2 px-3">
        {baseBands.map((band, i) => (
          <div key={i} className="text-[9px] text-gray-500 font-mono text-center" style={{ width: '10%' }}>
            {band.label}
          </div>
        ))}
      </div>
    </motion.div>
  );
}

interface FrequencyBarProps {
  band: FrequencyBand;
  index: number;
  isActive: boolean;
}

function FrequencyBar({ band, index, isActive }: FrequencyBarProps) {
  const [animatedLevel, setAnimatedLevel] = useState(band.level);

  useEffect(() => {
    if (isActive) {
      // Animate to new level
      const animation = setInterval(() => {
        setAnimatedLevel(prev => {
          const diff = band.level - prev;
          if (Math.abs(diff) < 0.01) return band.level;
          return prev + diff * 0.3;
        });
      }, 50);
      
      return () => clearInterval(animation);
    } else {
      setAnimatedLevel(band.level);
    }
  }, [band.level, isActive]);

  // Determine bar color based on problem and level
  const getBarColor = () => {
    if (band.hasProblem) {
      if (band.problemType === 'cut') {
        return 'from-red-500/80 to-red-600/80';
      } else {
        return 'from-purple-500/80 to-purple-600/80';
      }
    }
    
    // Normal spectrum colors based on level
    if (animatedLevel > 0.8) return 'from-yellow-500/60 to-orange-500/60';
    if (animatedLevel > 0.5) return 'from-blue-500/60 to-cyan-500/60';
    return 'from-green-500/60 to-emerald-500/60';
  };

  const barHeight = animatedLevel * 100;
  const barColor = getBarColor();

  return (
    <div className="flex-1 h-full flex flex-col justify-end relative">
      {/* Bar */}
      <motion.div
        key={`bar-${band.freq}-${band.level}`}
        initial={{ height: 0 }}
        animate={{ height: `${barHeight}%` }}
        transition={{
          type: "spring",
          stiffness: isActive ? 200 : 300,
          damping: isActive ? 15 : 25,
          delay: index * 0.02
        }}
        className={`w-full bg-gradient-to-t ${barColor} rounded-t-sm relative`}
        style={{ minHeight: '2px' }}
      >
        {/* Peak indicator */}
        {animatedLevel > 0.85 && (
          <motion.div
            animate={{ opacity: [1, 0.5, 1] }}
            transition={{ duration: 0.5, repeat: Infinity }}
            className="absolute -top-0.5 left-0 right-0 h-0.5 bg-white rounded-full"
          />
        )}
      </motion.div>

      {/* Problem indicator at base */}
      {band.hasProblem && (
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full bg-red-500"
        />
      )}
    </div>
  );
}

// Helper functions

function gainToY(gain: number): number {
  // Map gain (-12 to +12 dB) to Y position (0% to 100%)
  const minGain = -12;
  const maxGain = 12;
  return ((maxGain - gain) / (maxGain - minGain)) * 100;
}

function generateFrequencyBands(
  problems: DiagnosticProblem[],
  isLive: boolean,
  state: string
): FrequencyBand[] {
  const bands: FrequencyBand[] = [];
  
  // Extract frequency adjustments from problems
  const adjustments = extractFrequencyAdjustments(problems);
  
  baseBands.forEach((baseBand, index) => {
    // Generate realistic level based on state
    let level = 0.3; // Default idle level
    
    if (state === 'listening' || state === 'monitoring') {
      // Simulate live audio levels
      level = 0.4 + Math.random() * 0.4 + Math.sin(Date.now() / 500 + index) * 0.15;
      
      // Mid frequencies typically louder
      if (baseBand.freq >= 500 && baseBand.freq <= 4000) {
        level += 0.1;
      }
    } else if (state === 'result') {
      // Static representation for result state
      level = 0.5 + Math.random() * 0.2;
      if (baseBand.freq >= 500 && baseBand.freq <= 4000) {
        level += 0.1;
      }
    }
    
    // Clamp level
    level = Math.max(0.1, Math.min(0.95, level));
    
    // Check if this frequency has adjustments
    const adjustment = findAdjustmentForFrequency(baseBand.freq, adjustments);
    
    bands.push({
      freq: baseBand.freq,
      label: baseBand.label,
      level,
      gain: adjustment?.gain || 0,
      hasProblem: !!adjustment,
      problemType: adjustment ? (adjustment.gain < 0 ? 'cut' : 'boost') : undefined
    });
  });
  
  return bands;
}

function extractFrequencyAdjustments(problems: DiagnosticProblem[]): Array<{ freqStart: number; freqEnd: number; gain: number }> {
  const adjustments: Array<{ freqStart: number; freqEnd: number; gain: number }> = [];
  
  problems.forEach(problem => {
    problem.actions.forEach(action => {
      const parsed = parseFrequencyAction(action);
      if (parsed) {
        adjustments.push(parsed);
      }
    });
  });
  
  return adjustments;
}

function parseFrequencyAction(action: string): { freqStart: number; freqEnd: number; gain: number } | null {
  // Parse actions like "100–250Hz -3dB" or "3kHz +2dB"
  const rangeMatch = action.match(/(\d+)(?:–|-)(\d+)?(?:Hz|kHz)\s*([+-]?\d+(?:\.\d+)?)\s*dB/i);
  
  if (rangeMatch) {
    let start = parseInt(rangeMatch[1]);
    let end = rangeMatch[2] ? parseInt(rangeMatch[2]) : start;
    const gain = parseFloat(rangeMatch[3]);
    
    // Convert kHz to Hz
    if (action.toLowerCase().includes('khz')) {
      start *= 1000;
      end *= 1000;
    }
    
    return { freqStart: start, freqEnd: end, gain };
  }
  
  return null;
}

function findAdjustmentForFrequency(
  freq: number,
  adjustments: Array<{ freqStart: number; freqEnd: number; gain: number }>
): { gain: number } | null {
  for (const adj of adjustments) {
    if (freq >= adj.freqStart && freq <= adj.freqEnd) {
      return { gain: adj.gain };
    }
    
    // Check if frequency is close enough to the range (within 50% tolerance)
    const center = (adj.freqStart + adj.freqEnd) / 2;
    const range = adj.freqEnd - adj.freqStart;
    const tolerance = range * 0.5;
    
    if (Math.abs(freq - center) <= tolerance + range / 2) {
      // Interpolate gain based on distance
      const distance = Math.abs(freq - center);
      const maxDistance = tolerance + range / 2;
      const factor = 1 - (distance / maxDistance);
      return { gain: adj.gain * factor };
    }
  }
  
  return null;
}

function generateEQCurvePath(bands: FrequencyBand[]): string {
  if (bands.length === 0) return '';
  
  const points = bands.map((band, index) => ({
    x: ((index + 0.5) / bands.length) * 100,
    y: gainToY(band.gain)
  }));
  
  // Start path
  let path = `M ${points[0].x} ${points[0].y}`;
  
  // Create smooth curve using bezier curves
  for (let i = 0; i < points.length - 1; i++) {
    const current = points[i];
    const next = points[i + 1];
    
    // Control points for smooth curve
    const cp1x = current.x + (next.x - current.x) * 0.5;
    const cp1y = current.y;
    const cp2x = current.x + (next.x - current.x) * 0.5;
    const cp2y = next.y;
    
    path += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${next.x} ${next.y}`;
  }
  
  return path;
}