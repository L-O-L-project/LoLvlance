import { motion } from 'motion/react';
import { useEffect, useMemo, useRef } from 'react';
import type { DiagnosticProblem } from '../types';
import { TARGET_SAMPLE_RATE } from '../audio/audioUtils';

interface EQVisualizationProps {
  analyserNode?: AnalyserNode | null;
  problems: DiagnosticProblem[];
  isLive?: boolean;
  state?: 'idle' | 'listening' | 'result' | 'monitoring';
}

interface FrequencyBandAdjustment {
  freq: number;
  label: string;
  gain: number;
  hasProblem: boolean;
  problemType?: 'cut' | 'boost';
}

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

export function EQVisualization({
  analyserNode = null,
  problems,
  isLive = false,
  state = 'idle'
}: EQVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const adjustmentBands = useMemo(() => generateAdjustmentBands(problems), [problems]);

  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas) {
      return;
    }

    const resizeCanvas = () => {
      const context = canvas.getContext('2d');

      if (!context) {
        return;
      }

      const { width, height } = canvas.getBoundingClientRect();
      const devicePixelRatio = window.devicePixelRatio || 1;

      canvas.width = Math.max(1, Math.floor(width * devicePixelRatio));
      canvas.height = Math.max(1, Math.floor(height * devicePixelRatio));
      context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    };

    resizeCanvas();

    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', resizeCanvas);

      return () => {
        window.removeEventListener('resize', resizeCanvas);
      };
    }

    const resizeObserver = new ResizeObserver(() => {
      resizeCanvas();
    });

    resizeObserver.observe(canvas);

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');

    if (!canvas || !context) {
      return;
    }

    const renderFrame = () => {
      const isActive = isLive || state === 'listening';

      if (!isActive || !analyserNode) {
        const { width, height } = canvas.getBoundingClientRect();
        drawVisualizationFrame(context, width, height, null, null, TARGET_SAMPLE_RATE);
        return;
      }

      const timeDomainData = new Uint8Array(analyserNode.fftSize);
      const frequencyData = new Uint8Array(analyserNode.frequencyBinCount);

      const render = () => {
        const { width, height } = canvas.getBoundingClientRect();
        analyserNode.getByteTimeDomainData(timeDomainData);
        analyserNode.getByteFrequencyData(frequencyData);
        drawVisualizationFrame(
          context,
          width,
          height,
          timeDomainData,
          frequencyData,
          analyserNode.context.sampleRate
        );
        animationFrameRef.current = window.requestAnimationFrame(render);
      };

      render();
    };

    renderFrame();

    return () => {
      if (animationFrameRef.current !== null) {
        window.cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [analyserNode, isLive, state]);

  const statusInfo = getStatusInfo(state, problems.length);

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

      <div className="relative w-full h-40 bg-gray-950/50 rounded-lg border border-gray-800/50 overflow-hidden p-3">
        <div className="absolute inset-0 flex flex-col justify-between py-3 px-3 pointer-events-none">
          {['+12', '+6', '0', '-6', '-12'].map((db) => (
            <div key={db} className="flex items-center">
              <div className="text-[8px] text-gray-600 font-mono w-6">{db}</div>
              <div
                className="flex-1 h-px bg-gray-800/50"
                style={{
                  backgroundColor: db === '0' ? '#374151' : '#1f2937',
                  height: db === '0' ? '1.5px' : '0.5px'
                }}
              />
            </div>
          ))}
        </div>

        <div className="absolute inset-3">
          <canvas ref={canvasRef} className="w-full h-full" />

          {adjustmentBands.length > 0 && (
            <svg className="absolute inset-0 w-full h-full pointer-events-none">
              <defs>
                <linearGradient id="eq-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#8b5cf6" />
                  <stop offset="50%" stopColor="#6366f1" />
                  <stop offset="100%" stopColor="#3b82f6" />
                </linearGradient>
              </defs>

              <motion.path
                key={`curve-${adjustmentBands.map((band) => band.gain.toFixed(2)).join('-')}`}
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 0.85 }}
                transition={{
                  duration: isLive ? 0.35 : 0.75,
                  ease: 'easeInOut'
                }}
                d={generateEQCurvePath(adjustmentBands)}
                fill="none"
                stroke="url(#eq-gradient)"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {adjustmentBands.map((band, index) => {
                if (Math.abs(band.gain) < 0.5) {
                  return null;
                }

                const x = ((index + 0.5) / adjustmentBands.length) * 100;
                const y = gainToY(band.gain);

                return (
                  <motion.g key={`${band.freq}-${band.gain}`}>
                    <motion.circle
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{
                        delay: index * 0.04,
                        type: 'spring',
                        stiffness: 280,
                        damping: 22
                      }}
                      cx={`${x}%`}
                      cy={`${y}%`}
                      r="4"
                      fill={band.problemType === 'cut' ? '#ef4444' : '#8b5cf6'}
                      stroke="#111827"
                      strokeWidth="2"
                    />
                    <text
                      x={`${x}%`}
                      y={`${y}%`}
                      dy="-10"
                      fontSize="8"
                      fill={band.problemType === 'cut' ? '#f87171' : '#a78bfa'}
                      textAnchor="middle"
                      className="font-mono font-bold"
                    >
                      {band.gain > 0 ? '+' : ''}
                      {band.gain.toFixed(1)}
                    </text>
                  </motion.g>
                );
              })}
            </svg>
          )}
        </div>
      </div>

      <div className="flex justify-between mt-2 px-3">
        {baseBands.map((band) => (
          <div
            key={band.freq}
            className="text-[9px] text-gray-500 font-mono text-center"
            style={{ width: '10%' }}
          >
            {band.label}
          </div>
        ))}
      </div>
    </motion.div>
  );
}

function getStatusInfo(state: string, problemCount: number) {
  if (state === 'monitoring') {
    return { text: 'Real-time Monitoring', color: 'text-green-400', dotColor: 'bg-green-400' };
  }

  if (state === 'listening') {
    return { text: 'Analyzing...', color: 'text-purple-400', dotColor: 'bg-purple-400' };
  }

  if (state === 'result' && problemCount > 0) {
    return { text: 'Analysis Complete', color: 'text-blue-400', dotColor: 'bg-blue-400' };
  }

  if (state === 'result') {
    return { text: 'No Issues Detected', color: 'text-green-400', dotColor: 'bg-green-400' };
  }

  return { text: 'Ready', color: 'text-gray-400', dotColor: 'bg-gray-400' };
}

function drawVisualizationFrame(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  timeDomainData: Uint8Array | null,
  frequencyData: Uint8Array | null,
  sampleRate: number
) {
  context.clearRect(0, 0, width, height);
  drawFrequencyBars(context, width, height, frequencyData, sampleRate);
  drawWaveform(context, width, height, timeDomainData);
}

function drawFrequencyBars(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  frequencyData: Uint8Array | null,
  sampleRate: number
) {
  const slotWidth = width / baseBands.length;
  const maxBarHeight = height * 0.78;
  const floorY = height;

  for (let index = 0; index < baseBands.length; index += 1) {
    const x = index * slotWidth;
    const barWidth = Math.max(6, slotWidth * 0.58);
    const barX = x + (slotWidth - barWidth) / 2;
    const level = frequencyData
      ? getFrequencyBandLevel(frequencyData, sampleRate, baseBands[index].freq)
      : 0.16 + Math.sin(index * 0.7) * 0.02;
    const normalizedLevel = clamp(level * 0.92 + 0.08, 0.08, 1);
    const barHeight = normalizedLevel * maxBarHeight;
    const barY = floorY - barHeight;

    const gradient = context.createLinearGradient(0, barY, 0, floorY);

    if (normalizedLevel > 0.75) {
      gradient.addColorStop(0, 'rgba(251, 191, 36, 0.85)');
      gradient.addColorStop(1, 'rgba(249, 115, 22, 0.35)');
    } else if (normalizedLevel > 0.45) {
      gradient.addColorStop(0, 'rgba(59, 130, 246, 0.82)');
      gradient.addColorStop(1, 'rgba(34, 211, 238, 0.32)');
    } else {
      gradient.addColorStop(0, 'rgba(16, 185, 129, 0.78)');
      gradient.addColorStop(1, 'rgba(6, 182, 212, 0.22)');
    }

    context.fillStyle = gradient;
    drawRoundedRectPath(context, barX, barY, barWidth, barHeight, 4);
    context.fill();

    if (normalizedLevel > 0.88) {
      context.fillStyle = 'rgba(255, 255, 255, 0.85)';
      context.fillRect(barX, Math.max(0, barY - 1), barWidth, 1.5);
    }
  }
}

function drawWaveform(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  timeDomainData: Uint8Array | null
) {
  const midY = height * 0.5;
  const amplitude = height * 0.23;
  const points = Math.max(64, Math.floor(width));

  context.beginPath();

  for (let pointIndex = 0; pointIndex < points; pointIndex += 1) {
    const ratio = pointIndex / (points - 1);
    const samplePosition = ratio * ((timeDomainData?.length ?? points) - 1);
    let normalizedValue = 0;

    if (timeDomainData) {
      const baseIndex = Math.floor(samplePosition);
      const nextIndex = Math.min(baseIndex + 1, timeDomainData.length - 1);
      const fraction = samplePosition - baseIndex;
      const value = timeDomainData[baseIndex]
        + (timeDomainData[nextIndex] - timeDomainData[baseIndex]) * fraction;

      normalizedValue = value / 128 - 1;
    } else {
      normalizedValue = Math.sin(ratio * Math.PI * 2) * 0.03;
    }

    const x = ratio * width;
    const y = midY + normalizedValue * amplitude;

    if (pointIndex === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  }

  context.strokeStyle = 'rgba(196, 181, 253, 0.95)';
  context.lineWidth = 2;
  context.lineJoin = 'round';
  context.lineCap = 'round';
  context.shadowColor = 'rgba(99, 102, 241, 0.45)';
  context.shadowBlur = 10;
  context.stroke();
  context.shadowBlur = 0;

  context.beginPath();
  context.moveTo(0, midY);
  context.lineTo(width, midY);
  context.strokeStyle = 'rgba(148, 163, 184, 0.14)';
  context.lineWidth = 1;
  context.stroke();
}

function getFrequencyBandLevel(
  frequencyData: Uint8Array,
  sampleRate: number,
  targetFrequency: number
) {
  const nyquist = sampleRate / 2;
  const normalizedFrequency = clamp(targetFrequency / nyquist, 0, 1);
  const centerIndex = Math.round(normalizedFrequency * (frequencyData.length - 1));
  const startIndex = Math.max(0, centerIndex - 2);
  const endIndex = Math.min(frequencyData.length - 1, centerIndex + 2);
  let total = 0;
  let count = 0;

  for (let index = startIndex; index <= endIndex; index += 1) {
    total += frequencyData[index];
    count += 1;
  }

  return count === 0 ? 0 : total / count / 255;
}

function drawRoundedRectPath(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
) {
  const effectiveRadius = Math.min(radius, width / 2, height / 2);

  context.beginPath();
  context.moveTo(x, y + height);
  context.lineTo(x, y + effectiveRadius);
  context.quadraticCurveTo(x, y, x + effectiveRadius, y);
  context.lineTo(x + width - effectiveRadius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + effectiveRadius);
  context.lineTo(x + width, y + height);
  context.closePath();
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function gainToY(gain: number) {
  const minGain = -12;
  const maxGain = 12;
  return ((maxGain - gain) / (maxGain - minGain)) * 100;
}

function generateAdjustmentBands(problems: DiagnosticProblem[]): FrequencyBandAdjustment[] {
  const adjustments = extractFrequencyAdjustments(problems);

  return baseBands.map((band) => {
    const adjustment = findAdjustmentForFrequency(band.freq, adjustments);

    return {
      freq: band.freq,
      label: band.label,
      gain: adjustment?.gain || 0,
      hasProblem: Boolean(adjustment),
      problemType: adjustment ? (adjustment.gain < 0 ? 'cut' : 'boost') : undefined
    };
  });
}

function extractFrequencyAdjustments(
  problems: DiagnosticProblem[]
): Array<{ freqStart: number; freqEnd: number; gain: number }> {
  const adjustments: Array<{ freqStart: number; freqEnd: number; gain: number }> = [];

  problems.forEach((problem) => {
    problem.actions.forEach((action) => {
      const parsedAdjustment = parseFrequencyAction(action);

      if (parsedAdjustment) {
        adjustments.push(parsedAdjustment);
      }
    });
  });

  return adjustments;
}

function parseFrequencyAction(action: string) {
  const rangePattern = /(\d+(?:\.\d+)?)(k?Hz)?\s*(?:–|-)\s*(\d+(?:\.\d+)?)(k?Hz)?\s*([+-]?\d+(?:\.\d+)?)\s*dB/i;
  const singlePattern = /(\d+(?:\.\d+)?)(k?Hz)?\s*([+-]?\d+(?:\.\d+)?)\s*dB/i;

  const rangeMatch = action.match(rangePattern);

  if (rangeMatch) {
    const start = parseFrequencyValue(rangeMatch[1], rangeMatch[2]);
    const end = parseFrequencyValue(rangeMatch[3], rangeMatch[4]);

    return {
      freqStart: Math.min(start, end),
      freqEnd: Math.max(start, end),
      gain: parseFloat(rangeMatch[5])
    };
  }

  const singleMatch = action.match(singlePattern);

  if (!singleMatch) {
    return null;
  }

  const frequency = parseFrequencyValue(singleMatch[1], singleMatch[2]);

  return {
    freqStart: frequency,
    freqEnd: frequency,
    gain: parseFloat(singleMatch[3])
  };
}

function parseFrequencyValue(value: string, unit: string | undefined) {
  const numericValue = parseFloat(value);
  return unit?.toLowerCase() === 'khz' ? numericValue * 1000 : numericValue;
}

function findAdjustmentForFrequency(
  frequency: number,
  adjustments: Array<{ freqStart: number; freqEnd: number; gain: number }>
) {
  for (const adjustment of adjustments) {
    if (frequency >= adjustment.freqStart && frequency <= adjustment.freqEnd) {
      return { gain: adjustment.gain };
    }

    const center = (adjustment.freqStart + adjustment.freqEnd) / 2;
    const span = Math.max(adjustment.freqEnd - adjustment.freqStart, center * 0.2);
    const distance = Math.abs(frequency - center);

    if (distance <= span) {
      return {
        gain: adjustment.gain * (1 - distance / span)
      };
    }
  }

  return null;
}

function generateEQCurvePath(bands: FrequencyBandAdjustment[]) {
  if (bands.length === 0) {
    return '';
  }

  const points = bands.map((band, index) => ({
    x: ((index + 0.5) / bands.length) * 100,
    y: gainToY(band.gain)
  }));

  let path = `M ${points[0].x} ${points[0].y}`;

  for (let index = 0; index < points.length - 1; index += 1) {
    const current = points[index];
    const next = points[index + 1];
    const controlPointX = current.x + (next.x - current.x) * 0.5;

    path += ` C ${controlPointX} ${current.y}, ${controlPointX} ${next.y}, ${next.x} ${next.y}`;
  }

  return path;
}
