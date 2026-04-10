import { motion } from 'motion/react';
import { useEffect, useId, useMemo, useRef } from 'react';
import type { DiagnosticProblem } from '../types';
import { TARGET_SAMPLE_RATE } from '../audio/audioUtils';

interface EQVisualizationProps {
  analyserNode?: AnalyserNode | null;
  featureRms?: number | null;
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

const scopeScaleLabels = ['+1', '0', '-1'];
const analyzerScaleLabels = ['0', '-12', '-24', '-36', '-48'];
const meterScaleLabels = ['0', '-6', '-12', '-18', '-24'];

export function EQVisualization({
  analyserNode = null,
  featureRms = null,
  problems,
  isLive = false,
  state = 'idle'
}: EQVisualizationProps) {
  const scopeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const analyzerCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const meterCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef(0);
  const bandPeakHoldRef = useRef<number[]>(Array(baseBands.length).fill(0));
  const outputPeakHoldRef = useRef(0);
  const gradientId = useId().replace(/:/g, '-');

  const adjustmentBands = useMemo(() => generateAdjustmentBands(problems), [problems]);
  const isActive = isLive || state === 'listening';
  const statusInfo = getStatusInfo(state, problems.length);

  useEffect(() => {
    const canvases = [
      scopeCanvasRef.current,
      analyzerCanvasRef.current,
      meterCanvasRef.current
    ].filter(Boolean) as HTMLCanvasElement[];

    if (canvases.length === 0) {
      return;
    }

    const syncAllCanvases = () => {
      canvases.forEach(syncCanvasToDisplaySize);
    };

    syncAllCanvases();

    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', syncAllCanvases);

      return () => {
        window.removeEventListener('resize', syncAllCanvases);
      };
    }

    const resizeObserver = new ResizeObserver(() => {
      syncAllCanvases();
    });

    canvases.forEach((canvas) => resizeObserver.observe(canvas));

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  useEffect(() => {
    const scopeCanvas = scopeCanvasRef.current;
    const analyzerCanvas = analyzerCanvasRef.current;
    const meterCanvas = meterCanvasRef.current;

    if (!scopeCanvas || !analyzerCanvas || !meterCanvas) {
      return;
    }

    const scopeContext = scopeCanvas.getContext('2d');
    const analyzerContext = analyzerCanvas.getContext('2d');
    const meterContext = meterCanvas.getContext('2d');

    if (!scopeContext || !analyzerContext || !meterContext) {
      return;
    }

    const timeDomainData = analyserNode ? new Uint8Array(analyserNode.fftSize) : null;
    const frequencyData = analyserNode ? new Uint8Array(analyserNode.frequencyBinCount) : null;

    const renderFrame = (timestamp: number) => {
      const deltaSeconds = lastFrameTimeRef.current > 0
        ? Math.min(0.05, (timestamp - lastFrameTimeRef.current) / 1000)
        : 1 / 60;
      lastFrameTimeRef.current = timestamp;

      const liveTimeDomainData = isActive && analyserNode && timeDomainData ? timeDomainData : null;
      const liveFrequencyData = isActive && analyserNode && frequencyData ? frequencyData : null;

      if (liveTimeDomainData && liveFrequencyData) {
        analyserNode.getByteTimeDomainData(liveTimeDomainData);
        analyserNode.getByteFrequencyData(liveFrequencyData);
      }

      const analyzerLevels = getDisplayBandLevels(
        liveFrequencyData,
        analyserNode?.context.sampleRate ?? TARGET_SAMPLE_RATE,
        adjustmentBands,
        state
      );
      const outputLevel = getOutputLevel(liveTimeDomainData, analyzerLevels);

      drawScopePanel(
        scopeContext,
        scopeCanvas.getBoundingClientRect().width,
        scopeCanvas.getBoundingClientRect().height,
        liveTimeDomainData,
        state
      );
      drawAnalyzerPanel(
        analyzerContext,
        analyzerCanvas.getBoundingClientRect().width,
        analyzerCanvas.getBoundingClientRect().height,
        analyzerLevels,
        adjustmentBands,
        bandPeakHoldRef.current,
        deltaSeconds
      );
      drawOutputMeter(
        meterContext,
        meterCanvas.getBoundingClientRect().width,
        meterCanvas.getBoundingClientRect().height,
        outputLevel,
        featureRms,
        outputPeakHoldRef,
        deltaSeconds
      );

      if (isActive && analyserNode) {
        animationFrameRef.current = window.requestAnimationFrame(renderFrame);
      }
    };

    animationFrameRef.current = window.requestAnimationFrame(renderFrame);

    return () => {
      if (animationFrameRef.current !== null) {
        window.cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      lastFrameTimeRef.current = 0;
    };
  }, [adjustmentBands, analyserNode, featureRms, isActive, state]);

  return (
    <motion.div
      layout
      className="w-full bg-[#101419] border border-gray-700/60 rounded-xl p-4 shadow-[0_12px_30px_rgba(0,0,0,0.28)]"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="text-[10px] font-semibold tracking-[0.24em] text-gray-400 uppercase">
            Main Bus
          </div>
          <div className="px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-[9px] font-mono text-emerald-300">
            OSC
          </div>
          <div className="px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/20 text-[9px] font-mono text-cyan-300">
            RTA
          </div>
          <div className="px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/20 text-[9px] font-mono text-amber-300">
            EQ
          </div>
        </div>

        <div className={`flex items-center gap-2 text-xs font-medium ${statusInfo.color}`}>
          {isActive && (
            <div className={`w-1.5 h-1.5 rounded-full ${statusInfo.dotColor} animate-pulse`} />
          )}
          {statusInfo.text}
        </div>
      </div>

      <div className="rounded-lg border border-gray-800/80 bg-[#07090c] p-2.5 space-y-2">
        <div className="grid grid-cols-[24px_1fr] gap-2">
          <div className="flex flex-col justify-between py-1 text-[8px] font-mono text-gray-500">
            {scopeScaleLabels.map((label) => (
              <span key={label}>{label}</span>
            ))}
          </div>

          <div className="rounded-md border border-emerald-500/15 bg-[#040606] px-2 py-1.5">
            <div className="flex items-center justify-between mb-1">
              <div className="text-[9px] font-mono uppercase tracking-[0.18em] text-gray-400">
                Oscilloscope
              </div>
              <div className="text-[9px] font-mono text-emerald-300/80">
                Waveform
              </div>
            </div>

            <div className="relative h-16">
              <canvas ref={scopeCanvasRef} className="absolute inset-0 w-full h-full" />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-[24px_1fr_30px] gap-2">
          <div className="flex flex-col justify-between py-1 text-[8px] font-mono text-gray-500">
            {analyzerScaleLabels.map((label) => (
              <span key={label}>{label}</span>
            ))}
          </div>

          <div className="rounded-md border border-cyan-500/15 bg-[#040607] px-2 py-1.5">
            <div className="flex items-center justify-between mb-1">
              <div className="text-[9px] font-mono uppercase tracking-[0.18em] text-gray-400">
                Graphic EQ
              </div>
              <div className="text-[9px] font-mono text-cyan-300/80">
                31Hz - 16kHz
              </div>
            </div>

            <div className="relative h-24">
              <canvas ref={analyzerCanvasRef} className="absolute inset-0 w-full h-full" />

              <svg className="absolute inset-0 w-full h-full pointer-events-none">
                <defs>
                  <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#f59e0b" />
                    <stop offset="40%" stopColor="#f97316" />
                    <stop offset="100%" stopColor="#fb7185" />
                  </linearGradient>
                </defs>

                <motion.path
                  key={`eq-curve-${adjustmentBands.map((band) => band.gain.toFixed(2)).join('-')}`}
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 0.85 }}
                  transition={{ duration: isActive ? 0.35 : 0.65, ease: 'easeInOut' }}
                  d={generateEQCurvePath(adjustmentBands)}
                  fill="none"
                  stroke={`url(#${gradientId})`}
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
                      <motion.rect
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{
                          delay: index * 0.03,
                          type: 'spring',
                          stiffness: 280,
                          damping: 22
                        }}
                        x={`${x - 1.5}%`}
                        y={`${y - 1.8}%`}
                        width="7"
                        height="7"
                        rx="1.5"
                        fill={band.problemType === 'cut' ? '#ef4444' : '#f59e0b'}
                        stroke="#020617"
                        strokeWidth="1.5"
                      />
                    </motion.g>
                  );
                })}
              </svg>
            </div>

            <div className="flex justify-between mt-2">
              {baseBands.map((band) => (
                <div
                  key={band.freq}
                  className="text-[9px] text-gray-500 font-mono text-center"
                  style={{ width: `${100 / baseBands.length}%` }}
                >
                  {band.label}
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-md border border-lime-500/15 bg-[#040607] px-1.5 py-1.5">
            <div className="flex flex-col justify-between h-full">
              <div className="text-[8px] font-mono text-center uppercase tracking-[0.16em] text-gray-400">
                RMS
              </div>

              <div className="relative h-24">
                <canvas ref={meterCanvasRef} className="absolute inset-0 w-full h-full" />
              </div>

              <div className="text-[7px] font-mono text-gray-500 text-center leading-tight whitespace-pre-line">
                {meterScaleLabels.join('\n')}
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function getStatusInfo(state: string, problemCount: number) {
  if (state === 'monitoring') {
    return { text: 'Live Mix', color: 'text-green-400', dotColor: 'bg-green-400' };
  }

  if (state === 'listening') {
    return { text: 'Reading Input', color: 'text-purple-400', dotColor: 'bg-purple-400' };
  }

  if (state === 'result' && problemCount > 0) {
    return { text: 'EQ Suggestions Ready', color: 'text-blue-400', dotColor: 'bg-blue-400' };
  }

  if (state === 'result') {
    return { text: 'Signal Clean', color: 'text-green-400', dotColor: 'bg-green-400' };
  }

  return { text: 'Standby', color: 'text-gray-400', dotColor: 'bg-gray-400' };
}

function syncCanvasToDisplaySize(canvas: HTMLCanvasElement) {
  const context = canvas.getContext('2d');

  if (!context) {
    return;
  }

  const { width, height } = canvas.getBoundingClientRect();
  const devicePixelRatio = window.devicePixelRatio || 1;
  const pixelWidth = Math.max(1, Math.floor(width * devicePixelRatio));
  const pixelHeight = Math.max(1, Math.floor(height * devicePixelRatio));

  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }

  context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}

function drawScopePanel(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  timeDomainData: Uint8Array | null,
  state: string
) {
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#050809';
  context.fillRect(0, 0, width, height);

  drawScopeGrid(context, width, height);

  const midY = height / 2;
  const amplitude = height * 0.33;
  const pointCount = Math.max(64, Math.floor(width));
  const values = timeDomainData
    ? buildScopeValues(timeDomainData, pointCount)
    : buildIdleScopeValues(pointCount, state);

  context.beginPath();
  context.moveTo(0, midY);
  context.lineTo(width, midY);
  context.strokeStyle = 'rgba(74, 222, 128, 0.18)';
  context.lineWidth = 1;
  context.stroke();

  context.beginPath();

  for (let index = 0; index < pointCount; index += 1) {
    const x = (index / (pointCount - 1)) * width;
    const y = midY + values[index] * amplitude;

    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  }

  context.strokeStyle = 'rgba(74, 222, 128, 0.96)';
  context.lineWidth = 1.8;
  context.lineJoin = 'round';
  context.lineCap = 'round';
  context.shadowColor = 'rgba(34, 197, 94, 0.45)';
  context.shadowBlur = 9;
  context.stroke();
  context.shadowBlur = 0;
}

function drawScopeGrid(context: CanvasRenderingContext2D, width: number, height: number) {
  context.strokeStyle = 'rgba(52, 211, 153, 0.08)';
  context.lineWidth = 1;

  for (let column = 0; column <= 8; column += 1) {
    const x = (column / 8) * width;
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }

  for (let row = 0; row <= 4; row += 1) {
    const y = (row / 4) * height;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
}

function drawAnalyzerPanel(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  levels: number[],
  adjustments: FrequencyBandAdjustment[],
  peakHoldLevels: number[],
  deltaSeconds: number
) {
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#050709';
  context.fillRect(0, 0, width, height);

  drawAnalyzerGrid(context, width, height);

  const slotWidth = width / levels.length;
  const segmentGap = 2;
  const segmentCount = 14;
  const segmentHeight = (height - segmentGap * (segmentCount - 1)) / segmentCount;

  for (let index = 0; index < levels.length; index += 1) {
    const x = index * slotWidth;
    const trackWidth = Math.max(8, slotWidth * 0.58);
    const trackX = x + (slotWidth - trackWidth) / 2;
    const normalizedLevel = clamp(levels[index], 0.02, 1);
    const litSegments = Math.round(normalizedLevel * segmentCount);

    peakHoldLevels[index] = Math.max(
      normalizedLevel,
      peakHoldLevels[index] - deltaSeconds * 0.42
    );

    if (adjustments[index]?.hasProblem) {
      context.fillStyle = adjustments[index].problemType === 'cut'
        ? 'rgba(239, 68, 68, 0.08)'
        : 'rgba(245, 158, 11, 0.08)';
      context.fillRect(trackX - 3, 0, trackWidth + 6, height);
    }

    for (let segmentIndex = 0; segmentIndex < segmentCount; segmentIndex += 1) {
      const y = height - (segmentIndex + 1) * segmentHeight - segmentIndex * segmentGap;
      const isLit = segmentIndex < litSegments;
      context.fillStyle = getMeterSegmentColor(segmentIndex, segmentCount, isLit);
      context.fillRect(trackX, y, trackWidth, Math.max(2, segmentHeight));
    }

    const peakY = height - peakHoldLevels[index] * height;
    context.fillStyle = 'rgba(255, 255, 255, 0.92)';
    context.fillRect(trackX - 1, clamp(peakY, 1, height - 2), trackWidth + 2, 1.5);
  }
}

function drawAnalyzerGrid(context: CanvasRenderingContext2D, width: number, height: number) {
  context.lineWidth = 1;

  for (let column = 0; column <= baseBands.length; column += 1) {
    const x = (column / baseBands.length) * width;
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.strokeStyle = 'rgba(51, 65, 85, 0.24)';
    context.stroke();
  }

  for (let row = 0; row <= 4; row += 1) {
    const y = (row / 4) * height;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.strokeStyle = row === 2 ? 'rgba(71, 85, 105, 0.38)' : 'rgba(51, 65, 85, 0.18)';
    context.stroke();
  }
}

function drawOutputMeter(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  level: number,
  featureRms: number | null,
  peakHoldRef: { current: number },
  deltaSeconds: number
) {
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#050709';
  context.fillRect(0, 0, width, height);

  const segmentGap = 2;
  const segmentCount = 14;
  const segmentHeight = (height - segmentGap * (segmentCount - 1)) / segmentCount;
  const normalizedLevel = clamp(level, 0.02, 1);
  const litSegments = Math.round(normalizedLevel * segmentCount);

  peakHoldRef.current = Math.max(normalizedLevel, peakHoldRef.current - deltaSeconds * 0.36);

  for (let segmentIndex = 0; segmentIndex < segmentCount; segmentIndex += 1) {
    const y = height - (segmentIndex + 1) * segmentHeight - segmentIndex * segmentGap;
    const isLit = segmentIndex < litSegments;
    context.fillStyle = getMeterSegmentColor(segmentIndex, segmentCount, isLit);
    context.fillRect(0, y, width, Math.max(2, segmentHeight));
  }

  const peakY = height - peakHoldRef.current * height;
  context.fillStyle = 'rgba(255, 255, 255, 0.9)';
  context.fillRect(0, clamp(peakY, 1, height - 2), width, 1.5);

  if (featureRms !== null) {
    const rmsY = height - clamp(featureRms, 0, 1) * height;
    context.fillStyle = 'rgba(34, 211, 238, 0.9)';
    context.fillRect(0, clamp(rmsY, 1, height - 2), width, 1.5);
  }
}

function getMeterSegmentColor(segmentIndex: number, segmentCount: number, isLit: boolean) {
  if (!isLit) {
    return 'rgba(30, 41, 59, 0.38)';
  }

  const ratio = segmentIndex / (segmentCount - 1);

  if (ratio > 0.82) {
    return 'rgba(239, 68, 68, 0.92)';
  }

  if (ratio > 0.62) {
    return 'rgba(250, 204, 21, 0.92)';
  }

  return 'rgba(74, 222, 128, 0.9)';
}

function buildScopeValues(timeDomainData: Uint8Array, pointCount: number) {
  const values = new Array<number>(pointCount);

  for (let index = 0; index < pointCount; index += 1) {
    const ratio = index / (pointCount - 1);
    const samplePosition = ratio * (timeDomainData.length - 1);
    const baseIndex = Math.floor(samplePosition);
    const nextIndex = Math.min(baseIndex + 1, timeDomainData.length - 1);
    const fraction = samplePosition - baseIndex;
    const value = timeDomainData[baseIndex]
      + (timeDomainData[nextIndex] - timeDomainData[baseIndex]) * fraction;

    values[index] = value / 128 - 1;
  }

  return values;
}

function buildIdleScopeValues(pointCount: number, state: string) {
  const amplitude = state === 'result' ? 0.045 : 0.015;
  const values = new Array<number>(pointCount);

  for (let index = 0; index < pointCount; index += 1) {
    const ratio = index / (pointCount - 1);
    values[index] = Math.sin(ratio * Math.PI * 3.2) * amplitude;
  }

  return values;
}

function getDisplayBandLevels(
  frequencyData: Uint8Array | null,
  sampleRate: number,
  adjustments: FrequencyBandAdjustment[],
  state: string
) {
  return baseBands.map((band, index) => {
    if (frequencyData) {
      const level = getFrequencyBandLevel(frequencyData, sampleRate, band.freq);
      return clamp(Math.pow(level, 0.78) * 1.08, 0.04, 1);
    }

    const adjustmentBias = Math.min(0.26, Math.abs(adjustments[index].gain) / 28);
    const midBias = band.freq >= 250 && band.freq <= 4000 ? 0.06 : 0.02;

    if (state === 'result') {
      return clamp(0.12 + adjustmentBias + midBias, 0.08, 0.42);
    }

    return 0.05 + adjustmentBias * 0.2;
  });
}

function getOutputLevel(timeDomainData: Uint8Array | null, analyzerLevels: number[]) {
  if (timeDomainData) {
    let sum = 0;

    for (let index = 0; index < timeDomainData.length; index += 1) {
      const normalizedValue = timeDomainData[index] / 128 - 1;
      sum += normalizedValue * normalizedValue;
    }

    return clamp(Math.sqrt(sum / timeDomainData.length) * 2.8, 0.04, 1);
  }

  const averageLevel = analyzerLevels.reduce((total, level) => total + level, 0) / analyzerLevels.length;
  return clamp(averageLevel * 1.1, 0.04, 0.45);
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
