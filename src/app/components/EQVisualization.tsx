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

// EQ bands retained as reference points for parametric filter computation
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

// X32-style frequency axis labels (log-positioned in JSX)
const freqDisplayLabels = [
  { freq: 20, label: '20' },
  { freq: 50, label: '50' },
  { freq: 100, label: '100' },
  { freq: 200, label: '200' },
  { freq: 500, label: '500' },
  { freq: 1000, label: '1k' },
  { freq: 2000, label: '2k' },
  { freq: 5000, label: '5k' },
  { freq: 10000, label: '10k' },
  { freq: 20000, label: '20k' },
];

// ±15 dB range matching X32 default view
const EQ_DB_RANGE = 15;
const MIN_DISPLAY_FREQ = 20;
const MAX_DISPLAY_FREQ = 20000;

const scopeScaleLabels = ['+1', '0', '-1'];
// dB scale matching EQ_DB_RANGE
const analyzerScaleLabels = ['+12', '+6', '0', '-6', '-12'];
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
  const outputPeakHoldRef = useRef(0);
  const smoothedSpectrumRef = useRef<Float32Array | null>(null);
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

      if (liveTimeDomainData && liveFrequencyData && analyserNode) {
        analyserNode.getByteTimeDomainData(liveTimeDomainData);
        analyserNode.getByteFrequencyData(liveFrequencyData);
        if (!smoothedSpectrumRef.current || smoothedSpectrumRef.current.length !== liveFrequencyData.length) {
          smoothedSpectrumRef.current = new Float32Array(liveFrequencyData.length);
        }
        updateSmoothedSpectrum(smoothedSpectrumRef.current, liveFrequencyData);
      } else if (!isActive) {
        smoothedSpectrumRef.current = null;
      }

      const sampleRate = analyserNode?.context.sampleRate ?? TARGET_SAMPLE_RATE;
      const outputLevel = getOutputLevel(liveTimeDomainData);

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
        smoothedSpectrumRef.current,
        sampleRate
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

      <div className="rounded-lg border border-gray-800/80 bg-[#07090c] p-2 md:p-2.5 space-y-2">
        {/* Oscilloscope row */}
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

            <div className="relative h-12 md:h-16 lg:h-20">
              <canvas ref={scopeCanvasRef} className="absolute inset-0 w-full h-full" />
            </div>
          </div>
        </div>

        {/* Parametric EQ + RTA row */}
        <div className="grid grid-cols-[24px_1fr_30px] gap-2">
          {/* dB scale labels — aligned to ±12 dB ticks inside the grid */}
          <div className="flex flex-col justify-between py-1 text-[8px] font-mono text-gray-500">
            {analyzerScaleLabels.map((label) => (
              <span key={label}>{label}</span>
            ))}
          </div>

          <div className="rounded-md border border-cyan-500/15 bg-[#040607] px-2 py-1.5">
            <div className="flex items-center justify-between mb-1">
              <div className="text-[9px] font-mono uppercase tracking-[0.18em] text-gray-400">
                Parametric EQ
              </div>
              <div className="text-[9px] font-mono text-cyan-300/80">
                20Hz – 20kHz
              </div>
            </div>

            <div className="relative h-20 md:h-24 lg:h-32">
              {/* Canvas: EQ grid + RTA spectrum */}
              <canvas ref={analyzerCanvasRef} className="absolute inset-0 w-full h-full" />

              {/* SVG: EQ response curve + band markers */}
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
              >
                <defs>
                  <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#f59e0b" />
                    <stop offset="40%" stopColor="#f97316" />
                    <stop offset="100%" stopColor="#fb7185" />
                  </linearGradient>
                </defs>

                {/* EQ response curve — proper parametric filter shape */}
                <motion.path
                  key={`eq-curve-${adjustmentBands.map((b) => b.gain.toFixed(2)).join('-')}`}
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 0.88 }}
                  transition={{ duration: isActive ? 0.35 : 0.65, ease: 'easeInOut' }}
                  d={generateEQCurvePath(adjustmentBands)}
                  fill="none"
                  stroke={`url(#${gradientId})`}
                  strokeWidth="0.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  vectorEffect="non-scaling-stroke"
                />

                {/* Band markers at each adjusted frequency */}
                {adjustmentBands.map((band, index) => {
                  if (Math.abs(band.gain) < 0.5) return null;

                  const cx = freqToXPercent(band.freq);
                  const cy = dbToYPercent(band.gain);

                  return (
                    <motion.g key={`${band.freq}-${band.gain}`}>
                      {/* Vertical stem from 0 dB line to band point */}
                      <line
                        x1={`${cx}%`}
                        y1="50%"
                        x2={`${cx}%`}
                        y2={`${cy}%`}
                        stroke={band.problemType === 'cut' ? 'rgba(239,68,68,0.3)' : 'rgba(245,158,11,0.3)'}
                        strokeWidth="0.4"
                        vectorEffect="non-scaling-stroke"
                      />
                      {/* Diamond marker */}
                      <motion.rect
                        initial={{ scale: 0, rotate: 45 }}
                        animate={{ scale: 1, rotate: 45 }}
                        transition={{
                          delay: index * 0.04,
                          type: 'spring',
                          stiffness: 300,
                          damping: 20
                        }}
                        x={`${cx - 1.2}%`}
                        y={`${cy - 1.2}%`}
                        width="2.4%"
                        height="2.4%"
                        fill={band.problemType === 'cut' ? '#ef4444' : '#f59e0b'}
                        stroke="#020617"
                        strokeWidth="0.8"
                        vectorEffect="non-scaling-stroke"
                      />
                      {/* Frequency label below marker */}
                      <text
                        x={`${cx}%`}
                        y={`${cy + (band.gain < 0 ? -4 : 4.5)}%`}
                        textAnchor="middle"
                        fontSize="3.5"
                        fill={band.problemType === 'cut' ? 'rgba(239,68,68,0.85)' : 'rgba(245,158,11,0.85)'}
                        vectorEffect="non-scaling-stroke"
                      >
                        {band.label}
                      </text>
                    </motion.g>
                  );
                })}
              </svg>
            </div>

            {/* Frequency labels — log-positioned */}
            <div className="relative mt-1 h-3">
              {freqDisplayLabels.map(({ freq, label }) => (
                <div
                  key={freq}
                  className="absolute text-[8px] text-gray-600 font-mono"
                  style={{
                    left: `${freqToXPercent(freq)}%`,
                    transform: 'translateX(-50%)'
                  }}
                >
                  {label}
                </div>
              ))}
            </div>
          </div>

          {/* Output level meter */}
          <div className="rounded-md border border-lime-500/15 bg-[#040607] px-1.5 py-1.5">
            <div className="flex flex-col justify-between h-full">
              <div className="text-[8px] font-mono text-center uppercase tracking-[0.16em] text-gray-400">
                RMS
              </div>

              <div className="relative h-20 md:h-24 lg:h-32">
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

// ─── Coordinate helpers ──────────────────────────────────────────────────────

/** Map a frequency (Hz) to an SVG/CSS X percentage using a log scale. */
function freqToXPercent(
  freq: number,
  minFreq = MIN_DISPLAY_FREQ,
  maxFreq = MAX_DISPLAY_FREQ
): number {
  const logMin = Math.log10(minFreq);
  const logMax = Math.log10(maxFreq);
  return ((Math.log10(clamp(freq, minFreq, maxFreq)) - logMin) / (logMax - logMin)) * 100;
}

/** Map a gain value (dB) to a Y percentage within the ±EQ_DB_RANGE window. */
function dbToYPercent(dB: number, range = EQ_DB_RANGE): number {
  return ((range - clamp(dB, -range, range)) / (2 * range)) * 100;
}

// ─── EQ response ─────────────────────────────────────────────────────────────

/**
 * Gaussian bell-filter approximation (log frequency domain).
 * Matches the shape of a 2nd-order peaking IIR filter for display purposes.
 * Q controls bandwidth: higher Q = narrower bell.
 */
function peakingFilterGain(
  freq: number,
  centerFreq: number,
  gainDB: number,
  Q = 1.4
): number {
  if (Math.abs(gainDB) < 0.01 || centerFreq <= 0) return 0;
  const octaves = Math.log2(freq / centerFreq);
  const bwOctaves = 1 / Q;
  return gainDB * Math.exp(-0.5 * ((octaves / (bwOctaves / 2)) ** 2));
}

/**
 * Compute the combined EQ transfer function across 256 log-spaced frequencies.
 * Returns an SVG path string using the viewBox "0 0 100 100" coordinate space.
 */
function generateEQCurvePath(bands: FrequencyBandAdjustment[]): string {
  const POINTS = 256;
  const logMin = Math.log10(MIN_DISPLAY_FREQ);
  const logMax = Math.log10(MAX_DISPLAY_FREQ);
  const parts: string[] = [];

  for (let i = 0; i < POINTS; i += 1) {
    const logFreq = logMin + (i / (POINTS - 1)) * (logMax - logMin);
    const freq = Math.pow(10, logFreq);

    let totalGain = 0;
    for (const band of bands) {
      if (Math.abs(band.gain) >= 0.1) {
        totalGain += peakingFilterGain(freq, band.freq, band.gain);
      }
    }
    totalGain = clamp(totalGain, -EQ_DB_RANGE, EQ_DB_RANGE);

    const x = (i / (POINTS - 1)) * 100;
    const y = dbToYPercent(totalGain);
    parts.push(`${i === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`);
  }

  return parts.join(' ');
}

// ─── Status ───────────────────────────────────────────────────────────────────

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

// ─── Canvas utilities ─────────────────────────────────────────────────────────

function syncCanvasToDisplaySize(canvas: HTMLCanvasElement) {
  const context = canvas.getContext('2d');

  if (!context) return;

  const { width, height } = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const pixelWidth = Math.max(1, Math.floor(width * dpr));
  const pixelHeight = Math.max(1, Math.floor(height * dpr));

  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }

  context.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// ─── Oscilloscope ─────────────────────────────────────────────────────────────

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

  // 0 dB centre line
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

  for (let col = 0; col <= 8; col += 1) {
    const x = (col / 8) * width;
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

function buildScopeValues(timeDomainData: Uint8Array, pointCount: number) {
  const values = new Array<number>(pointCount);

  for (let index = 0; index < pointCount; index += 1) {
    const ratio = index / (pointCount - 1);
    const samplePosition = ratio * (timeDomainData.length - 1);
    const baseIndex = Math.floor(samplePosition);
    const nextIndex = Math.min(baseIndex + 1, timeDomainData.length - 1);
    const fraction = samplePosition - baseIndex;
    const value =
      timeDomainData[baseIndex] +
      (timeDomainData[nextIndex] - timeDomainData[baseIndex]) * fraction;

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

// ─── Parametric EQ panel ──────────────────────────────────────────────────────

/**
 * Draws the X32-style EQ background onto the canvas:
 *  • Log-frequency + dB grid
 *  • Real-time RTA spectrum (when live)
 * The EQ response curve itself is drawn in SVG on top so it can be animated.
 */
function drawAnalyzerPanel(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  smoothedSpectrum: Float32Array | null,
  sampleRate: number
) {
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#050709';
  context.fillRect(0, 0, width, height);

  drawEQGrid(context, width, height);

  if (smoothedSpectrum && smoothedSpectrum.length > 0) {
    drawSpectrumRTA(context, width, height, smoothedSpectrum, sampleRate);
  }
}

/**
 * X32-style grid:
 *  • Horizontal lines at ±12, ±6, 0 dB
 *  • Vertical lines at decade/octave frequency markers (log-spaced)
 */
function drawEQGrid(context: CanvasRenderingContext2D, width: number, height: number) {
  const logMin = Math.log10(MIN_DISPLAY_FREQ);
  const logMax = Math.log10(MAX_DISPLAY_FREQ);

  // --- Horizontal dB lines ---
  const dBLines = [-12, -6, 0, 6, 12];
  for (const dB of dBLines) {
    // map dB to Y: 0dB → height/2
    const y = height / 2 - (dB / EQ_DB_RANGE) * (height / 2);
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.strokeStyle =
      dB === 0 ? 'rgba(71, 85, 105, 0.55)' : 'rgba(51, 65, 85, 0.22)';
    context.lineWidth = dB === 0 ? 1.5 : 1;
    context.stroke();
  }

  // --- Vertical frequency lines (log scale) ---
  const freqLines = [
    { freq: 50,    major: false },
    { freq: 100,   major: true  },
    { freq: 200,   major: false },
    { freq: 500,   major: false },
    { freq: 1000,  major: true  },
    { freq: 2000,  major: false },
    { freq: 5000,  major: false },
    { freq: 10000, major: true  },
  ];

  for (const { freq, major } of freqLines) {
    const x = ((Math.log10(freq) - logMin) / (logMax - logMin)) * width;
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.strokeStyle = major ? 'rgba(51, 65, 85, 0.35)' : 'rgba(51, 65, 85, 0.16)';
    context.lineWidth = 1;
    context.stroke();
  }
}

/**
 * Draws the real-time spectrum as a filled waveform behind the EQ curve.
 * Uses logarithmic frequency mapping so the display matches the grid.
 */
function drawSpectrumRTA(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  spectrum: Float32Array,
  sampleRate: number
) {
  const nyquist = sampleRate / 2;
  const logMin = Math.log10(MIN_DISPLAY_FREQ);
  const logMax = Math.log10(Math.min(nyquist, MAX_DISPLAY_FREQ));

  const pts: Array<[number, number]> = [];

  for (let px = 0; px < width; px += 1) {
    const logFreq = logMin + (px / (width - 1)) * (logMax - logMin);
    const freq = Math.pow(10, logFreq);
    const binIndex = (freq / nyquist) * (spectrum.length - 1);
    const binFloor = Math.floor(binIndex);
    const binCeil = Math.min(spectrum.length - 1, binFloor + 1);
    const fraction = binIndex - binFloor;
    const val = spectrum[binFloor] + (spectrum[binCeil] - spectrum[binFloor]) * fraction;
    // Map amplitude 0-1 to the lower 80 % of the canvas height (so it
    // doesn't cover the boost region of the EQ display)
    const y = height - val * height * 0.78;
    pts.push([px, y]);
  }

  const gradient = context.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, 'rgba(6, 182, 212, 0.28)');
  gradient.addColorStop(0.6, 'rgba(6, 182, 212, 0.10)');
  gradient.addColorStop(1, 'rgba(6, 182, 212, 0.01)');

  // Filled area
  context.beginPath();
  context.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i += 1) context.lineTo(pts[i][0], pts[i][1]);
  context.lineTo(width, height);
  context.lineTo(0, height);
  context.closePath();
  context.fillStyle = gradient;
  context.fill();

  // Stroke line
  context.beginPath();
  context.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i += 1) context.lineTo(pts[i][0], pts[i][1]);
  context.strokeStyle = 'rgba(6, 182, 212, 0.75)';
  context.lineWidth = 1.2;
  context.lineJoin = 'round';
  context.stroke();
}

// ─── Output level meter ───────────────────────────────────────────────────────

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
  if (!isLit) return 'rgba(30, 41, 59, 0.38)';

  const ratio = segmentIndex / (segmentCount - 1);

  if (ratio > 0.82) return 'rgba(239, 68, 68, 0.92)';
  if (ratio > 0.62) return 'rgba(250, 204, 21, 0.92)';
  return 'rgba(74, 222, 128, 0.9)';
}

// ─── Level helpers ────────────────────────────────────────────────────────────

function getOutputLevel(timeDomainData: Uint8Array | null): number {
  if (!timeDomainData) return 0.04;

  let sum = 0;

  for (let index = 0; index < timeDomainData.length; index += 1) {
    const v = timeDomainData[index] / 128 - 1;
    sum += v * v;
  }

  return clamp(Math.sqrt(sum / timeDomainData.length) * 2.8, 0.04, 1);
}

// ─── Spectrum smoothing ───────────────────────────────────────────────────────

/**
 * Exponential moving average with asymmetric time constants:
 *  fast attack  → level rises quickly
 *  slow release → level decays slowly (classic ballistics)
 */
function updateSmoothedSpectrum(smoothed: Float32Array, current: Uint8Array) {
  for (let i = 0; i < smoothed.length; i += 1) {
    const newVal = current[i] / 255;
    const prevVal = smoothed[i];
    const coeff = newVal > prevVal ? 0.3 : 0.85; // fast attack, slow release
    smoothed[i] = coeff * prevVal + (1 - coeff) * newVal;
  }
}

// ─── EQ band generation ───────────────────────────────────────────────────────

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
  const rangePattern =
    /(\d+(?:\.\d+)?)(k?Hz)?\s*(?:–|-)\s*(\d+(?:\.\d+)?)(k?Hz)?\s*([+-]?\d+(?:\.\d+)?)\s*dB/i;
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

  if (!singleMatch) return null;

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
      return { gain: adjustment.gain * (1 - distance / span) };
    }
  }

  return null;
}

// ─── Shared utility ───────────────────────────────────────────────────────────

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
