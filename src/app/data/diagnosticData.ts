import type { DiagnosticProblem, ProblemType, DetailedCause, Instrument } from '../types';

// ===================================
// PROBLEM → DETAILED CAUSES MAPPING
// ===================================

export const problemCauseMap: Record<ProblemType, DetailedCause[]> = {
  // Primary
  muddy: [
    'low_frequency_buildup',
    'low_mid_overlap',
    'boxy_resonance',
    'room_resonance',
    'overlapping_sources'
  ],
  harsh: [
    'high_frequency_spike',
    'sibilance',
    'cymbal_dominance',
    'guitar_presence_peak'
  ],
  buried: [
    'mid_range_masking',
    'lack_of_presence',
    'competing_sources',
    'level_too_low'
  ],
  imbalance: [
    'level_imbalance',
    'tonal_imbalance',
    'bass_overpower',
    'drums_dominance',
    'missing_low_end',
    'missing_high_end'
  ],
  // Secondary
  boomy: ['low_frequency_buildup', 'boomy_resonance', 'room_resonance'],
  thin: ['missing_low_end', 'frequency_gap'],
  boxy: ['boxy_resonance', 'low_mid_overlap'],
  nasal: ['nasal_peak', 'mid_range_masking'],
  sibilant: ['sibilance'],
  dull: ['missing_high_end', 'frequency_gap'],
  // Source-specific
  vocal_buried: ['mid_range_masking', 'lack_of_presence', 'competing_sources'],
  guitar_harsh: ['high_frequency_spike', 'guitar_presence_peak'],
  bass_muddy: ['low_frequency_buildup', 'low_mid_overlap'],
  drums_overpower: ['level_imbalance', 'cymbal_dominance', 'transient_overload'],
  keys_masking: ['mid_range_masking', 'competing_sources', 'frequency_gap']
};

// ===================================
// DETAILED CAUSE → ACTIONS MAPPING
// ===================================

export const causeActionMap: Record<DetailedCause, string[]> = {
  // Muddy causes
  low_frequency_buildup: [
    '100–250Hz -3dB',
    'High-pass filter at 80Hz'
  ],
  low_mid_overlap: [
    '250–500Hz -3dB',
    'reduce_competing_sources'
  ],
  boxy_resonance: [
    '300–800Hz -3dB',
    'Narrow cut at resonant frequency'
  ],
  room_resonance: [
    'Identify resonant frequency',
    'Apply narrow cut -4dB'
  ],
  overlapping_sources: [
    'reduce_competing_sources',
    'adjust_frequency_overlap',
    'Pan sources apart'
  ],
  
  // Harsh causes
  high_frequency_spike: [
    '3–8kHz -3dB',
    'Identify peak frequency',
    'Apply narrow cut'
  ],
  sibilance: [
    '5–10kHz -4dB',
    'control_sibilance',
    'De-esser on vocals'
  ],
  cymbal_dominance: [
    '8–12kHz -2dB',
    'Reduce cymbal volume',
    'isolate_problem_source'
  ],
  guitar_presence_peak: [
    '2–4kHz -3dB',
    'boost_presence (other instruments)'
  ],
  
  // Buried causes
  mid_range_masking: [
    '3kHz +2dB (target)',
    'reduce_competing_sources (2–3kHz)',
    'adjust_frequency_overlap'
  ],
  lack_of_presence: [
    '2–5kHz +3dB',
    'boost_presence',
    'Enhance clarity range'
  ],
  competing_sources: [
    'reduce_competing_sources',
    'Pan separation',
    'EQ carving'
  ],
  level_too_low: [
    'increase_target_level (+3dB)',
    'Check gain staging',
    'Reduce competing levels'
  ],
  
  // Imbalance causes
  level_imbalance: [
    'rebalance_levels',
    'Check gain staging',
    'Adjust fader positions'
  ],
  tonal_imbalance: [
    'Review EQ across mix',
    'Balance frequency spectrum',
    'Match reference track'
  ],
  bass_overpower: [
    '80–120Hz -4dB',
    'Reduce bass level',
    'rebalance_levels'
  ],
  drums_dominance: [
    'Reduce drum bus level',
    'Control transients',
    'rebalance_levels'
  ],
  missing_low_end: [
    '80–150Hz +3dB',
    'boost_low_end',
    'Add low-end warmth'
  ],
  missing_high_end: [
    '8–12kHz +2dB',
    'boost_high',
    'Add air and sparkle'
  ],
  
  // Secondary causes
  boomy_resonance: [
    '80–180Hz -3dB',
    'cut_boomy',
    'Narrow cut at boom frequency'
  ],
  nasal_peak: [
    '800Hz–1.5kHz -3dB',
    'reduce_nasal',
    'Smooth mid-range'
  ],
  transient_overload: [
    'Apply compression',
    'Control peak levels',
    'Reduce attack transients'
  ],
  frequency_gap: [
    'Identify missing range',
    'Fill frequency gap',
    'Review arrangement'
  ]
};

// ===================================
// PROBLEM → LIKELY SOURCES MAPPING
// ===================================

export const problemSourceMap: Record<ProblemType, Instrument[]> = {
  // Primary
  muddy: ['bass', 'guitar', 'keys', 'overall'],
  harsh: ['drums', 'guitar', 'vocal'],
  buried: ['vocal', 'guitar', 'keys'],
  imbalance: ['overall', 'drums', 'bass'],
  // Secondary
  boomy: ['bass', 'drums', 'overall'],
  thin: ['bass', 'guitar', 'overall'],
  boxy: ['vocal', 'guitar', 'keys'],
  nasal: ['vocal', 'guitar'],
  sibilant: ['vocal'],
  dull: ['guitar', 'keys', 'drums', 'overall'],
  // Source-specific
  vocal_buried: ['vocal', 'guitar', 'keys'],
  guitar_harsh: ['guitar'],
  bass_muddy: ['bass'],
  drums_overpower: ['drums'],
  keys_masking: ['keys', 'guitar']
};

// ===================================
// MOCK DATA GENERATOR
// ===================================

export function generateMockProblems(count: number = 1): DiagnosticProblem[] {
  const primaryProblems: ProblemType[] = ['muddy', 'harsh', 'buried', 'imbalance'];
  const secondaryProblems: ProblemType[] = ['boomy', 'thin', 'boxy', 'nasal', 'sibilant', 'dull'];
  const sourceSpecificProblems: ProblemType[] = [
    'vocal_buried',
    'guitar_harsh',
    'bass_muddy',
    'drums_overpower',
    'keys_masking'
  ];
  
  const problems: DiagnosticProblem[] = [];
  
  for (let i = 0; i < count; i++) {
    // Weight towards primary problems
    let problemType: ProblemType;
    const rand = Math.random();
    
    if (rand < 0.6) {
      // 60% primary
      problemType = primaryProblems[Math.floor(Math.random() * primaryProblems.length)];
    } else if (rand < 0.85) {
      // 25% source-specific
      problemType = sourceSpecificProblems[Math.floor(Math.random() * sourceSpecificProblems.length)];
    } else {
      // 15% secondary
      problemType = secondaryProblems[Math.floor(Math.random() * secondaryProblems.length)];
    }
    
    // Get possible causes for this problem
    const possibleCauses = problemCauseMap[problemType];
    
    // Select 1-2 detailed causes
    const numCauses = Math.random() < 0.7 ? 1 : 2;
    const selectedCauses = possibleCauses
      .sort(() => Math.random() - 0.5)
      .slice(0, Math.min(numCauses, possibleCauses.length));
    
    // Get sources
    const possibleSources = problemSourceMap[problemType];
    const numSources = Math.random() < 0.6 ? 1 : Math.random() < 0.9 ? 2 : 3;
    const selectedSources = possibleSources
      .sort(() => Math.random() - 0.5)
      .slice(0, Math.min(numSources, possibleSources.length));
    
    // Gather all actions from selected causes
    const allActions: string[] = [];
    selectedCauses.forEach(cause => {
      const actions = causeActionMap[cause];
      allActions.push(...actions);
    });
    
    // Remove duplicates and limit to 2-4 actions
    const uniqueActions = Array.from(new Set(allActions));
    const finalActions = uniqueActions.slice(0, Math.min(4, uniqueActions.length));
    
    // Calculate confidence (higher for primary, lower for secondary)
    const baseConfidence = primaryProblems.includes(problemType) 
      ? 0.65 
      : secondaryProblems.includes(problemType) 
      ? 0.45 
      : 0.55;
    
    const confidence = Math.min(baseConfidence + Math.random() * 0.35, 1);
    
    problems.push({
      type: problemType,
      confidence: Math.round(confidence * 100) / 100, // Round to 2 decimals
      sources: selectedSources,
      details: selectedCauses,
      actions: finalActions
    });
  }
  
  // Sort by confidence (highest first)
  return problems.sort((a, b) => b.confidence - a.confidence);
}
