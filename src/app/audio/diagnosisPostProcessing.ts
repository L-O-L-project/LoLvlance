import type {
  DerivedDiagnosisResult,
  DerivedDiagnosisType,
  MlInferenceOutput,
  SourceLabel,
  TrainableIssueLabel
} from '../types';
import {
  DERIVED_DEFAULT_THRESHOLDS,
  DERIVED_DIAGNOSIS_LABELS,
  ISSUE_DEFAULT_THRESHOLDS,
  ISSUE_LABELS,
  ML_SCHEMA_VERSION,
  SOURCE_DEFAULT_THRESHOLDS,
  SOURCE_LABELS
} from './mlSchema';

interface DerivedDiagnosisCandidate {
  label: DerivedDiagnosisType;
  score: number;
  reasons: string[];
  explanation: string;
}

export function buildMlInferenceOutput(
  issueScores: Record<TrainableIssueLabel, number>,
  sourceScores: Record<SourceLabel, number>
): MlInferenceOutput {
  const derived_diagnoses = deriveSourceSpecificDiagnoses(issueScores, sourceScores);

  return {
    schema_version: ML_SCHEMA_VERSION,
    issues: issueScores,
    sources: sourceScores,
    derived_diagnoses,
    metadata: {
      thresholds_used: {
        ...Object.fromEntries(ISSUE_LABELS.map((label) => [`issue:${label}`, ISSUE_DEFAULT_THRESHOLDS[label]])),
        ...Object.fromEntries(SOURCE_LABELS.map((label) => [`source:${label}`, SOURCE_DEFAULT_THRESHOLDS[label]])),
        ...Object.fromEntries(
          DERIVED_DIAGNOSIS_LABELS.map((label) => [`derived:${label}`, DERIVED_DEFAULT_THRESHOLDS[label]])
        )
      },
      label_quality: {
        ...Object.fromEntries(ISSUE_LABELS.map((label) => [label, 'weak'])),
        ...Object.fromEntries(SOURCE_LABELS.map((label) => [label, 'weak'])),
        ...Object.fromEntries(DERIVED_DIAGNOSIS_LABELS.map((label) => [label, 'derived']))
      }
    }
  };
}

export function deriveSourceSpecificDiagnoses(
  issueScores: Record<TrainableIssueLabel, number>,
  sourceScores: Record<SourceLabel, number>
): MlInferenceOutput['derived_diagnoses'] {
  const candidates: DerivedDiagnosisCandidate[] = [
    {
      label: 'vocal_buried',
      score: combineSignals(issueScores.buried, sourceScores.vocal, Math.max(issueScores.dull, issueScores.boxy) * 0.1),
      reasons: collectReasons(
        ['buried_high', issueScores.buried, ISSUE_DEFAULT_THRESHOLDS.buried],
        ['vocal_present', sourceScores.vocal, SOURCE_DEFAULT_THRESHOLDS.vocal],
        ['relative_presence_low', Math.max(issueScores.dull, issueScores.boxy), 0.45]
      ),
      explanation: 'Buried vocal likelihood comes from buried issue evidence plus strong vocal presence.'
    },
    {
      label: 'guitar_harsh',
      score: combineSignals(issueScores.harsh, sourceScores.guitar, issueScores.sibilant * 0.05),
      reasons: collectReasons(
        ['harsh_high', issueScores.harsh, ISSUE_DEFAULT_THRESHOLDS.harsh],
        ['guitar_present', sourceScores.guitar, SOURCE_DEFAULT_THRESHOLDS.guitar],
        ['presence_peak_active', issueScores.sibilant, 0.42]
      ),
      explanation: 'Guitar harshness is derived from harsh issue evidence combined with guitar presence.'
    },
    {
      label: 'bass_muddy',
      score: combineSignals(issueScores.muddy, sourceScores.bass, issueScores.boomy * 0.12),
      reasons: collectReasons(
        ['muddy_high', issueScores.muddy, ISSUE_DEFAULT_THRESHOLDS.muddy],
        ['bass_present', sourceScores.bass, SOURCE_DEFAULT_THRESHOLDS.bass],
        ['low_end_bloom', issueScores.boomy, ISSUE_DEFAULT_THRESHOLDS.boomy]
      ),
      explanation: 'Bass muddiness is derived from muddy issue evidence reinforced by bass presence and low-end bloom.'
    },
    {
      label: 'drums_overpower',
      score: combineSignals(Math.max(issueScores.harsh, issueScores.boomy), sourceScores.drums, issueScores.thin * 0.08),
      reasons: collectReasons(
        ['drums_dominant', sourceScores.drums, SOURCE_DEFAULT_THRESHOLDS.drums],
        ['harsh_or_boomy_high', Math.max(issueScores.harsh, issueScores.boomy), 0.5],
        ['mix_feels_thin', issueScores.thin, ISSUE_DEFAULT_THRESHOLDS.thin]
      ),
      explanation: 'Drum overpower is derived from strong drum evidence alongside harsh or boomy mix traits.'
    },
    {
      label: 'keys_masking',
      score: combineSignals(Math.max(issueScores.buried, issueScores.boxy), sourceScores.keys, issueScores.nasal * 0.05),
      reasons: collectReasons(
        ['keys_present', sourceScores.keys, SOURCE_DEFAULT_THRESHOLDS.keys],
        ['buried_or_boxy_high', Math.max(issueScores.buried, issueScores.boxy), 0.5],
        ['midrange_masking_pattern', issueScores.nasal, 0.45]
      ),
      explanation: 'Keys masking is derived from buried or boxy mix evidence with strong keyboard presence.'
    }
  ];

  return Object.fromEntries(
    candidates
      .filter((candidate) => candidate.score >= DERIVED_DEFAULT_THRESHOLDS[candidate.label] && candidate.reasons.length >= 2)
      .map((candidate) => [
        candidate.label,
        {
          score: Number(candidate.score.toFixed(4)),
          reasons: candidate.reasons,
          explanation: candidate.explanation
        } satisfies DerivedDiagnosisResult
      ])
  ) as MlInferenceOutput['derived_diagnoses'];
}

function combineSignals(issueScore: number, sourceScore: number, bonus = 0) {
  return clamp(issueScore * 0.62 + sourceScore * 0.38 + bonus, 0, 1);
}

function collectReasons(...rules: [string, number, number][]) {
  return rules.filter(([, score, threshold]) => score >= threshold).map(([reason]) => reason);
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
