from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

try:
    from .label_schema import (
        DEFAULT_DERIVED_THRESHOLDS,
        DEFAULT_ISSUE_THRESHOLDS,
        DEFAULT_SOURCE_THRESHOLDS,
        DERIVED_DIAGNOSIS_LABELS,
        ISSUE_LABELS,
        SCHEMA_VERSION,
        SOURCE_LABELS,
        build_label_quality_map,
    )
except ImportError:
    from label_schema import (
        DEFAULT_DERIVED_THRESHOLDS,
        DEFAULT_ISSUE_THRESHOLDS,
        DEFAULT_SOURCE_THRESHOLDS,
        DERIVED_DIAGNOSIS_LABELS,
        ISSUE_LABELS,
        SCHEMA_VERSION,
        SOURCE_LABELS,
        build_label_quality_map,
    )


@dataclass(frozen=True)
class DerivedDiagnosis:
    score: float
    reasons: list[str]
    explanation: str


def build_hierarchical_output(
    issue_scores: Mapping[str, float],
    source_scores: Mapping[str, float],
    *,
    issue_thresholds: Mapping[str, float] | None = None,
    source_thresholds: Mapping[str, float] | None = None,
    derived_thresholds: Mapping[str, float] | None = None,
    external_source_scores: Mapping[str, float] | None = None,
) -> dict[str, object]:
    active_issue_thresholds = dict(DEFAULT_ISSUE_THRESHOLDS if issue_thresholds is None else issue_thresholds)
    active_source_thresholds = dict(DEFAULT_SOURCE_THRESHOLDS if source_thresholds is None else source_thresholds)
    active_derived_thresholds = dict(DEFAULT_DERIVED_THRESHOLDS if derived_thresholds is None else derived_thresholds)
    normalized_issue_scores = {label: float(issue_scores.get(label, 0.0)) for label in ISSUE_LABELS}
    normalized_source_scores = {
        label: max(float(source_scores.get(label, 0.0)), float((external_source_scores or {}).get(label, 0.0)))
        for label in SOURCE_LABELS
    }
    derived = derive_source_specific_diagnoses(
        issue_scores=normalized_issue_scores,
        source_scores=normalized_source_scores,
        issue_thresholds=active_issue_thresholds,
        source_thresholds=active_source_thresholds,
        derived_thresholds=active_derived_thresholds,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "issues": normalized_issue_scores,
        "sources": normalized_source_scores,
        "derived_diagnoses": {
            label: {
                "score": round(diagnosis.score, 4),
                "reasons": diagnosis.reasons,
                "explanation": diagnosis.explanation,
            }
            for label, diagnosis in derived.items()
        },
        "metadata": {
            "thresholds_used": {
                **{f"issue:{label}": threshold for label, threshold in active_issue_thresholds.items()},
                **{f"source:{label}": threshold for label, threshold in active_source_thresholds.items()},
                **{f"derived:{label}": threshold for label, threshold in active_derived_thresholds.items()},
            },
            "label_quality": build_label_quality_map(),
        },
    }


def derive_source_specific_diagnoses(
    *,
    issue_scores: Mapping[str, float],
    source_scores: Mapping[str, float],
    issue_thresholds: Mapping[str, float],
    source_thresholds: Mapping[str, float],
    derived_thresholds: Mapping[str, float],
) -> dict[str, DerivedDiagnosis]:
    derived: dict[str, DerivedDiagnosis] = {}

    derived.update(
        maybe_add_diagnosis(
            label="vocal_buried",
            score=combine_signals(issue_scores["buried"], source_scores["vocal"], bonus=max(issue_scores["dull"], issue_scores["boxy"]) * 0.1),
            reasons=collect_reasons(
                ("buried_high", issue_scores["buried"], issue_thresholds["buried"]),
                ("vocal_present", source_scores["vocal"], source_thresholds["vocal"]),
                ("relative_presence_low", max(issue_scores["dull"], issue_scores["boxy"]), 0.45),
            ),
            explanation="Buried vocal likelihood comes from buried issue evidence plus strong vocal presence.",
            threshold=derived_thresholds["vocal_buried"],
        )
    )
    derived.update(
        maybe_add_diagnosis(
            label="guitar_harsh",
            score=combine_signals(issue_scores["harsh"], source_scores["guitar"], bonus=issue_scores["sibilant"] * 0.05),
            reasons=collect_reasons(
                ("harsh_high", issue_scores["harsh"], issue_thresholds["harsh"]),
                ("guitar_present", source_scores["guitar"], source_thresholds["guitar"]),
                ("presence_peak_active", issue_scores["sibilant"], 0.42),
            ),
            explanation="Guitar harshness is derived from harsh issue evidence combined with guitar presence.",
            threshold=derived_thresholds["guitar_harsh"],
        )
    )
    derived.update(
        maybe_add_diagnosis(
            label="bass_muddy",
            score=combine_signals(issue_scores["muddy"], source_scores["bass"], bonus=issue_scores["boomy"] * 0.12),
            reasons=collect_reasons(
                ("muddy_high", issue_scores["muddy"], issue_thresholds["muddy"]),
                ("bass_present", source_scores["bass"], source_thresholds["bass"]),
                ("low_end_bloom", issue_scores["boomy"], issue_thresholds["boomy"]),
            ),
            explanation="Bass muddiness is derived from muddy issue evidence reinforced by bass presence and low-end bloom.",
            threshold=derived_thresholds["bass_muddy"],
        )
    )
    derived.update(
        maybe_add_diagnosis(
            label="drums_overpower",
            score=combine_signals(max(issue_scores["harsh"], issue_scores["boomy"]), source_scores["drums"], bonus=issue_scores["thin"] * 0.08),
            reasons=collect_reasons(
                ("drums_dominant", source_scores["drums"], source_thresholds["drums"]),
                ("harsh_or_boomy_high", max(issue_scores["harsh"], issue_scores["boomy"]), 0.5),
                ("mix_feels_thin", issue_scores["thin"], issue_thresholds["thin"]),
            ),
            explanation="Drum overpower is derived from strong drum evidence alongside harsh or boomy mix traits.",
            threshold=derived_thresholds["drums_overpower"],
        )
    )
    derived.update(
        maybe_add_diagnosis(
            label="keys_masking",
            score=combine_signals(max(issue_scores["buried"], issue_scores["boxy"]), source_scores["keys"], bonus=issue_scores["nasal"] * 0.05),
            reasons=collect_reasons(
                ("keys_present", source_scores["keys"], source_thresholds["keys"]),
                ("buried_or_boxy_high", max(issue_scores["buried"], issue_scores["boxy"]), 0.5),
                ("midrange_masking_pattern", issue_scores["nasal"], 0.45),
            ),
            explanation="Keys masking is derived from buried or boxy mix evidence with strong keyboard presence.",
            threshold=derived_thresholds["keys_masking"],
        )
    )

    return derived


def maybe_add_diagnosis(
    *,
    label: str,
    score: float,
    reasons: list[str],
    explanation: str,
    threshold: float,
) -> dict[str, DerivedDiagnosis]:
    if label not in DERIVED_DIAGNOSIS_LABELS:
        return {}

    if score < threshold or len(reasons) < 2:
        return {}

    return {
        label: DerivedDiagnosis(
            score=round(score, 4),
            reasons=reasons,
            explanation=explanation,
        )
    }


def combine_signals(issue_score: float, source_score: float, *, bonus: float = 0.0) -> float:
    return min(1.0, issue_score * 0.62 + source_score * 0.38 + bonus)


def collect_reasons(*items: tuple[str, float, float]) -> list[str]:
    return [name for name, score, threshold in items if score >= threshold]
