from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = "2.0.0"

ISSUE_LABELS = (
    "muddy",
    "harsh",
    "buried",
    "boomy",
    "thin",
    "boxy",
    "nasal",
    "sibilant",
    "dull",
)

PRIMARY_ISSUE_LABELS = (
    "muddy",
    "harsh",
    "buried",
)

SOURCE_LABELS = (
    "vocal",
    "guitar",
    "bass",
    "drums",
    "keys",
)

DERIVED_DIAGNOSIS_LABELS = (
    "vocal_buried",
    "guitar_harsh",
    "bass_muddy",
    "drums_overpower",
    "keys_masking",
)

CAUSE_METADATA_LABELS = (
    "low_frequency_buildup",
    "low_mid_overlap",
    "boxy_resonance",
    "room_resonance",
    "overlapping_sources",
    "high_frequency_spike",
    "sibilance",
    "cymbal_dominance",
    "guitar_presence_peak",
    "mid_range_masking",
    "lack_of_presence",
    "competing_sources",
    "level_too_low",
    "level_imbalance",
    "tonal_imbalance",
    "bass_overpower",
    "drums_dominance",
    "missing_low_end",
    "missing_high_end",
    "boomy_resonance",
    "nasal_peak",
    "transient_overload",
    "frequency_gap",
)

LABEL_QUALITY_VALUES = ("weak", "reviewed", "derived", "unavailable")

ISSUE_TO_CAUSES: dict[str, tuple[str, ...]] = {
    "muddy": ("low_frequency_buildup", "low_mid_overlap", "boxy_resonance", "overlapping_sources"),
    "harsh": ("high_frequency_spike", "sibilance", "guitar_presence_peak"),
    "buried": ("mid_range_masking", "lack_of_presence", "competing_sources", "level_too_low"),
    "boomy": ("low_frequency_buildup", "boomy_resonance", "room_resonance"),
    "thin": ("missing_low_end", "frequency_gap"),
    "boxy": ("boxy_resonance", "low_mid_overlap"),
    "nasal": ("nasal_peak", "mid_range_masking"),
    "sibilant": ("sibilance",),
    "dull": ("missing_high_end", "frequency_gap"),
}

ISSUE_TO_SOURCE_AFFINITY: dict[str, tuple[str, ...]] = {
    "muddy": ("bass", "guitar", "keys"),
    "harsh": ("vocal", "guitar", "drums"),
    "buried": ("vocal", "guitar", "keys"),
    "boomy": ("bass", "drums"),
    "thin": ("bass", "guitar"),
    "boxy": ("vocal", "guitar", "keys"),
    "nasal": ("vocal", "guitar"),
    "sibilant": ("vocal",),
    "dull": ("guitar", "keys", "drums"),
}

DEFAULT_ISSUE_THRESHOLDS: dict[str, float] = {
    "muddy": 0.52,
    "harsh": 0.50,
    "buried": 0.50,
    "boomy": 0.52,
    "thin": 0.50,
    "boxy": 0.50,
    "nasal": 0.50,
    "sibilant": 0.48,
    "dull": 0.50,
}

DEFAULT_SOURCE_THRESHOLDS: dict[str, float] = {
    "vocal": 0.48,
    "guitar": 0.50,
    "bass": 0.48,
    "drums": 0.48,
    "keys": 0.50,
}

DEFAULT_DERIVED_THRESHOLDS: dict[str, float] = {
    "vocal_buried": 0.55,
    "guitar_harsh": 0.55,
    "bass_muddy": 0.55,
    "drums_overpower": 0.58,
    "keys_masking": 0.56,
}

ISSUE_FALLBACK_EQ: dict[str, dict[str, float | str]] = {
    "muddy": {"frequency_hz": 315.0, "gain_db": -3.0, "reason": "reduce low-mid buildup"},
    "harsh": {"frequency_hz": 5600.0, "gain_db": -2.5, "reason": "tame upper-mid harshness"},
    "buried": {"frequency_hz": 3000.0, "gain_db": 2.5, "reason": "restore presence and intelligibility"},
    "boomy": {"frequency_hz": 120.0, "gain_db": -3.0, "reason": "tighten low-end resonance"},
    "thin": {"frequency_hz": 110.0, "gain_db": 2.5, "reason": "restore low-end weight"},
    "boxy": {"frequency_hz": 650.0, "gain_db": -2.5, "reason": "reduce boxy midrange resonance"},
    "nasal": {"frequency_hz": 1100.0, "gain_db": -2.0, "reason": "soften nasal midrange focus"},
    "sibilant": {"frequency_hz": 6500.0, "gain_db": -2.5, "reason": "control sibilant edge"},
    "dull": {"frequency_hz": 6500.0, "gain_db": 2.0, "reason": "restore brightness and air"},
}

SOURCE_HINT_TERMS: dict[str, tuple[str, ...]] = {
    "vocal": ("vocal", "vocals", "voice", "speech", "singer", "singing", "choir", "rap", "rapping"),
    "guitar": ("guitar", "acoustic_guitar", "electric_guitar", "strum", "mandolin", "ukulele", "banjo"),
    "bass": ("bass", "bass_guitar", "double_bass", "contrabass"),
    "drums": ("drum", "drums", "percussion", "cymbal", "snare", "kick", "hihat", "toms", "tabla"),
    "keys": ("keys", "keyboard", "piano", "organ", "synth", "synthesizer", "electric_piano", "harpsichord"),
}


@dataclass(frozen=True)
class ThresholdBundle:
    issue_thresholds: dict[str, float]
    source_thresholds: dict[str, float]
    derived_thresholds: dict[str, float]

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "issue_thresholds": dict(self.issue_thresholds),
            "source_thresholds": dict(self.source_thresholds),
            "derived_thresholds": dict(self.derived_thresholds),
        }


@dataclass(frozen=True)
class LabelSchema:
    schema_version: str
    issue_labels: tuple[str, ...]
    primary_issue_labels: tuple[str, ...]
    source_labels: tuple[str, ...]
    derived_diagnosis_labels: tuple[str, ...]
    cause_metadata_labels: tuple[str, ...]
    thresholds: ThresholdBundle
    issue_to_causes: dict[str, tuple[str, ...]]
    issue_to_source_affinity: dict[str, tuple[str, ...]]
    issue_fallback_eq: dict[str, dict[str, float | str]]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["thresholds"] = self.thresholds.to_dict()
        return payload


def get_threshold_bundle(
    issue_thresholds: dict[str, float] | None = None,
    source_thresholds: dict[str, float] | None = None,
    derived_thresholds: dict[str, float] | None = None,
) -> ThresholdBundle:
    return ThresholdBundle(
        issue_thresholds=dict(DEFAULT_ISSUE_THRESHOLDS if issue_thresholds is None else issue_thresholds),
        source_thresholds=dict(DEFAULT_SOURCE_THRESHOLDS if source_thresholds is None else source_thresholds),
        derived_thresholds=dict(DEFAULT_DERIVED_THRESHOLDS if derived_thresholds is None else derived_thresholds),
    )


def get_label_schema(
    issue_thresholds: dict[str, float] | None = None,
    source_thresholds: dict[str, float] | None = None,
    derived_thresholds: dict[str, float] | None = None,
) -> LabelSchema:
    return LabelSchema(
        schema_version=SCHEMA_VERSION,
        issue_labels=ISSUE_LABELS,
        primary_issue_labels=PRIMARY_ISSUE_LABELS,
        source_labels=SOURCE_LABELS,
        derived_diagnosis_labels=DERIVED_DIAGNOSIS_LABELS,
        cause_metadata_labels=CAUSE_METADATA_LABELS,
        thresholds=get_threshold_bundle(issue_thresholds, source_thresholds, derived_thresholds),
        issue_to_causes=ISSUE_TO_CAUSES,
        issue_to_source_affinity=ISSUE_TO_SOURCE_AFFINITY,
        issue_fallback_eq=ISSUE_FALLBACK_EQ,
    )


def build_label_quality_map() -> dict[str, str]:
    label_quality = {label: "weak" for label in ISSUE_LABELS}
    label_quality.update({label: "weak" for label in SOURCE_LABELS})
    label_quality.update({label: "derived" for label in DERIVED_DIAGNOSIS_LABELS})
    return label_quality
