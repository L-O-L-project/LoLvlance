from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

try:
    from .label_schema import ISSUE_LABELS, SCHEMA_VERSION, SOURCE_HINT_TERMS, SOURCE_LABELS
    from .preprocessing import AudioFeatures, PreprocessingConfig, extract_audio_features_from_path
except ImportError:
    from label_schema import ISSUE_LABELS, SCHEMA_VERSION, SOURCE_HINT_TERMS, SOURCE_LABELS
    from preprocessing import AudioFeatures, PreprocessingConfig, extract_audio_features_from_path

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
ISSUE_LABEL_QUALITY = "weak"
SOURCE_LABEL_QUALITY = "weak"
UNAVAILABLE_LABEL_QUALITY = "unavailable"


@dataclass(frozen=True)
class DatasetRoots:
    openmic: Path | None = None
    slakh: Path | None = None
    musan: Path | None = None
    fsd50k: Path | None = None


@dataclass
class SourceAnnotation:
    values: dict[str, float]
    mask: dict[str, float]
    quality: dict[str, str]
    evidence: dict[str, list[str]]
    support: str

    @classmethod
    def empty(cls) -> "SourceAnnotation":
        return cls(
            values={label: 0.0 for label in SOURCE_LABELS},
            mask={label: 0.0 for label in SOURCE_LABELS},
            quality={label: UNAVAILABLE_LABEL_QUALITY for label in SOURCE_LABELS},
            evidence={label: [] for label in SOURCE_LABELS},
            support="unavailable",
        )

    def merge(self, other: "SourceAnnotation") -> "SourceAnnotation":
        merged = SourceAnnotation.empty()
        merged.support = choose_stronger_support(self.support, other.support)

        for label in SOURCE_LABELS:
            merged.values[label] = max(self.values[label], other.values[label])
            merged.mask[label] = max(self.mask[label], other.mask[label])
            merged.quality[label] = choose_label_quality(self.quality[label], other.quality[label], merged.mask[label])
            merged.evidence[label] = unique_in_order([*self.evidence[label], *other.evidence[label]])

        return merged


class LoLvlanceAudioDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        preprocessing_config: PreprocessingConfig | None = None,
    ) -> None:
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
        self.entries = [entry for entry in load_manifest(manifest_path) if entry["split"] == split]

        if not self.entries:
            raise ValueError(f"No entries found for split '{split}' in {manifest_path}.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self.entries[index]
        features = extract_audio_features_from_path(
            audio_path=entry["audio_path"],
            start_seconds=float(entry["start_seconds"]),
            duration_seconds=float(entry["duration_seconds"]),
            config=self.preprocessing_config,
        )

        return {
            "log_mel_spectrogram": torch.from_numpy(features.log_mel_spectrogram).float(),
            "issue_targets": torch.tensor(entry["issue_targets"]["values"], dtype=torch.float32),
            "issue_target_mask": torch.tensor(entry["issue_targets"]["mask"], dtype=torch.float32),
            "source_targets": torch.tensor(entry["source_targets"]["values"], dtype=torch.float32),
            "source_target_mask": torch.tensor(entry["source_targets"]["mask"], dtype=torch.float32),
        }


def build_public_manifest(
    dataset_roots: DatasetRoots,
    output_path: str | Path,
    preprocessing_config: PreprocessingConfig | None = None,
    clips_per_file: int = 2,
    max_files_per_dataset: int | None = None,
) -> list[dict]:
    config = preprocessing_config or PreprocessingConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    entries.extend(
        scan_dataset_root(
            dataset_roots.openmic,
            dataset_name="openmic",
            config=config,
            clips_per_file=clips_per_file,
            max_files=max_files_per_dataset,
        )
    )
    entries.extend(
        scan_dataset_root(
            dataset_roots.slakh,
            dataset_name="slakh",
            config=config,
            clips_per_file=clips_per_file,
            max_files=max_files_per_dataset,
        )
    )
    entries.extend(
        scan_dataset_root(
            dataset_roots.musan,
            dataset_name="musan",
            config=config,
            clips_per_file=clips_per_file,
            max_files=max_files_per_dataset,
        )
    )
    entries.extend(
        scan_dataset_root(
            dataset_roots.fsd50k,
            dataset_name="fsd50k",
            config=config,
            clips_per_file=clips_per_file,
            max_files=max_files_per_dataset,
        )
    )

    if not entries:
        raise ValueError("No dataset clips were discovered. Provide at least one public dataset root.")

    with output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    return entries


def scan_dataset_root(
    root: Path | None,
    dataset_name: str,
    config: PreprocessingConfig,
    clips_per_file: int,
    max_files: int | None,
) -> list[dict]:
    if root is None or not root.exists():
        return []

    csv_annotations = load_source_annotations_from_csv(root)
    musan_music_annotations: dict[str, tuple[set[str], bool]] | None = None

    if dataset_name == "musan":
        musan_music_annotations = load_musan_music_annotations(root)

    if dataset_name == "slakh":
        audio_files = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in AUDIO_EXTENSIONS
            and path.stem.lower() == "mix"
        ]
    else:
        audio_files = collect_audio_files(root)

    audio_files.sort()
    if max_files is not None:
        audio_files = audio_files[:max_files]

    entries: list[dict] = []

    for audio_path in audio_files:
        clip_id = normalize_clip_id(audio_path)
        csv_annotation = csv_annotations.get(clip_id, SourceAnnotation.empty())
        stem_paths = []

        if dataset_name == "slakh":
            stem_dir = audio_path.parent / "stems"
            stem_paths = collect_audio_files(stem_dir) if stem_dir.exists() else []

        if dataset_name == "musan":
            musan_annotation = infer_musan_source_annotation(audio_path, musan_music_annotations)
            csv_annotation = csv_annotation.merge(musan_annotation)

        entries.extend(
            build_entries_for_file(
                audio_path=audio_path,
                dataset_name=dataset_name,
                config=config,
                clips_per_file=clips_per_file,
                csv_annotation=csv_annotation,
                stem_paths=stem_paths,
            )
        )

    return entries


def build_entries_for_file(
    audio_path: Path,
    dataset_name: str,
    config: PreprocessingConfig,
    clips_per_file: int,
    csv_annotation: SourceAnnotation,
    stem_paths: list[Path] | None = None,
) -> list[dict]:
    import soundfile as sf

    audio_info = sf.info(audio_path)
    duration_seconds = float(audio_info.frames) / float(audio_info.samplerate)
    starts = choose_segment_starts(duration_seconds, config.clip_seconds, clips_per_file)
    track_group_id = infer_track_group_id(audio_path, dataset_name)
    split = infer_split(audio_path, track_group_id)
    filename_annotation = infer_source_annotation_from_path(audio_path, support="filename_partial")
    stem_annotation = infer_source_annotation_from_stems(stem_paths or [])
    combined_annotation = csv_annotation.merge(filename_annotation).merge(stem_annotation)
    vocal_hint = combined_annotation.values["vocal"] > 0.5 or any(
        "vocal" in evidence or "singer" in evidence for evidence in combined_annotation.evidence["vocal"]
    )
    entries: list[dict] = []

    for clip_index, start_seconds in enumerate(starts):
        try:
            features = extract_audio_features_from_path(
                audio_path=audio_path,
                start_seconds=start_seconds,
                duration_seconds=config.clip_seconds,
                config=config,
            )
        except RuntimeError:
            continue

        low_mid_overlap = estimate_low_mid_overlap(
            stem_paths=stem_paths or [],
            start_seconds=start_seconds,
            clip_duration=config.clip_seconds,
            config=config,
        )
        issue_values, issue_reasons = infer_issue_targets(
            dataset_name=dataset_name,
            features=features,
            vocal_hint=vocal_hint,
            low_mid_overlap=low_mid_overlap,
            path=audio_path,
        )

        entries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "clip_id": f"{dataset_name}:{track_group_id}:{clip_index}",
                "track_group_id": track_group_id,
                "dataset": dataset_name,
                "audio_path": audio_path.resolve().as_posix(),
                "start_seconds": round(start_seconds, 3),
                "duration_seconds": config.clip_seconds,
                "split": split,
                "issue_targets": {
                    "labels": list(ISSUE_LABELS),
                    "values": [issue_values[label] for label in ISSUE_LABELS],
                    "mask": [1.0 for _ in ISSUE_LABELS],
                    "quality": [ISSUE_LABEL_QUALITY for _ in ISSUE_LABELS],
                },
                "source_targets": {
                    "labels": list(SOURCE_LABELS),
                    "values": [combined_annotation.values[label] for label in SOURCE_LABELS],
                    "mask": [combined_annotation.mask[label] for label in SOURCE_LABELS],
                    "quality": [combined_annotation.quality[label] for label in SOURCE_LABELS],
                },
                "metadata": {
                    "vocal_hint": vocal_hint,
                    "issue_reasons": issue_reasons,
                    "source_evidence": combined_annotation.evidence,
                    "source_support": combined_annotation.support,
                    "features": {
                        "rms": round(features.rms, 6),
                        "centroid_hz": round(features.spectral_centroid_hz, 2),
                        "rolloff_hz": round(features.spectral_rolloff_hz, 2),
                        "boom_ratio": round(features.boom_ratio, 4),
                        "low_mid_ratio": round(features.low_mid_ratio, 4),
                        "boxy_ratio": round(features.boxy_ratio, 4),
                        "presence_ratio": round(features.presence_ratio, 4),
                        "harsh_ratio": round(features.harsh_ratio, 4),
                        "nasal_ratio": round(features.nasal_ratio, 4),
                        "sibilant_ratio": round(features.sibilant_ratio, 4),
                        "air_ratio": round(features.air_ratio, 4),
                        "low_mid_overlap": low_mid_overlap,
                    },
                },
            }
        )

    return entries


def infer_issue_targets(
    dataset_name: str,
    features: AudioFeatures,
    vocal_hint: bool,
    low_mid_overlap: int,
    path: Path,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    normalized_path = normalize_text(path.as_posix())
    harsh_keyword = any(term in normalized_path for term in ("alarm", "siren", "scream", "crash", "noise"))
    muddy_keyword = any(term in normalized_path for term in ("bass", "organ", "piano", "guitar", "keys"))

    values = {label: 0.0 for label in ISSUE_LABELS}
    reasons = {label: [] for label in ISSUE_LABELS}

    def activate(label: str, *why: str) -> None:
        values[label] = 1.0
        reasons[label] = unique_in_order([*reasons[label], *why])

    if features.low_mid_ratio >= 0.24 or low_mid_overlap >= 2 or (muddy_keyword and features.low_mid_ratio >= 0.2):
        activate("muddy", "low_mid_ratio_high", "harmonic_overlap")

    if features.harsh_ratio >= 0.16 or features.spectral_centroid_hz >= 2_600.0 or harsh_keyword:
        activate("harsh", "upper_band_energy_high", "spectral_centroid_high")

    if vocal_hint and features.presence_ratio <= 0.18 and features.rms >= 0.005:
        activate("buried", "vocal_present", "presence_band_low")

    if features.boom_ratio >= 0.22 and (features.spectral_centroid_hz <= 2_000.0 or low_mid_overlap >= 1):
        activate("boomy", "low_end_ratio_high")

    if features.boom_ratio <= 0.08 and features.low_mid_ratio <= 0.17 and features.rms >= 0.005:
        activate("thin", "low_end_ratio_low")

    if features.boxy_ratio >= 0.24 and features.low_mid_ratio >= 0.2:
        activate("boxy", "boxy_band_high")

    if vocal_hint and features.nasal_ratio >= 0.22 and features.presence_ratio <= 0.28:
        activate("nasal", "nasal_band_high", "vocal_present")

    if vocal_hint and features.sibilant_ratio >= 0.1 and features.harsh_ratio >= 0.12:
        activate("sibilant", "sibilant_band_high", "vocal_present")

    if features.air_ratio <= 0.035 and features.spectral_rolloff_hz <= 2_200.0 and features.rms >= 0.005:
        activate("dull", "high_end_rolloff_low")

    if dataset_name == "slakh" and low_mid_overlap >= 2:
        activate("muddy", "stem_overlap")
        if features.boom_ratio >= 0.18:
            activate("boomy", "stem_overlap_low_end")

    if dataset_name == "openmic" and vocal_hint and features.low_mid_ratio >= 0.22 and features.presence_ratio <= 0.2:
        activate("buried", "vocal_present", "low_presence_contrast")

    return values, reasons


def estimate_low_mid_overlap(
    stem_paths: list[Path],
    start_seconds: float,
    clip_duration: float,
    config: PreprocessingConfig,
) -> int:
    overlap_count = 0

    for stem_path in stem_paths:
        try:
            features = extract_audio_features_from_path(
                audio_path=stem_path,
                start_seconds=start_seconds,
                duration_seconds=clip_duration,
                config=config,
            )
        except RuntimeError:
            continue

        if features.rms >= 0.005 and features.low_mid_ratio >= 0.2:
            overlap_count += 1

    return overlap_count


def summarize_manifest(entries: Iterable[dict]) -> dict[str, object]:
    entries = list(entries)
    split_counts: dict[str, int] = {}
    issue_positives = {label: 0 for label in ISSUE_LABELS}
    source_support = {label: 0.0 for label in SOURCE_LABELS}
    source_positives = {label: 0.0 for label in SOURCE_LABELS}

    for entry in entries:
        split_counts[entry["split"]] = split_counts.get(entry["split"], 0) + 1

        for label, value in zip(ISSUE_LABELS, entry["issue_targets"]["values"]):
            if value >= 0.5:
                issue_positives[label] += 1

        for label, mask, value in zip(
            SOURCE_LABELS,
            entry["source_targets"]["mask"],
            entry["source_targets"]["values"],
        ):
            source_support[label] += float(mask)
            source_positives[label] += float(value * mask)

    return {
        "schema_version": SCHEMA_VERSION,
        "total": len(entries),
        "splits": split_counts,
        "issue_positives": issue_positives,
        "source_support": {label: round(source_support[label], 2) for label in SOURCE_LABELS},
        "source_positives": {label: round(source_positives[label], 2) for label in SOURCE_LABELS},
    }


def load_manifest(manifest_path: str | Path) -> list[dict]:
    manifest = Path(manifest_path)

    with manifest.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# OpenMIC-2018 instrument name → our SOURCE_LABELS mapping
_OPENMIC_INSTRUMENT_TO_SOURCE: dict[str, str] = {
    "voice": "vocal",
    "guitar": "guitar",
    "banjo": "guitar",
    "mandolin": "guitar",
    "ukulele": "guitar",
    "bass": "bass",
    "drums": "drums",
    "cymbals": "drums",
    "piano": "keys",
    "organ": "keys",
    "synthesizer": "keys",
}
# All 20 OpenMIC instruments — used to set mask=1 for known-absent sources
_OPENMIC_ALL_INSTRUMENTS: frozenset[str] = frozenset([
    "accordion", "banjo", "bass", "cello", "clarinet", "cymbals", "drums",
    "flute", "guitar", "mallet_percussion", "mandolin", "organ", "piano",
    "saxophone", "synthesizer", "trombone", "trumpet", "ukulele", "violin", "voice",
])
_OPENMIC_RELEVANCE_THRESHOLD = 0.5


def _is_openmic_longformat_csv(fieldnames: list[str]) -> bool:
    lowered = {f.lower() for f in fieldnames}
    return {"sample_key", "instrument", "relevance"}.issubset(lowered)


def _parse_openmic_longformat_csv(csv_path: Path) -> dict[str, SourceAnnotation]:
    """Parse OpenMIC-2018 aggregated-labels CSV (long format).

    Each row: sample_key, instrument, relevance, num_responses
    We collect all rows per sample_key, then build a SourceAnnotation that:
    - sets mask=1.0 for every SOURCE_LABEL that has at least one observed row
    - sets value=1.0 for labels above the relevance threshold
    """
    # Accumulate per sample: {sample_key: {source_label: max_relevance}}
    sample_source_relevance: dict[str, dict[str, float]] = {}
    # Track which source labels were actually observed (have any row) per sample
    sample_observed: dict[str, set[str]] = {}

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_key = row.get("sample_key", "").strip()
            instrument = row.get("instrument", "").strip().lower().replace("-", "_").replace(" ", "_")
            try:
                relevance = float(row.get("relevance", 0.0))
            except ValueError:
                relevance = 0.0

            if not raw_key or not instrument:
                continue

            clip_id = normalize_clip_id(raw_key)
            source_label = _OPENMIC_INSTRUMENT_TO_SOURCE.get(instrument)

            if clip_id not in sample_source_relevance:
                sample_source_relevance[clip_id] = {}
                sample_observed[clip_id] = set()

            # Track which source labels we have any observation for
            if source_label:
                sample_observed[clip_id].add(source_label)
                prev = sample_source_relevance[clip_id].get(source_label, 0.0)
                sample_source_relevance[clip_id][source_label] = max(prev, relevance)

            # Any OpenMIC instrument row means we have full coverage for this sample
            # (OpenMIC annotates all 20 instruments per sample)
            if instrument in _OPENMIC_ALL_INSTRUMENTS:
                for sl in SOURCE_LABELS:
                    sample_observed[clip_id].add(sl)

    annotations: dict[str, SourceAnnotation] = {}
    for clip_id, relevance_map in sample_source_relevance.items():
        annotation = SourceAnnotation.empty()
        annotation.support = "csv_structured"
        observed = sample_observed.get(clip_id, set())

        for source_label in SOURCE_LABELS:
            if source_label not in observed:
                continue
            annotation.mask[source_label] = 1.0
            annotation.quality[source_label] = SOURCE_LABEL_QUALITY
            rel = relevance_map.get(source_label, 0.0)
            annotation.values[source_label] = 1.0 if rel >= _OPENMIC_RELEVANCE_THRESHOLD else 0.0
            if annotation.values[source_label] > 0:
                annotation.evidence[source_label] = [f"openmic_relevance={rel:.2f}"]

        annotations[clip_id] = annotation

    return annotations


def load_source_annotations_from_csv(root: Path) -> dict[str, SourceAnnotation]:
    annotations: dict[str, SourceAnnotation] = {}

    for csv_path in sorted(root.rglob("*.csv")):
        try:
            with csv_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue

                # OpenMIC-2018 long-format CSV
                if _is_openmic_longformat_csv(list(reader.fieldnames)):
                    for clip_id, annotation in _parse_openmic_longformat_csv(csv_path).items():
                        annotations[clip_id] = annotations.get(clip_id, SourceAnnotation.empty()).merge(annotation)
                    continue

                fieldnames = [field.lower() for field in reader.fieldnames]
                id_field = next(
                    (
                        original
                        for original, lowered in zip(reader.fieldnames, fieldnames)
                        if lowered in {"sample_key", "clip_id", "track_id", "fname", "filename", "id"}
                    ),
                    None,
                )

                if id_field is None:
                    continue

                grouped_fields = {
                    source_label: [
                        original
                        for original, lowered in zip(reader.fieldnames, fieldnames)
                        if lowered not in {"label", "labels", "tag", "tags", "category", "categories"}
                        and any(term in lowered for term in SOURCE_HINT_TERMS[source_label])
                    ]
                    for source_label in SOURCE_LABELS
                }
                generic_tag_fields = [
                    original
                    for original, lowered in zip(reader.fieldnames, fieldnames)
                    if lowered in {"label", "labels", "tag", "tags", "category", "categories"}
                ]

                for row in reader:
                    clip_id = normalize_clip_id(row.get(id_field, ""))
                    if not clip_id:
                        continue

                    row_annotation = SourceAnnotation.empty()
                    row_annotation.support = "unavailable"

                    for source_label, source_fields in grouped_fields.items():
                        if not source_fields:
                            continue

                        row_annotation.mask[source_label] = 1.0
                        row_annotation.quality[source_label] = SOURCE_LABEL_QUALITY
                        row_annotation.support = choose_stronger_support(row_annotation.support, "csv_structured")
                        values = [row.get(field, "") for field in source_fields]
                        is_present = any(is_truthy(value) for value in values)
                        row_annotation.values[source_label] = 1.0 if is_present else 0.0
                        if is_present:
                            row_annotation.evidence[source_label] = unique_in_order(
                                [*row_annotation.evidence[source_label], *[str(value) for value in values if str(value).strip()]]
                            )

                    if generic_tag_fields:
                        tag_text = " ".join(str(row.get(field, "")) for field in generic_tag_fields).strip()
                        if tag_text:
                            tag_hits = detect_source_labels_in_text(tag_text)
                            for source_label in SOURCE_LABELS:
                                row_annotation.mask[source_label] = 1.0
                                row_annotation.quality[source_label] = SOURCE_LABEL_QUALITY
                                if source_label in tag_hits:
                                    row_annotation.values[source_label] = 1.0
                                    row_annotation.evidence[source_label] = unique_in_order(
                                        [*row_annotation.evidence[source_label], tag_text]
                                    )
                            row_annotation.support = choose_stronger_support(row_annotation.support, "csv_tags")

                    annotations[clip_id] = annotations.get(clip_id, SourceAnnotation.empty()).merge(row_annotation)
        except UnicodeDecodeError:
            continue

    return annotations


# MUSAN music genre → source presence signals.
# Sources listed here get mask=1; value=1.0 means "present", 0.0 means "absent".
# Sources NOT listed get mask=0 (unknown).
_MUSAN_GENRE_PRESENT: dict[str, set[str]] = {
    "blues":       {"guitar", "bass", "drums"},
    "pop":         {"guitar", "bass", "drums"},
    "rock":        {"guitar", "bass", "drums"},
    "metal":       {"guitar", "bass", "drums"},
    "indierock":   {"guitar", "bass", "drums"},
    "country":     {"guitar", "bass", "drums"},
    "folk":        {"guitar"},
    "funk":        {"guitar", "bass", "drums"},
    "jazz":        {"guitar", "bass", "drums", "keys"},
    "hiphop":      {"drums", "bass"},
    "rap":         {"drums", "bass"},
    "electronica": {"keys", "bass", "drums"},
    "dance":       {"keys", "bass", "drums"},
    "gospel":      {"keys", "bass", "drums"},
    "soul":        {"keys", "guitar", "bass", "drums"},
    "rnb":         {"keys", "bass", "drums"},
    "latin":       {"keys", "bass", "drums"},
    "baroque":     {"keys"},
    "westernart":  {"keys"},
    "romantic":    {"keys"},
    "classical":   {"keys"},
    "adventure":   {"keys"},
    "mondernist":  {"keys"},
}
# Sources that are definitively absent for a genre (e.g. classical has no drum kit)
_MUSAN_GENRE_ABSENT: dict[str, set[str]] = {
    "baroque":    {"guitar", "bass", "drums"},
    "westernart": {"guitar", "bass", "drums"},
    "romantic":   {"guitar", "bass", "drums"},
    "classical":  {"guitar", "bass", "drums"},
    "adventure":  {"guitar", "bass", "drums"},
    "mondernist": {"guitar", "bass", "drums"},
}


def load_musan_music_annotations(musan_root: Path) -> dict[str, tuple[set[str], bool]]:
    """Load per-file annotations from MUSAN music/*/ANNOTATIONS files.

    Returns a dict mapping normalised stem → (genre_set, vocal_present).
    ANNOTATIONS line format: ``filename genre1[,genre2] Y|N artist [composer]``
    """
    result: dict[str, tuple[set[str], bool]] = {}
    music_root = musan_root / "music"

    if not music_root.exists():
        return result

    for annotations_path in sorted(music_root.rglob("ANNOTATIONS")):
        try:
            with annotations_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    stem = normalize_clip_id(parts[0])
                    genres: set[str] = {g.strip().lower() for g in parts[1].split(",") if g.strip()}
                    vocal_present = parts[2].strip().upper() == "Y"
                    result[stem] = (genres, vocal_present)
        except (OSError, UnicodeDecodeError):
            continue

    return result


def infer_musan_source_annotation(
    audio_path: Path,
    musan_annotations: dict[str, tuple[set[str], bool]] | None = None,
) -> SourceAnnotation:
    """MUSAN uses a 3-tier directory structure to signal content type.

    musan/speech/* → pure vocal; all other sources definitely absent
    musan/noise/*  → non-musical; all sources definitely absent
    musan/music/*  → use per-file ANNOTATIONS (genre + vocal flag) when available
    """
    annotation = SourceAnnotation.empty()
    parts_lower = {p.lower() for p in audio_path.parts}

    if "speech" in parts_lower:
        # We know: vocal present, no instruments
        for label in SOURCE_LABELS:
            annotation.mask[label] = 1.0
            annotation.quality[label] = SOURCE_LABEL_QUALITY
            annotation.values[label] = 1.0 if label == "vocal" else 0.0
        annotation.support = "filename_partial"
        annotation.evidence["vocal"] = ["musan/speech"]
        return annotation

    if "noise" in parts_lower:
        # We know: no musical sources at all
        for label in SOURCE_LABELS:
            annotation.mask[label] = 1.0
            annotation.quality[label] = SOURCE_LABEL_QUALITY
            annotation.values[label] = 0.0
        annotation.support = "filename_partial"
        return annotation

    # music/ path — try to use per-file ANNOTATIONS data
    if musan_annotations is None or "music" not in parts_lower:
        return annotation  # mask=0, fully unknown

    stem = normalize_clip_id(audio_path)
    entry = musan_annotations.get(stem)

    if entry is None:
        return annotation  # no ANNOTATIONS entry found, leave unknown

    genres, vocal_present = entry

    # Collect which sources are clearly present or absent across all genres
    confirmed_present: set[str] = set()
    confirmed_absent: set[str] = set()

    for genre in genres:
        confirmed_present.update(_MUSAN_GENRE_PRESENT.get(genre, set()))
        confirmed_absent.update(_MUSAN_GENRE_ABSENT.get(genre, set()))

    # A source is only absent if confirmed absent by all genres AND never present
    truly_absent = confirmed_absent - confirmed_present

    # Vocal: always from the explicit Y/N flag
    annotation.mask["vocal"] = 1.0
    annotation.quality["vocal"] = SOURCE_LABEL_QUALITY
    annotation.values["vocal"] = 1.0 if vocal_present else 0.0
    if vocal_present:
        annotation.evidence["vocal"] = ["musan_annotation:vocal_Y"]

    # Instrument sources
    for label in SOURCE_LABELS:
        if label == "vocal":
            continue
        if label in confirmed_present:
            annotation.mask[label] = 1.0
            annotation.quality[label] = SOURCE_LABEL_QUALITY
            annotation.values[label] = 1.0
            annotation.evidence[label] = [f"musan_genre:{','.join(sorted(genres))}"]
        elif label in truly_absent:
            annotation.mask[label] = 1.0
            annotation.quality[label] = SOURCE_LABEL_QUALITY
            annotation.values[label] = 0.0
        # else: mask stays 0 (unknown)

    annotation.support = "csv_structured"
    return annotation


def infer_source_annotation_from_path(audio_path: Path, support: str) -> SourceAnnotation:
    annotation = SourceAnnotation.empty()
    normalized = normalize_text(audio_path.as_posix())
    positives = detect_source_labels_in_text(normalized)

    if not positives:
        return annotation

    annotation.support = support

    for source_label in positives:
        annotation.values[source_label] = 1.0
        annotation.mask[source_label] = 1.0
        annotation.quality[source_label] = SOURCE_LABEL_QUALITY
        annotation.evidence[source_label] = [audio_path.name]

    return annotation


def infer_source_annotation_from_stems(stem_paths: list[Path]) -> SourceAnnotation:
    annotation = SourceAnnotation.empty()

    for stem_path in stem_paths:
        positives = detect_source_labels_in_text(normalize_text(stem_path.as_posix()))

        for source_label in positives:
            annotation.values[source_label] = 1.0
            annotation.mask[source_label] = 1.0
            annotation.quality[source_label] = SOURCE_LABEL_QUALITY
            annotation.evidence[source_label] = unique_in_order([*annotation.evidence[source_label], stem_path.name])

    if any(annotation.mask[label] > 0 for label in SOURCE_LABELS):
        annotation.support = "stems_partial"

    return annotation


def detect_source_labels_in_text(value: str) -> list[str]:
    normalized = normalize_text(value)
    hits: list[str] = []

    for source_label, terms in SOURCE_HINT_TERMS.items():
        if any(term in normalized for term in terms):
            hits.append(source_label)

    return hits


def choose_segment_starts(duration_seconds: float, clip_duration: float, clips_per_file: int) -> list[float]:
    if duration_seconds <= clip_duration or clips_per_file <= 1:
        return [0.0]

    available = max(0.0, duration_seconds - clip_duration)
    if clips_per_file == 2:
        return [0.0, available]

    return [available * (index / (clips_per_file - 1)) for index in range(clips_per_file)]


def collect_audio_files(root: Path | None) -> list[Path]:
    if root is None or not root.exists():
        return []

    paths = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS]
    paths.sort()
    return paths


def infer_track_group_id(audio_path: Path, dataset_name: str) -> str:
    lowered_parts = [part.lower() for part in audio_path.parts]

    if dataset_name == "slakh":
        for part in reversed(audio_path.parts[:-1]):
            lowered = part.lower()
            if lowered.startswith("track") or lowered not in {"train", "validation", "valid", "val", "test"}:
                return f"{dataset_name}:{normalize_text(part)}"

    return f"{dataset_name}:{normalize_clip_id(audio_path)}"


def infer_split(audio_path: Path, track_group_id: str) -> str:
    lowered_parts = {part.lower() for part in audio_path.parts}

    if lowered_parts & {"test", "eval"}:
        return "test"

    if lowered_parts & {"valid", "validation", "val"}:
        return "val"

    if lowered_parts & {"train", "training", "dev"}:
        return "train"

    digest = hashlib.md5(track_group_id.encode("utf-8")).hexdigest()
    return "val" if int(digest[:2], 16) < 51 else "train"


def normalize_clip_id(path_like: str | Path) -> str:
    if not path_like:
        return ""
    path = Path(path_like)
    return normalize_text(path.stem)


def normalize_text(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def is_truthy(value: str | float | int) -> bool:
    normalized = normalize_text(str(value))

    if normalized in {"1", "true", "yes", "y", "present"}:
        return True

    try:
        return float(normalized) > 0.0
    except ValueError:
        return any(term in normalized for terms in SOURCE_HINT_TERMS.values() for term in terms)


def choose_stronger_support(left: str, right: str) -> str:
    support_priority = {
        "unavailable": 0,
        "filename_partial": 1,
        "stems_partial": 2,
        "csv_tags": 3,
        "csv_structured": 4,
    }
    return left if support_priority[left] >= support_priority[right] else right


def choose_label_quality(left: str, right: str, mask_value: float) -> str:
    if mask_value <= 0:
        return UNAVAILABLE_LABEL_QUALITY

    if "reviewed" in {left, right}:
        return "reviewed"

    return SOURCE_LABEL_QUALITY


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []

    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)

    return unique
