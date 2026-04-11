from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .preprocessing import (
    AudioFeatures,
    PreprocessingConfig,
    extract_audio_features_from_path,
)

PROBLEM_LABELS = ("muddy", "harsh", "buried")
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
VOCAL_HINT_TERMS = {"singer", "singing", "song", "speech", "vocal", "vocals", "choir"}


@dataclass(frozen=True)
class DatasetRoots:
    openmic: Path | None = None
    slakh: Path | None = None
    musan: Path | None = None
    fsd50k: Path | None = None


class LoLvlanceAudioDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        preprocessing_config: PreprocessingConfig | None = None,
    ) -> None:
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
        self.entries = [
            entry for entry in load_manifest(manifest_path) if entry["split"] == split
        ]

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
            "labels": torch.tensor(entry["label_vector"], dtype=torch.float32),
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
        scan_openmic(
            dataset_roots.openmic,
            config=config,
            clips_per_file=clips_per_file,
            max_files=max_files_per_dataset,
        )
    )
    entries.extend(
        scan_slakh(
            dataset_roots.slakh,
            config=config,
            clips_per_file=clips_per_file,
            max_files=max_files_per_dataset,
        )
    )
    entries.extend(
        scan_generic_dataset(
            dataset_roots.musan,
            dataset_name="musan",
            config=config,
            clips_per_file=clips_per_file,
            max_files=max_files_per_dataset,
        )
    )
    entries.extend(
        scan_generic_dataset(
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


def load_manifest(manifest_path: str | Path) -> list[dict]:
    manifest = Path(manifest_path)

    with manifest.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_manifest(entries: Iterable[dict]) -> dict[str, object]:
    entries = list(entries)
    split_counts: dict[str, int] = {}
    positive_counts = {label: 0 for label in PROBLEM_LABELS}

    for entry in entries:
        split_counts[entry["split"]] = split_counts.get(entry["split"], 0) + 1

        for label, value in zip(PROBLEM_LABELS, entry["label_vector"]):
            if value >= 0.5:
                positive_counts[label] += 1

    return {
        "total": len(entries),
        "splits": split_counts,
        "positives": positive_counts,
    }


def scan_openmic(
    root: Path | None,
    config: PreprocessingConfig,
    clips_per_file: int,
    max_files: int | None,
) -> list[dict]:
    if root is None or not root.exists():
        return []

    vocal_hints = load_vocal_hints_from_csv(root)
    audio_files = collect_audio_files(root)
    if max_files is not None:
        audio_files = audio_files[:max_files]

    entries: list[dict] = []

    for audio_path in audio_files:
        sample_id = normalize_clip_id(audio_path)
        vocal_hint = vocal_hints.get(sample_id, filename_has_vocal_hint(audio_path))
        entries.extend(
            build_entries_for_file(
                audio_path=audio_path,
                dataset_name="openmic",
                config=config,
                clips_per_file=clips_per_file,
                vocal_hint=vocal_hint,
            )
        )

    return entries


def scan_slakh(
    root: Path | None,
    config: PreprocessingConfig,
    clips_per_file: int,
    max_files: int | None,
) -> list[dict]:
    if root is None or not root.exists():
        return []

    mix_files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
        and path.stem.lower() == "mix"
    ]
    mix_files.sort()
    if max_files is not None:
        mix_files = mix_files[:max_files]

    entries: list[dict] = []

    for mix_path in mix_files:
        stem_dir = mix_path.parent / "stems"
        stem_paths = collect_audio_files(stem_dir) if stem_dir.exists() else []
        entries.extend(
            build_entries_for_file(
                audio_path=mix_path,
                dataset_name="slakh",
                config=config,
                clips_per_file=clips_per_file,
                vocal_hint=False,
                stem_paths=stem_paths[:8],
            )
        )

    return entries


def scan_generic_dataset(
    root: Path | None,
    dataset_name: str,
    config: PreprocessingConfig,
    clips_per_file: int,
    max_files: int | None,
) -> list[dict]:
    if root is None or not root.exists():
        return []

    vocal_hints = load_vocal_hints_from_csv(root)
    audio_files = collect_audio_files(root)
    if max_files is not None:
        audio_files = audio_files[:max_files]

    entries: list[dict] = []

    for audio_path in audio_files:
        sample_id = normalize_clip_id(audio_path)
        vocal_hint = vocal_hints.get(sample_id, filename_has_vocal_hint(audio_path))
        entries.extend(
            build_entries_for_file(
                audio_path=audio_path,
                dataset_name=dataset_name,
                config=config,
                clips_per_file=clips_per_file,
                vocal_hint=vocal_hint,
            )
        )

    return entries


def build_entries_for_file(
    audio_path: Path,
    dataset_name: str,
    config: PreprocessingConfig,
    clips_per_file: int,
    vocal_hint: bool,
    stem_paths: list[Path] | None = None,
) -> list[dict]:
    import soundfile as sf

    audio_info = sf.info(audio_path)
    duration_seconds = float(audio_info.frames) / float(audio_info.samplerate)
    starts = choose_segment_starts(duration_seconds, config.clip_seconds, clips_per_file)

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
        label_vector = infer_problem_labels(
            dataset_name=dataset_name,
            features=features,
            vocal_hint=vocal_hint,
            low_mid_overlap=low_mid_overlap,
            path=audio_path,
        )
        split = infer_split(audio_path)
        clip_id = f"{dataset_name}:{normalize_clip_id(audio_path)}:{clip_index}"

        entries.append(
            {
                "clip_id": clip_id,
                "dataset": dataset_name,
                "audio_path": audio_path.resolve().as_posix(),
                "start_seconds": round(start_seconds, 3),
                "duration_seconds": config.clip_seconds,
                "split": split,
                "label_vector": label_vector,
                "positive_labels": [
                    label for label, value in zip(PROBLEM_LABELS, label_vector) if value >= 0.5
                ],
                "vocal_hint": vocal_hint,
                "stats": {
                    "rms": round(features.rms, 6),
                    "centroid_hz": round(features.spectral_centroid_hz, 2),
                    "low_mid_ratio": round(features.low_mid_ratio, 4),
                    "presence_ratio": round(features.presence_ratio, 4),
                    "harsh_ratio": round(features.harsh_ratio, 4),
                    "low_mid_overlap": low_mid_overlap,
                },
            }
        )

    return entries


def infer_problem_labels(
    dataset_name: str,
    features: AudioFeatures,
    vocal_hint: bool,
    low_mid_overlap: int,
    path: Path,
) -> list[float]:
    harsh_keyword = any(term in normalize_text(path.as_posix()) for term in ("alarm", "siren", "scream", "crash"))
    muddy_keyword = any(term in normalize_text(path.as_posix()) for term in ("bass", "organ", "piano", "guitar", "keys"))

    muddy = features.low_mid_ratio >= 0.24 or low_mid_overlap >= 2
    harsh = features.harsh_ratio >= 0.16 or features.spectral_centroid_hz >= 2_600.0 or harsh_keyword
    buried = vocal_hint and features.presence_ratio <= 0.18 and features.rms >= 0.005

    if dataset_name == "slakh":
        muddy = muddy or low_mid_overlap >= 2

    if dataset_name == "musan":
        harsh = harsh or "noise" in normalize_text(path.as_posix())

    if dataset_name == "fsd50k":
        harsh = harsh or harsh_keyword

    if dataset_name == "openmic":
        buried = buried or (vocal_hint and features.low_mid_ratio >= 0.22 and features.presence_ratio <= 0.2)

    if muddy_keyword and features.low_mid_ratio >= 0.2:
        muddy = True

    return [float(muddy), float(harsh), float(buried)]


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

    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    paths.sort()
    return paths


def load_vocal_hints_from_csv(root: Path) -> dict[str, bool]:
    hints: dict[str, bool] = {}

    for csv_path in sorted(root.rglob("*.csv")):
        try:
            with csv_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
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
                vocal_fields = [
                    original
                    for original, lowered in zip(reader.fieldnames, fieldnames)
                    if any(term in lowered for term in VOCAL_HINT_TERMS)
                    or lowered in {"label", "labels", "tag", "tags"}
                ]

                if id_field is None or not vocal_fields:
                    continue

                for row in reader:
                    clip_id = normalize_clip_id(Path(row[id_field]))
                    hint = any(is_truthy(row.get(field, "")) for field in vocal_fields)
                    if hint:
                        hints[clip_id] = True
        except UnicodeDecodeError:
            continue

    return hints


def infer_split(path: Path) -> str:
    lowered_parts = {part.lower() for part in path.parts}

    if lowered_parts & {"valid", "validation", "val", "test", "eval"}:
        return "val"

    if lowered_parts & {"train", "training", "dev"}:
        return "train"

    digest = hashlib.md5(path.as_posix().encode("utf-8")).hexdigest()
    return "val" if int(digest[:2], 16) < 51 else "train"


def normalize_clip_id(path_like: str | Path) -> str:
    path = Path(path_like)
    return normalize_text(path.stem)


def filename_has_vocal_hint(path: Path) -> bool:
    normalized = normalize_text(path.as_posix())
    return any(term in normalized for term in VOCAL_HINT_TERMS)


def normalize_text(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def is_truthy(value: str) -> bool:
    normalized = normalize_text(str(value))
    return normalized in {"1", "true", "yes", "y", "present", "vocal", "vocals", "singer"} or any(
        term in normalized for term in VOCAL_HINT_TERMS
    )
