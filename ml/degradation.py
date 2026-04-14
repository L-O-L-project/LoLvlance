from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .dataset import AUDIO_EXTENSIONS, load_manifest
    from .label_schema import ISSUE_FALLBACK_EQ, ISSUE_LABELS, SOURCE_HINT_TERMS, SOURCE_LABELS
    from .preprocessing import (
        DEFAULT_SAMPLE_RATE,
        PreprocessingConfig,
        extract_audio_features_from_waveform,
        load_audio_segment,
        pad_or_trim,
    )
except ImportError:
    from dataset import AUDIO_EXTENSIONS, load_manifest
    from label_schema import ISSUE_FALLBACK_EQ, ISSUE_LABELS, SOURCE_HINT_TERMS, SOURCE_LABELS
    from preprocessing import (
        DEFAULT_SAMPLE_RATE,
        PreprocessingConfig,
        extract_audio_features_from_waveform,
        load_audio_segment,
        pad_or_trim,
    )


@dataclass(frozen=True)
class EqBand:
    center_hz: float
    gain_db: float
    width_octaves: float
    issue_label: str


@dataclass(frozen=True)
class CompressionSettings:
    threshold: float
    ratio: float
    makeup_gain_db: float


@dataclass(frozen=True)
class ReverbSettings:
    wet: float
    decay_seconds: float
    pre_delay_seconds: float


@dataclass(frozen=True)
class FilterSettings:
    mode: str
    cutoff_hz: float
    slope_db: float
    issue_label: str


@dataclass(frozen=True)
class DegradationRecipe:
    inverse_eq_bands: tuple[EqBand, ...]
    compression: CompressionSettings | None = None
    reverb: ReverbSettings | None = None
    filter_error: FilterSettings | None = None
    issue_labels: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "inverse_eq_bands": [asdict(band) for band in self.inverse_eq_bands],
            "compression": None if self.compression is None else asdict(self.compression),
            "reverb": None if self.reverb is None else asdict(self.reverb),
            "filter_error": None if self.filter_error is None else asdict(self.filter_error),
            "issue_labels": list(self.issue_labels),
        }


@dataclass(frozen=True)
class DegradationConfig:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    clip_seconds: float = 3.0
    eq_band_count: int = 5
    max_active_bands: int = 3
    max_eq_gain_db: float = 9.0
    min_eq_gain_db: float = 2.0
    reverb_probability: float = 0.35
    compression_probability: float = 0.40
    filter_error_probability: float = 0.30
    seed: int = 7


def build_real_source_manifest(
    *,
    audio_roots: Iterable[Path],
    output_path: Path,
    clip_seconds: float,
    clips_per_file: int = 2,
) -> list[dict]:
    entries: list[dict] = []

    for root in audio_roots:
        if not root.exists():
            continue

        for audio_path in sorted(path for path in root.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS):
            track_group_id = infer_track_group_id(audio_path, root)
            split = infer_split(track_group_id)
            source_values, source_mask = infer_source_targets_from_path(audio_path)
            starts = choose_segment_starts(audio_path, clip_seconds, clips_per_file)

            for clip_index, start_seconds in enumerate(starts):
                entries.append(
                    {
                        "clip_id": f"{track_group_id}:{clip_index}",
                        "track_group_id": track_group_id,
                        "audio_path": audio_path.resolve().as_posix(),
                        "start_seconds": round(start_seconds, 3),
                        "duration_seconds": clip_seconds,
                        "split": split,
                        "issue_targets": {
                            "labels": list(ISSUE_LABELS),
                            "values": [0.0 for _ in ISSUE_LABELS],
                            "mask": [0.0 for _ in ISSUE_LABELS],
                        },
                        "source_targets": {
                            "labels": list(SOURCE_LABELS),
                            "values": source_values,
                            "mask": source_mask,
                        },
                    }
                )

    if not entries:
        raise ValueError("No audio files found while building the real-source manifest.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(entry, ensure_ascii=True) for entry in entries), encoding="utf-8")
    return entries


class RealAudioDegradationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        *,
        manifest_path: str | Path,
        split: str,
        preprocessing_config: PreprocessingConfig | None = None,
        degradation_config: DegradationConfig | None = None,
    ) -> None:
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
        self.degradation_config = degradation_config or DegradationConfig(
            sample_rate=self.preprocessing_config.sample_rate,
            clip_seconds=self.preprocessing_config.clip_seconds,
        )
        self.entries = [entry for entry in load_manifest(manifest_path) if entry.get("split") == split]
        if not self.entries:
            raise ValueError(f"No entries found for split '{split}' in {manifest_path}.")

        self.base_seed = self.degradation_config.seed
        self.issue_sampling_weights = build_issue_sampling_weights(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self.entries[index]
        rng = np.random.default_rng(self.base_seed + index)
        clean_waveform = load_audio_segment(
            audio_path=entry["audio_path"],
            start_seconds=float(entry.get("start_seconds", 0.0)),
            duration_seconds=float(entry.get("duration_seconds", self.preprocessing_config.clip_seconds)),
            target_sample_rate=self.preprocessing_config.sample_rate,
        )
        target_length = int(round(self.preprocessing_config.sample_rate * self.preprocessing_config.clip_seconds))
        clean_waveform = pad_or_trim(clean_waveform, target_length)
        recipe = sample_degradation_recipe(
            rng=rng,
            source_targets=entry.get("source_targets", {}).get("values", [0.0 for _ in SOURCE_LABELS]),
            config=self.degradation_config,
            sample_rate=self.preprocessing_config.sample_rate,
            issue_sampling_weights=self.issue_sampling_weights,
        )
        degraded_waveform = apply_degradation_recipe(clean_waveform, recipe, self.preprocessing_config.sample_rate, rng)

        clean_features = extract_audio_features_from_waveform(clean_waveform, self.preprocessing_config)
        degraded_features = extract_audio_features_from_waveform(degraded_waveform, self.preprocessing_config)
        eq_params, eq_params_normalized, eq_mask = recipe_to_eq_targets(
            recipe,
            band_count=self.degradation_config.eq_band_count,
            min_hz=60.0,
            max_hz=min(self.preprocessing_config.sample_rate / 2.0, 8_000.0),
            max_gain_db=self.degradation_config.max_eq_gain_db,
        )
        issue_targets = [1.0 if label in recipe.issue_labels else 0.0 for label in ISSUE_LABELS]
        source_values = entry.get("source_targets", {}).get("values", [0.0 for _ in SOURCE_LABELS])
        source_mask = entry.get("source_targets", {}).get("mask", [0.0 for _ in SOURCE_LABELS])

        return {
            "log_mel_spectrogram": torch.from_numpy(degraded_features.log_mel_spectrogram).float(),
            "clean_log_mel_spectrogram": torch.from_numpy(clean_features.log_mel_spectrogram).float(),
            "issue_targets": torch.tensor(issue_targets, dtype=torch.float32),
            "issue_target_mask": torch.ones(len(ISSUE_LABELS), dtype=torch.float32),
            "source_targets": torch.tensor(source_values, dtype=torch.float32),
            "source_target_mask": torch.tensor(source_mask, dtype=torch.float32),
            "eq_params": torch.from_numpy(eq_params).float(),
            "eq_params_normalized": torch.from_numpy(eq_params_normalized).float(),
            "eq_mask": torch.from_numpy(eq_mask).float(),
        }


def build_issue_sampling_weights(entries: list[dict]) -> dict[str, float]:
    weights = {label: 1.0 for label in ISSUE_LABELS}

    for entry in entries:
        source_values = entry.get("source_targets", {}).get("values", [])
        if len(source_values) != len(SOURCE_LABELS):
            continue

        if source_values[SOURCE_LABELS.index("vocal")] > 0.5:
            weights["buried"] += 0.4
            weights["nasal"] += 0.2
            weights["sibilant"] += 0.3

        if source_values[SOURCE_LABELS.index("drums")] > 0.5:
            weights["harsh"] += 0.2

        if source_values[SOURCE_LABELS.index("bass")] > 0.5:
            weights["boomy"] += 0.25
            weights["thin"] += 0.15

    return weights


def sample_degradation_recipe(
    *,
    rng: np.random.Generator,
    source_targets: list[float],
    config: DegradationConfig,
    sample_rate: int,
    issue_sampling_weights: dict[str, float],
) -> DegradationRecipe:
    active_band_count = int(rng.integers(1, min(config.max_active_bands, config.eq_band_count) + 1))
    issues = weighted_choice_without_replacement(rng, issue_sampling_weights, active_band_count)
    inverse_eq_bands = []
    issue_labels = set(issues)

    vocal_present = len(source_targets) > SOURCE_LABELS.index("vocal") and source_targets[SOURCE_LABELS.index("vocal")] > 0.5

    for issue in issues:
        fallback = ISSUE_FALLBACK_EQ[issue]
        target_gain_db = float(abs(float(fallback["gain_db"])) + rng.uniform(0.5, 2.5))
        target_gain_db = float(np.clip(target_gain_db, config.min_eq_gain_db, config.max_eq_gain_db))
        jitter = float(np.exp(rng.normal(0.0, 0.12)))
        center_hz = float(np.clip(float(fallback["frequency_hz"]) * jitter, 60.0, sample_rate / 2.2))
        inverse_eq_bands.append(
            EqBand(
                center_hz=center_hz,
                gain_db=float(np.sign(float(fallback["gain_db"])) * target_gain_db),
                width_octaves=float(rng.uniform(0.35, 0.9)),
                issue_label=issue,
            )
        )

    reverb = None
    if rng.random() < config.reverb_probability:
        reverb = ReverbSettings(
            wet=float(rng.uniform(0.08, 0.24)),
            decay_seconds=float(rng.uniform(0.18, 0.6)),
            pre_delay_seconds=float(rng.uniform(0.0, 0.04)),
        )
        issue_labels.update({"muddy", "buried"} if vocal_present else {"muddy", "dull"})

    compression = None
    if rng.random() < config.compression_probability:
        compression = CompressionSettings(
            threshold=float(rng.uniform(0.18, 0.4)),
            ratio=float(rng.uniform(2.0, 6.0)),
            makeup_gain_db=float(rng.uniform(0.5, 3.5)),
        )
        issue_labels.add("harsh" if rng.random() > 0.5 else "buried")

    filter_error = None
    if rng.random() < config.filter_error_probability:
        mode = "lowpass" if rng.random() > 0.5 else "highpass"
        issue_label = "dull" if mode == "lowpass" else "thin"
        cutoff_hz = float(rng.uniform(2_400.0, 4_500.0) if mode == "lowpass" else rng.uniform(80.0, 180.0))
        filter_error = FilterSettings(
            mode=mode,
            cutoff_hz=cutoff_hz,
            slope_db=float(rng.uniform(6.0, 18.0)),
            issue_label=issue_label,
        )
        issue_labels.add(issue_label)

    inverse_eq_bands = tuple(sorted(inverse_eq_bands, key=lambda band: band.center_hz))
    return DegradationRecipe(
        inverse_eq_bands=inverse_eq_bands,
        compression=compression,
        reverb=reverb,
        filter_error=filter_error,
        issue_labels=tuple(sorted(issue_labels, key=lambda label: ISSUE_LABELS.index(label))),
    )


def apply_degradation_recipe(
    clean_waveform: np.ndarray,
    recipe: DegradationRecipe,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    degraded = clean_waveform.astype(np.float32).copy()
    degradation_bands = [
        EqBand(
            center_hz=band.center_hz,
            gain_db=-band.gain_db,
            width_octaves=band.width_octaves,
            issue_label=band.issue_label,
        )
        for band in recipe.inverse_eq_bands
    ]
    degraded = apply_eq_bands(degraded, sample_rate, degradation_bands)

    if recipe.filter_error is not None:
        degraded = apply_filter_error(degraded, sample_rate, recipe.filter_error)

    if recipe.compression is not None:
        degraded = apply_compression(degraded, recipe.compression)

    if recipe.reverb is not None:
        degraded = apply_synthetic_reverb(degraded, sample_rate, recipe.reverb, rng)

    peak = float(np.max(np.abs(degraded)))
    if peak > 1.0:
        degraded = degraded / peak
    return degraded.astype(np.float32)


def apply_eq_bands(waveform: np.ndarray, sample_rate: int, bands: list[EqBand]) -> np.ndarray:
    if not bands:
        return waveform

    fft = np.fft.rfft(waveform)
    frequencies = np.fft.rfftfreq(waveform.shape[0], d=1.0 / sample_rate)
    curve_db = np.zeros_like(frequencies, dtype=np.float32)

    for band in bands:
        safe_frequencies = np.maximum(frequencies, 1.0)
        log_distance = np.log2(safe_frequencies / max(1.0, band.center_hz))
        sigma = max(0.08, band.width_octaves / 2.355)
        curve_db += band.gain_db * np.exp(-0.5 * np.square(log_distance / sigma))

    magnitude = np.power(10.0, curve_db / 20.0).astype(np.float32)
    processed = np.fft.irfft(fft * magnitude, n=waveform.shape[0])
    return processed.astype(np.float32)


def apply_filter_error(waveform: np.ndarray, sample_rate: int, settings: FilterSettings) -> np.ndarray:
    fft = np.fft.rfft(waveform)
    frequencies = np.fft.rfftfreq(waveform.shape[0], d=1.0 / sample_rate)
    slope = settings.slope_db / 20.0

    if settings.mode == "lowpass":
        attenuation = np.where(
            frequencies > settings.cutoff_hz,
            -slope * np.log2(np.maximum(frequencies, 1.0) / settings.cutoff_hz),
            0.0,
        )
    else:
        attenuation = np.where(
            frequencies < settings.cutoff_hz,
            -slope * np.log2(settings.cutoff_hz / np.maximum(frequencies, 1.0)),
            0.0,
        )

    magnitude = np.power(10.0, attenuation / 20.0).astype(np.float32)
    processed = np.fft.irfft(fft * magnitude, n=waveform.shape[0])
    return processed.astype(np.float32)


def apply_compression(waveform: np.ndarray, settings: CompressionSettings) -> np.ndarray:
    amplitude = np.abs(waveform)
    over_threshold = np.maximum(amplitude - settings.threshold, 0.0)
    compressed = settings.threshold + (over_threshold / max(settings.ratio, 1.0))
    gain = compressed / np.maximum(amplitude, 1e-5)
    output = np.sign(waveform) * amplitude * np.where(amplitude > settings.threshold, gain, 1.0)
    makeup = float(np.power(10.0, settings.makeup_gain_db / 20.0))
    return (output * makeup).astype(np.float32)


def apply_synthetic_reverb(
    waveform: np.ndarray,
    sample_rate: int,
    settings: ReverbSettings,
    rng: np.random.Generator,
) -> np.ndarray:
    ir_length = max(1, int(round(settings.decay_seconds * sample_rate)))
    timeline = np.arange(ir_length, dtype=np.float32) / sample_rate
    impulse = np.exp(-timeline / max(settings.decay_seconds, 1e-3)).astype(np.float32)
    impulse *= rng.uniform(0.85, 1.15, size=ir_length).astype(np.float32)
    pre_delay = max(0, int(round(settings.pre_delay_seconds * sample_rate)))
    if pre_delay > 0:
        impulse = np.pad(impulse, (pre_delay, 0))
    impulse[0] = 1.0
    wet = np.convolve(waveform, impulse, mode="full")[: waveform.shape[0]].astype(np.float32)
    wet /= max(float(np.max(np.abs(wet))), 1.0)
    return ((1.0 - settings.wet) * waveform + settings.wet * wet).astype(np.float32)


def recipe_to_eq_targets(
    recipe: DegradationRecipe,
    *,
    band_count: int,
    min_hz: float,
    max_hz: float,
    max_gain_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eq_params = np.zeros((band_count, 2), dtype=np.float32)
    eq_params_normalized = np.zeros((band_count, 2), dtype=np.float32)
    eq_mask = np.zeros((band_count, 2), dtype=np.float32)

    for index, band in enumerate(recipe.inverse_eq_bands[:band_count]):
        eq_params[index, 0] = band.center_hz
        eq_params[index, 1] = band.gain_db
        eq_params_normalized[index, 0] = normalize_frequency(band.center_hz, min_hz=min_hz, max_hz=max_hz)
        eq_params_normalized[index, 1] = float(np.clip(band.gain_db / max_gain_db, -1.0, 1.0))
        eq_mask[index, :] = 1.0

    return eq_params, eq_params_normalized, eq_mask


def normalize_frequency(frequency_hz: float, *, min_hz: float, max_hz: float) -> float:
    clamped = float(np.clip(frequency_hz, min_hz, max_hz))
    return float(np.log(clamped / min_hz) / np.log(max_hz / min_hz))


def weighted_choice_without_replacement(
    rng: np.random.Generator,
    weights: dict[str, float],
    count: int,
) -> list[str]:
    labels = list(weights.keys())
    probabilities = np.asarray([max(1e-6, weights[label]) for label in labels], dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    chosen = rng.choice(labels, size=min(count, len(labels)), replace=False, p=probabilities)
    return [str(label) for label in np.atleast_1d(chosen)]


def infer_source_targets_from_path(audio_path: Path) -> tuple[list[float], list[float]]:
    normalized = audio_path.as_posix().lower().replace("-", "_").replace(" ", "_")
    values = []
    mask = []

    for label in SOURCE_LABELS:
        matches = any(term in normalized for term in SOURCE_HINT_TERMS[label])
        values.append(1.0 if matches else 0.0)
        mask.append(1.0 if matches else 0.0)

    return values, mask


def infer_track_group_id(audio_path: Path, root: Path) -> str:
    relative = audio_path.relative_to(root).with_suffix("")
    return relative.as_posix().replace("/", "_").lower()


def infer_split(track_group_id: str) -> str:
    digest = hashlib.md5(track_group_id.encode("utf-8")).hexdigest()
    return "val" if int(digest[:2], 16) < 26 else "train"


def choose_segment_starts(audio_path: Path, clip_seconds: float, clips_per_file: int) -> list[float]:
    import soundfile as sf

    info = sf.info(audio_path)
    duration_seconds = float(info.frames) / float(info.samplerate)
    if duration_seconds <= clip_seconds or clips_per_file <= 1:
        return [0.0]

    available = max(0.0, duration_seconds - clip_seconds)
    if clips_per_file == 2:
        return [0.0, available]

    return [available * (index / (clips_per_file - 1)) for index in range(clips_per_file)]
