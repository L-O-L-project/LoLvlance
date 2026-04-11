from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CLIP_SECONDS = 3.0
DEFAULT_WINDOW_MS = 25
DEFAULT_HOP_MS = 10
DEFAULT_FFT_SIZE = 512
DEFAULT_MEL_BIN_COUNT = 64
MIN_MEL_FREQUENCY = 20.0
LOG_EPSILON = 1e-6


@dataclass(frozen=True)
class PreprocessingConfig:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    clip_seconds: float = DEFAULT_CLIP_SECONDS
    window_ms: int = DEFAULT_WINDOW_MS
    hop_ms: int = DEFAULT_HOP_MS
    fft_size: int = DEFAULT_FFT_SIZE
    mel_bin_count: int = DEFAULT_MEL_BIN_COUNT


@dataclass
class AudioFeatures:
    waveform: np.ndarray
    log_mel_spectrogram: np.ndarray
    rms: float
    spectral_centroid_hz: float
    spectral_rolloff_hz: float
    boom_ratio: float
    low_mid_ratio: float
    boxy_ratio: float
    presence_ratio: float
    harsh_ratio: float
    nasal_ratio: float
    sibilant_ratio: float
    air_ratio: float


def load_audio_segment(
    audio_path: str | Path,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    path = Path(audio_path)
    info = sf.info(path)
    start_frame = max(0, int(round(start_seconds * info.samplerate)))
    frame_count = -1 if duration_seconds is None else max(0, int(round(duration_seconds * info.samplerate)))

    waveform, sample_rate = sf.read(
        path,
        start=start_frame,
        frames=frame_count,
        dtype="float32",
        always_2d=False,
    )

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if sample_rate != target_sample_rate:
        waveform = resample_audio(waveform, sample_rate, target_sample_rate)

    return np.asarray(waveform, dtype=np.float32)


def extract_audio_features_from_path(
    audio_path: str | Path,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    config: PreprocessingConfig | None = None,
) -> AudioFeatures:
    active_config = config or PreprocessingConfig()
    waveform = load_audio_segment(
        audio_path=audio_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        target_sample_rate=active_config.sample_rate,
    )
    return extract_audio_features_from_waveform(waveform, active_config)


def extract_audio_features_from_waveform(
    waveform: np.ndarray,
    config: PreprocessingConfig | None = None,
) -> AudioFeatures:
    active_config = config or PreprocessingConfig()
    target_length = int(round(active_config.sample_rate * active_config.clip_seconds))
    normalized_waveform = pad_or_trim(waveform, target_length)
    log_mel = waveform_to_log_mel_spectrogram(normalized_waveform, active_config)

    rms = float(np.sqrt(np.mean(np.square(normalized_waveform)) + 1e-12))
    stats = compute_spectral_statistics(normalized_waveform, active_config.sample_rate)

    return AudioFeatures(
        waveform=normalized_waveform,
        log_mel_spectrogram=log_mel,
        rms=rms,
        spectral_centroid_hz=stats["spectral_centroid_hz"],
        spectral_rolloff_hz=stats["spectral_rolloff_hz"],
        boom_ratio=stats["boom_ratio"],
        low_mid_ratio=stats["low_mid_ratio"],
        boxy_ratio=stats["boxy_ratio"],
        presence_ratio=stats["presence_ratio"],
        harsh_ratio=stats["harsh_ratio"],
        nasal_ratio=stats["nasal_ratio"],
        sibilant_ratio=stats["sibilant_ratio"],
        air_ratio=stats["air_ratio"],
    )


def waveform_to_log_mel_spectrogram(
    waveform: np.ndarray,
    config: PreprocessingConfig | None = None,
) -> np.ndarray:
    active_config = config or PreprocessingConfig()
    frame_size = max(1, int(round((active_config.window_ms / 1000.0) * active_config.sample_rate)))
    hop_size = max(1, int(round((active_config.hop_ms / 1000.0) * active_config.sample_rate)))
    fft_size = max(next_power_of_two(frame_size), active_config.fft_size)
    spectrum_bin_count = fft_size // 2 + 1
    frame_count = max(1, math.floor((len(waveform) - frame_size) / hop_size) + 1)
    hann_window = get_hann_window(frame_size)
    mel_filter_bank = get_mel_filter_bank(
        sample_rate=active_config.sample_rate,
        fft_size=fft_size,
        mel_bin_count=active_config.mel_bin_count,
    )
    log_mel = np.zeros((frame_count, active_config.mel_bin_count), dtype=np.float32)

    for frame_index in range(frame_count):
        frame_offset = frame_index * hop_size
        frame = np.zeros(fft_size, dtype=np.float32)
        frame_slice = waveform[frame_offset : frame_offset + frame_size]
        frame[: len(frame_slice)] = frame_slice * hann_window[: len(frame_slice)]

        power_spectrum = (np.abs(np.fft.rfft(frame, n=fft_size)) ** 2).astype(np.float32)
        power_spectrum /= float(max(1, fft_size))

        if power_spectrum.shape[0] != spectrum_bin_count:
            raise RuntimeError("Unexpected power spectrum size while computing log-mel features.")

        mel_energy = mel_filter_bank @ power_spectrum
        log_mel[frame_index] = np.log(np.maximum(LOG_EPSILON, mel_energy))

    return log_mel


def compute_spectral_statistics(waveform: np.ndarray, sample_rate: int) -> dict[str, float]:
    if waveform.size == 0:
        return {
            "spectral_centroid_hz": 0.0,
            "spectral_rolloff_hz": 0.0,
            "boom_ratio": 0.0,
            "low_mid_ratio": 0.0,
            "boxy_ratio": 0.0,
            "presence_ratio": 0.0,
            "harsh_ratio": 0.0,
            "nasal_ratio": 0.0,
            "sibilant_ratio": 0.0,
            "air_ratio": 0.0,
        }

    window = np.hanning(max(2, waveform.size)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(waveform * window[: waveform.size])) ** 2
    freqs = np.fft.rfftfreq(waveform.size, d=1.0 / sample_rate)
    total_energy = float(np.sum(spectrum) + 1e-12)
    centroid = float(np.sum(freqs * spectrum) / total_energy)
    cumulative = np.cumsum(spectrum)
    rolloff_index = int(np.searchsorted(cumulative, total_energy * 0.85, side="left"))
    rolloff_index = min(rolloff_index, freqs.shape[0] - 1)

    return {
        "spectral_centroid_hz": centroid,
        "spectral_rolloff_hz": float(freqs[rolloff_index]),
        "boom_ratio": band_energy_ratio(freqs, spectrum, 60.0, 180.0, total_energy),
        "low_mid_ratio": band_energy_ratio(freqs, spectrum, 200.0, 500.0, total_energy),
        "boxy_ratio": band_energy_ratio(freqs, spectrum, 350.0, 900.0, total_energy),
        "presence_ratio": band_energy_ratio(freqs, spectrum, 1_000.0, 4_000.0, total_energy),
        "harsh_ratio": band_energy_ratio(freqs, spectrum, 4_000.0, 8_000.0, total_energy),
        "nasal_ratio": band_energy_ratio(freqs, spectrum, 700.0, 1_600.0, total_energy),
        "sibilant_ratio": band_energy_ratio(freqs, spectrum, 5_000.0, 8_000.0, total_energy),
        "air_ratio": band_energy_ratio(freqs, spectrum, 6_000.0, 8_000.0, total_energy),
    }


def band_energy_ratio(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    min_hz: float,
    max_hz: float,
    total_energy: float | None = None,
) -> float:
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    band_energy = float(np.sum(spectrum[mask]))
    denominator = float(total_energy if total_energy is not None else np.sum(spectrum) + 1e-12)
    return band_energy / max(1e-12, denominator)


def pad_or_trim(waveform: np.ndarray, target_length: int) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.shape[0] == target_length:
        return waveform.copy()

    if waveform.shape[0] > target_length:
        return waveform[-target_length:].copy()

    padded = np.zeros(target_length, dtype=np.float32)
    padded[-waveform.shape[0] :] = waveform
    return padded


def resample_audio(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or waveform.size == 0:
        return np.asarray(waveform, dtype=np.float32)

    duration_seconds = waveform.shape[0] / float(source_rate)
    target_length = max(1, int(round(duration_seconds * target_rate)))
    source_positions = np.linspace(0.0, duration_seconds, num=waveform.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, duration_seconds, num=target_length, endpoint=False)
    resampled = np.interp(target_positions, source_positions, waveform)
    return np.asarray(resampled, dtype=np.float32)


def next_power_of_two(value: int) -> int:
    power = 1

    while power < value:
        power <<= 1

    return power


@lru_cache(maxsize=32)
def get_hann_window(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((max(1, length),), dtype=np.float32)

    return np.hanning(length).astype(np.float32)


@lru_cache(maxsize=64)
def get_mel_filter_bank(sample_rate: int, fft_size: int, mel_bin_count: int) -> np.ndarray:
    spectrum_bin_count = fft_size // 2 + 1
    max_frequency = sample_rate / 2.0
    mel_min = hz_to_mel(MIN_MEL_FREQUENCY)
    mel_max = hz_to_mel(max_frequency)
    mel_points = np.linspace(mel_min, mel_max, num=mel_bin_count + 2, dtype=np.float32)
    hz_points = mel_to_hz(mel_points)
    fft_bins = np.floor(((fft_size + 1) * hz_points) / sample_rate).astype(np.int32)
    fft_bins = np.clip(fft_bins, 0, spectrum_bin_count - 1)
    filter_bank = np.zeros((mel_bin_count, spectrum_bin_count), dtype=np.float32)

    for mel_index in range(mel_bin_count):
        start_bin = int(fft_bins[mel_index])
        center_bin = max(start_bin + 1, int(fft_bins[mel_index + 1]))
        end_bin = max(center_bin + 1, int(fft_bins[mel_index + 2]))

        for bin_index in range(start_bin, min(center_bin, spectrum_bin_count)):
            denominator = max(1, center_bin - start_bin)
            filter_bank[mel_index, bin_index] = (bin_index - start_bin) / denominator

        for bin_index in range(center_bin, min(end_bin, spectrum_bin_count)):
            denominator = max(1, end_bin - center_bin)
            filter_bank[mel_index, bin_index] = (end_bin - bin_index) / denominator

    return filter_bank


def hz_to_mel(frequency_hz: float) -> float:
    return 2595.0 * math.log10(1.0 + frequency_hz / 700.0)


def mel_to_hz(mel_value: np.ndarray | float) -> np.ndarray:
    mel_array = np.asarray(mel_value, dtype=np.float32)
    return 700.0 * (np.power(10.0, mel_array / 2595.0) - 1.0)
