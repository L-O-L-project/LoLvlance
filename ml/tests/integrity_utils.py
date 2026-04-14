from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ml.dataset import load_manifest
from ml.preprocessing import (
    DEFAULT_FFT_SIZE,
    DEFAULT_HOP_MS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WINDOW_MS,
    MIN_MEL_FREQUENCY,
    PreprocessingConfig,
    load_audio_segment,
    pad_or_trim,
    resample_audio,
    waveform_to_log_mel_spectrogram,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_AUDIO_PATH = Path(
    os.getenv(
        "LOLVLANCE_REFERENCE_AUDIO",
        (PROJECT_ROOT / "eval/goldens/vocal_sibilant_01/vocal_sibilant_01.wav").as_posix(),
    )
)
MANIFEST_PATH = Path(
    os.getenv(
        "LOLVLANCE_MANIFEST_PATH",
        (PROJECT_ROOT / "ml/artifacts/public_dataset_manifest.jsonl").as_posix(),
    )
)

TARGET_SAMPLE_RATE = DEFAULT_SAMPLE_RATE
ROLLING_BUFFER_SECONDS = 3
MODEL_MEL_BIN_COUNT = 64
REFERENCE_SOURCE_SAMPLE_RATES = (48_000, 44_100)
SOURCE_DURATIONS_SECONDS = (ROLLING_BUFFER_SECONDS, ROLLING_BUFFER_SECONDS + 0.2)

WAVEFORM_MSE_THRESHOLD = float(os.getenv("LOLVLANCE_WAVEFORM_MSE_THRESHOLD", "0.0001"))
WAVEFORM_MAE_THRESHOLD = float(os.getenv("LOLVLANCE_WAVEFORM_MAE_THRESHOLD", "0.001"))
WAVEFORM_MAX_ABS_THRESHOLD = float(os.getenv("LOLVLANCE_WAVEFORM_MAX_ABS_THRESHOLD", "0.01"))

FEATURE_MSE_THRESHOLD = float(os.getenv("LOLVLANCE_FEATURE_MSE_THRESHOLD", "0.0001"))
FEATURE_MAE_THRESHOLD = float(os.getenv("LOLVLANCE_FEATURE_MAE_THRESHOLD", "0.001"))
FEATURE_MEAN_DELTA_THRESHOLD = float(os.getenv("LOLVLANCE_FEATURE_MEAN_DELTA_THRESHOLD", "0.001"))
FEATURE_STD_DELTA_THRESHOLD = float(os.getenv("LOLVLANCE_FEATURE_STD_DELTA_THRESHOLD", "0.001"))

RESAMPLER_MEAN_DELTA_THRESHOLD = float(os.getenv("LOLVLANCE_RESAMPLER_MEAN_DELTA_THRESHOLD", "0.001"))
RESAMPLER_RMS_DELTA_THRESHOLD = float(os.getenv("LOLVLANCE_RESAMPLER_RMS_DELTA_THRESHOLD", "0.005"))
RESAMPLER_PEAK_DELTA_THRESHOLD = float(os.getenv("LOLVLANCE_RESAMPLER_PEAK_DELTA_THRESHOLD", "0.01"))

_SPLIT_TOKENS = {
    "train",
    "val",
    "valid",
    "validation",
    "test",
    "dev",
    "eval",
}


@dataclass(frozen=True)
class ErrorMetrics:
    mse: float
    mae: float
    max_abs: float
    mean_delta: float
    std_delta: float


def load_reference_waveform() -> np.ndarray:
    if not REFERENCE_AUDIO_PATH.exists():
        raise FileNotFoundError(f"Reference audio not found: {REFERENCE_AUDIO_PATH}")
    return load_audio_segment(REFERENCE_AUDIO_PATH, target_sample_rate=TARGET_SAMPLE_RATE)


def simulate_browser_capture(source_sample_rate: int, duration_seconds: float) -> np.ndarray:
    reference = load_reference_waveform()
    desired_length = max(1, int(round(TARGET_SAMPLE_RATE * duration_seconds)))
    padded_reference = pad_or_trim(reference, desired_length)
    return resample_audio(padded_reference, TARGET_SAMPLE_RATE, source_sample_rate)


def python_preprocessed_waveform(
    capture_waveform: np.ndarray,
    capture_sample_rate: int,
    *,
    clip_seconds: float = ROLLING_BUFFER_SECONDS,
) -> np.ndarray:
    target_length = int(round(TARGET_SAMPLE_RATE * clip_seconds))
    resampled = resample_audio(capture_waveform, capture_sample_rate, TARGET_SAMPLE_RATE)
    return pad_or_trim(resampled, target_length)


def python_model_input_features(
    capture_waveform: np.ndarray,
    capture_sample_rate: int,
    *,
    clip_seconds: float = ROLLING_BUFFER_SECONDS,
    mel_bin_count: int = MODEL_MEL_BIN_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    waveform = python_preprocessed_waveform(
        capture_waveform,
        capture_sample_rate,
        clip_seconds=clip_seconds,
    )
    features = waveform_to_log_mel_spectrogram(
        waveform,
        PreprocessingConfig(
            sample_rate=TARGET_SAMPLE_RATE,
            clip_seconds=clip_seconds,
            window_ms=DEFAULT_WINDOW_MS,
            hop_ms=DEFAULT_HOP_MS,
            fft_size=DEFAULT_FFT_SIZE,
            mel_bin_count=mel_bin_count,
        ),
    )
    return waveform, features


def browser_resample_mono_buffer(
    input_waveform: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    waveform = np.asarray(input_waveform, dtype=np.float32)

    if waveform.size == 0 or source_sample_rate <= 0:
        return np.zeros((0,), dtype=np.float32)

    if source_sample_rate == target_sample_rate:
        return waveform.copy()

    if source_sample_rate < target_sample_rate:
        ratio = source_sample_rate / target_sample_rate
        output_length = max(1, round(waveform.shape[0] / ratio))
        output = np.zeros((output_length,), dtype=np.float32)

        for index in range(output_length):
            position = index * ratio
            base_index = int(math.floor(position))
            next_index = min(base_index + 1, waveform.shape[0] - 1)
            fraction = position - base_index
            output[index] = waveform[base_index] + (waveform[next_index] - waveform[base_index]) * fraction

        return output

    ratio = source_sample_rate / target_sample_rate
    output_length = max(1, round(waveform.shape[0] / ratio))
    output = np.zeros((output_length,), dtype=np.float32)
    input_offset = 0

    for index in range(output_length):
        next_offset = min(waveform.shape[0], round((index + 1) * ratio))
        segment = waveform[input_offset:next_offset]
        output[index] = float(segment.mean()) if segment.size else float(waveform[min(input_offset, waveform.shape[0] - 1)])
        input_offset = next_offset

    return output


def browser_normalize_snapshot_to_fixed_length(
    samples: np.ndarray,
    target_length: int = TARGET_SAMPLE_RATE * ROLLING_BUFFER_SECONDS,
) -> np.ndarray:
    waveform = np.asarray(samples, dtype=np.float32)

    if waveform.shape[0] == target_length:
        return waveform.copy()

    if waveform.shape[0] > target_length:
        return waveform[-target_length:].copy()

    normalized = np.zeros((target_length,), dtype=np.float32)
    normalized[target_length - waveform.shape[0] :] = waveform
    return normalized


def browser_preprocessed_waveform(
    capture_waveform: np.ndarray,
    capture_sample_rate: int,
    *,
    clip_seconds: float = ROLLING_BUFFER_SECONDS,
) -> np.ndarray:
    target_length = int(round(TARGET_SAMPLE_RATE * clip_seconds))
    resampled = browser_resample_mono_buffer(capture_waveform, capture_sample_rate, TARGET_SAMPLE_RATE)
    return browser_normalize_snapshot_to_fixed_length(resampled, target_length)


def browser_model_input_features(
    capture_waveform: np.ndarray,
    capture_sample_rate: int,
    *,
    clip_seconds: float = ROLLING_BUFFER_SECONDS,
    mel_bin_count: int = MODEL_MEL_BIN_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    waveform = browser_preprocessed_waveform(
        capture_waveform,
        capture_sample_rate,
        clip_seconds=clip_seconds,
    )
    frame_size = max(1, round((DEFAULT_WINDOW_MS / 1000.0) * TARGET_SAMPLE_RATE))
    hop_size = max(1, round((DEFAULT_HOP_MS / 1000.0) * TARGET_SAMPLE_RATE))
    fft_size = max(next_power_of_two(frame_size), DEFAULT_FFT_SIZE)
    frame_count = math.floor((waveform.shape[0] - frame_size) / hop_size) + 1
    hann_window = browser_hann_window(frame_size)
    mel_filter_bank = browser_mel_filter_bank(TARGET_SAMPLE_RATE, fft_size, mel_bin_count)

    log_mel = np.zeros((frame_count, mel_bin_count), dtype=np.float32)

    for frame_index in range(frame_count):
        frame_offset = frame_index * hop_size
        fft_input = np.zeros((fft_size,), dtype=np.float32)
        fft_input[:frame_size] = waveform[frame_offset : frame_offset + frame_size] * hann_window
        power_spectrum = (np.abs(np.fft.rfft(fft_input, n=fft_size)) ** 2).astype(np.float32) / fft_input.shape[0]
        mel_energy = mel_filter_bank @ power_spectrum
        log_mel[frame_index] = np.log(np.maximum(1e-6, mel_energy))

    return waveform, log_mel


def browser_hann_window(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((max(1, length),), dtype=np.float32)

    indices = np.arange(length, dtype=np.float32)
    return (0.5 - 0.5 * np.cos((2 * np.pi * indices) / max(1, length - 1))).astype(np.float32)


def browser_mel_filter_bank(sample_rate: int, fft_size: int, mel_bin_count: int) -> np.ndarray:
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


def next_power_of_two(value: int) -> int:
    power = 1
    while power < value:
        power <<= 1
    return power


def hz_to_mel(frequency_hz: float) -> float:
    return 2595.0 * math.log10(1.0 + frequency_hz / 700.0)


def mel_to_hz(mel_value: np.ndarray | float) -> np.ndarray:
    mel_array = np.asarray(mel_value, dtype=np.float32)
    return 700.0 * (np.power(10.0, mel_array / 2595.0) - 1.0)


def compute_error_metrics(reference: np.ndarray, candidate: np.ndarray) -> ErrorMetrics:
    reference_array = np.asarray(reference, dtype=np.float32)
    candidate_array = np.asarray(candidate, dtype=np.float32)

    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            f"Shape mismatch while computing error metrics: {reference_array.shape} vs {candidate_array.shape}"
        )

    delta = reference_array - candidate_array
    return ErrorMetrics(
        mse=float(np.mean(np.square(delta))),
        mae=float(np.mean(np.abs(delta))),
        max_abs=float(np.max(np.abs(delta))),
        mean_delta=float(abs(reference_array.mean() - candidate_array.mean())),
        std_delta=float(abs(reference_array.std() - candidate_array.std())),
    )


def assert_all_finite(array: np.ndarray, *, context: str) -> None:
    if not np.isfinite(array).all():
        raise AssertionError(f"{context} contains NaN or Inf values.")


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(signal, dtype=np.float32)))))


def peak(signal: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(signal, dtype=np.float32))))


def generate_synthetic_capture(source_sample_rate: int, duration_seconds: float = ROLLING_BUFFER_SECONDS + 0.25) -> np.ndarray:
    sample_count = max(1, int(round(source_sample_rate * duration_seconds)))
    timeline = np.arange(sample_count, dtype=np.float32) / float(source_sample_rate)
    waveform = (
        0.35 * np.sin(2 * np.pi * 180.0 * timeline)
        + 0.22 * np.sin(2 * np.pi * 920.0 * timeline)
        + 0.12 * np.sin(2 * np.pi * 3400.0 * timeline)
        + 0.05 * np.sign(np.sin(2 * np.pi * 3.0 * timeline))
    )
    return np.clip(waveform, -1.0, 1.0).astype(np.float32)


def load_manifest_entries() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    return load_manifest(MANIFEST_PATH)


def split_manifest_entries(entries: Iterable[dict]) -> dict[str, list[dict]]:
    split_map: dict[str, list[dict]] = {}
    for entry in entries:
        split_map.setdefault(entry["split"], []).append(entry)
    return split_map


def derive_weak_collision_key(entry: dict) -> str:
    audio_path = Path(entry["audio_path"])
    stem = audio_path.stem.lower()
    if stem == "mix" and audio_path.parent.name.lower() == "stems" and audio_path.parent.parent.name:
        stem = audio_path.parent.parent.name.lower()
    elif stem == "mix" and audio_path.parent.name:
        stem = f"{audio_path.parent.name.lower()}_{stem}"

    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", stem)
        if token and token not in _SPLIT_TOKENS
    ]
    return "_".join(tokens)

