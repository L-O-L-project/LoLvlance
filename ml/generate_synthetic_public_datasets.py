from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic fallback dataset laid out like the public LoLvlance training sources."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ml/artifacts/synthetic_public_datasets"),
        help="Directory where the synthetic dataset tree will be created.",
    )
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Waveform sample rate.")
    parser.add_argument("--duration-seconds", type=float, default=3.2, help="Duration for each synthetic clip.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = generate_synthetic_public_datasets(
        output_root=args.output_root,
        sample_rate=args.sample_rate,
        duration_seconds=args.duration_seconds,
    )
    print(json.dumps({name: path.as_posix() for name, path in roots.items()}, indent=2))


def generate_synthetic_public_datasets(
    *,
    output_root: Path,
    sample_rate: int,
    duration_seconds: float,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    openmic_root = output_root / "openmic"
    slakh_root = output_root / "slakh"
    musan_root = output_root / "musan"
    fsd50k_root = output_root / "fsd50k"

    build_openmic_dataset(openmic_root, sample_rate, duration_seconds)
    build_slakh_dataset(slakh_root, sample_rate, duration_seconds)
    build_musan_dataset(musan_root, sample_rate, duration_seconds)
    build_fsd50k_dataset(fsd50k_root, sample_rate, duration_seconds)

    return {
        "openmic": openmic_root,
        "slakh": slakh_root,
        "musan": musan_root,
        "fsd50k": fsd50k_root,
    }


def build_openmic_dataset(root: Path, sample_rate: int, duration_seconds: float) -> None:
    clips = {
        "train": [
            ("buried_vocal_train", [180.0, 260.0, 340.0], 0.004, {"singer": "1", "guitar": "0"}),
            ("nasal_vocal_train", [900.0, 1100.0, 1300.0], 0.003, {"singer": "1", "guitar": "0"}),
            ("sibilant_vocal_train", [6200.0, 7000.0], 0.01, {"singer": "1", "guitar": "0"}),
            ("guitar_presence_train", [1700.0, 2600.0, 3400.0], 0.002, {"singer": "0", "guitar": "1"}),
        ],
        "validation": [
            ("buried_vocal_val", [200.0, 280.0, 360.0], 0.004, {"singer": "1", "guitar": "0"}),
            ("nasal_vocal_val", [850.0, 1020.0, 1180.0], 0.003, {"singer": "1", "guitar": "0"}),
            ("sibilant_vocal_val", [5800.0, 6800.0], 0.01, {"singer": "1", "guitar": "0"}),
            ("guitar_presence_val", [1600.0, 2450.0, 3200.0], 0.002, {"singer": "0", "guitar": "1"}),
        ],
    }

    rows: list[dict[str, str]] = []
    for split, examples in clips.items():
        for clip_id, frequencies, noise, labels in examples:
            write_wave(
                root / split / "audio" / f"{clip_id}.wav",
                frequencies,
                sample_rate,
                duration_seconds,
                noise=noise,
                tremolo_rate_hz=2.0 if labels["singer"] == "1" else 0.0,
            )
            rows.append({"sample_key": clip_id, **labels})

    with (root / "openmic_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_key", "singer", "guitar"])
        writer.writeheader()
        writer.writerows(rows)


def build_slakh_dataset(root: Path, sample_rate: int, duration_seconds: float) -> None:
    tracks = {
        "train": {
            "Track000": {
                "bass_stem": [90.0, 140.0],
                "guitar_stem": [320.0, 420.0],
                "keys_stem": [280.0, 360.0],
            },
            "Track001": {
                "bass_stem": [80.0, 120.0],
                "drums_stem": [70.0, 180.0, 3200.0],
                "guitar_stem": [450.0, 620.0],
            },
        },
        "validation": {
            "Track002": {
                "bass_stem": [100.0, 150.0],
                "guitar_stem": [300.0, 410.0],
                "keys_stem": [260.0, 340.0],
            },
            "Track003": {
                "bass_stem": [85.0, 130.0],
                "drums_stem": [75.0, 210.0, 3400.0],
                "guitar_stem": [480.0, 680.0],
            },
        },
    }

    for split, split_tracks in tracks.items():
        for track_name, stems in split_tracks.items():
            mix_frequencies: list[float] = []
            for stem_name, frequencies in stems.items():
                write_wave(
                    root / split / track_name / "stems" / f"{stem_name}.wav",
                    frequencies,
                    sample_rate,
                    duration_seconds,
                    noise=0.002,
                )
                mix_frequencies.extend(frequencies)

            write_wave(
                root / split / track_name / "mix.wav",
                mix_frequencies,
                sample_rate,
                duration_seconds,
                noise=0.003,
            )


def build_musan_dataset(root: Path, sample_rate: int, duration_seconds: float) -> None:
    clips = {
        "train": [
            ("harsh_noise_train", [5200.0, 6400.0, 7600.0], 0.02),
            ("thin_texture_train", [2000.0, 3200.0, 4200.0], 0.006),
        ],
        "validation": [
            ("harsh_noise_val", [5000.0, 6200.0, 7200.0], 0.02),
            ("thin_texture_val", [1900.0, 3000.0, 3900.0], 0.006),
        ],
    }

    for split, examples in clips.items():
        for clip_id, frequencies, noise in examples:
            write_wave(
                root / split / "noise" / f"{clip_id}.wav",
                frequencies,
                sample_rate,
                duration_seconds,
                noise=noise,
            )


def build_fsd50k_dataset(root: Path, sample_rate: int, duration_seconds: float) -> None:
    clips = {
        "train": [
            ("dull_piano_train", [450.0, 700.0, 950.0], 0.002, "piano"),
            ("boxy_piano_train", [350.0, 500.0, 750.0], 0.002, "piano"),
            ("thin_synth_train", [2200.0, 3200.0, 4200.0], 0.003, "synth"),
        ],
        "validation": [
            ("dull_piano_val", [420.0, 650.0, 900.0], 0.002, "piano"),
            ("boxy_piano_val", [330.0, 520.0, 780.0], 0.002, "piano"),
            ("thin_synth_val", [2100.0, 3050.0, 4000.0], 0.003, "synth"),
        ],
    }

    annotation_rows: list[dict[str, str]] = []
    for split, examples in clips.items():
        for clip_id, frequencies, noise, labels in examples:
            write_wave(
                root / split / f"{clip_id}.wav",
                frequencies,
                sample_rate,
                duration_seconds,
                noise=noise,
            )
            annotation_rows.append({"fname": clip_id, "labels": labels})

    with (root / "annotations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fname", "labels"])
        writer.writeheader()
        writer.writerows(annotation_rows)


def write_wave(
    path: Path,
    frequencies_hz: list[float],
    sample_rate: int,
    duration_seconds: float,
    *,
    noise: float = 0.0,
    tremolo_rate_hz: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timeline = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    waveform = np.zeros_like(timeline, dtype=np.float32)

    for index, frequency_hz in enumerate(frequencies_hz):
        amplitude = 1.0 / max(1, index + 1)
        waveform += amplitude * np.sin((2.0 * np.pi * frequency_hz * timeline) + (index * 0.37)).astype(np.float32)

    if tremolo_rate_hz > 0.0:
        tremolo = 0.65 + 0.35 * np.sin(2.0 * np.pi * tremolo_rate_hz * timeline)
        waveform *= tremolo.astype(np.float32)

    if noise > 0.0:
        rng = np.random.default_rng(abs(hash(path.stem)) % (2**32))
        waveform += (noise * rng.standard_normal(waveform.shape[0])).astype(np.float32)

    fade_length = max(32, int(sample_rate * 0.02))
    fade_in = np.linspace(0.0, 1.0, fade_length, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_length, dtype=np.float32)
    waveform[:fade_length] *= fade_in
    waveform[-fade_length:] *= fade_out

    peak = float(np.max(np.abs(waveform))) or 1.0
    normalized = 0.45 * waveform / peak
    sf.write(path.as_posix(), normalized, sample_rate)


if __name__ == "__main__":
    main()
