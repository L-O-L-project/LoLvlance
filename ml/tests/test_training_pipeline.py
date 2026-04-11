import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from ml.dataset import LoLvlanceAudioDataset, load_manifest
from ml.export_to_onnx import export_to_onnx
from ml.train import run_training


class TrainingPipelineTest(unittest.TestCase):
    def test_end_to_end_training_manifest_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_roots = create_fake_public_datasets(root)
            manifest_path = root / "artifacts" / "manifest.jsonl"
            checkpoint_dir = root / "checkpoints"

            train_args = Namespace(
                openmic_root=dataset_roots["openmic"],
                slakh_root=dataset_roots["slakh"],
                musan_root=dataset_roots["musan"],
                fsd50k_root=dataset_roots["fsd50k"],
                manifest_path=manifest_path,
                rebuild_manifest=True,
                clips_per_file=1,
                max_files_per_dataset=None,
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                weight_decay=1e-4,
                dropout=0.1,
                num_workers=0,
                seed=7,
                checkpoint_dir=checkpoint_dir,
                device="cpu",
                export_onnx=False,
                onnx_output=root / "model.onnx",
            )

            best_checkpoint_path, summary = run_training(train_args)

            self.assertTrue(best_checkpoint_path.exists())
            self.assertEqual(summary["device"], "cpu")

            manifest_entries = load_manifest(manifest_path)
            self.assertGreaterEqual(len(manifest_entries), 6)

            val_dataset = LoLvlanceAudioDataset(manifest_path=manifest_path, split="val")
            sample = val_dataset[0]["log_mel_spectrogram"].unsqueeze(0).numpy()

            onnx_path = root / "exported_model.onnx"
            export_args = Namespace(
                checkpoint=best_checkpoint_path,
                output=onnx_path,
                time_steps=sample.shape[1],
                mel_bins=sample.shape[2],
                opset=18,
                verify=True,
            )
            export_to_onnx(export_args)

            self.assertTrue(onnx_path.exists())

            session = ort.InferenceSession(
                onnx_path.as_posix(),
                providers=["CPUExecutionProvider"],
            )
            outputs = session.run(None, {"log_mel_spectrogram": sample})
            self.assertEqual(outputs[0].shape, (1, 3))


def create_fake_public_datasets(root: Path) -> dict[str, Path]:
    sample_rate = 16_000
    duration_seconds = 4.0

    openmic_root = root / "openmic"
    slakh_root = root / "slakh"
    musan_root = root / "musan"
    fsd50k_root = root / "fsd50k"

    write_wave(openmic_root / "train" / "audio" / "buried_train.wav", [220.0, 330.0], sample_rate, duration_seconds)
    write_wave(openmic_root / "validation" / "audio" / "buried_val.wav", [240.0, 360.0], sample_rate, duration_seconds)
    with (openmic_root / "openmic_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_key", "singer"])
        writer.writeheader()
        writer.writerow({"sample_key": "buried_train", "singer": "1"})
        writer.writerow({"sample_key": "buried_val", "singer": "1"})

    slakh_train = slakh_root / "train" / "Track000"
    slakh_val = slakh_root / "validation" / "Track001"
    write_wave(slakh_train / "stems" / "stem_01.wav", [260.0], sample_rate, duration_seconds)
    write_wave(slakh_train / "stems" / "stem_02.wav", [420.0], sample_rate, duration_seconds)
    write_wave(slakh_train / "mix.wav", [260.0, 420.0], sample_rate, duration_seconds)
    write_wave(slakh_val / "stems" / "stem_01.wav", [280.0], sample_rate, duration_seconds)
    write_wave(slakh_val / "stems" / "stem_02.wav", [440.0], sample_rate, duration_seconds)
    write_wave(slakh_val / "mix.wav", [280.0, 440.0], sample_rate, duration_seconds)

    write_wave(musan_root / "train" / "noise" / "harsh_noise_train.wav", [5_500.0, 6_800.0], sample_rate, duration_seconds, noise=0.01)
    write_wave(musan_root / "validation" / "noise" / "harsh_noise_val.wav", [5_200.0, 6_400.0], sample_rate, duration_seconds, noise=0.01)

    write_wave(fsd50k_root / "train" / "neutral_train.wav", [1_100.0], sample_rate, duration_seconds)
    write_wave(fsd50k_root / "validation" / "neutral_val.wav", [1_300.0], sample_rate, duration_seconds)

    return {
        "openmic": openmic_root,
        "slakh": slakh_root,
        "musan": musan_root,
        "fsd50k": fsd50k_root,
    }


def write_wave(
    path: Path,
    frequencies_hz: list[float],
    sample_rate: int,
    duration_seconds: float,
    noise: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timeline = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    waveform = np.zeros_like(timeline, dtype=np.float32)

    for frequency_hz in frequencies_hz:
        waveform += np.sin(2.0 * np.pi * frequency_hz * timeline).astype(np.float32)

    if noise > 0:
        rng = np.random.default_rng(7)
        waveform += (noise * rng.standard_normal(waveform.shape[0])).astype(np.float32)

    peak = float(np.max(np.abs(waveform))) or 1.0
    normalized = 0.4 * waveform / peak
    sf.write(path.as_posix(), normalized, sample_rate)


if __name__ == "__main__":
    unittest.main()
