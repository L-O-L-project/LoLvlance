from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

try:
    from .label_schema import ISSUE_LABELS, SOURCE_LABELS
    from .model import load_model_from_checkpoint
    from .preprocessing import (
        DEFAULT_MEL_BIN_COUNT,
        PreprocessingConfig,
        extract_audio_features_from_path,
        extract_audio_features_from_waveform,
    )
except ImportError:
    from label_schema import ISSUE_LABELS, SOURCE_LABELS
    from model import load_model_from_checkpoint
    from preprocessing import (
        DEFAULT_MEL_BIN_COUNT,
        PreprocessingConfig,
        extract_audio_features_from_path,
        extract_audio_features_from_waveform,
    )


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(normalized)


class AudioIntelligenceInference:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        device: str = "auto",
        mel_bins: int = DEFAULT_MEL_BIN_COUNT,
    ) -> None:
        self.device = resolve_device(device)
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model = load_model_from_checkpoint(self.checkpoint, mel_bins=mel_bins).to(self.device).eval()
        self.preprocessing_config = PreprocessingConfig(mel_bin_count=mel_bins)

    def predict(self, audio: str | Path | Any) -> dict[str, object]:
        if isinstance(audio, (str, Path)):
            features = extract_audio_features_from_path(audio, config=self.preprocessing_config)
        else:
            features = extract_audio_features_from_waveform(audio, config=self.preprocessing_config)

        inputs = torch.from_numpy(features.log_mel_spectrogram).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)

        issue_probs = outputs["issue_probs"][0].detach().cpu().tolist()
        source_probs = outputs["source_probs"][0].detach().cpu().tolist()
        eq_params = outputs["eq_params"][0].detach().cpu().tolist()

        return {
            "issue_probs": {
                label: round(float(score), 6)
                for label, score in zip(ISSUE_LABELS, issue_probs, strict=True)
            },
            "source_probs": {
                label: round(float(score), 6)
                for label, score in zip(SOURCE_LABELS, source_probs, strict=True)
            },
            "eq_params": [
                [round(float(freq_hz), 2), round(float(gain_db), 2)]
                for freq_hz, gain_db in eq_params
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run checkpoint inference for the LoLvlance production model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--audio", type=Path, required=True, help="Path to the input audio file.")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, mps, or auto.")
    parser.add_argument("--mel-bins", type=int, default=DEFAULT_MEL_BIN_COUNT, help="Model mel-bin count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = AudioIntelligenceInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        mel_bins=args.mel_bins,
    )
    prediction = runtime.predict(args.audio)
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
