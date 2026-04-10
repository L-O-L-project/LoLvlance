from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from lightweight_audio_model import LightweightAudioAnalysisNet, ModelConfig


class OnnxExportWrapper(torch.nn.Module):
    """
    ONNX-friendly wrapper that exports tensor outputs instead of a Python dict.
    """

    def __init__(self, model: LightweightAudioAnalysisNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.model(x)
        return (
            outputs["problem_probs"],
            outputs["instrument_probs"],
            outputs["eq_freq"],
            outputs["eq_gain_db"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the lightweight audio analysis model to ONNX for onnxruntime-web."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a trained PyTorch checkpoint (.pt or .pth).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/lightweight_audio_model.onnx"),
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=128,
        help="Example input sequence length T used during export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version. 17 is a safe target for onnxruntime-web.",
    )
    parser.add_argument(
        "--disable-transformer",
        action="store_true",
        help="Instantiate the model without transformer layers.",
    )
    parser.add_argument(
        "--encoder-dim",
        type=int,
        default=128,
        help="Encoder width used when reconstructing the model config.",
    )
    parser.add_argument(
        "--transformer-layers",
        type=int,
        default=2,
        help="Transformer layer count used when reconstructing the model config.",
    )
    parser.add_argument(
        "--transformer-heads",
        type=int,
        default=4,
        help="Transformer attention head count.",
    )
    parser.add_argument(
        "--transformer-ffn-dim",
        type=int,
        default=256,
        help="Transformer feed-forward width.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout used when reconstructing the model config.",
    )
    parser.add_argument(
        "--mel-bins",
        type=int,
        default=64,
        help="Input mel-bin dimension. Expected to stay at 64 for the current frontend.",
    )
    return parser.parse_args()


def build_model_from_args(args: argparse.Namespace) -> LightweightAudioAnalysisNet:
    config = ModelConfig(
        mel_bins=args.mel_bins,
        encoder_dim=args.encoder_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ffn_dim=args.transformer_ffn_dim,
        dropout=args.dropout,
        use_transformer=not args.disable_transformer,
    )
    return LightweightAudioAnalysisNet(config)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        raise RuntimeError(f"Missing keys while loading checkpoint: {missing_keys}")

    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading checkpoint: {unexpected_keys}")


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return normalize_state_dict_keys(checkpoint["state_dict"])

        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            return normalize_state_dict_keys(checkpoint["model_state_dict"])

        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return normalize_state_dict_keys(checkpoint)

    raise ValueError(
        "Unsupported checkpoint format. Expected a raw state_dict or a dict with "
        "'state_dict' / 'model_state_dict'."
    )


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        normalized[key.removeprefix("module.")] = value

    return normalized


def export_to_onnx(args: argparse.Namespace) -> Path:
    model = build_model_from_args(args)
    load_checkpoint(model, args.checkpoint)
    model.eval().cpu()

    export_model = OnnxExportWrapper(model).eval().cpu()
    example_input = torch.randn(1, args.time_steps, args.mel_bins, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Example input tensor for export and quick validation.
    print(f"Example input tensor shape: {tuple(example_input.shape)}")

    torch.onnx.export(
        export_model,
        example_input,
        args.output.as_posix(),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["log_mel_spectrogram"],
        output_names=[
            "problem_probs",
            "instrument_probs",
            "eq_freq",
            "eq_gain_db",
        ],
        dynamic_axes={
            "log_mel_spectrogram": {
                0: "batch_size",
                1: "time_steps",
            },
            "problem_probs": {0: "batch_size"},
            "instrument_probs": {0: "batch_size"},
            "eq_freq": {0: "batch_size"},
            "eq_gain_db": {0: "batch_size"},
        },
    )

    return args.output


def main() -> None:
    args = parse_args()
    output_path = export_to_onnx(args)
    print(f"Exported ONNX model to: {output_path}")


if __name__ == "__main__":
    main()
