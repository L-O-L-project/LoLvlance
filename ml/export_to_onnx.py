from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional unless verify is enabled
    ort = None

try:
    from .model import LightweightAudioAnalysisNet, ModelConfig
except ImportError:
    from model import LightweightAudioAnalysisNet, ModelConfig


class OnnxExportWrapper(torch.nn.Module):
    def __init__(self, model: LightweightAudioAnalysisNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["problem_probs"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the LoLvlance lightweight issue classifier to ONNX."
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
        default=298,
        help="Example time dimension for export. The ONNX graph keeps it dynamic.",
    )
    parser.add_argument(
        "--mel-bins",
        type=int,
        default=64,
        help="Input mel-bin dimension. Must stay aligned with the frontend.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a post-export onnxruntime check and verify output shape is (batch, 3).",
    )
    return parser.parse_args()


def export_to_onnx(args: argparse.Namespace) -> Path:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(checkpoint=checkpoint, mel_bins=args.mel_bins)
    load_checkpoint(model, checkpoint)
    model.eval().cpu()

    export_model = OnnxExportWrapper(model).eval().cpu()
    example_input = torch.randn(1, args.time_steps, args.mel_bins, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_model,
        example_input,
        args.output.as_posix(),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        dynamo=False,
        input_names=["log_mel_spectrogram"],
        output_names=["problem_probs"],
        dynamic_axes={
            "log_mel_spectrogram": {0: "batch_size", 1: "time_steps"},
            "problem_probs": {0: "batch_size"},
        },
    )

    if args.verify:
        verify_export(args.output, export_model, example_input)

    return args.output


def build_model(checkpoint: Any, mel_bins: int) -> LightweightAudioAnalysisNet:
    config_payload = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    config = ModelConfig.from_dict(config_payload)
    config.mel_bins = mel_bins
    return LightweightAudioAnalysisNet(config)


def load_checkpoint(model: torch.nn.Module, checkpoint: Any) -> None:
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
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def verify_export(
    onnx_path: Path,
    export_model: torch.nn.Module,
    example_input: torch.Tensor,
) -> None:
    if ort is None:
        raise RuntimeError("onnxruntime is required for --verify but is not installed.")

    session = ort.InferenceSession(
        onnx_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    outputs = session.run(None, {"log_mel_spectrogram": example_input.numpy()})

    if len(outputs) != 1:
        raise RuntimeError(f"Expected one ONNX output tensor, received {len(outputs)}.")

    if outputs[0].shape != (1, 3):
        raise RuntimeError(f"Expected ONNX output shape (1, 3), received {outputs[0].shape}.")

    with torch.no_grad():
        reference_output = export_model(example_input).cpu().numpy()

    np.testing.assert_allclose(outputs[0], reference_output, rtol=1e-3, atol=1e-4)


def main() -> None:
    args = parse_args()
    output_path = export_to_onnx(args)
    print(f"Exported ONNX model to: {output_path}")


if __name__ == "__main__":
    main()
