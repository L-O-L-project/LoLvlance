from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

try:
    from .label_schema import get_label_schema
    from .model import LightweightAudioAnalysisNet, ModelConfig
    from .onnx_schema_adapter import HierarchicalEqProjection
except ImportError:
    from label_schema import get_label_schema
    from model import LightweightAudioAnalysisNet, ModelConfig
    from onnx_schema_adapter import HierarchicalEqProjection


class OnnxExportWrapper(torch.nn.Module):
    def __init__(self, model: LightweightAudioAnalysisNet) -> None:
        super().__init__()
        self.model = model
        self.eq_projection = HierarchicalEqProjection()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(x)
        eq_freq, eq_gain_db = self.eq_projection(outputs["issue_probs"], outputs["source_probs"])
        return outputs["issue_probs"], outputs["source_probs"], eq_freq, eq_gain_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the hierarchical LoLvlance model to ONNX."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a trained checkpoint.")
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
        help="Example time dimension used during export. The ONNX graph keeps it dynamic.",
    )
    parser.add_argument("--mel-bins", type=int, default=64, help="Input mel-bin dimension.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a post-export onnxruntime verification.",
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
        output_names=["issue_probs", "source_probs", "eq_freq", "eq_gain_db"],
        dynamic_axes={
            "log_mel_spectrogram": {0: "batch_size", 1: "time_steps"},
            "issue_probs": {0: "batch_size"},
            "source_probs": {0: "batch_size"},
            "eq_freq": {0: "batch_size"},
            "eq_gain_db": {0: "batch_size"},
        },
    )

    export_metadata(args.output, checkpoint)

    if args.verify:
        verify_export(args.output, export_model, example_input)

    return args.output


def export_metadata(onnx_path: Path, checkpoint: Any) -> None:
    threshold_payload = checkpoint.get("thresholds", {}) if isinstance(checkpoint, dict) else {}
    schema = get_label_schema(
        issue_thresholds=threshold_payload.get("issue_thresholds"),
        source_thresholds=threshold_payload.get("source_thresholds"),
    )
    metadata_path = onnx_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(schema.to_dict(), indent=2), encoding="utf-8")


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
        "Unsupported checkpoint format. Expected a raw state_dict or a dict with 'state_dict' / 'model_state_dict'."
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

    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    issue_probs, source_probs, eq_freq, eq_gain_db = session.run(None, {"log_mel_spectrogram": example_input.numpy()})

    if issue_probs.shape[1] != 9:
        raise RuntimeError(f"Expected issue_probs shape (batch, 9), received {issue_probs.shape}.")
    if source_probs.shape[1] != 5:
        raise RuntimeError(f"Expected source_probs shape (batch, 5), received {source_probs.shape}.")
    if eq_freq.shape[1] != 1:
        raise RuntimeError(f"Expected eq_freq shape (batch, 1), received {eq_freq.shape}.")
    if eq_gain_db.shape[1] != 1:
        raise RuntimeError(f"Expected eq_gain_db shape (batch, 1), received {eq_gain_db.shape}.")
    if not np.isfinite(eq_freq).all():
        raise RuntimeError("eq_freq contains non-finite values.")
    if not np.isfinite(eq_gain_db).all():
        raise RuntimeError("eq_gain_db contains non-finite values.")

    with torch.no_grad():
        reference_issue_probs, reference_source_probs, reference_eq_freq, reference_eq_gain_db = export_model(example_input)

    np.testing.assert_allclose(issue_probs, reference_issue_probs.cpu().numpy(), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(source_probs, reference_source_probs.cpu().numpy(), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(eq_freq, reference_eq_freq.cpu().numpy(), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(eq_gain_db, reference_eq_gain_db.cpu().numpy(), rtol=1e-3, atol=1e-4)


def main() -> None:
    args = parse_args()
    output_path = export_to_onnx(args)
    print(f"Exported ONNX model to: {output_path}")


if __name__ == "__main__":
    main()
