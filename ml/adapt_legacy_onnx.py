from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

try:
    from .onnx_schema_adapter import adapt_legacy_browser_onnx_to_hierarchical_schema
except ImportError:
    from onnx_schema_adapter import adapt_legacy_browser_onnx_to_hierarchical_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrap a legacy browser ONNX model with the hierarchical LoLvlance output schema."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to the legacy ONNX model.")
    parser.add_argument("--output", type=Path, required=True, help="Path for the adapted ONNX model.")
    parser.add_argument("--opset", type=int, default=18, help="Adapter opset version.")
    parser.add_argument("--verify", action="store_true", help="Validate the adapted ONNX with onnxruntime.")
    return parser.parse_args()


def verify_adapted_model(onnx_path: Path) -> None:
    if ort is None:
        raise RuntimeError("onnxruntime is required for --verify but is not installed.")

    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    time_steps = 128 if not isinstance(input_shape[1], int) else max(32, input_shape[1])
    mel_bins = 64 if not isinstance(input_shape[2], int) else input_shape[2]
    sample = np.random.standard_normal((1, time_steps, mel_bins)).astype(np.float32)
    issue_probs, source_probs, eq_freq, eq_gain_db = session.run(None, {"log_mel_spectrogram": sample})

    if issue_probs.shape != (1, 9):
        raise RuntimeError(f"Expected issue_probs shape (1, 9), received {issue_probs.shape}.")
    if source_probs.shape[0] != 1 or source_probs.shape[1] < 1:
        raise RuntimeError(f"Expected source_probs shape (1, N), received {source_probs.shape}.")
    if eq_freq.shape != (1, 1):
        raise RuntimeError(f"Expected eq_freq shape (1, 1), received {eq_freq.shape}.")
    if eq_gain_db.shape != (1, 1):
        raise RuntimeError(f"Expected eq_gain_db shape (1, 1), received {eq_gain_db.shape}.")
    if not np.isfinite(issue_probs).all():
        raise RuntimeError("issue_probs contains non-finite values.")
    if not np.isfinite(source_probs).all():
        raise RuntimeError("source_probs contains non-finite values.")
    if not np.isfinite(eq_freq).all():
        raise RuntimeError("eq_freq contains non-finite values.")
    if not np.isfinite(eq_gain_db).all():
        raise RuntimeError("eq_gain_db contains non-finite values.")


def main() -> None:
    args = parse_args()
    output_path = adapt_legacy_browser_onnx_to_hierarchical_schema(
        legacy_onnx_path=args.input,
        output_path=args.output,
        opset_version=args.opset,
    )

    if args.verify:
        verify_adapted_model(output_path)

    print(f"Adapted ONNX model written to: {output_path}")


if __name__ == "__main__":
    main()
