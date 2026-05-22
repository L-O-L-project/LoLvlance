from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.label_schema import ISSUE_LABELS, SOURCE_LABELS  # noqa: E402

EXPECTED_INPUT_NAME = "log_mel_spectrogram"
EXPECTED_OUTPUTS = {
    "issue_probs": (1, len(ISSUE_LABELS)),
    "source_probs": (1, len(SOURCE_LABELS)),
    "eq_freq": (1, 1),
    "eq_gain_db": (1, 1),
}
EXPECTED_MEL_BINS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the browser ONNX contract for a LoLvlance model.")
    parser.add_argument("--model-path", type=Path, default=Path("public/models/lightweight_audio_model.production.onnx"))
    parser.add_argument("--time-steps", type=int, default=298)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def validate_model(model_path: Path, time_steps: int) -> dict[str, Any]:
    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    input_by_name = {entry.name: entry for entry in inputs}
    output_by_name = {entry.name: entry for entry in outputs}

    if EXPECTED_INPUT_NAME not in input_by_name:
        raise RuntimeError(f"Missing input {EXPECTED_INPUT_NAME!r}; found {sorted(input_by_name)}")

    model_input = input_by_name[EXPECTED_INPUT_NAME]
    if model_input.type != "tensor(float)":
        raise RuntimeError(f"Expected input dtype tensor(float), received {model_input.type}.")
    if len(model_input.shape) != 3 or model_input.shape[2] != EXPECTED_MEL_BINS:
        raise RuntimeError(f"Expected input shape [batch, time, {EXPECTED_MEL_BINS}], received {model_input.shape}.")

    for output_name in EXPECTED_OUTPUTS:
        if output_name not in output_by_name:
            raise RuntimeError(f"Missing output {output_name!r}; found {sorted(output_by_name)}")
        if output_by_name[output_name].type != "tensor(float)":
            raise RuntimeError(f"Expected {output_name} dtype tensor(float), received {output_by_name[output_name].type}.")

    sample = np.zeros((1, time_steps, EXPECTED_MEL_BINS), dtype=np.float32)
    output_values = session.run(list(EXPECTED_OUTPUTS), {EXPECTED_INPUT_NAME: sample})
    runtime_shapes: dict[str, list[int]] = {}

    for output_name, value in zip(EXPECTED_OUTPUTS, output_values, strict=True):
        expected_shape = EXPECTED_OUTPUTS[output_name]
        observed = np.asarray(value)
        if tuple(observed.shape) != expected_shape:
            raise RuntimeError(f"Expected {output_name} runtime shape {expected_shape}, received {tuple(observed.shape)}.")
        if observed.dtype != np.float32:
            raise RuntimeError(f"Expected {output_name} runtime dtype float32, received {observed.dtype}.")
        if not np.isfinite(observed).all():
            raise RuntimeError(f"{output_name} contains non-finite values.")
        if output_name.endswith("_probs") and ((observed < -1e-4).any() or (observed > 1.0001).any()):
            raise RuntimeError(f"{output_name} contains values outside probability range [0, 1].")
        runtime_shapes[output_name] = list(observed.shape)

    return {
        "status": "ok",
        "model_path": model_path.as_posix(),
        "input": {
            "name": model_input.name,
            "type": model_input.type,
            "shape": list(model_input.shape),
        },
        "outputs": {
            output.name: {
                "type": output.type,
                "static_shape": list(output.shape),
                "runtime_shape": runtime_shapes[output.name],
            }
            for output in outputs
            if output.name in EXPECTED_OUTPUTS
        },
        "issue_labels": list(ISSUE_LABELS),
        "source_labels": list(SOURCE_LABELS),
    }


def main() -> int:
    args = parse_args()
    report = validate_model(args.model_path, args.time_steps)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"ONNX contract: {report['status']}")
        print(f"model: {report['model_path']}")
        print(f"input: {report['input']}")
        for name, output in report["outputs"].items():
            print(f"output {name}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
