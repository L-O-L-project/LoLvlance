import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml.export_to_onnx import export_to_onnx
from ml.lightweight_audio_model import LightweightAudioAnalysisNet, PROBLEM_LABELS


class OnnxExportTest(unittest.TestCase):
    def test_export_and_runtime_match_pytorch_shapes_and_ranges(self) -> None:
        torch.manual_seed(11)
        model = LightweightAudioAnalysisNet().eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "dummy_checkpoint.pt"
            onnx_path = temp_path / "lightweight_audio_model.onnx"
            time_steps = 128

            torch.save(model.state_dict(), checkpoint_path)

            export_args = Namespace(
                checkpoint=checkpoint_path,
                output=onnx_path,
                time_steps=time_steps,
                opset=18,
                verify=True,
                mel_bins=64,
            )

            export_to_onnx(export_args)

            self.assertTrue(onnx_path.exists())

            input_tensor = torch.randn(1, time_steps, 64)
            with torch.no_grad():
                pytorch_output = model(input_tensor)

            session = ort.InferenceSession(
                onnx_path.as_posix(),
                providers=["CPUExecutionProvider"],
            )
            onnx_output = session.run(
                None,
                {"log_mel_spectrogram": input_tensor.cpu().numpy()},
            )

            self.assertEqual(onnx_output[0].shape, (1, len(PROBLEM_LABELS)))

            np.testing.assert_allclose(
                onnx_output[0],
                pytorch_output["problem_probs"].cpu().numpy(),
                rtol=1e-3,
                atol=1e-4,
            )

            self.assertTrue(np.all((onnx_output[0] >= 0) & (onnx_output[0] <= 1)))


if __name__ == "__main__":
    unittest.main()
