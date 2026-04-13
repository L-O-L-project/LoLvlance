import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml.export_to_onnx import export_to_onnx
from ml.lightweight_audio_model import ISSUE_LABELS, SOURCE_LABELS, LightweightAudioAnalysisNet


class OnnxExportTest(unittest.TestCase):
    def test_export_and_runtime_match_pytorch_shapes_and_ranges(self) -> None:
        torch.manual_seed(11)
        model = LightweightAudioAnalysisNet().eval()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "dummy_checkpoint.pt"
            onnx_path = temp_path / "hierarchical_audio_model.onnx"
            metadata_path = temp_path / "hierarchical_audio_model.metadata.json"
            time_steps = 128

            torch.save({"state_dict": model.state_dict(), "config": model.config.to_dict()}, checkpoint_path)

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
            self.assertTrue(metadata_path.exists())

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["issue_labels"], list(ISSUE_LABELS))
            self.assertEqual(metadata["source_labels"], list(SOURCE_LABELS))

            input_tensor = torch.randn(1, time_steps, 64)
            with torch.no_grad():
                pytorch_output = model(input_tensor)

            session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
            issue_probs, source_probs, eq_freq, eq_gain_db = session.run(
                None,
                {"log_mel_spectrogram": input_tensor.cpu().numpy()},
            )

            self.assertEqual(issue_probs.shape, (1, len(ISSUE_LABELS)))
            self.assertEqual(source_probs.shape, (1, len(SOURCE_LABELS)))
            self.assertEqual(eq_freq.shape, (1, 1))
            self.assertEqual(eq_gain_db.shape, (1, 1))
            self.assertTrue(np.isfinite(eq_freq).all())
            self.assertTrue(np.isfinite(eq_gain_db).all())
            np.testing.assert_allclose(
                issue_probs,
                pytorch_output["issue_probs"].cpu().numpy(),
                rtol=1e-3,
                atol=1e-4,
            )
            np.testing.assert_allclose(
                source_probs,
                pytorch_output["source_probs"].cpu().numpy(),
                rtol=1e-3,
                atol=1e-4,
            )


if __name__ == "__main__":
    unittest.main()
