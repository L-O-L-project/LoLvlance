import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml.onnx_schema_adapter import (
    LegacyHierarchicalOutputAdapter,
    adapt_legacy_browser_onnx_to_hierarchical_schema,
)


class DummyLegacyBrowserModel(torch.nn.Module):
    def forward(
        self,
        log_mel_spectrogram: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = log_mel_spectrogram.shape[0]
        mean_activation = log_mel_spectrogram.mean(dim=(1, 2), keepdim=True)
        positive_drive = torch.sigmoid(mean_activation)
        negative_drive = torch.sigmoid(-mean_activation)
        energy_drive = torch.sigmoid(log_mel_spectrogram.square().mean(dim=(1, 2), keepdim=True))
        problem_logits = torch.cat(
            [
                positive_drive,
                negative_drive,
                0.7 * energy_drive,
                torch.full_like(mean_activation, 0.15),
            ],
            dim=-1,
        )
        problem_probs = torch.softmax(problem_logits.squeeze(1), dim=-1)
        source_probs = torch.sigmoid(
            torch.cat(
                [
                    positive_drive,
                    energy_drive,
                    negative_drive,
                    0.5 * positive_drive + 0.5 * negative_drive,
                    0.7 * energy_drive + 0.1,
                ],
                dim=-1,
            ).squeeze(1)
        )
        eq_freq = positive_drive.squeeze(1)
        eq_gain_db = (positive_drive - negative_drive).squeeze(1) * 6.0
        return problem_probs, source_probs, eq_freq, eq_gain_db


class LegacyOnnxAdapterTest(unittest.TestCase):
    def test_adapted_model_exposes_hierarchical_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy_model_path = root / "legacy_browser_model.onnx"
            adapted_model_path = root / "hierarchical_browser_model.onnx"

            dummy_model = DummyLegacyBrowserModel().eval()
            dummy_input = torch.randn(1, 128, 64)
            torch.onnx.export(
                dummy_model,
                dummy_input,
                legacy_model_path.as_posix(),
                export_params=True,
                do_constant_folding=True,
                opset_version=18,
                dynamo=False,
                input_names=["log_mel_spectrogram"],
                output_names=["problem_probs", "instrument_probs", "eq_freq", "eq_gain_db"],
                dynamic_axes={
                    "log_mel_spectrogram": {0: "batch_size", 1: "time_steps"},
                    "problem_probs": {0: "batch_size"},
                    "instrument_probs": {0: "batch_size"},
                    "eq_freq": {0: "batch_size"},
                    "eq_gain_db": {0: "batch_size"},
                },
            )

            adapt_legacy_browser_onnx_to_hierarchical_schema(
                legacy_onnx_path=legacy_model_path,
                output_path=adapted_model_path,
            )

            legacy_session = ort.InferenceSession(legacy_model_path.as_posix(), providers=["CPUExecutionProvider"])
            adapted_session = ort.InferenceSession(adapted_model_path.as_posix(), providers=["CPUExecutionProvider"])
            sample = np.random.standard_normal((1, 128, 64)).astype(np.float32)
            problem_probs, instrument_probs, eq_freq, eq_gain_db = legacy_session.run(
                None,
                {"log_mel_spectrogram": sample},
            )
            issue_probs, source_probs, adapted_eq_freq, adapted_eq_gain_db = adapted_session.run(
                None,
                {"log_mel_spectrogram": sample},
            )

            adapter = LegacyHierarchicalOutputAdapter().eval()
            with torch.no_grad():
                expected_issue_probs, expected_source_probs, expected_eq_freq, expected_eq_gain_db = adapter(
                    torch.from_numpy(problem_probs),
                    torch.from_numpy(instrument_probs),
                    torch.from_numpy(eq_freq),
                    torch.from_numpy(eq_gain_db),
                )

            self.assertEqual(issue_probs.shape, (1, 9))
            self.assertEqual(source_probs.shape, (1, 5))
            self.assertEqual(adapted_eq_freq.shape, (1, 1))
            self.assertEqual(adapted_eq_gain_db.shape, (1, 1))
            self.assertTrue(np.isfinite(issue_probs).all())
            self.assertTrue(np.isfinite(source_probs).all())
            self.assertTrue(np.isfinite(adapted_eq_freq).all())
            self.assertTrue(np.isfinite(adapted_eq_gain_db).all())
            np.testing.assert_allclose(
                issue_probs,
                expected_issue_probs.cpu().numpy(),
                rtol=1e-3,
                atol=1e-4,
            )
            np.testing.assert_allclose(
                source_probs,
                expected_source_probs.cpu().numpy(),
                rtol=1e-3,
                atol=1e-4,
            )
            np.testing.assert_allclose(
                adapted_eq_freq,
                expected_eq_freq.cpu().numpy(),
                rtol=1e-3,
                atol=1e-4,
            )
            np.testing.assert_allclose(
                adapted_eq_gain_db,
                expected_eq_gain_db.cpu().numpy(),
                rtol=1e-3,
                atol=1e-4,
            )
            del legacy_session
            del adapted_session


if __name__ == "__main__":
    unittest.main()
