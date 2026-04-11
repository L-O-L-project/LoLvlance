import unittest

import torch

from ml.lightweight_audio_model import LightweightAudioAnalysisNet


class LightweightAudioModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = LightweightAudioAnalysisNet().eval()

    def test_forward_supports_variable_length_inputs(self) -> None:
        for time_steps in (100, 128, 150):
            with self.subTest(time_steps=time_steps):
                x = torch.randn(1, time_steps, 64)

                with torch.no_grad():
                    outputs = self.model(x)

                self.assertEqual(outputs["problem_probs"].shape, (1, 4))
                self.assertEqual(outputs["instrument_probs"].shape, (1, 5))
                self.assertEqual(outputs["eq_freq"].shape, (1, 1))
                self.assertEqual(outputs["eq_gain_db"].shape, (1, 1))
                self.assertEqual(outputs["embedding"].shape[-1], 256)

    def test_forward_supports_single_example_without_batch_dimension(self) -> None:
        x = torch.randn(128, 64)

        with torch.no_grad():
            outputs = self.model(x)

        self.assertEqual(outputs["problem_probs"].shape, (1, 4))
        self.assertEqual(outputs["instrument_probs"].shape, (1, 5))

    def test_output_ranges_match_head_contracts(self) -> None:
        x = torch.randn(2, 128, 64)

        with torch.no_grad():
            outputs = self.model(x)

        problem_probs = outputs["problem_probs"]
        instrument_probs = outputs["instrument_probs"]
        eq_freq = outputs["eq_freq"]
        eq_gain_db = outputs["eq_gain_db"]

        self.assertTrue(torch.allclose(problem_probs.sum(dim=-1), torch.ones(2), atol=1e-5))
        self.assertTrue(bool(((instrument_probs >= 0) & (instrument_probs <= 1)).all()))
        self.assertTrue(bool(((eq_freq >= 0) & (eq_freq <= 1)).all()))
        self.assertTrue(bool(((eq_gain_db >= -6) & (eq_gain_db <= 6)).all()))


if __name__ == "__main__":
    unittest.main()
