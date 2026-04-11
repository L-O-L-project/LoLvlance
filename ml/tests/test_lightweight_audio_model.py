import unittest

import torch

from ml.lightweight_audio_model import LightweightAudioAnalysisNet, PROBLEM_LABELS


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

                self.assertEqual(outputs["problem_probs"].shape, (1, len(PROBLEM_LABELS)))
                self.assertEqual(outputs["problem_logits"].shape, (1, len(PROBLEM_LABELS)))
                self.assertEqual(outputs["embedding"].shape[-1], 192)

    def test_forward_supports_single_example_without_batch_dimension(self) -> None:
        x = torch.randn(128, 64)

        with torch.no_grad():
            outputs = self.model(x)

        self.assertEqual(outputs["problem_probs"].shape, (1, len(PROBLEM_LABELS)))

    def test_output_ranges_match_head_contracts(self) -> None:
        x = torch.randn(2, 128, 64)

        with torch.no_grad():
            outputs = self.model(x)

        problem_probs = outputs["problem_probs"]
        self.assertTrue(bool(((problem_probs >= 0) & (problem_probs <= 1)).all()))


if __name__ == "__main__":
    unittest.main()
