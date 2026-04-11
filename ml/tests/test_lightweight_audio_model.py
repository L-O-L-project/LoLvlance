import unittest

import torch

from ml.lightweight_audio_model import ISSUE_LABELS, SOURCE_LABELS, LightweightAudioAnalysisNet


class LightweightAudioModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = LightweightAudioAnalysisNet().eval()

    def test_forward_supports_variable_length_inputs(self) -> None:
        for time_steps in (100, 128, 150):
            with self.subTest(time_steps=time_steps):
                inputs = torch.randn(1, time_steps, 64)

                with torch.no_grad():
                    outputs = self.model(inputs)

                self.assertEqual(outputs["issue_probs"].shape, (1, len(ISSUE_LABELS)))
                self.assertEqual(outputs["source_probs"].shape, (1, len(SOURCE_LABELS)))
                self.assertEqual(outputs["embedding"].shape[-1], 192)

    def test_forward_supports_single_example_without_batch_dimension(self) -> None:
        inputs = torch.randn(128, 64)

        with torch.no_grad():
            outputs = self.model(inputs)

        self.assertEqual(outputs["issue_probs"].shape, (1, len(ISSUE_LABELS)))
        self.assertEqual(outputs["source_probs"].shape, (1, len(SOURCE_LABELS)))

    def test_output_ranges_match_head_contracts(self) -> None:
        inputs = torch.randn(2, 128, 64)

        with torch.no_grad():
            outputs = self.model(inputs)

        self.assertTrue(bool(((outputs["issue_probs"] >= 0) & (outputs["issue_probs"] <= 1)).all()))
        self.assertTrue(bool(((outputs["source_probs"] >= 0) & (outputs["source_probs"] <= 1)).all()))
        self.assertTrue(torch.equal(outputs["problem_probs"], outputs["issue_probs"]))


if __name__ == "__main__":
    unittest.main()
