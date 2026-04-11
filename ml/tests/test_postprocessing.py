import unittest

from ml.postprocessing import build_hierarchical_output


class PostProcessingTest(unittest.TestCase):
    def test_builds_derived_diagnoses_from_issue_and_source_scores(self) -> None:
        output = build_hierarchical_output(
            issue_scores={
                "muddy": 0.81,
                "harsh": 0.2,
                "buried": 0.78,
                "boomy": 0.64,
                "thin": 0.1,
                "boxy": 0.41,
                "nasal": 0.12,
                "sibilant": 0.08,
                "dull": 0.15,
            },
            source_scores={
                "vocal": 0.87,
                "guitar": 0.22,
                "bass": 0.79,
                "drums": 0.18,
                "keys": 0.2,
            },
        )

        self.assertEqual(output["schema_version"], "2.0.0")
        self.assertIn("vocal_buried", output["derived_diagnoses"])
        self.assertIn("bass_muddy", output["derived_diagnoses"])


if __name__ == "__main__":
    unittest.main()
