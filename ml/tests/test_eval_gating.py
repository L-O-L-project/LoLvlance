import copy
import unittest

from ml.eval.evaluate import build_gate_report


def make_gate_config() -> dict[str, float | int]:
    return {
        "macro_epsilon": 0.02,
        "per_label_epsilon": 0.03,
        "weak_label_f1_threshold": 0.4,
        "weak_label_epsilon": 0.02,
        "max_ratio_per_label": 0.5,
        "distribution_slack": 0.2,
        "entropy_epsilon": 0.12,
        "top_confusion_deltas": 5,
    }


def make_report() -> dict:
    return {
        "groups": {
            "combined": {
                "macro": {"f1": 0.8},
                "per_label": {
                    "issue:harsh": {
                        "f1": 0.82,
                        "precision": 0.82,
                        "recall": 0.82,
                        "support": 3,
                        "predicted_positive": 3,
                    },
                    "issue:thin": {
                        "f1": 0.35,
                        "precision": 0.35,
                        "recall": 0.35,
                        "support": 1,
                        "predicted_positive": 1,
                    },
                    "source:keys": {
                        "f1": 0.78,
                        "precision": 0.78,
                        "recall": 0.78,
                        "support": 3,
                        "predicted_positive": 3,
                    },
                    "source:vocal": {
                        "f1": 0.84,
                        "precision": 0.84,
                        "recall": 0.84,
                        "support": 2,
                        "predicted_positive": 2,
                    },
                },
            }
        },
        "prediction_distribution": {
            "issue": {
                "by_label": {
                    "issue:harsh": {
                        "predicted_ratio": 0.3,
                        "expected_ratio": 0.3,
                        "top1_ratio": 0.3,
                    },
                    "issue:thin": {
                        "predicted_ratio": 0.1,
                        "expected_ratio": 0.1,
                        "top1_ratio": 0.1,
                    },
                },
                "positive_entropy": {"normalized": 0.72},
                "dominant_label": "issue:harsh",
                "dominant_ratio": 0.3,
            },
            "source": {
                "by_label": {
                    "source:keys": {
                        "predicted_ratio": 0.6,
                        "expected_ratio": 0.45,
                        "top1_ratio": 0.6,
                    },
                    "source:vocal": {
                        "predicted_ratio": 0.4,
                        "expected_ratio": 0.35,
                        "top1_ratio": 0.4,
                    },
                },
                "positive_entropy": {"normalized": 0.81},
                "dominant_label": "source:keys",
                "dominant_ratio": 0.6,
            },
        },
        "confusion_summary": {"confusion_matrix": {"source:vocal->keys": 1}},
    }


class EvalGatingTest(unittest.TestCase):
    def test_passes_for_small_non_regression(self) -> None:
        baseline = make_report()
        current = copy.deepcopy(baseline)
        current["groups"]["combined"]["macro"]["f1"] = 0.79
        current["groups"]["combined"]["per_label"]["issue:harsh"]["f1"] = 0.8
        current["groups"]["combined"]["per_label"]["source:vocal"]["f1"] = 0.82

        gate = build_gate_report(current, baseline, gate_config=make_gate_config())

        self.assertEqual(gate["status"], "pass")
        self.assertFalse(gate["failures"])

    def test_fails_when_macro_regresses_beyond_epsilon(self) -> None:
        baseline = make_report()
        current = copy.deepcopy(baseline)
        current["groups"]["combined"]["macro"]["f1"] = 0.75

        gate = build_gate_report(current, baseline, gate_config=make_gate_config())

        self.assertEqual(gate["status"], "fail")
        self.assertTrue(gate["regression_flags"]["macro_regression"])

    def test_fails_when_single_label_collapses_even_if_macro_improves(self) -> None:
        baseline = make_report()
        current = copy.deepcopy(baseline)
        current["groups"]["combined"]["macro"]["f1"] = 0.82
        current["groups"]["combined"]["per_label"]["source:keys"]["f1"] = 0.6

        gate = build_gate_report(current, baseline, gate_config=make_gate_config())

        self.assertEqual(gate["status"], "fail")
        self.assertTrue(
            any(entry["label"] == "source:keys" for entry in gate["regression_flags"]["per_label_regressions"])
        )

    def test_weak_labels_use_stricter_epsilon(self) -> None:
        baseline = make_report()
        current = copy.deepcopy(baseline)
        current["groups"]["combined"]["macro"]["f1"] = 0.79
        current["groups"]["combined"]["per_label"]["issue:thin"]["f1"] = 0.32

        gate = build_gate_report(current, baseline, gate_config=make_gate_config())

        self.assertEqual(gate["status"], "fail")
        self.assertTrue(
            any(entry["label"] == "issue:thin" for entry in gate["regression_flags"]["weak_label_regressions"])
        )

    def test_fails_when_prediction_distribution_is_biased(self) -> None:
        baseline = make_report()
        current = copy.deepcopy(baseline)
        current["prediction_distribution"]["issue"]["by_label"]["issue:harsh"]["predicted_ratio"] = 0.95
        current["prediction_distribution"]["issue"]["dominant_ratio"] = 0.95

        gate = build_gate_report(current, baseline, gate_config=make_gate_config())

        self.assertEqual(gate["status"], "fail")
        self.assertTrue(gate["regression_flags"]["bias_violations"])

    def test_fails_when_entropy_collapses_vs_baseline(self) -> None:
        baseline = make_report()
        current = copy.deepcopy(baseline)
        current["prediction_distribution"]["source"]["positive_entropy"]["normalized"] = 0.5

        gate = build_gate_report(current, baseline, gate_config=make_gate_config())

        self.assertEqual(gate["status"], "fail")
        self.assertTrue(gate["regression_flags"]["entropy_regressions"])


if __name__ == "__main__":
    unittest.main()
