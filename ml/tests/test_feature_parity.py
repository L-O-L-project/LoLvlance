from __future__ import annotations

import unittest

from ml.tests.integrity_utils import (
    FEATURE_MAE_THRESHOLD,
    FEATURE_MEAN_DELTA_THRESHOLD,
    FEATURE_MSE_THRESHOLD,
    FEATURE_STD_DELTA_THRESHOLD,
    REFERENCE_SOURCE_SAMPLE_RATES,
    SOURCE_DURATIONS_SECONDS,
    assert_all_finite,
    browser_model_input_features,
    compute_error_metrics,
    python_model_input_features,
    simulate_browser_capture,
)


class TestFeatureParity(unittest.TestCase):
    def test_browser_and_python_model_inputs_match(self) -> None:
        for source_sample_rate in REFERENCE_SOURCE_SAMPLE_RATES:
            for duration_seconds in SOURCE_DURATIONS_SECONDS:
                with self.subTest(source_sample_rate=source_sample_rate, duration_seconds=duration_seconds):
                    capture_waveform = simulate_browser_capture(source_sample_rate, duration_seconds)
                    python_waveform, python_features = python_model_input_features(capture_waveform, source_sample_rate)
                    browser_waveform, browser_features = browser_model_input_features(capture_waveform, source_sample_rate)

                    self.assertEqual(
                        python_waveform.shape,
                        browser_waveform.shape,
                        (
                            "Waveform shape mismatch before feature comparison. "
                            f"python_shape={python_waveform.shape} browser_shape={browser_waveform.shape}"
                        ),
                    )
                    self.assertEqual(
                        python_features.shape,
                        browser_features.shape,
                        (
                            "Feature parity shape mismatch. "
                            f"python_shape={python_features.shape} browser_shape={browser_features.shape}"
                        ),
                    )

                    assert_all_finite(python_features, context="Python feature tensor")
                    assert_all_finite(browser_features, context="Browser-equivalent feature tensor")

                    metrics = compute_error_metrics(python_features, browser_features)

                    self.assertLessEqual(
                        metrics.mse,
                        FEATURE_MSE_THRESHOLD,
                        (
                            "Feature parity MSE exceeded threshold. "
                            f"source_rate={source_sample_rate} duration={duration_seconds}s "
                            f"mse={metrics.mse:.8f} threshold={FEATURE_MSE_THRESHOLD:.8f}"
                        ),
                    )
                    self.assertLessEqual(
                        metrics.mae,
                        FEATURE_MAE_THRESHOLD,
                        (
                            "Feature parity MAE exceeded threshold. "
                            f"source_rate={source_sample_rate} duration={duration_seconds}s "
                            f"mae={metrics.mae:.8f} threshold={FEATURE_MAE_THRESHOLD:.8f}"
                        ),
                    )
                    self.assertLessEqual(
                        metrics.mean_delta,
                        FEATURE_MEAN_DELTA_THRESHOLD,
                        (
                            "Feature parity mean-delta exceeded threshold. "
                            f"source_rate={source_sample_rate} duration={duration_seconds}s "
                            f"mean_delta={metrics.mean_delta:.8f} threshold={FEATURE_MEAN_DELTA_THRESHOLD:.8f}"
                        ),
                    )
                    self.assertLessEqual(
                        metrics.std_delta,
                        FEATURE_STD_DELTA_THRESHOLD,
                        (
                            "Feature parity std-delta exceeded threshold. "
                            f"source_rate={source_sample_rate} duration={duration_seconds}s "
                            f"std_delta={metrics.std_delta:.8f} threshold={FEATURE_STD_DELTA_THRESHOLD:.8f}"
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
