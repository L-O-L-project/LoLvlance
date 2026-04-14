from __future__ import annotations

import unittest

from ml.tests.integrity_utils import (
    REFERENCE_AUDIO_PATH,
    REFERENCE_SOURCE_SAMPLE_RATES,
    SOURCE_DURATIONS_SECONDS,
    WAVEFORM_MAE_THRESHOLD,
    WAVEFORM_MAX_ABS_THRESHOLD,
    WAVEFORM_MSE_THRESHOLD,
    assert_all_finite,
    browser_preprocessed_waveform,
    compute_error_metrics,
    python_preprocessed_waveform,
    simulate_browser_capture,
)


class TestWaveformParity(unittest.TestCase):
    def test_reference_audio_exists(self) -> None:
        self.assertTrue(
            REFERENCE_AUDIO_PATH.exists(),
            f"Reference audio file is missing: {REFERENCE_AUDIO_PATH}",
        )

    def test_browser_and_python_waveforms_match(self) -> None:
        for source_sample_rate in REFERENCE_SOURCE_SAMPLE_RATES:
            for duration_seconds in SOURCE_DURATIONS_SECONDS:
                with self.subTest(source_sample_rate=source_sample_rate, duration_seconds=duration_seconds):
                    capture_waveform = simulate_browser_capture(source_sample_rate, duration_seconds)
                    python_waveform = python_preprocessed_waveform(capture_waveform, source_sample_rate)
                    browser_waveform = browser_preprocessed_waveform(capture_waveform, source_sample_rate)

                    self.assertEqual(
                        python_waveform.shape,
                        browser_waveform.shape,
                        (
                            "Waveform parity shape mismatch. "
                            f"python_shape={python_waveform.shape} browser_shape={browser_waveform.shape}"
                        ),
                    )
                    assert_all_finite(python_waveform, context="Python waveform preprocessing output")
                    assert_all_finite(browser_waveform, context="Browser-equivalent waveform preprocessing output")

                    metrics = compute_error_metrics(python_waveform, browser_waveform)

                    self.assertLessEqual(
                        metrics.mse,
                        WAVEFORM_MSE_THRESHOLD,
                        (
                            "Waveform parity MSE exceeded threshold. "
                            f"source_rate={source_sample_rate} duration={duration_seconds}s "
                            f"mse={metrics.mse:.8f} threshold={WAVEFORM_MSE_THRESHOLD:.8f}"
                        ),
                    )
                    self.assertLessEqual(
                        metrics.mae,
                        WAVEFORM_MAE_THRESHOLD,
                        (
                            "Waveform parity MAE exceeded threshold. "
                            f"source_rate={source_sample_rate} duration={duration_seconds}s "
                            f"mae={metrics.mae:.8f} threshold={WAVEFORM_MAE_THRESHOLD:.8f}"
                        ),
                    )
                    self.assertLessEqual(
                        metrics.max_abs,
                        WAVEFORM_MAX_ABS_THRESHOLD,
                        (
                            "Waveform parity max-abs error exceeded threshold. "
                            f"source_rate={source_sample_rate} duration={duration_seconds}s "
                            f"max_abs={metrics.max_abs:.8f} threshold={WAVEFORM_MAX_ABS_THRESHOLD:.8f}"
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
