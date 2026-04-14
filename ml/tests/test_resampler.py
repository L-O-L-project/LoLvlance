from __future__ import annotations

import unittest

import numpy as np

from ml.tests.integrity_utils import (
    REFERENCE_SOURCE_SAMPLE_RATES,
    RESAMPLER_MEAN_DELTA_THRESHOLD,
    RESAMPLER_PEAK_DELTA_THRESHOLD,
    RESAMPLER_RMS_DELTA_THRESHOLD,
    TARGET_SAMPLE_RATE,
    assert_all_finite,
    browser_resample_mono_buffer,
    generate_synthetic_capture,
    peak,
    rms,
    simulate_browser_capture,
)
from ml.preprocessing import resample_audio


class TestResamplerConsistency(unittest.TestCase):
    def test_resampler_output_is_finite_and_safe(self) -> None:
        for source_sample_rate in REFERENCE_SOURCE_SAMPLE_RATES:
            with self.subTest(source_sample_rate=source_sample_rate):
                capture_waveform = generate_synthetic_capture(source_sample_rate)
                browser_waveform = browser_resample_mono_buffer(capture_waveform, source_sample_rate, TARGET_SAMPLE_RATE)
                python_waveform = resample_audio(capture_waveform, source_sample_rate, TARGET_SAMPLE_RATE)

                expected_length = max(1, round(capture_waveform.shape[0] / (source_sample_rate / TARGET_SAMPLE_RATE)))
                self.assertEqual(
                    browser_waveform.shape[0],
                    expected_length,
                    (
                        "Browser-equivalent resampler returned an unexpected output length. "
                        f"source_rate={source_sample_rate} expected_length={expected_length} "
                        f"actual_length={browser_waveform.shape[0]}"
                    ),
                )
                self.assertEqual(
                    python_waveform.shape[0],
                    expected_length,
                    (
                        "Python resampler returned an unexpected output length. "
                        f"source_rate={source_sample_rate} expected_length={expected_length} "
                        f"actual_length={python_waveform.shape[0]}"
                    ),
                )

                assert_all_finite(browser_waveform, context="Browser-equivalent resampler output")
                assert_all_finite(python_waveform, context="Python resampler output")

                self.assertLessEqual(
                    float(np.max(np.abs(browser_waveform))),
                    1.0 + 1e-6,
                    f"Browser-equivalent resampler clipped the waveform at source_rate={source_sample_rate}.",
                )
                self.assertLessEqual(
                    float(np.max(np.abs(python_waveform))),
                    1.0 + 1e-6,
                    f"Python resampler clipped the waveform at source_rate={source_sample_rate}.",
                )

    def test_resampler_statistics_match_reference_path(self) -> None:
        for source_sample_rate in REFERENCE_SOURCE_SAMPLE_RATES:
            with self.subTest(source_sample_rate=source_sample_rate):
                capture_waveform = simulate_browser_capture(source_sample_rate, 3.2)
                browser_waveform = browser_resample_mono_buffer(capture_waveform, source_sample_rate, TARGET_SAMPLE_RATE)
                python_waveform = resample_audio(capture_waveform, source_sample_rate, TARGET_SAMPLE_RATE)

                mean_delta = abs(float(browser_waveform.mean()) - float(python_waveform.mean()))
                rms_delta = abs(rms(browser_waveform) - rms(python_waveform))
                peak_delta = abs(peak(browser_waveform) - peak(python_waveform))

                self.assertLessEqual(
                    mean_delta,
                    RESAMPLER_MEAN_DELTA_THRESHOLD,
                    (
                        "Resampler mean drift exceeded threshold. "
                        f"source_rate={source_sample_rate} mean_delta={mean_delta:.8f} "
                        f"threshold={RESAMPLER_MEAN_DELTA_THRESHOLD:.8f}"
                    ),
                )
                self.assertLessEqual(
                    rms_delta,
                    RESAMPLER_RMS_DELTA_THRESHOLD,
                    (
                        "Resampler RMS drift exceeded threshold. "
                        f"source_rate={source_sample_rate} rms_delta={rms_delta:.8f} "
                        f"threshold={RESAMPLER_RMS_DELTA_THRESHOLD:.8f}"
                    ),
                )
                self.assertLessEqual(
                    peak_delta,
                    RESAMPLER_PEAK_DELTA_THRESHOLD,
                    (
                        "Resampler peak drift exceeded threshold. "
                        f"source_rate={source_sample_rate} peak_delta={peak_delta:.8f} "
                        f"threshold={RESAMPLER_PEAK_DELTA_THRESHOLD:.8f}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
