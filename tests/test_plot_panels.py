import unittest

import numpy as np

from app.plot_panels import Plot2DPanel


class XtDisplaySeriesTests(unittest.TestCase):
    def test_raw_series_is_unchanged(self) -> None:
        series = np.array([1.0, 3.5, 7.0], dtype=float)

        derived = Plot2DPanel._build_xt_display_series(series, "raw")

        np.testing.assert_allclose(derived, series)

    def test_first_difference_keeps_alignment_with_leading_nan(self) -> None:
        series = np.array([2.0, 5.0, 9.5, 10.0], dtype=float)

        derived = Plot2DPanel._build_xt_display_series(series, "diff1")

        self.assertTrue(np.isnan(derived[0]))
        np.testing.assert_allclose(derived[1:], np.array([3.0, 4.5, 0.5], dtype=float))

    def test_second_difference_keeps_alignment_with_two_leading_nans(self) -> None:
        series = np.array([1.0, 4.0, 9.0, 16.0], dtype=float)

        derived = Plot2DPanel._build_xt_display_series(series, "diff2")

        self.assertTrue(np.isnan(derived[0]))
        self.assertTrue(np.isnan(derived[1]))
        np.testing.assert_allclose(derived[2:], np.array([2.0, 2.0], dtype=float))


if __name__ == "__main__":
    unittest.main()
