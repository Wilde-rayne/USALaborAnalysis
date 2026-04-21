"""
Pure-numpy tests for the trend-analytics helpers. No heavy deps.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from utils.forecasting.trend import (
    TrendSummary,
    rolling_statistics,
    summarize_trend,
)


class TestSummarizeTrend:
    def test_basic_series(self) -> None:
        y = list(range(120))  # 10 years of monthly data, 0..119
        s = summarize_trend(y)
        assert isinstance(s, TrendSummary)
        assert s.current == 119
        assert s.n_obs == 120
        assert s.all_time_high == 119 and s.all_time_low == 0
        assert s.delta_1y == 12  # position 119 vs 119-12=107
        assert s.pct_1y == pytest.approx((12 / 107) * 100.0)
        assert s.delta_5y == 60
        assert s.volatility > 0

    def test_returns_none_for_all_nan(self) -> None:
        assert summarize_trend([float("nan"), float("nan")]) is None

    def test_skips_nan_in_stats(self) -> None:
        y = [1.0, 2.0, float("nan"), 3.0, 4.0]
        s = summarize_trend(y)
        assert s is not None
        assert s.n_obs == 4
        assert s.current == 4.0

    def test_delta_1y_none_when_series_too_short(self) -> None:
        s = summarize_trend([1.0, 2.0, 3.0])
        assert s is not None
        assert s.delta_1y is None
        assert s.pct_1y is None

    def test_delta_5y_computed_when_possible(self) -> None:
        # 61 monthly obs ⇒ 5-year delta is valid.
        y = [float(i) for i in range(61)]
        s = summarize_trend(y)
        assert s is not None
        assert s.delta_5y == 60.0

    def test_zero_base_value_yields_none_pct(self) -> None:
        y = [0.0] * 12 + [5.0]
        s = summarize_trend(y)
        assert s is not None
        assert s.delta_1y == 5.0
        # Division by zero base ⇒ percent change undefined.
        assert s.pct_1y is None

    def test_periods_per_year_override(self) -> None:
        # Quarterly data — a "1-year" lookback is 4 periods.
        y = [float(i) for i in range(20)]
        s = summarize_trend(y, periods_per_year=4)
        assert s is not None
        assert s.delta_1y == 4
        assert s.delta_5y is None  # 5*4 = 20 > 19 usable step offset


class TestRollingStatistics:
    def test_returns_same_length_arrays(self) -> None:
        y = list(range(10))
        m, s = rolling_statistics(y, window=3)
        assert m.shape == (10,) and s.shape == (10,)

    def test_first_positions_are_nan_until_min_obs(self) -> None:
        y = list(range(10))
        m, _ = rolling_statistics(y, window=5, min_obs=5)
        # First 4 positions don't have 5 obs.
        assert np.isnan(m[:4]).all()
        assert np.isfinite(m[4])
        # Fifth window (indices 0-4) mean = (0+1+2+3+4)/5 = 2.0
        assert m[4] == pytest.approx(2.0)

    def test_min_obs_default_allows_partial_windows(self) -> None:
        y = [0.0, 2.0, 4.0]
        m, _ = rolling_statistics(y, window=12)  # default min_obs = 6
        # All positions have < 6 obs → all NaN.
        assert np.isnan(m).all()
        m2, _ = rolling_statistics(y, window=4, min_obs=2)
        # min_obs=2 so position 1 (two obs: 0, 2) → mean = 1.
        assert m2[0] == m2[0] or math.isnan(m2[0])  # may be nan (1 obs < 2)
        assert np.isnan(m2[0])
        assert m2[1] == pytest.approx(1.0)

    def test_nan_in_window_is_skipped(self) -> None:
        y = [1.0, float("nan"), 3.0, 5.0]
        m, _ = rolling_statistics(y, window=3, min_obs=2)
        # Position 3 window = [nan, 3, 5], valid mean of two = 4.
        assert m[3] == pytest.approx(4.0)

    def test_std_is_population_not_sample(self) -> None:
        """ddof=0 — population std matches numpy.std default."""
        y = [1.0, 2.0, 3.0, 4.0]
        _, s = rolling_statistics(y, window=4, min_obs=4)
        assert s[3] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=0))

    def test_rejects_bad_window(self) -> None:
        with pytest.raises(ValueError):
            rolling_statistics([1.0], window=0)
        with pytest.raises(ValueError):
            rolling_statistics([1.0], window=5, min_obs=10)
