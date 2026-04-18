"""
Numerical correctness tests for ``utils.model_utils``.

These tests cover the pure-numpy primitives only (``create_sliding_windows``).
Heavier TensorFlow-backed routines live behind the ``integration`` marker
and are exercised in a separate suite so CI can run this file on a minimal
Python image without GPU / TF.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from utils.model_utils import create_sliding_windows


class TestCreateSlidingWindowsShape:
    def test_basic_shape(self) -> None:
        series = np.arange(10, dtype=float)
        X, Y = create_sliding_windows(series, window_size=3)
        # 10 - 3 = 7 windows; each row has window_size elements.
        assert X.shape == (7, 3)
        assert Y.shape == (7,)

    def test_window_equals_length_minus_one(self) -> None:
        series = np.arange(10, dtype=float)
        X, Y = create_sliding_windows(series, window_size=9)
        # Only a single window fits.
        assert X.shape == (1, 9)
        assert Y.shape == (1,)

    @pytest.mark.parametrize(
        ("n", "w"),
        [
            (5, 5),   # window equals length → no room for Y
            (5, 6),   # window longer than series
            (3, 10),  # much longer
            (0, 3),   # empty series
        ],
    )
    def test_insufficient_data_returns_empty(self, n: int, w: int) -> None:
        series = np.arange(n, dtype=float)
        X, Y = create_sliding_windows(series, window_size=w)
        assert len(X) == 0
        assert len(Y) == 0


class TestCreateSlidingWindowsValues:
    def test_first_and_last_windows_match_source(self) -> None:
        series = np.arange(10, dtype=float)
        X, Y = create_sliding_windows(series, window_size=3)
        # First window covers series[0:3], target is series[3].
        np.testing.assert_array_equal(X[0], [0.0, 1.0, 2.0])
        assert Y[0] == 3.0
        # Last window covers series[6:9], target is series[9].
        np.testing.assert_array_equal(X[-1], [6.0, 7.0, 8.0])
        assert Y[-1] == 9.0

    def test_window_one_step_forecast(self) -> None:
        series = np.array([10.0, 20.0, 30.0])
        X, Y = create_sliding_windows(series, window_size=1)
        assert X.shape == (2, 1)
        np.testing.assert_array_equal(X.ravel(), [10.0, 20.0])
        np.testing.assert_array_equal(Y, [20.0, 30.0])

    def test_targets_are_strict_future_of_windows(self) -> None:
        """Y[i] must come strictly after X[i] — no leakage."""
        series = np.arange(20, dtype=float)
        w = 4
        X, Y = create_sliding_windows(series, window_size=w)
        for i in range(len(X)):
            # Target is the element at position i + w.
            assert Y[i] == series[i + w]
            # Window is series[i : i + w]; its last element is series[i + w - 1].
            assert X[i][-1] == series[i + w - 1]
            # The target is never part of its own window.
            assert Y[i] not in X[i] or len(np.unique(series[i : i + w + 1])) < w + 1


class TestCreateSlidingWindowsProperty:
    """Property-based checks via hypothesis — catch off-by-one and shape drift."""

    @pytest.mark.numerical
    @given(
        series=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=200),
            elements=st.floats(
                min_value=-1e6, max_value=1e6,
                allow_nan=False, allow_infinity=False,
            ),
        ),
        window_size=st.integers(min_value=1, max_value=24),
    )
    @settings(max_examples=100, deadline=None)
    def test_shape_invariants(self, series: np.ndarray, window_size: int) -> None:
        X, Y = create_sliding_windows(series, window_size=window_size)
        n = len(series)
        expected = max(0, n - window_size)
        assert X.shape[0] == expected
        assert Y.shape[0] == expected
        if expected > 0:
            assert X.shape[1] == window_size

    @pytest.mark.numerical
    @given(
        series=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=20, max_value=100),
            elements=st.floats(
                min_value=-1000, max_value=1000,
                allow_nan=False, allow_infinity=False,
            ),
        ),
        window_size=st.integers(min_value=1, max_value=12),
    )
    @settings(max_examples=50, deadline=None)
    def test_targets_equal_series_tail(self, series: np.ndarray, window_size: int) -> None:
        """Y is exactly series[window_size:] when there's enough data."""
        X, Y = create_sliding_windows(series, window_size=window_size)
        np.testing.assert_array_equal(Y, series[window_size:])

    @pytest.mark.numerical
    @given(
        series=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=20, max_value=100),
            elements=st.floats(
                min_value=-1000, max_value=1000,
                allow_nan=False, allow_infinity=False,
            ),
        ),
        window_size=st.integers(min_value=1, max_value=12),
    )
    @settings(max_examples=50, deadline=None)
    def test_every_window_matches_slice(self, series: np.ndarray, window_size: int) -> None:
        """X[i] is exactly series[i : i + window_size]."""
        X, Y = create_sliding_windows(series, window_size=window_size)
        for i in range(len(X)):
            np.testing.assert_array_equal(X[i], series[i : i + window_size])
