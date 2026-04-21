"""
Core forecasting tests — baselines, metric functions, and the selector.

Runs without statsmodels / tensorflow — only numpy + the in-repo code.
Tests that need heavy deps live in test_forecasting_statsmodels.py.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from utils.forecasting.base import BaseForecaster, ForecastResult
from utils.forecasting.models import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from utils.forecasting.selection import (
    _expanding_window_folds,
    compute_metrics,
    select_forecaster,
)


# --------------------------------------------------------------------------
# NaiveForecaster
# --------------------------------------------------------------------------
class TestNaiveForecaster:
    def test_predict_is_last_value_repeated(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        f = NaiveForecaster().fit(y, None)
        preds = f.predict(6)
        assert preds.shape == (6,)
        assert np.all(preds == 5.0)

    def test_rejects_empty_series(self) -> None:
        with pytest.raises(ValueError):
            NaiveForecaster().fit(np.array([]), None)

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            NaiveForecaster().predict(3)

    def test_residuals_are_first_differences(self) -> None:
        y = np.array([10.0, 12.0, 15.0, 14.0])
        f = NaiveForecaster().fit(y, None)
        np.testing.assert_allclose(f.residuals, [2.0, 3.0, -1.0])

    def test_is_subclass_of_base_forecaster(self) -> None:
        assert issubclass(NaiveForecaster, BaseForecaster)


# --------------------------------------------------------------------------
# SeasonalNaiveForecaster
# --------------------------------------------------------------------------
class TestSeasonalNaiveForecaster:
    def test_predict_repeats_last_season_block(self) -> None:
        # Season = 3, so predictions cycle through y[-3:] = [7, 8, 9].
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
        f = SeasonalNaiveForecaster(season=3).fit(y, None)
        preds = f.predict(7)
        # horizon=7 → 3 full seasons would be 9; slice to 7.
        np.testing.assert_array_equal(preds, [7, 8, 9, 7, 8, 9, 7])

    def test_rejects_too_short_series(self) -> None:
        with pytest.raises(ValueError):
            # Need at least season+1 = 13 observations.
            SeasonalNaiveForecaster(season=12).fit(np.arange(8, dtype=float), None)

    def test_rejects_non_positive_season(self) -> None:
        with pytest.raises(ValueError):
            SeasonalNaiveForecaster(season=0)

    def test_residuals_drop_first_season(self) -> None:
        y = np.array([1, 2, 3, 10, 12, 14], dtype=float)
        f = SeasonalNaiveForecaster(season=3).fit(y, None)
        np.testing.assert_allclose(f.residuals, [9, 10, 11])


# --------------------------------------------------------------------------
# Metric functions
# --------------------------------------------------------------------------
class TestMetrics:
    def test_perfect_forecast_zero_metrics(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = compute_metrics(y, y.copy(), model="perfect", horizon=4)
        assert m.mae == 0.0
        assert m.rmse == 0.0
        assert m.bias == 0.0
        assert m.smape == 0.0

    def test_mae_rmse_match_definitions(self) -> None:
        y_true = np.array([10.0, 10.0, 10.0, 10.0])
        y_pred = np.array([8.0, 12.0, 9.0, 11.0])
        m = compute_metrics(y_true, y_pred, model="test", horizon=4)
        # Errors: [2, -2, 1, -1]
        assert m.mae == pytest.approx(1.5)  # mean(|e|) = (2+2+1+1)/4
        assert m.rmse == pytest.approx(np.sqrt(2.5))  # sqrt(mean(e^2))=sqrt((4+4+1+1)/4)
        assert m.bias == pytest.approx(0.0)  # mean(e) = 0

    def test_bias_sign_convention(self) -> None:
        """bias > 0 means we under-forecast (y_true > y_pred on average)."""
        y_true = np.array([10.0, 10.0])
        y_pred = np.array([8.0, 8.0])
        m = compute_metrics(y_true, y_pred, model="under", horizon=2)
        assert m.bias > 0

    def test_mape_safe_against_zero_truth(self) -> None:
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(y_true, y_pred, model="allzero", horizon=3)
        assert np.isnan(m.mape)

    def test_smape_bounded(self) -> None:
        y_true = np.array([1.0, 2.0, 0.0, 100.0])
        y_pred = np.array([0.0, 4.0, 0.0, 90.0])
        m = compute_metrics(y_true, y_pred, model="b", horizon=4)
        assert 0.0 <= m.smape <= 200.0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_metrics(
                np.array([1.0, 2.0]), np.array([1.0]), model="x", horizon=2
            )


# --------------------------------------------------------------------------
# Expanding-window fold generator
# --------------------------------------------------------------------------
class TestExpandingWindowFolds:
    def test_fold_count_matches_n_folds(self) -> None:
        folds = list(_expanding_window_folds(n=60, horizon=12, n_folds=3))
        assert len(folds) == 3

    def test_last_fold_ends_at_series_end(self) -> None:
        folds = list(_expanding_window_folds(n=60, horizon=12, n_folds=3))
        assert folds[-1][1] == 60

    def test_training_sets_grow_by_horizon(self) -> None:
        folds = list(_expanding_window_folds(n=60, horizon=12, n_folds=3))
        # Train ends: 24, 36, 48 (first_train_end = 60 - 36 = 24).
        assert [f[0] for f in folds] == [24, 36, 48]

    def test_rejects_series_too_short(self) -> None:
        with pytest.raises(ValueError):
            list(_expanding_window_folds(n=20, horizon=12, n_folds=3))

    def test_rejects_nonpositive_parameters(self) -> None:
        with pytest.raises(ValueError):
            list(_expanding_window_folds(n=60, horizon=0, n_folds=3))
        with pytest.raises(ValueError):
            list(_expanding_window_folds(n=60, horizon=12, n_folds=0))


# --------------------------------------------------------------------------
# select_forecaster
# --------------------------------------------------------------------------
class TestSelectForecaster:
    def test_picks_seasonal_on_strongly_seasonal_series(self) -> None:
        """On a clean sin-wave, seasonal-naive should beat flat-naive."""
        t = np.arange(120, dtype=float)
        y = np.sin(2 * np.pi * t / 12) * 100 + 500
        result = select_forecaster(
            y,
            horizon=12,
            candidates=[NaiveForecaster(), SeasonalNaiveForecaster(season=12)],
            n_folds=3,
        )
        assert result.model.name == "seasonal_naive"
        # Two candidates scored.
        assert len(result.candidates) == 2
        # Winner's RMSE must be the minimum across candidates.
        best_rmse = min(c.rmse for c in result.candidates)
        assert result.metrics.rmse == best_rmse

    def test_result_has_fitted_winner(self) -> None:
        y = np.arange(50, dtype=float)
        result = select_forecaster(
            y,
            horizon=6,
            candidates=[NaiveForecaster()],
            n_folds=2,
        )
        # The winner is refit on the full series and ready to forecast.
        preds = result.model.predict(6)
        assert preds.shape == (6,)
        assert np.all(preds == y[-1])  # Naive on arange → last value

    def test_raises_when_all_candidates_fail(self) -> None:
        """Every candidate fails in every fold → RuntimeError from selector."""

        class AlwaysFail(NaiveForecaster):
            name = "always_fail"

            def fit(self, y, dates):  # noqa: ANN001
                raise RuntimeError("nope")

        with pytest.raises(RuntimeError, match="every candidate failed"):
            select_forecaster(
                np.arange(40, dtype=float),
                horizon=5,
                candidates=[AlwaysFail()],
                n_folds=3,
            )

    def test_custom_score_can_flip_the_winner(self) -> None:
        """
        Verify the scoring rule actually drives selection.

        On y = arange(60) with horizon=6, the biased +5 variant has
        *lower* RMSE than flat-Naive (it partially corrects for the
        monotonic trend). Under default scoring (min RMSE) biased wins.
        Under a flipped score (min bias value — i.e. most-negative bias),
        biased also wins because its bias is negative while naive's is
        positive. Under `score=-bias` the ranking flips to naive.
        """
        y = np.arange(60, dtype=float)

        class BiasedNaive(NaiveForecaster):
            name = "biased_naive"

            def predict(self, horizon):
                return super().predict(horizon) + 5.0

        # Default RMSE-min scoring.
        default = select_forecaster(
            y,
            horizon=6,
            candidates=[NaiveForecaster(), BiasedNaive()],
            n_folds=3,
        )
        # Custom rule: lower bias-value wins (bias can be negative).
        lower_bias = select_forecaster(
            y,
            horizon=6,
            candidates=[NaiveForecaster(), BiasedNaive()],
            n_folds=3,
            score=lambda m: m.bias,
        )
        # Flipped bias rule — highest bias value wins.
        higher_bias = select_forecaster(
            y,
            horizon=6,
            candidates=[NaiveForecaster(), BiasedNaive()],
            n_folds=3,
            score=lambda m: -m.bias,
        )
        assert default.model.name == "biased_naive"
        assert lower_bias.model.name == "biased_naive"
        assert higher_bias.model.name == "naive"

    def test_returns_forecast_result_dataclass(self) -> None:
        y = np.arange(50, dtype=float)
        result = select_forecaster(
            y,
            horizon=5,
            candidates=[NaiveForecaster()],
            n_folds=2,
        )
        assert isinstance(result, ForecastResult)
        assert result.metrics is not None
        assert result.diagnostics is not None
        assert isinstance(result.candidates, list)


# --------------------------------------------------------------------------
# Property-based: selector invariants
# --------------------------------------------------------------------------
class TestSelectorProperty:
    @pytest.mark.numerical
    @given(
        y=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=40, max_value=200),
            elements=st.floats(
                min_value=-1000, max_value=1000,
                allow_nan=False, allow_infinity=False,
            ),
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_winner_rmse_is_minimum(self, y: np.ndarray) -> None:
        # Baselines only — hypothesis shouldn't pay the statsmodels cost.
        result = select_forecaster(
            y,
            horizon=6,
            candidates=[NaiveForecaster(), SeasonalNaiveForecaster(season=3)],
            n_folds=3,
        )
        min_rmse = min(c.rmse for c in result.candidates)
        assert result.metrics.rmse == pytest.approx(min_rmse)

    @pytest.mark.numerical
    @given(
        y=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=40, max_value=200),
            elements=st.floats(
                min_value=-1000, max_value=1000,
                allow_nan=False, allow_infinity=False,
            ),
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_winner_is_always_refit_on_full_series(self, y: np.ndarray) -> None:
        result = select_forecaster(
            y,
            horizon=6,
            candidates=[NaiveForecaster()],
            n_folds=3,
        )
        # Naive fit on full series → _last == y[-1].
        assert result.model._last == float(y[-1])
