"""
Forecasting tests that require statsmodels (+ scipy).

Covered here: ETS forecaster, the stationarity and residual
diagnostics, and the Diebold-Mariano pairwise test. Skipped cleanly if
the optional deps aren't available (e.g. the lightweight CI path).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("statsmodels")
pytest.importorskip("scipy")

from utils.forecasting.diagnostics import (  # noqa: E402
    adf_pvalue,
    diebold_mariano,
    jarquebera_pvalue,
    kpss_pvalue,
    ljungbox_pvalue,
    run_diagnostics,
)
from utils.forecasting.models import ETSForecaster  # noqa: E402


# --------------------------------------------------------------------------
# ETS
# --------------------------------------------------------------------------
class TestETS:
    def test_fits_and_predicts_on_seasonal_series(self) -> None:
        rng = np.random.default_rng(42)
        t = np.arange(60, dtype=float)
        y = 10 + 0.1 * t + 2 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.2, 60)
        f = ETSForecaster(seasonal_periods=12).fit(y, None)
        preds = f.predict(12)
        assert preds.shape == (12,)
        # Predictions should be in the reasonable neighborhood of the series.
        assert preds.min() > y.min() - 5
        assert preds.max() < y.max() + 5

    def test_falls_back_to_non_seasonal_when_short(self) -> None:
        """Less than 2 seasonal periods → no seasonal component, but still fits."""
        y = np.arange(18, dtype=float)  # < 24 = 2 * 12
        f = ETSForecaster(seasonal_periods=12).fit(y, None)
        preds = f.predict(6)
        assert preds.shape == (6,)
        assert np.isfinite(preds).all()

    def test_residuals_populated_after_fit(self) -> None:
        y = np.arange(60, dtype=float) + np.random.default_rng(0).normal(0, 1, 60)
        f = ETSForecaster(seasonal_periods=12).fit(y, None)
        assert f.residuals is not None
        assert f.residuals.shape[0] == 60

    def test_aic_bic_exposed_after_fit(self) -> None:
        y = np.arange(60, dtype=float) + np.random.default_rng(0).normal(0, 1, 60)
        f = ETSForecaster(seasonal_periods=12).fit(y, None)
        assert f.aic is not None and np.isfinite(f.aic)
        assert f.bic is not None and np.isfinite(f.bic)


# --------------------------------------------------------------------------
# Stationarity tests
# --------------------------------------------------------------------------
class TestStationarityDiagnostics:
    def test_adf_rejects_unit_root_on_stationary_series(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.normal(size=200)  # i.i.d. → stationary
        p = adf_pvalue(y)
        assert p is not None
        assert p < 0.05  # reject H0 (unit root)

    def test_adf_fails_to_reject_on_random_walk(self) -> None:
        """
        Random walks are genuinely non-stationary, so ADF should leave the
        null standing most of the time. On a short path, the test can fail
        the null by chance (~5%); averaging p-values across several
        independent walks is far more robust.
        """
        ps = []
        for seed in range(10):
            rng = np.random.default_rng(100 + seed)
            y = np.cumsum(rng.normal(size=500))
            p = adf_pvalue(y)
            assert p is not None
            ps.append(p)
        # Across 10 independent random walks the average p-value must be
        # well above the 0.05 rejection threshold.
        assert float(np.mean(ps)) > 0.3

    def test_adf_returns_none_on_too_short_series(self) -> None:
        assert adf_pvalue(np.array([1.0, 2.0, 3.0])) is None

    def test_kpss_accepts_stationary_series(self) -> None:
        rng = np.random.default_rng(3)
        y = rng.normal(size=200)
        p = kpss_pvalue(y)
        # statsmodels clips at 0.1 for strongly stationary series.
        assert p is not None and p >= 0.05


# --------------------------------------------------------------------------
# Residual diagnostics
# --------------------------------------------------------------------------
class TestResidualDiagnostics:
    def test_ljungbox_on_white_noise(self) -> None:
        rng = np.random.default_rng(4)
        e = rng.normal(size=200)
        p = ljungbox_pvalue(e)
        assert p is not None and p > 0.05  # null not rejected

    def test_ljungbox_on_autocorrelated(self) -> None:
        rng = np.random.default_rng(5)
        e = np.zeros(200)
        e[0] = rng.normal()
        for i in range(1, 200):
            e[i] = 0.8 * e[i - 1] + rng.normal()  # AR(1)
        p = ljungbox_pvalue(e)
        assert p is not None and p < 0.05  # reject white-noise null

    def test_jarquebera_on_normal(self) -> None:
        rng = np.random.default_rng(6)
        e = rng.normal(size=500)
        p = jarquebera_pvalue(e)
        assert p is not None and p > 0.05

    def test_jarquebera_on_uniform(self) -> None:
        rng = np.random.default_rng(7)
        e = rng.uniform(-1, 1, size=500)
        p = jarquebera_pvalue(e)
        assert p is not None and p < 0.05


# --------------------------------------------------------------------------
# Diebold-Mariano
# --------------------------------------------------------------------------
class TestDieboldMariano:
    def test_good_model_beats_bad(self) -> None:
        """Model A's residuals are tighter — DM stat should be negative."""
        rng = np.random.default_rng(8)
        a = rng.normal(0, 1, size=200)   # good: σ² = 1
        b = rng.normal(0, 3, size=200)   # bad:  σ² = 9
        stat, pval = diebold_mariano(a, b, horizon=1)
        assert stat is not None
        assert stat < 0
        assert pval is not None and pval < 0.05

    def test_symmetric_when_swapped(self) -> None:
        rng = np.random.default_rng(9)
        a = rng.normal(0, 1, size=200)
        b = rng.normal(0, 3, size=200)
        s1, _ = diebold_mariano(a, b)
        s2, _ = diebold_mariano(b, a)
        assert s1 is not None and s2 is not None
        assert s1 == pytest.approx(-s2)

    def test_identical_residuals_gives_zero_stat(self) -> None:
        rng = np.random.default_rng(10)
        a = rng.normal(size=200)
        stat, pval = diebold_mariano(a, a.copy())
        assert stat == 0.0
        assert pval == 1.0

    def test_returns_none_when_residuals_missing(self) -> None:
        assert diebold_mariano(None, np.zeros(200)) == (None, None)
        assert diebold_mariano(np.zeros(200), None) == (None, None)


# --------------------------------------------------------------------------
# Composite
# --------------------------------------------------------------------------
class TestRunDiagnostics:
    def test_all_fields_populated_with_enough_data(self) -> None:
        rng = np.random.default_rng(11)
        y = rng.normal(size=200)
        resid = rng.normal(size=200)
        baseline = rng.normal(size=200) * 3  # deliberately worse
        d = run_diagnostics(
            name="test_model", y=y, residuals=resid, baseline_residuals=baseline
        )
        assert d.model == "test_model"
        assert d.adf_pvalue is not None
        assert d.ljungbox_pvalue is not None
        assert d.jarquebera_pvalue is not None
        assert d.dm_stat_vs_baseline is not None

    def test_without_residuals_has_none_dm(self) -> None:
        rng = np.random.default_rng(12)
        y = rng.normal(size=200)
        d = run_diagnostics(name="no_resid", y=y, residuals=None)
        assert d.dm_stat_vs_baseline is None
        assert d.dm_pvalue_vs_baseline is None
