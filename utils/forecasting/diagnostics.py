"""
Statistical diagnostics for forecast models.

Every helper here takes plain numpy arrays and returns either a float
p-value or ``None`` (if the test couldn't be run — e.g. too few obs,
missing optional dep). Nothing in here mutates state.

Tests implemented:
- ADF (Augmented Dickey-Fuller): stationarity of the input series.
- KPSS: complementary stationarity test — use ADF & KPSS together to
  avoid each test's individual blind spots.
- Ljung-Box: autocorrelation in residuals. Large p => white noise.
- Jarque-Bera: normality of residuals. Large p => normal.
- Diebold-Mariano: pairwise comparison of forecast losses. The
  Harvey-Leybourne-Newbold (1997) small-sample correction is applied
  so short series still give sensible p-values.

``run_diagnostics`` composes all of these into a
:class:`ForecastDiagnostics` record.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from utils.forecasting.base import ForecastDiagnostics

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _finite(arr: Any) -> np.ndarray | None:
    """Return ``arr`` as float ndarray with NaN/Inf dropped, or None if empty."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float).ravel()
    a = a[np.isfinite(a)]
    return a if a.size else None


# --------------------------------------------------------------------------
# Stationarity
# --------------------------------------------------------------------------
def adf_pvalue(y: Any) -> float | None:
    """
    Augmented Dickey-Fuller. Null: y has a unit root (non-stationary).

    Returns None if statsmodels isn't installed or the series is too
    short for a reliable test (< 10 obs).
    """
    a = _finite(y)
    if a is None or a.size < 10:
        return None
    try:
        from statsmodels.tsa.stattools import adfuller  # noqa: PLC0415
    except ImportError:
        logger.info("[diag] statsmodels not installed; skipping ADF")
        return None
    try:
        _, pval, *_ = adfuller(a, autolag="AIC")
        return float(pval)
    except Exception as exc:  # noqa: BLE001
        logger.info(f"[diag] ADF failed: {exc}")
        return None


def kpss_pvalue(y: Any) -> float | None:
    """
    KPSS. Null: y is stationary. Complement to ADF.

    Note: statsmodels clips KPSS p-values at {0.01, 0.1}; treat this as
    ordinal, not a continuous p.
    """
    a = _finite(y)
    if a is None or a.size < 15:
        return None
    try:
        from statsmodels.tsa.stattools import kpss  # noqa: PLC0415
    except ImportError:
        return None
    try:
        import warnings  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # clips at bound -> InterpolationWarning
            _, pval, *_ = kpss(a, regression="c", nlags="auto")
        return float(pval)
    except Exception as exc:  # noqa: BLE001
        logger.info(f"[diag] KPSS failed: {exc}")
        return None


# --------------------------------------------------------------------------
# Residual whiteness / normality
# --------------------------------------------------------------------------
def ljungbox_pvalue(resid: Any, lags: int = 10) -> float | None:
    """
    Ljung-Box Q. Null: residuals are uncorrelated up to ``lags``.

    Interpret large p (>0.05) as "residuals look like white noise" =
    the model captured the serial structure.
    """
    a = _finite(resid)
    if a is None or a.size < lags + 2:
        return None
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: PLC0415
    except ImportError:
        return None
    try:
        # return_df is the modern API; take the p-value at the chosen lag.
        lags_used = min(lags, a.size - 2)
        df = acorr_ljungbox(a, lags=[lags_used], return_df=True)
        return float(df["lb_pvalue"].iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.info(f"[diag] Ljung-Box failed: {exc}")
        return None


def jarquebera_pvalue(resid: Any) -> float | None:
    """
    Jarque-Bera. Null: residuals are normally distributed.

    Asymptotic; prefer at least 30 observations.
    """
    a = _finite(resid)
    if a is None or a.size < 8:
        return None
    try:
        from scipy.stats import jarque_bera  # noqa: PLC0415
    except ImportError:
        return None
    try:
        _, pval = jarque_bera(a)
        return float(pval)
    except Exception as exc:  # noqa: BLE001
        logger.info(f"[diag] Jarque-Bera failed: {exc}")
        return None


# --------------------------------------------------------------------------
# Diebold-Mariano pairwise test
# --------------------------------------------------------------------------
def diebold_mariano(
    resid_a: Any,
    resid_b: Any,
    horizon: int = 1,
    loss: str = "squared",
) -> tuple[float | None, float | None]:
    """
    DM test: are the expected forecast losses of A and B equal?

    Negative stat ⇒ model A (whose residuals are ``resid_a``) has lower
    expected loss, i.e. A beats B. Small p (<0.05) ⇒ the difference is
    statistically meaningful.

    Uses HAC variance with truncation lag ``horizon - 1`` (Newey-West
    style) and the Harvey-Leybourne-Newbold small-sample correction.
    Returns ``(stat, pvalue)`` or ``(None, None)`` if the test cannot
    be run.
    """
    a = _finite(resid_a)
    b = _finite(resid_b)
    if a is None or b is None:
        return None, None

    n = min(a.size, b.size)
    if n < 8:
        return None, None
    a, b = a[-n:], b[-n:]

    if loss == "squared":
        d = a ** 2 - b ** 2
    elif loss == "absolute":
        d = np.abs(a) - np.abs(b)
    else:
        raise ValueError(f"unknown loss: {loss}")

    mean_d = float(np.mean(d))
    gamma0 = float(np.var(d, ddof=0))
    if gamma0 == 0:
        return 0.0, 1.0
    var_d = gamma0
    for k in range(1, max(horizon, 1)):
        if k >= n:
            break
        gamma_k = float(np.cov(d[k:], d[:-k], ddof=0)[0, 1])
        var_d += 2 * gamma_k
    var_d = max(var_d / n, 1e-12)
    stat = mean_d / np.sqrt(var_d)

    # Harvey, Leybourne, Newbold (1997) small-sample correction.
    correction = np.sqrt(
        max((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n, 1e-12)
    )
    stat *= correction

    try:
        from scipy.stats import t as student_t  # noqa: PLC0415

        pval = 2.0 * (1.0 - student_t.cdf(abs(stat), df=n - 1))
    except ImportError:
        # Normal approximation fallback.
        from math import erf, sqrt  # noqa: PLC0415

        pval = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(stat) / sqrt(2.0))))

    return float(stat), float(pval)


# --------------------------------------------------------------------------
# Composite
# --------------------------------------------------------------------------
def run_diagnostics(
    name: str,
    y: Any,
    residuals: Any = None,
    baseline_residuals: Any = None,
    horizon: int = 1,
) -> ForecastDiagnostics:
    """Bundle every diagnostic into a single record for the UI."""
    dm_stat, dm_p = diebold_mariano(residuals, baseline_residuals, horizon=horizon) \
        if residuals is not None and baseline_residuals is not None \
        else (None, None)

    return ForecastDiagnostics(
        model=name,
        adf_pvalue=adf_pvalue(y),
        kpss_pvalue=kpss_pvalue(y),
        ljungbox_pvalue=ljungbox_pvalue(residuals),
        jarquebera_pvalue=jarquebera_pvalue(residuals),
        dm_stat_vs_baseline=dm_stat,
        dm_pvalue_vs_baseline=dm_p,
    )
