"""
Model selection via expanding-window backtest.

The key function is :func:`select_forecaster`: runs every candidate
through a K-fold expanding-window backtest, averages hold-out metrics,
picks a winner by a scoring rule (default: minimum RMSE), refits the
winner on the full series, and returns a :class:`ForecastResult` with
everything the UI needs to explain the choice.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterator, Sequence

import numpy as np

from utils.forecasting.base import (
    BaseForecaster,
    ForecastMetrics,
    ForecastResult,
)
from utils.forecasting.models import default_candidates

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Metric computation
# --------------------------------------------------------------------------
def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE with a near-zero guard. NaN when y_true has no non-zero values."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return float("nan")
    return float(
        np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) * 100.0
    )


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE, bounded in [0, 200]. Safe when y_true ≈ 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-12
    if not mask.any():
        return 0.0
    return float(
        np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model: str,
    horizon: int,
    aic: float | None = None,
    bic: float | None = None,
) -> ForecastMetrics:
    """Compute point-forecast metrics (MAE / RMSE / MAPE / sMAPE / bias).

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Equally-shaped 1-D arrays of actual and predicted values.
    model : str
        Forecaster name (carried into the result).
    horizon : int
        Forecast horizon length in steps.
    aic, bic : float, optional
        Information criteria from the fitted forecaster, if available.

    Returns
    -------
    ForecastMetrics
        Scored metrics for the window.

    Raises
    ------
    ValueError
        On a shape mismatch between ``y_true`` and ``y_pred``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape} y_pred={y_pred.shape}")

    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    return ForecastMetrics(
        model=model,
        mae=mae,
        rmse=rmse,
        mape=_safe_mape(y_true, y_pred),
        smape=_smape(y_true, y_pred),
        bias=bias,
        n_eval=int(y_true.size),
        horizon=horizon,
        aic=aic,
        bic=bic,
    )


# --------------------------------------------------------------------------
# Expanding-window backtest
# --------------------------------------------------------------------------
def _expanding_window_folds(
    n: int, horizon: int, n_folds: int
) -> Iterator[tuple[int, int]]:
    """Yield ``(train_end_exclusive, test_end_exclusive)`` index pairs.

    Fold ``k`` trains on ``y[:train_end_k]`` and scores against
    ``y[train_end_k : test_end_k]``. Each successive fold grows the
    training set by one horizon.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    total_test = n_folds * horizon
    first_train_end = n - total_test
    if first_train_end <= 0:
        raise ValueError(
            f"series too short: n={n}, need > {total_test} "
            f"(horizon={horizon} * n_folds={n_folds})"
        )
    for k in range(n_folds):
        train_end = first_train_end + k * horizon
        yield train_end, train_end + horizon


def _fold_metrics(
    candidate: BaseForecaster,
    y: np.ndarray,
    dates: np.ndarray,
    horizon: int,
    n_folds: int,
) -> ForecastMetrics | None:
    """Run ``candidate`` through every backtest fold and average the metrics.

    Returns ``None`` if the candidate failed in every fold.
    """
    per_fold: list[ForecastMetrics] = []
    for fold_i, (tr_end, te_end) in enumerate(
        _expanding_window_folds(len(y), horizon, n_folds)
    ):
        try:
            # Fresh model per fold — some backends (e.g. tf.keras) keep
            # state across fits that we don't want carried over.
            fresh = candidate.__class__(**_clone_kwargs(candidate))
            fresh.fit(y[:tr_end], dates[:tr_end] if dates is not None else None)
            preds = fresh.predict(horizon)
            actual = y[tr_end:te_end]
            per_fold.append(
                compute_metrics(
                    actual,
                    preds,
                    model=candidate.name,
                    horizon=horizon,
                    aic=fresh.aic,
                    bic=fresh.bic,
                )
            )
        except Exception as exc:  # noqa: BLE001 — one failing fold shouldn't kill the bakeoff
            logger.warning(
                f"[select] {candidate.name} fold={fold_i} failed: {exc.__class__.__name__}: {exc}"
            )

    if not per_fold:
        return None

    # Fold-averaged metrics. AIC/BIC averaging is a rough summary; we
    # keep the last fold's values because they reflect the largest
    # training sample. MAPE can legitimately be NaN for all folds (y_true
    # near zero everywhere), so compute its mean manually to avoid the
    # "mean of empty slice" warning nanmean raises.
    mapes_ok = [m.mape for m in per_fold if not np.isnan(m.mape)]
    mape_avg = float(np.mean(mapes_ok)) if mapes_ok else float("nan")
    return ForecastMetrics(
        model=candidate.name,
        mae=float(np.mean([m.mae for m in per_fold])),
        rmse=float(np.mean([m.rmse for m in per_fold])),
        mape=mape_avg,
        smape=float(np.mean([m.smape for m in per_fold])),
        bias=float(np.mean([m.bias for m in per_fold])),
        n_eval=int(sum(m.n_eval for m in per_fold)),
        horizon=horizon,
        aic=per_fold[-1].aic,
        bic=per_fold[-1].bic,
    )


def _clone_kwargs(model: BaseForecaster) -> dict:
    """Return ``model``'s public, non-fitted attributes for sibling instantiation."""
    return {
        k: v
        for k, v in vars(model).items()
        if not k.startswith("_") and k != "name"
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def select_forecaster(
    y: np.ndarray,
    dates: Sequence | None = None,
    *,
    horizon: int = 12,
    candidates: list[BaseForecaster] | None = None,
    n_folds: int = 3,
    score: Callable[[ForecastMetrics], float] | None = None,
) -> ForecastResult:
    """
    Run the bakeoff and return the winning forecaster refit on the full
    series, plus per-candidate metrics and diagnostics.

    Parameters
    ----------
    y, dates
        The univariate series. ``dates`` is optional; baseline/ETS
        ignore it, LSTM only uses it for logging.
    horizon
        Forecast length in steps. Also the size of each backtest fold.
    candidates
        Models to evaluate. Defaults to the built-in set.
    n_folds
        Number of expanding-window folds. More folds = more reliable
        ranking but more compute.
    score
        ``ForecastMetrics -> float``; lower is better. Default: RMSE.

    Returns
    -------
    ForecastResult
        ``result.model`` is fit on the full series and ready to
        ``.predict(horizon)``; ``result.metrics`` is the winner's fold-
        averaged hold-out performance; ``result.candidates`` is the
        full table so the UI can show why this one won.

    Raises
    ------
    RuntimeError
        If every candidate failed in every fold.
    """
    if candidates is None:
        candidates = default_candidates()
    if score is None:
        def score(m: ForecastMetrics) -> float:  # RMSE by default
            return m.rmse

    y = np.asarray(y, dtype=float)
    if dates is None:
        dates_arr = np.arange(y.size)
    else:
        dates_arr = np.asarray(dates)

    scored: list[tuple[BaseForecaster, ForecastMetrics]] = []
    for cand in candidates:
        metrics = _fold_metrics(cand, y, dates_arr, horizon, n_folds)
        if metrics is None:
            continue
        scored.append((cand, metrics))

    if not scored:
        raise RuntimeError(
            "select_forecaster: every candidate failed in every fold — "
            "check the series length and candidate compatibility."
        )

    # Pick the winner by the scoring rule.
    winner_cand, winner_metrics = min(scored, key=lambda p: score(p[1]))

    # Refit the winner on the full series for production predictions.
    final = winner_cand.__class__(**_clone_kwargs(winner_cand))
    final.fit(y, dates_arr)

    # Diagnostics go next to the winner. The diagnostics module is
    # imported here so callers without statsmodels can still use the
    # selector with only baselines (they'll get a near-empty diagnostics
    # object).
    from utils.forecasting.diagnostics import run_diagnostics  # noqa: PLC0415

    diag = run_diagnostics(
        name=final.name,
        y=y,
        residuals=final.residuals,
        baseline_residuals=_baseline_residuals(y, candidates),
    )

    return ForecastResult(
        model=final,
        metrics=winner_metrics,
        diagnostics=diag,
        candidates=[m for _, m in scored],
    )


def _baseline_residuals(
    y: np.ndarray, candidates: list[BaseForecaster]
) -> np.ndarray | None:
    """Return the in-sample residuals of a freshly fit :class:`NaiveForecaster`.

    These are the reference residuals for the Diebold-Mariano test. We
    always fit a fresh ``NaiveForecaster`` regardless of whether one is
    present in ``candidates``: refitting is O(n) and dropping the branch
    removes a dead conditional that did the same work twice.
    """
    from utils.forecasting.models import NaiveForecaster  # noqa: PLC0415

    baseline = NaiveForecaster().fit(y, None)
    return baseline.residuals
