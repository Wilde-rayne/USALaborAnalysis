"""
Base types for the forecasting framework.

Everything else in ``utils.forecasting`` depends on the abstractions here
and nothing else in the project. Keep this file free of pandas/scipy
imports — the data types live here so they can be safely imported and
pickled without dragging in a numerical stack.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


# --------------------------------------------------------------------------
# Metrics + diagnostics data classes
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ForecastMetrics:
    """Point-forecast quality on a held-out window."""

    model: str
    mae: float              # mean absolute error
    rmse: float             # root mean squared error
    mape: float             # mean absolute percentage error
    smape: float            # symmetric MAPE — bounded, robust when y≈0
    bias: float             # mean(y_true - y_pred); >0 means under-forecast
    n_eval: int             # number of (y_true, y_pred) pairs scored
    horizon: int            # forecast horizon used in scoring, in steps
    aic: float | None = None
    bic: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ForecastDiagnostics:
    """
    Statistical receipts for a fitted model.

    All p-values are floats in [0, 1]; ``None`` if the test could not be
    run (e.g. residuals too short). Higher-is-better or lower-is-better
    depends on the test — consult the individual attribute docstrings.
    """

    model: str
    # Null: y has a unit root (non-stationary). Small p => stationary.
    adf_pvalue: float | None = None
    # Null: y is stationary. Large p => stationary.
    kpss_pvalue: float | None = None
    # Null: residuals are uncorrelated (white noise). Large p => good fit.
    ljungbox_pvalue: float | None = None
    # Null: residuals are normally distributed. Large p => normal.
    jarquebera_pvalue: float | None = None
    # Diebold-Mariano vs a baseline model. p<0.05 => this model is
    # statistically different from baseline. Sign-convention: negative
    # DM stat means THIS model beats the baseline.
    dm_stat_vs_baseline: float | None = None
    dm_pvalue_vs_baseline: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ForecastResult:
    """
    The full output of ``select_forecaster``: the chosen model (fit on
    the entire series), the winner's hold-out metrics, diagnostics, and
    the losing candidates' metrics for display/audit.
    """

    model: "BaseForecaster"
    metrics: ForecastMetrics
    diagnostics: ForecastDiagnostics
    candidates: list[ForecastMetrics] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.model.name


# --------------------------------------------------------------------------
# Forecaster ABC
# --------------------------------------------------------------------------
class BaseForecaster(ABC):
    """
    Minimal sklearn-style interface for univariate time-series models.

    Concrete subclasses must:
    - set ``name`` (used in logs / UI / metric tables);
    - implement :meth:`fit` to consume a 1-D float array + matching
      ``DatetimeIndex``-like array of timestamps;
    - implement :meth:`predict` to produce the next ``horizon`` values;
    - optionally expose :attr:`residuals` after ``fit`` so the diagnostics
      module can run Ljung-Box / Jarque-Bera on them.
    """

    name: str = "BaseForecaster"

    def __init__(self) -> None:
        self._y: np.ndarray | None = None
        self._fitted: bool = False

    @abstractmethod
    def fit(self, y: np.ndarray, dates) -> "BaseForecaster":
        ...

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        ...

    # Subclasses that can expose residuals override this. Default is the
    # (trivial) in-sample zero-residual, which is never useful for
    # diagnostics — so the diagnostics module checks for None explicitly.
    @property
    def residuals(self) -> np.ndarray | None:
        return None

    # Some models have an informative fit likelihood; override to surface it.
    @property
    def aic(self) -> float | None:
        return None

    @property
    def bic(self) -> float | None:
        return None

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name} must be .fit() before .predict()")
