"""
Multi-model forecasting with principled model selection.

Public API::

    from utils.forecasting import (
        select_forecaster,       # run a bakeoff and return the winner
        ForecastMetrics,         # per-model hold-out metrics
        ForecastDiagnostics,     # residual + stationarity tests
        BaseForecaster,          # ABC for custom models
        ALL_FORECASTERS,         # built-in candidate set
    )

Design goals:
- Every candidate implements ``fit(y, dates) -> self`` and
  ``predict(horizon) -> np.ndarray``, so the selector is model-agnostic.
- Evaluation uses *expanding-window* backtesting — more realistic than a
  single train/test split for monthly economic data.
- Selection is driven by a scoring rule over :class:`ForecastMetrics`
  (default: minimum RMSE). Users can pass their own to trade off bias,
  calibration, or likelihood-style metrics (AIC/BIC).
- Diagnostics (stationarity, residual whiteness, normality, pairwise
  Diebold-Mariano) live next to the winner so the UI can show the
  statistical receipts behind each forecast.
"""
from utils.forecasting.base import (
    BaseForecaster,
    ForecastDiagnostics,
    ForecastMetrics,
    ForecastResult,
)
from utils.forecasting.models import (
    ALL_FORECASTERS,
    ARIMAForecaster,
    ETSForecaster,
    LSTMForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from utils.forecasting.selection import select_forecaster

__all__ = [
    "ALL_FORECASTERS",
    "ARIMAForecaster",
    "BaseForecaster",
    "ETSForecaster",
    "ForecastDiagnostics",
    "ForecastMetrics",
    "ForecastResult",
    "LSTMForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "select_forecaster",
]
