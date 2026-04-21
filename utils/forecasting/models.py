"""
Concrete forecasters behind :class:`BaseForecaster`.

Heavy dependencies (statsmodels, tensorflow) are imported **inside**
``fit`` so importing this module is cheap — a test file that only
exercises the naive baselines doesn't pay the TF startup cost.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from utils.forecasting.base import BaseForecaster

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Baselines — pure numpy, zero dependencies
# --------------------------------------------------------------------------
class NaiveForecaster(BaseForecaster):
    """
    Persistence model: the next ``horizon`` values equal ``y[-1]``.

    This is the weakest sensible baseline. Any model worth shipping must
    beat it; otherwise your "forecast" is just a flat line.
    """

    name = "naive"

    def __init__(self) -> None:
        super().__init__()
        self._last: float = 0.0

    def fit(self, y: np.ndarray, dates) -> "NaiveForecaster":
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            raise ValueError(f"{self.name}: empty series")
        self._last = float(y[-1])
        self._y = y
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._require_fitted()
        return np.full(horizon, self._last, dtype=float)

    @property
    def residuals(self) -> np.ndarray | None:
        # In-sample residuals for a persistence model: y[t] - y[t-1].
        if self._y is None or self._y.size < 2:
            return None
        return self._y[1:] - self._y[:-1]


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal persistence: predicts ``y[t] = y[t - season]``.

    For monthly employment data ``season=12`` captures the obvious
    annual cycle and is often shockingly hard to beat with complex
    models — precisely why it belongs in the bakeoff.
    """

    name = "seasonal_naive"

    def __init__(self, season: int = 12) -> None:
        super().__init__()
        if season < 1:
            raise ValueError("season must be >= 1")
        self.season = season

    def fit(self, y: np.ndarray, dates) -> "SeasonalNaiveForecaster":
        y = np.asarray(y, dtype=float)
        if y.size < self.season + 1:
            raise ValueError(
                f"{self.name}: need at least season+1={self.season + 1} obs, got {y.size}"
            )
        self._y = y
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._require_fitted()
        assert self._y is not None
        tail = self._y[-self.season :]
        # Repeat the last season block to cover the horizon.
        reps = int(np.ceil(horizon / self.season))
        return np.tile(tail, reps)[:horizon]

    @property
    def residuals(self) -> np.ndarray | None:
        if self._y is None or self._y.size <= self.season:
            return None
        return self._y[self.season :] - self._y[: -self.season]


# --------------------------------------------------------------------------
# Exponential smoothing (statsmodels)
# --------------------------------------------------------------------------
class ETSForecaster(BaseForecaster):
    """
    Holt-Winters exponential smoothing. Strong on monthly seasonal data
    with limited history — typically the baseline to beat in labor
    forecasting.

    ``trend`` and ``seasonal`` follow statsmodels' conventions:
    "add" for additive, "mul" for multiplicative, or ``None`` to
    disable.
    """

    name = "ets"

    def __init__(
        self,
        trend: str | None = "add",
        seasonal: str | None = "add",
        seasonal_periods: int = 12,
    ) -> None:
        super().__init__()
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._fitted_model: Any = None

    def fit(self, y: np.ndarray, dates) -> "ETSForecaster":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa: PLC0415

        y = np.asarray(y, dtype=float)
        if y.size < self.seasonal_periods * 2:
            # Not enough to estimate a seasonal component — fall back to
            # non-seasonal Holt. ETS with a too-short series silently
            # produces nonsense, so we downgrade explicitly.
            seasonal = None
            seasonal_periods = None
            logger.info(
                f"[ETS] series length {y.size} < 2 * seasonal_periods; "
                "falling back to non-seasonal."
            )
        else:
            seasonal = self.seasonal
            seasonal_periods = self.seasonal_periods

        model = ExponentialSmoothing(
            y,
            trend=self.trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        self._fitted_model = model.fit(optimized=True)
        self._y = y
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._require_fitted()
        return np.asarray(self._fitted_model.forecast(horizon), dtype=float)

    @property
    def residuals(self) -> np.ndarray | None:
        if self._fitted_model is None:
            return None
        return np.asarray(self._fitted_model.resid, dtype=float)

    @property
    def aic(self) -> float | None:
        return None if self._fitted_model is None else float(self._fitted_model.aic)

    @property
    def bic(self) -> float | None:
        return None if self._fitted_model is None else float(self._fitted_model.bic)


# --------------------------------------------------------------------------
# LSTM — recurrent neural network
# --------------------------------------------------------------------------
class LSTMForecaster(BaseForecaster):
    """
    Single-layer LSTM (32 units) with a dense read-out, trained on a
    sliding window of the series.

    Deterministic subject to TensorFlow's seeding caveats: we set NumPy,
    Python, and TF global seeds in :meth:`fit` so two runs with the
    same data produce the same forecast within numerical tolerance.
    """

    name = "lstm"

    def __init__(
        self,
        window: int = 12,
        epochs: int = 10,
        batch_size: int = 16,
        units: int = 32,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.units = units
        self.seed = seed
        self._model: Any = None
        self._last_window: np.ndarray | None = None
        self._resid: np.ndarray | None = None

    def fit(self, y: np.ndarray, dates) -> "LSTMForecaster":
        import os  # noqa: PLC0415
        import random  # noqa: PLC0415

        # Deterministic seeding for reproducibility.
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        import tensorflow as tf  # noqa: PLC0415

        tf.random.set_seed(self.seed)

        y = np.asarray(y, dtype=float)
        if y.size < self.window + 2:
            raise ValueError(
                f"{self.name}: need at least window+2={self.window + 2} obs, got {y.size}"
            )

        # Sliding windows: X[i] = y[i : i+w], Y[i] = y[i+w].
        X = np.stack([y[i : i + self.window] for i in range(len(y) - self.window)])
        Y = y[self.window :]
        X = X.reshape(-1, self.window, 1)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.window, 1)),
                tf.keras.layers.LSTM(self.units, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, Y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        # In-sample residuals for diagnostics.
        yhat = model.predict(X, verbose=0).flatten()
        self._resid = Y - yhat

        self._model = model
        self._last_window = y[-self.window :].copy()
        self._y = y
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        self._require_fitted()
        assert self._model is not None and self._last_window is not None

        window = self._last_window.copy()
        preds = []
        for _ in range(horizon):
            inp = window[-self.window :].reshape(1, self.window, 1)
            yhat = float(self._model.predict(inp, verbose=0)[0, 0])
            preds.append(yhat)
            window = np.append(window, yhat)
        return np.asarray(preds, dtype=float)

    @property
    def residuals(self) -> np.ndarray | None:
        return self._resid


# --------------------------------------------------------------------------
# Default candidate set
# --------------------------------------------------------------------------
def default_candidates(seasonal_periods: int = 12) -> list[BaseForecaster]:
    """
    Build the default bakeoff line-up. Kept as a function (not a module-
    level constant) because each selector call needs fresh instances —
    these models keep fitted state on ``self``.
    """
    return [
        NaiveForecaster(),
        SeasonalNaiveForecaster(season=seasonal_periods),
        ETSForecaster(seasonal_periods=seasonal_periods),
        LSTMForecaster(window=seasonal_periods),
    ]


#: Read-only tuple of the built-in candidate classes, for discoverability.
ALL_FORECASTERS: tuple[type[BaseForecaster], ...] = (
    NaiveForecaster,
    SeasonalNaiveForecaster,
    ETSForecaster,
    LSTMForecaster,
)
