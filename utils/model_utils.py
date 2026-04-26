"""
Lightweight numerical helpers for time-series forecasting code.

Historically this module also held the original LSTM training pipeline
(``build_lstm_rnn`` / ``train_test_rnn`` / ``forecast_with_model``)
that the early dashboard called directly. The ``utils.forecasting``
package (Phase A) replaced that flow with a model-bakeoff harness;
the LSTM lives there now as ``utils.forecasting.models.LSTMForecaster``.

Only :func:`create_sliding_windows` survived the migration — it's a
pure-numpy primitive used by every forecaster that consumes a sliding
window of past observations, so it's worth keeping discoverable here
without dragging tensorflow into module load.
"""
from __future__ import annotations

import numpy as np


def create_sliding_windows(
    series: np.ndarray, window_size: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a 1-D series into ``(X, Y)`` arrays for one-step-ahead training.

    For input ``series = [a, b, c, d, e]`` and ``window_size = 2``
    returns::

        X = [[a, b], [b, c], [c, d]]
        Y = [c, d, e]

    The function is fully NumPy and side-effect-free; consumers
    (LSTM, sliding-window XGBoost, etc.) reshape ``X`` for their own
    input layout downstream.
    """
    X, Y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        Y.append(series[i + window_size])
    return np.array(X), np.array(Y)
