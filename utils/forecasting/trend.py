"""
Pure-function trend analytics used by the tab UIs.

Two kinds of output:

- :func:`summarize_trend` returns a :class:`TrendSummary` describing
  level / deltas / range / volatility. The LFP tab renders one row
  per series in its trend-summary table.
- :func:`rolling_statistics` returns `(mean, std)` arrays for the
  input series with an ``NaN``-safe rolling window — fed straight
  into the chart as an overlay + shaded band.

Everything here is numpy + Python stdlib; no pandas, no plotting.
That keeps the module cheap to test and easy to reason about.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class TrendSummary:
    """Compact level / change / range / volatility stats for a series."""

    current: float          #: latest observed value
    delta_1y: float | None  #: absolute change vs 12 periods back; None if series is shorter
    pct_1y: float | None    #: percent change vs 12 periods back
    delta_5y: float | None  #: absolute change vs 60 periods back
    pct_5y: float | None    #: percent change vs 60 periods back
    all_time_high: float
    all_time_low: float
    volatility: float       #: population stdev over the full span
    n_obs: int              #: count of finite observations actually used


def _finite_array(values: Sequence[float]) -> np.ndarray:
    """Coerce to float64 ndarray and drop NaN/Inf."""
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def summarize_trend(values: Sequence[float], *, periods_per_year: int = 12) -> TrendSummary | None:
    """
    Summary stats for a time-series. Returns ``None`` when the input
    has no finite observations. Deltas over windows longer than the
    series length return ``None`` rather than an extrapolated guess.
    """
    arr = _finite_array(values)
    if arr.size == 0:
        return None

    def _step_delta(step: int) -> tuple[float | None, float | None]:
        if arr.size <= step:
            return None, None
        base = arr[-(step + 1)]
        current = arr[-1]
        abs_d = float(current - base)
        pct = float((current - base) / base * 100.0) if base else None
        return abs_d, pct

    d1, p1 = _step_delta(periods_per_year)
    d5, p5 = _step_delta(5 * periods_per_year)

    return TrendSummary(
        current=float(arr[-1]),
        delta_1y=d1,
        pct_1y=p1,
        delta_5y=d5,
        pct_5y=p5,
        all_time_high=float(np.max(arr)),
        all_time_low=float(np.min(arr)),
        volatility=float(np.std(arr, ddof=0)),
        n_obs=int(arr.size),
    )


def rolling_statistics(
    values: Sequence[float], *, window: int = 12, min_obs: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trailing rolling mean + std, NaN-safe (NaN in ⇒ NaN out for that window).

    Returns two float arrays of the same length as the input. Each
    position holds the stat computed over ``[i - window + 1 .. i]``
    (inclusive) when that window has at least ``min_obs`` finite
    observations, otherwise NaN. ``min_obs`` defaults to
    ``ceil(window / 2)`` so the output starts sooner than a strict
    ``window``-point requirement would allow.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if min_obs is None:
        min_obs = (window + 1) // 2
    if min_obs < 1 or min_obs > window:
        raise ValueError(f"min_obs out of range: {min_obs}")

    arr = np.asarray(list(values), dtype=float)
    n = arr.size
    means = np.full(n, np.nan, dtype=float)
    stds = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo = max(0, i - window + 1)
        chunk = arr[lo : i + 1]
        mask = np.isfinite(chunk)
        count = int(mask.sum())
        if count < min_obs:
            continue
        vals = chunk[mask]
        means[i] = float(vals.mean())
        stds[i] = float(vals.std(ddof=0))
    return means, stds
