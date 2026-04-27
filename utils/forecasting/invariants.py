"""
Mini statistical-rigor invariants for forecast correctness.

These are pure-Python guard checks the orchestrator can run on any
``ForecastResult`` (or any forecast point/CI/diagnostics tuple) before
the AI writes a narrative about it. They're cheap (microseconds), pure
(no I/O), and deterministic (no LLM calls), so they can sit on the hot
path of every Run-Forecast click.

Each invariant returns ``(ok: bool, message: str)``. Composite checks
collect every failure so the reviewer agent can include them in its
critique prompt — a forecast that violates "point ∈ historical range
± 2σ" should never sail through quietly.

Academic-rigor citations
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*
  (3rd ed., 2021), §4.5 — residual diagnostics, §5.5 — prediction
  intervals.
- Diebold & Mariano (1995, *J. Bus. Econ. Stat.*) — comparative
  predictive accuracy.
- Ljung & Box (1978, *Biometrika*) — autocorrelation white-noise test.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


# --------------------------------------------------------------------------
# Atomic invariants
# --------------------------------------------------------------------------
def point_within_historical_envelope(
    point: float,
    history: Sequence[float],
    *,
    sigma_buffer: float = 3.0,
) -> tuple[bool, str]:
    """
    A 2-year forecast that lands outside ``[mean - 3σ, mean + 3σ]`` of
    the observed series is almost always model failure or a data-cleaning
    bug. ``sigma_buffer=3`` is the standard "extreme outlier" threshold
    (Tukey 1977 fences); tighten to 2 for "anomaly" or relax to 4 for
    rare-event tolerance.
    """
    history = [v for v in history if v is not None and not _isnan(v)]
    if len(history) < 12:
        return True, "too few observations to form an envelope"
    mean = sum(history) / len(history)
    var = sum((v - mean) ** 2 for v in history) / max(1, len(history) - 1)
    sigma = math.sqrt(var)
    lo, hi = mean - sigma_buffer * sigma, mean + sigma_buffer * sigma
    if not (lo <= point <= hi):
        return False, (
            f"point forecast {point:.2f} falls outside the historical "
            f"{sigma_buffer}σ envelope [{lo:.2f}, {hi:.2f}] "
            f"(history mean {mean:.2f}, σ {sigma:.2f})"
        )
    return True, ""


def ci_contains_point(
    point: float,
    ci_lower: float | None,
    ci_upper: float | None,
) -> tuple[bool, str]:
    """The 95 % prediction interval should always contain its own
    point estimate. A failure means the model's interval code is
    inverted or the point + interval came from different runs."""
    if ci_lower is None or ci_upper is None:
        return True, "no CI to check"
    if not (ci_lower <= point <= ci_upper):
        return False, (
            f"point {point:.2f} not contained in CI "
            f"[{ci_lower:.2f}, {ci_upper:.2f}] — model bug"
        )
    return True, ""


def ci_width_reasonable(
    ci_lower: float | None,
    ci_upper: float | None,
    history: Sequence[float],
    *,
    max_width_sigmas: float = 8.0,
) -> tuple[bool, str]:
    """
    A CI wider than ``max_width_sigmas × σ_history`` is essentially
    "model says anything"; usually a sign of an over-fit or
    diverging model. Default ``max_width_sigmas=8`` lets ETS at
    long horizons through but flags exploded ARIMA fits.
    """
    if ci_lower is None or ci_upper is None:
        return True, "no CI to check"
    history = [v for v in history if v is not None and not _isnan(v)]
    if len(history) < 12:
        return True, "too few observations to assess CI width"
    mean = sum(history) / len(history)
    var = sum((v - mean) ** 2 for v in history) / max(1, len(history) - 1)
    sigma = math.sqrt(var)
    width = ci_upper - ci_lower
    cap = max_width_sigmas * sigma
    if width > cap:
        return False, (
            f"CI width {width:.2f} exceeds {max_width_sigmas}σ of history "
            f"({cap:.2f}); model is essentially uninformative"
        )
    return True, ""


def rmse_against_baseline(
    rmse: float | None,
    naive_rmse: float | None,
    *,
    epsilon: float = 0.0,
) -> tuple[bool, str]:
    """
    A "winning" model whose RMSE beats the naive baseline by less
    than ``epsilon`` should not really win — call it out so the
    reviewer prompt notes the caveat. Common in low-volatility
    series where naive is hard to beat.
    """
    if rmse is None or naive_rmse is None:
        return True, ""
    if rmse >= naive_rmse - epsilon:
        return False, (
            f"winning model RMSE {rmse:.3f} does not beat naive "
            f"baseline ({naive_rmse:.3f}); the bake-off chose this "
            f"on a tie-break — narrative should reflect uncertainty"
        )
    return True, ""


# --------------------------------------------------------------------------
# Composite — check everything, return list of failures
# --------------------------------------------------------------------------
def audit_forecast(
    *,
    history: Sequence[float],
    point: float,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    rmse: float | None = None,
    naive_rmse: float | None = None,
) -> list[str]:
    """
    Run every invariant against a forecast. Returns the list of
    failure messages (empty list ⇒ everything passes). The orchestrator
    can stuff failures into the reviewer's critique so the AI knows
    not to claim certainty the data doesn't support.
    """
    failures: list[str] = []
    for ok, msg in (
        point_within_historical_envelope(point, history),
        ci_contains_point(point, ci_lower, ci_upper),
        ci_width_reasonable(ci_lower, ci_upper, history),
        rmse_against_baseline(rmse, naive_rmse),
    ):
        if not ok:
            failures.append(msg)
    return failures


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _isnan(v) -> bool:
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return False
