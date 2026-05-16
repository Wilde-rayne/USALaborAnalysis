"""
Labor Force Participation — tailored multi-state forecast.

Manual review asked to "tailor it to their specific needs" and to
include "if there is a match or requirements they are looking for".
This tab now lets the user:

    1. Pick one *focus* state (the one getting the headline forecast),
    2. Pick one or more *peer* states (overlaid on the chart and
       evaluated against the same requirements),
    3. Pick the *primary metric* — Labor Force Participation Rate or
       Unemployment Rate — both are standard LAUS outputs,
    4. Set numeric *requirements* — minimum for LFPR or maximum for
       Unemployment Rate — and
    5. See a "match table" showing which states clear the bar at the
       end of the forecast horizon.

Each selected state runs through the same bakeoff
(Naive / Seasonal-Naive / ETS / ARIMA by default) and the chart shows
point forecasts + 95% prediction intervals for every series.
"""
from __future__ import annotations

import logging

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from utils.constants import ALL_STATES
from utils.data_pipeline import ensure_data
from utils.forecasting import ForecastResult, select_forecaster
from utils.forecasting.trend import (
    TrendSummary,
    rolling_statistics,
    summarize_trend,
)
from utils.agents import blurb_async
from utils.ontology import ONTOLOGY
from tabs._methodology import LFPR_DENOMINATOR_NOTE, methodology_panel
from tabs._components import (
    PANEL_BLURB_TYPE,
    error_boundary,
    figure_panel,
    progress_strip,
    render_blurb,
    tab_recap,
)

logger = logging.getLogger(__name__)

def _current_year() -> int:
    """
    Indirection over ``datetime.now().year`` so tests around the
    year-boundary (gap-anchor computation in ``update_lfp``) can
    monkey-patch a fixed year without having to freezegun the whole
    clock. Production callers always read the wall clock.
    """
    return datetime.now().year


# ---------------------------------------------------------------------------
# Cache: (metric_key, state_code, years_ahead) → ForecastResult.
# Gets populated on demand inside the callback. Wrapped in a small
# OrderedDict-backed LRU bound so a long-lived gunicorn worker that
# sees many distinct (metric, state, horizon) clicks doesn't leak
# memory across its lifetime. 64 entries covers ~5 horizons × ~12
# states × the two metrics without churn.
# ---------------------------------------------------------------------------
from collections import OrderedDict


class _LRUCache(OrderedDict):
    """Bounded OrderedDict — newest insertion stays at the right end."""

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.maxsize:
            self.popitem(last=False)


lfp_model_cache: _LRUCache = _LRUCache(maxsize=64)

#: What the user can forecast on this tab. Both are LAUS-derived and
#: percent-scale, which simplifies the chart axis + requirement UX.
METRIC_OPTIONS: list[dict] = [
    {"label": "Labor Force Participation Rate", "value": "LFPR"},
    {"label": "Unemployment Rate",              "value": "Unemployment_Rate"},
]

#: Threshold direction per metric — "min" = state must meet or exceed
#: the value (LFPR is higher-is-better); "max" = state must be at or
#: below (Unemployment is lower-is-better).
METRIC_THRESHOLD_DIRECTION: dict[str, str] = {
    "LFPR": "min",
    "Unemployment_Rate": "max",
}


# ---------------------------------------------------------------------------
# Panel loading + derived columns
# ---------------------------------------------------------------------------
def _compute_unemployment_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    The widened panel has ``{state}_Unemployment`` and
    ``{state}_Labor_Force`` but not yet a precomputed rate. Derive it
    inline: ``rate = 100 * unemployment / labor_force``. Keeps the
    merger lean and lets the rate track any future data refresh.
    """
    state_codes = {c[:2] for c in df.columns if c.endswith("_Unemployment")}
    for st in state_codes:
        u = f"{st}_Unemployment"
        lf = f"{st}_Labor_Force"
        rate = f"{st}_Unemployment_Rate"
        if u in df.columns and lf in df.columns and rate not in df.columns:
            denom = pd.to_numeric(df[lf], errors="coerce")
            num = pd.to_numeric(df[u], errors="coerce")
            df[rate] = np.where(denom > 0, 100.0 * num / denom, np.nan)
    return df


def _load_lfp_panel() -> pd.DataFrame:
    """Load and dedupe the panel, then add derived unemployment rate columns."""
    from utils.data_pipeline import load_panel_df

    ensure_data()
    df = load_panel_df()
    df = _compute_unemployment_rate(df)
    return df


def _column_for(metric: str, state: str) -> str:
    """Resolve the wide-column name for a (metric, state) pair."""
    if metric == "LFPR":
        return f"{state}_Labor_Force_Participation_Rate"
    if metric == "Unemployment_Rate":
        return f"{state}_Unemployment_Rate"
    raise ValueError(f"unknown metric: {metric!r}")


def _metric_label(metric: str) -> str:
    return next(o["label"] for o in METRIC_OPTIONS if o["value"] == metric)


def _forecast_state(
    metric: str, state: str, years_ahead: int, df: pd.DataFrame
) -> ForecastResult | None:
    """Fit the bakeoff for (metric, state, horizon). Caches across clicks."""
    key = (metric, state, years_ahead)
    if key in lfp_model_cache:
        return lfp_model_cache[key]

    col = _column_for(metric, state)
    if col not in df.columns:
        logger.info(f"[LFP] {col} missing for {state}")
        return None
    series = df[["date", col]].dropna()
    if len(series) < 24:
        logger.info(f"[LFP] {state}/{metric}: only {len(series)} obs, skipping")
        return None

    horizon = years_ahead * 12
    y = series[col].astype(float).to_numpy()
    dates = series["date"].to_numpy()
    try:
        result = select_forecaster(y, dates=dates, horizon=horizon, n_folds=3)
    except RuntimeError as exc:
        logger.warning(f"[LFP] {state}/{metric}: bakeoff failed — {exc}")
        return None
    lfp_model_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Display helpers (tables + charts)
# ---------------------------------------------------------------------------
def _trend_summary_table(
    summaries: dict[str, TrendSummary | None], unit: str
) -> html.Div:
    """Level / change / range / volatility table — one row per series."""

    def _pct(p: float | None) -> str:
        if p is None:
            return "—"
        arrow = "▲" if p > 0 else "▼" if p < 0 else "–"
        return f"{arrow} {p:+.2f}%"

    rows = []
    for label, s in summaries.items():
        if s is None:
            continue
        rows.append(
            html.Tr(
                [
                    html.Td(label),
                    html.Td(f"{s.current:.2f}{unit}"),
                    html.Td(_pct(s.pct_1y)),
                    html.Td(_pct(s.pct_5y)),
                    html.Td(f"{s.all_time_low:.2f}{unit}"),
                    html.Td(f"{s.all_time_high:.2f}{unit}"),
                    html.Td(f"{s.volatility:.2f}"),
                    html.Td(f"{s.n_obs}"),
                ]
            )
        )
    if not rows:
        return html.Div()
    return html.Div(
        [
            html.H6("Trend summary"),
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th(c)
                                for c in (
                                    "Series",
                                    "Current",
                                    "1-yr Δ",
                                    "5-yr Δ",
                                    "All-time low",
                                    "All-time high",
                                    "σ (vol)",
                                    "n obs",
                                )
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
                className="table table-sm table-striped mb-3",
            ),
        ]
    )


def _model_rationale_table(col_label: str, result: ForecastResult) -> html.Div:
    """Per-series candidate table + diagnostics one-liner."""
    rows = []
    for m in sorted(result.candidates, key=lambda x: x.rmse):
        is_winner = m.model == result.name
        cells = [
            html.Td(("★ " if is_winner else "") + m.model),
            html.Td(f"{m.rmse:.3f}"),
            html.Td(f"{m.mae:.3f}"),
            html.Td(f"{m.mape:.2f}%" if m.mape == m.mape else "—"),
            html.Td(f"{m.bias:+.3f}"),
        ]
        rows.append(html.Tr(cells, style={"fontWeight": "bold"} if is_winner else {}))

    d = result.diagnostics

    def _fmt_p(p):
        return "—" if p is None else f"{p:.3f}"

    return html.Div(
        [
            html.H6(f"Model selection — {col_label}"),
            html.Table(
                [
                    html.Thead(
                        html.Tr([html.Th(c) for c in ("Model", "RMSE", "MAE", "MAPE", "Bias")])
                    ),
                    html.Tbody(rows),
                ],
                className="table table-sm table-striped mb-2",
            ),
            html.Div(
                html.Small(
                    f"Diagnostics — ADF p={_fmt_p(d.adf_pvalue)} · "
                    f"KPSS p={_fmt_p(d.kpss_pvalue)} · "
                    f"Ljung-Box p={_fmt_p(d.ljungbox_pvalue)} · "
                    f"Jarque-Bera p={_fmt_p(d.jarquebera_pvalue)} · "
                    f"Diebold-Mariano p={_fmt_p(d.dm_pvalue_vs_baseline)} "
                    "(vs naive; negative DM stat ⇒ winner beats baseline)",
                    className="text-muted",
                ),
                className="mb-3",
            ),
        ]
    )


def _compute_forecast_points(
    states_in_scope: list[str],
    state_forecasts: dict[str, ForecastResult],
    months: int,
) -> dict[str, dict | None]:
    """
    Resolve point estimate + 95% CI for every state once, so both the
    rendered match table and the AI's pass/fail classifier can read
    from the same dict instead of each calling ``predict_interval``
    independently. ``None`` for a state means the bake-off had no
    usable result for it (UI renders an em-dash row).
    """
    out: dict[str, dict | None] = {}
    for st in states_in_scope:
        result = state_forecasts.get(st)
        if result is None:
            out[st] = None
            continue
        interval = result.model.predict_interval(months, alpha=0.05)
        if interval is not None:
            preds, lower, upper = interval
            ci_lo, ci_hi = float(lower[-1]), float(upper[-1])
        else:
            preds = result.model.predict(months)
            ci_lo = ci_hi = float(preds[-1])
        out[st] = {
            "point":      float(preds[-1]),
            "ci_lo":      ci_lo,
            "ci_hi":      ci_hi,
            "model_name": result.name,
            "rmse":       float(result.metrics.rmse),
        }
    return out


def _requirements_match_table(
    states_in_scope: list[str],
    forecast_points: dict[str, dict | None],
    months: int,
    metric: str,
    threshold: float | None,
) -> html.Div:
    """Which states clear the user's requirement at the forecast horizon?"""
    direction = METRIC_THRESHOLD_DIRECTION[metric]
    unit = "%"
    rows = []
    for st in states_in_scope:
        info = forecast_points.get(st)
        if info is None:
            rows.append(
                html.Tr(
                    [
                        html.Td(st),
                        html.Td("—", colSpan=4, className="text-muted small"),
                    ]
                )
            )
            continue
        point, ci_lo, ci_hi = info["point"], info["ci_lo"], info["ci_hi"]
        passes: bool | None
        if threshold is None:
            passes = None
        elif direction == "min":
            passes = point >= threshold
        else:  # "max"
            passes = point <= threshold

        verdict_cell: html.Td
        if passes is None:
            verdict_cell = html.Td("(no requirement)", className="text-muted small")
        elif passes:
            verdict_cell = html.Td("✓ meets", className="text-success fw-bold")
        else:
            verdict_cell = html.Td("✗ misses", className="text-danger fw-bold")

        rows.append(
            html.Tr(
                [
                    html.Td(st),
                    html.Td(f"{point:.2f}{unit}"),
                    html.Td(f"[{ci_lo:.2f}, {ci_hi:.2f}]{unit}"),
                    html.Td(f"{info['model_name']} (RMSE {info['rmse']:.2f})"),
                    verdict_cell,
                ]
            )
        )

    threshold_str = (
        f"Requirement: {_metric_label(metric)} "
        f"{'≥' if direction == 'min' else '≤'} {threshold:.2f}{unit}"
        if threshold is not None
        else f"Requirement: (none set — threshold left blank)"
    )

    return html.Div(
        [
            html.H6("Requirements match"),
            html.Small(threshold_str, className="text-muted d-block mb-2"),
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th(c)
                                for c in (
                                    "State",
                                    f"Forecast at +{months} mo",
                                    "95% CI",
                                    "Model (RMSE)",
                                    "Verdict",
                                )
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
                className="table table-sm table-striped mb-3",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Layout + callbacks
# ---------------------------------------------------------------------------
def _state_options() -> list[dict]:
    return [
        {"label": f"{ONTOLOGY.state(c).name} ({c})", "value": c}
        for c in sorted(ALL_STATES)
    ]


def render_layout():
    options = _state_options()
    default_focus = "IA" if "IA" in ALL_STATES else ALL_STATES[0]
    default_peers = [c for c in ("IL", "MN", "WI") if c in ALL_STATES and c != default_focus][:2]
    return html.Div(
        [
            html.H5("Labor Force Participation — Tailored Forecast"),
            html.P(
                "Pick a focus state plus peers, choose the metric and a "
                "requirement threshold, and the forecast will highlight which "
                "states clear the bar at the end of your horizon. Each state "
                "runs a Naive / Seasonal-Naive / Holt-Winters / ARIMA bakeoff "
                "and the chart shows the winning model's point forecast with "
                "a 95 % prediction interval.",
                className="text-muted small",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Focus state", htmlFor="lfp-focus-state"),
                            dcc.Dropdown(
                                id="lfp-focus-state",
                                options=options,
                                value=default_focus,
                                clearable=False,
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Label("Peer states (comparisons)", htmlFor="lfp-peer-states"),
                            dcc.Dropdown(
                                id="lfp-peer-states",
                                options=options,
                                value=default_peers,
                                multi=True,
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Label("Primary metric", htmlFor="lfp-metric"),
                            dcc.Dropdown(
                                id="lfp-metric",
                                options=METRIC_OPTIONS,
                                value="LFPR",
                                clearable=False,
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Label("Years ahead", htmlFor="lfp-years-slider"),
                            dcc.Slider(
                                id="lfp-years-slider",
                                min=1,
                                max=5,
                                step=1,
                                marks={i: str(i) for i in range(1, 6)},
                                value=2,
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Requirement threshold (leave blank to skip)",
                                htmlFor="lfp-threshold",
                            ),
                            dcc.Input(
                                id="lfp-threshold",
                                type="number",
                                placeholder="e.g. 60 (LFPR %) or 5 (unemp rate %)",
                                debounce=True,
                            ),
                            html.Small(
                                "LFPR uses the value as a minimum; Unemployment "
                                "rate uses it as a maximum.",
                                className="pi-muted",
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Run Forecast",
                                id="lfp-run",
                                className="btn btn-primary",
                            ),
                        ],
                        className="mb-3",
                    ),
                ],
                className="pi-selector-grid",
            ),
            dcc.Loading(id="loading-lfp", children=html.Div(id="lfp-output")),
            # Run-id channel: the main click callback spawns a daemon
            # thread that fills ``utils.agents.blurb_async._run_state``
            # as each panel completes. The polling callback below
            # reads from there every ~2 s and updates only the panels
            # that have populated since the last tick — every panel
            # pops in as soon as its individual Ollama call returns.
            dcc.Store(id="lfp-blurb-payload", data=None),
            # Tick driver. Disabled by default; enabled by the click
            # callback and disabled again by the poll callback once
            # every panel has filled.
            dcc.Interval(
                id="lfp-progress-tick",
                interval=2_000,
                n_intervals=0,
                disabled=True,
            ),
        ]
    )


def _blurb_id(section: str) -> dict:
    """Compose the pattern-matching ID for a per-tab AI blurb placeholder."""
    return {"type": PANEL_BLURB_TYPE, "tab": "lfp", "section": section}


def _round2(value: float | None) -> float | None:
    """Round to two decimals; pass ``None`` through so view_states stay JSON-clean."""
    return None if value is None else round(float(value), 2)


def _classify_requirements(
    states_in_scope: list[str],
    forecast_points: dict[str, dict | None],
    direction: str,
    threshold: float | None,
) -> tuple[list[dict], list[dict]]:
    """
    Split states into ``passing`` / ``failing`` lists for the
    requirements view_state, reading from the same ``forecast_points``
    dict the rendered match table uses so the AI verdict can never
    disagree with what the user sees. ``([], [])`` when no threshold
    is set.
    """
    if threshold is None:
        return [], []
    passing: list[dict] = []
    failing: list[dict] = []
    for st in states_in_scope:
        info = forecast_points.get(st)
        if info is None:
            continue
        point = info["point"]
        passes = (
            (direction == "min" and point >= threshold)
            or (direction == "max" and point <= threshold)
        )
        (passing if passes else failing).append(
            {"code": st, "value": _round2(point)}
        )
    return passing, failing


def _trend_view_rows(
    summaries: dict[str, TrendSummary | None],
) -> list[dict]:
    """Compact per-state rows for the trend panel's view_state."""
    rows: list[dict] = []
    for code, summary in summaries.items():
        if summary is None:
            continue
        rows.append({
            "code": code,
            "current": _round2(summary.current),
            "five_yr_change": _round2(summary.delta_5y),
        })
    return rows


def register_callbacks(app):
    @app.callback(
        Output("lfp-output", "children"),
        Output("lfp-blurb-payload", "data"),
        Output("lfp-progress-tick", "disabled"),
        Output("lfp-progress-tick", "n_intervals"),
        Output("pi-active-view", "data", allow_duplicate=True),
        Input("lfp-run", "n_clicks"),
        State("lfp-focus-state", "value"),
        State("lfp-peer-states", "value"),
        State("lfp-metric", "value"),
        State("lfp-years-slider", "value"),
        State("lfp-threshold", "value"),
        prevent_initial_call=True,
    )
    @error_boundary(fallback_id="lfp-output", extra_outputs=4)
    def update_lfp(n_clicks, focus_state, peer_states, metric, years_ahead, threshold):
        if not n_clicks:
            raise PreventUpdate
        if not focus_state:
            return (
                html.Div("Pick a focus state first.", className="alert alert-warning"),
                None,
                True,   # interval disabled
                0,      # n_intervals reset
                None,   # active-view cleared
            )

        df = _load_lfp_panel()
        peer_states = [s for s in (peer_states or []) if s and s != focus_state]
        all_states = [focus_state] + peer_states
        months = years_ahead * 12
        metric_label = _metric_label(metric)

        # Fit bakeoffs for every state in scope.
        state_forecasts: dict[str, ForecastResult] = {}
        for st in all_states:
            result = _forecast_state(metric, st, years_ahead, df)
            if result is not None:
                state_forecasts[st] = result

        focus_result = state_forecasts.get(focus_state)
        if focus_result is None:
            return (
                html.Div(
                    f"No {metric_label} series available for {focus_state}.",
                    className="alert alert-warning",
                ),
                None,
                True,   # interval disabled
                0,      # n_intervals reset
                None,   # active-view cleared
            )

        # ----- Forecast horizon with the Phase-J gap -----
        # The user wants predictions to start at the *next-year boundary*
        # so the chart visually distinguishes data already collected, the
        # months that have elapsed but BLS hasn't published yet (the
        # "data lag"), and the model's forward look.
        last_date = df["date"].max()
        # ``_current_year()`` is the injection seam — tests monkey-patch
        # it to freeze the gap-anchor across year boundaries instead of
        # freezegunning ``datetime.now``.
        gap_anchor = pd.Timestamp(year=_current_year() + 1, month=1, day=1)
        gap_months = max(
            0,
            (gap_anchor.year - last_date.year) * 12 + (gap_anchor.month - last_date.month) - 1,
        )
        # Predict far enough ahead that the last ``months`` predictions
        # land on or after ``gap_anchor``; we slice off the gap portion
        # from the displayed forecast so the line starts at the
        # current-year boundary instead of immediately after the last
        # actual observation.
        total_horizon = gap_months + months
        gap_dates = [last_date + pd.DateOffset(months=i + 1) for i in range(gap_months)]
        forecast_dates = [last_date + pd.DateOffset(months=gap_months + i + 1) for i in range(months)]

        # Colour palette: focus brand-blue, peers cycle through Plotly D3.
        palette = [
            "rgb(44,160,44)",
            "rgb(214,39,40)",
            "rgb(148,103,189)",
            "rgb(140,86,75)",
            "rgb(227,119,194)",
            "rgb(127,127,127)",
        ]

        traces: list = []

        # Focus state — historic heavy + forecast + CI band.
        focus_col = _column_for(metric, focus_state)
        hist = df[["date", focus_col]].dropna()
        traces.append(
            go.Scatter(
                x=hist["date"],
                y=hist[focus_col],
                mode="lines",
                name=f"{focus_state} historic",
                line=dict(color="rgb(31,119,180)", width=2),
            )
        )
        roll_mean, _ = rolling_statistics(hist[focus_col].to_numpy(), window=12)
        traces.append(
            go.Scatter(
                x=hist["date"],
                y=roll_mean,
                mode="lines",
                name=f"{focus_state} 12-mo avg",
                line=dict(color="rgba(31,119,180,0.55)", width=2, dash="dash"),
            )
        )
        focus_interval = focus_result.model.predict_interval(total_horizon, alpha=0.05)
        if focus_interval is not None:
            preds_full, lower_full, upper_full = focus_interval
            preds = preds_full[gap_months:]
            lower = lower_full[gap_months:]
            upper = upper_full[gap_months:]
            traces.append(
                go.Scatter(
                    x=forecast_dates,
                    y=upper,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            traces.append(
                go.Scatter(
                    x=forecast_dates,
                    y=lower,
                    mode="lines",
                    name=f"{focus_state} 95% CI",
                    fill="tonexty",
                    fillcolor="rgba(31,119,180,0.18)",
                    line=dict(width=0),
                    hoverinfo="skip",
                )
            )
        else:
            preds = focus_result.model.predict(total_horizon)[gap_months:]
            lower = upper = preds
        traces.append(
            go.Scatter(
                x=forecast_dates,
                y=preds,
                mode="lines+markers",
                name=(
                    f"{focus_state} forecast "
                    f"({focus_result.name} · RMSE {focus_result.metrics.rmse:.2f})"
                ),
                line=dict(color="rgb(31,119,180)", width=2.5),
                marker=dict(size=6),
            )
        )

        # Peers — historic dashed + forecast dotted, no CI band to keep
        # the chart readable.
        for i, st in enumerate(peer_states):
            peer_col = _column_for(metric, st)
            if peer_col not in df.columns:
                continue
            peer_hist = df[["date", peer_col]].dropna()
            colour = palette[i % len(palette)]
            traces.append(
                go.Scatter(
                    x=peer_hist["date"],
                    y=peer_hist[peer_col],
                    mode="lines",
                    name=f"{st} historic",
                    line=dict(color=colour, width=1, dash="dot"),
                    opacity=0.85,
                )
            )
            peer_result = state_forecasts.get(st)
            if peer_result is None:
                continue
            peer_preds = peer_result.model.predict(total_horizon)[gap_months:]
            traces.append(
                go.Scatter(
                    x=forecast_dates,
                    y=peer_preds,
                    mode="lines+markers",
                    name=(
                        f"{st} forecast "
                        f"({peer_result.name} · RMSE {peer_result.metrics.rmse:.2f})"
                    ),
                    line=dict(color=colour, width=1.5, dash="dot"),
                    marker=dict(size=4, symbol="diamond"),
                )
            )

        # Prediction is the focus on the LFP tab — default x-window
        # is the last ~10 years of history + the full forecast, so the
        # forecast takes ~30 % of the visible width instead of getting
        # squeezed against the right edge. A range-slider lets the
        # user drag back into deeper history when they want it.
        x_default_start = pd.Timestamp(
            year=max(int(df["date"].min().year), gap_anchor.year - 10),
            month=1, day=1,
        )
        x_default_end = forecast_dates[-1] + pd.DateOffset(months=2)

        fig = go.Figure(traces)
        fig.update_layout(
            title=dict(
                text=(
                    f"{metric_label} — {focus_state} (forecast starts {gap_anchor.year})"
                ),
                x=0,
                xanchor="left",
                font=dict(size=15),
            ),
            xaxis=dict(
                title=dict(text="Date", standoff=12),
                showgrid=True,
                gridcolor="rgba(127,127,127,0.18)",
                zeroline=False,
                range=[x_default_start.strftime("%Y-%m-%d"),
                       x_default_end.strftime("%Y-%m-%d")],
                rangeselector=dict(
                    buttons=[
                        dict(label="3y",  count=3,  step="year", stepmode="backward"),
                        dict(label="10y", count=10, step="year", stepmode="backward"),
                        dict(label="All", step="all"),
                    ],
                    bgcolor="rgba(255,255,255,0.65)",
                    activecolor="#2c7be5",
                ),
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(
                title=dict(text=f"{metric_label} (%)", standoff=8),
                showgrid=True,
                gridcolor="rgba(127,127,127,0.18)",
                zeroline=False,
                ticksuffix="%",
            ),
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(size=11),
            ),
            hoverlabel=dict(bgcolor="white", font=dict(size=12)),
            margin=dict(l=70, r=30, t=80, b=60),
            plot_bgcolor="rgba(255,255,255,0)",
            paper_bgcolor="rgba(255,255,255,0)",
        )
        # Phase J: visualise the data-lag gap so the user can see the
        # months where data is real-but-unpublished (no line) vs the
        # forward forecast (full line + CI fan). Use the low-level
        # ``add_shape`` + ``add_annotation`` API — the convenience
        # ``add_vrect`` / ``add_vline`` helpers do internal integer-
        # date arithmetic that breaks on Timestamps and ISO strings.
        gap_anchor_iso = gap_anchor.strftime("%Y-%m-%d")
        if gap_months > 0:
            fig.add_shape(
                type="rect",
                xref="x", yref="paper",
                x0=gap_dates[0].strftime("%Y-%m-%d"),
                x1=gap_anchor_iso,
                y0=0, y1=1,
                fillcolor="rgba(120, 120, 120, 0.10)",
                line=dict(width=0),
                layer="below",
            )
            fig.add_annotation(
                xref="x", yref="paper",
                x=gap_dates[0].strftime("%Y-%m-%d"),
                y=0.97,
                text=f"Data lag — {gap_months} mo not yet published",
                showarrow=False,
                xanchor="left",
                font=dict(size=11, color="rgba(40, 40, 40, 0.85)"),
                bgcolor="rgba(255, 255, 255, 0.65)",
            )
        fig.add_shape(
            type="line",
            xref="x", yref="paper",
            x0=gap_anchor_iso, x1=gap_anchor_iso,
            y0=0, y1=1,
            line=dict(color="rgba(60, 60, 60, 0.55)", dash="dot", width=1.5),
        )
        fig.add_annotation(
            xref="x", yref="paper",
            x=gap_anchor_iso,
            y=0.97,
            text=f"Forecast starts {gap_anchor.year}",
            showarrow=False,
            xanchor="left",
            xshift=4,
            font=dict(size=11),
            bgcolor="rgba(255, 255, 255, 0.65)",
        )

        # ----- Image impact: reference lines for context -----
        # Two horizontal reference lines that put the forecast in
        # context at a glance: (a) the focus state's historical mean
        # over the displayed window, (b) the peer-median forecast at
        # the +horizon endpoint. Both help the user see whether the
        # forecast is "back to normal" or "above peers" without
        # reading the AI prose.
        focus_history_arr = (
            hist[focus_col].dropna().to_numpy()
            if not hist.empty and focus_col in hist
            else None
        )
        if focus_history_arr is not None and focus_history_arr.size >= 12:
            hist_mean_value = float(np.mean(focus_history_arr))
            fig.add_shape(
                type="line", xref="paper", yref="y",
                x0=0, x1=1, y0=hist_mean_value, y1=hist_mean_value,
                line=dict(color="rgba(31, 119, 180, 0.45)", dash="dot", width=1),
                layer="below",
            )
            fig.add_annotation(
                xref="paper", yref="y",
                x=0.99, y=hist_mean_value,
                text=f"{focus_state} historical mean ({hist_mean_value:.1f}%)",
                showarrow=False, xanchor="right", yanchor="bottom",
                font=dict(size=10, color="rgba(31, 119, 180, 0.85)"),
                bgcolor="rgba(255, 255, 255, 0.65)",
            )
        # Peer median line — only when there are peers and they have
        # forecasts. We compute again here from the predicted points
        # to keep the chart self-contained.
        peer_point_values = []
        for st in peer_states:
            peer_col_for_med = _column_for(metric, st)
            if peer_col_for_med not in df.columns:
                continue
            try:
                p_result = state_forecasts.get(st)
                if p_result is None:
                    continue
                p_preds = p_result.model.predict(total_horizon)
                if len(p_preds) > 0:
                    peer_point_values.append(float(p_preds[-1]))
            except Exception:  # noqa: BLE001 — chart annotation, never block render
                continue
        if peer_point_values:
            peer_median_value = float(np.median(peer_point_values))
            fig.add_shape(
                type="line", xref="paper", yref="y",
                x0=0.0, x1=1, y0=peer_median_value, y1=peer_median_value,
                line=dict(color="rgba(214, 39, 40, 0.45)", dash="dash", width=1),
                layer="below",
            )
            fig.add_annotation(
                xref="paper", yref="y",
                x=0.01, y=peer_median_value,
                text=f"peer median forecast ({peer_median_value:.1f}%)",
                showarrow=False, xanchor="left", yanchor="bottom",
                font=dict(size=10, color="rgba(214, 39, 40, 0.85)"),
                bgcolor="rgba(255, 255, 255, 0.65)",
            )

        # Trend summary across every state in scope.
        summaries: dict[str, TrendSummary | None] = {}
        for st in all_states:
            col = _column_for(metric, st)
            if col in df.columns:
                summaries[st] = summarize_trend(df[col].dropna().to_numpy())
        trend_panel = _trend_summary_table(summaries, unit="%")

        # Resolve every state's point + CI once so the requirements
        # table and the AI's pass/fail classifier read from the same
        # numbers as the chart. Uses ``total_horizon`` (gap + horizon)
        # so the verdict reflects the *current_year+1 → +N years*
        # window the user actually sees, not the BLS-data-lag
        # months we hide behind the gap band.
        forecast_points = _compute_forecast_points(
            all_states, state_forecasts, total_horizon
        )
        threshold_value = float(threshold) if threshold not in (None, "") else None
        match_panel = _requirements_match_table(
            states_in_scope=all_states,
            forecast_points=forecast_points,
            months=months,
            metric=metric,
            threshold=threshold_value,
        )

        # Model rationale for the focus state (peers stay summarised).
        rationale_panel = _model_rationale_table(focus_state, focus_result)

        # ----- Build per-panel view_state payloads -----
        # Every figure / table on the page gets a typed view_state dict
        # that SentenceRAGBuilder turns into ontology-aware sentences;
        # the LLM only ever sees prose, never raw key:value lines.
        last_date = df["date"].max()
        last_actual_year = int(last_date.year) if pd.notna(last_date) else None
        last_actual_value = (
            float(hist[focus_col].iloc[-1])
            if not hist.empty and focus_col in hist
            else None
        )
        focus_points = forecast_points.get(focus_state) or {}

        # Statistical context for the AI prompt: focus state's
        # historical mean + volatility (σ) over the full window, plus
        # the median of peer states' point forecasts. These give the
        # specialist something to compare against ("forecast is X pp
        # above the historical mean of Y, vs peer median of Z") so
        # the prose includes interpretation rather than restatement.
        focus_history = hist[focus_col].dropna().to_numpy() if not hist.empty else np.array([])
        if focus_history.size >= 12:
            historical_mean = float(np.mean(focus_history))
            historical_std = float(np.std(focus_history, ddof=1))
        else:
            historical_mean = historical_std = None
        peer_points = [
            forecast_points[p]["point"]
            for p in peer_states
            if forecast_points.get(p) is not None
        ]
        peer_median = float(np.median(peer_points)) if peer_points else None

        forecast_view = {
            "_kind": "forecast_panel",
            "title": f"{focus_state} {metric_label} forecast (starts {gap_anchor.year})",
            "focus_state": focus_state,
            "peer_states": list(peer_states),
            "metric_key": metric,
            "horizon_years": years_ahead,
            "last_actual_year": last_actual_year,
            "last_actual_value": _round2(last_actual_value),
            "forecast_start_year": int(gap_anchor.year),
            "forecast_end_year": int(gap_anchor.year) + years_ahead - 1,
            "data_lag_months": int(gap_months),
            "forecast_point": _round2(focus_points.get("point")),
            "forecast_ci": (
                [_round2(focus_points["ci_lo"]), _round2(focus_points["ci_hi"])]
                if "ci_lo" in focus_points
                else None
            ),
            "winning_model": focus_result.name,
            "rmse": _round2(focus_result.metrics.rmse),
            # Statistical context for the AI specialist
            "historical_mean": _round2(historical_mean),
            "historical_volatility": _round2(historical_std),
            "peer_median_forecast": _round2(peer_median),
        }

        passing, failing = _classify_requirements(
            all_states,
            forecast_points,
            METRIC_THRESHOLD_DIRECTION[metric],
            threshold_value,
        )
        # Always emit per-state forecast points so the AI panel has
        # something to say even when the user left the threshold blank.
        state_points = [
            {"code": st, "value": _round2(info["point"])}
            for st, info in forecast_points.items()
            if info is not None
        ]
        requirements_view = {
            "_kind": "requirements_panel",
            "title": f"Requirements match at +{years_ahead}-year horizon",
            "metric_key": metric,
            "threshold": threshold_value,
            "direction": METRIC_THRESHOLD_DIRECTION[metric],
            "states_passing": passing,
            "states_failing": failing,
            "state_points": state_points,
        }

        trend_view = {
            "_kind": "trend_panel",
            "title": f"Historic {metric_label} trend",
            "metric_key": metric,
            "states": _trend_view_rows(summaries),
        }

        recap_view = {
            "_kind": "recap",
            "title": f"{focus_state} {metric_label} outlook (+{years_ahead} yrs)",
            "panels": [forecast_view, requirements_view, trend_view],
        }

        # ----- LFPR-only caveat banner (denominator audit) -----
        caveat = (
            dcc.Markdown(
                LFPR_DENOMINATOR_NOTE,
                className="alert alert-warning small pi-method-caveat",
            )
            if metric == "LFPR"
            else None
        )

        # ----- Spawn the background blurb runner -----
        # The Day-1 single-callback architecture meant a 2-min "dead
        # zone" where every panel was generated sequentially behind
        # one HTTP fetch. Now we hand the view_states to a daemon
        # thread that fills a per-process state slot keyed by
        # ``run_id``; the polling callback below picks them up as
        # each panel completes so the user sees panels arrive
        # individually instead of all at once.
        run_id = blurb_async.start_run({
            "forecast":     forecast_view,
            "requirements": requirements_view,
            "trend":        trend_view,
            "recap":        recap_view,
        })

        # ----- Interleaved body: figure → AI → figure → AI → recap -----
        body: list = [progress_strip("lfp-progress-status")]
        if caveat is not None:
            body.append(caveat)
        body.extend([
            figure_panel(
                title=forecast_view["title"],
                figure=dcc.Graph(figure=fig),
                caption=(
                    f"Solid line: published {metric_label.lower()} for "
                    f"{focus_state}. Dashed: 12-month rolling mean. "
                    f"Shaded band: 95% prediction interval from the "
                    f"winning bake-off model."
                ),
                blurb_id=_blurb_id("forecast"),
            ),
            figure_panel(
                title=requirements_view["title"],
                figure=match_panel,
                caption=None,
                blurb_id=_blurb_id("requirements"),
            ),
            figure_panel(
                title=trend_view["title"],
                figure=trend_panel,
                caption=(
                    "Per-state level + 1-year and 5-year change vs the "
                    "all-time range and population standard deviation."
                ),
                blurb_id=_blurb_id("trend"),
            ),
            tab_recap(
                title="Recap & deeper detail",
                blurb_id=_blurb_id("recap"),
            ),
            html.Hr(),
            rationale_panel,
            html.Hr(),
            methodology_panel(),
        ])

        payload = {"run_id": run_id, "n": int(n_clicks)}

        # Active-view payload for cross-tab AI surfaces (chat drawer
        # reads this Store on every send so its answers reflect what
        # the user is currently looking at, not just their question).
        active_view = {
            "tab": "lfp",
            "panels": [forecast_view, requirements_view, trend_view],
            "recap": recap_view,
        }
        return (
            html.Div(body),
            payload,
            False,        # interval enabled
            0,            # n_intervals reset
            active_view,  # cross-tab view-state for chat drawer
        )

    @app.callback(
        Output({"type": PANEL_BLURB_TYPE, "tab": "lfp", "section": "forecast"},     "children"),
        Output({"type": PANEL_BLURB_TYPE, "tab": "lfp", "section": "requirements"}, "children"),
        Output({"type": PANEL_BLURB_TYPE, "tab": "lfp", "section": "trend"},        "children"),
        Output({"type": PANEL_BLURB_TYPE, "tab": "lfp", "section": "recap"},        "children"),
        Output("lfp-progress-status", "children"),
        Output("lfp-progress-tick", "disabled", allow_duplicate=True),
        Input("lfp-progress-tick", "n_intervals"),
        State("lfp-blurb-payload", "data"),
        prevent_initial_call=True,
    )
    @error_boundary(fallback_id="lfp-progress-status", extra_outputs=5)
    def poll_lfp_blurbs(_n_intervals, payload):
        """
        Tick-driven polling callback. Reads the per-process run state
        produced by :mod:`utils.agents.blurb_async` and updates only
        the panels that have populated since the last tick — each
        panel pops in as soon as its individual Ollama call returns,
        so the user never sees a 2-minute dead zone.
        Disables the interval once every panel has filled to stop the
        ~2 s ticks from continuing to fire.
        """
        if not payload or "run_id" not in payload:
            raise PreventUpdate
        snapshot = blurb_async.get_snapshot(payload["run_id"])
        if not snapshot:
            # Run was GC'd or never started — leave placeholders alone.
            raise PreventUpdate

        outs: list = []
        for section in blurb_async.PANEL_SECTIONS:
            text = snapshot.get(section)
            if text is None:
                outs.append(html.Em(
                    f"Generating {section}…",
                    className="pi-muted small",
                ))
            elif text == "":
                # Caller explicitly skipped this section.
                outs.append(html.Em(
                    "no panel context — re-run the forecast",
                    className="pi-muted small",
                ))
            else:
                outs.append(render_blurb(text))

        status_text = snapshot.get("status", "starting")
        is_done = status_text == "done"
        # Add a friendly "X of N done" header for screen readers and
        # sighted users alike.
        completed = snapshot.get("completed", 0)
        total = snapshot.get("total", 0)
        if is_done:
            progress_msg = f"All {total} panel(s) ready."
        else:
            progress_msg = (
                f"Generated {completed} of {total} panel(s). "
                f"Working on the next one — Ollama is single-threaded "
                f"and each panel takes ~30 s on CPU."
            )
        return (*outs, progress_msg, is_done)

    # Chat lives in the global chat drawer now — registered in app.py.
