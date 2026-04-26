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

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from utils.constants import ALL_STATES, MONTH_MAP
from utils.data_pipeline import OUTPUT_JSON, ensure_data
from utils.forecasting import ForecastResult, select_forecaster
from utils.forecasting.trend import (
    TrendSummary,
    rolling_statistics,
    summarize_trend,
)
from utils.llm_utils import generate_insight
from utils.ontology import ONTOLOGY
from tabs._methodology import LFPR_DENOMINATOR_NOTE, methodology_panel
from tabs._components import error_boundary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache: (metric_key, state_code, years_ahead) → ForecastResult.
# Gets populated on demand inside the callback.
# ---------------------------------------------------------------------------
lfp_model_cache: dict[tuple[str, str, int], ForecastResult] = {}

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
    ensure_data()
    df = pd.read_json(OUTPUT_JSON, orient="records")
    df["period"] = df["period"].map(MONTH_MAP).fillna(df["period"])
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["period"] + "-01",
        format="%Y-%B-%d",
        errors="coerce",
    )
    df.sort_values("date", inplace=True)
    df = df.drop_duplicates(subset="date")
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


def _requirements_match_table(
    states_in_scope: list[str],
    state_forecasts: dict[str, ForecastResult],
    months: int,
    metric: str,
    threshold: float | None,
) -> html.Div:
    """Which states clear the user's requirement at the forecast horizon?"""
    direction = METRIC_THRESHOLD_DIRECTION[metric]
    unit = "%"
    rows = []
    for st in states_in_scope:
        result = state_forecasts.get(st)
        if result is None:
            rows.append(
                html.Tr(
                    [
                        html.Td(st),
                        html.Td("—", colSpan=4, className="text-muted small"),
                    ]
                )
            )
            continue
        interval = result.model.predict_interval(months, alpha=0.05)
        if interval is not None:
            preds, lower, upper = interval
            point = float(preds[-1])
            ci_lo = float(lower[-1])
            ci_hi = float(upper[-1])
        else:
            preds = result.model.predict(months)
            point = float(preds[-1])
            ci_lo = ci_hi = point
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
                    html.Td(f"{result.name} (RMSE {result.metrics.rmse:.2f})"),
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
            # Cross-callback channel: the main callback drops a prompt
            # payload here, the deferred callback below picks it up,
            # routes it through generate_insight, and fills the inline
            # ``lfp-blurb-area`` placeholder. Splitting it that way lets
            # the chart appear in ~30 s on a cold click instead of
            # waiting for the ~30-60 s Ollama round-trip.
            dcc.Store(id="lfp-blurb-prompt", data=None),
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("lfp-output", "children"),
        Output("lfp-blurb-prompt", "data"),
        Input("lfp-run", "n_clicks"),
        State("lfp-focus-state", "value"),
        State("lfp-peer-states", "value"),
        State("lfp-metric", "value"),
        State("lfp-years-slider", "value"),
        State("lfp-threshold", "value"),
    )
    @error_boundary(fallback_id="lfp-output", extra_outputs=1)
    def update_lfp(n_clicks, focus_state, peer_states, metric, years_ahead, threshold):
        if not n_clicks:
            raise PreventUpdate
        if not focus_state:
            return (
                html.Div("Pick a focus state first.", className="alert alert-warning"),
                None,
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
            )

        # Build historic + forecast traces.
        last_date = df["date"].max()
        forecast_dates = [last_date + pd.DateOffset(months=i + 1) for i in range(months)]

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
        focus_interval = focus_result.model.predict_interval(months, alpha=0.05)
        if focus_interval is not None:
            preds, lower, upper = focus_interval
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
            preds = focus_result.model.predict(months)
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
            peer_preds = peer_result.model.predict(months)
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

        fig = go.Figure(traces)
        fig.update_layout(
            title=f"{metric_label} — {focus_state} (+{years_ahead} yrs)",
            xaxis_title="Date",
            yaxis_title=f"{metric_label} (%)",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        # Trend summary across every state in scope.
        summaries: dict[str, TrendSummary | None] = {}
        for st in all_states:
            col = _column_for(metric, st)
            if col in df.columns:
                summaries[st] = summarize_trend(df[col].dropna().to_numpy())
        trend_panel = _trend_summary_table(summaries, unit="%")

        # Requirements match table.
        match_panel = _requirements_match_table(
            states_in_scope=all_states,
            state_forecasts=state_forecasts,
            months=months,
            metric=metric,
            threshold=float(threshold) if threshold not in (None, "") else None,
        )

        # Model rationale for the focus state (peers stay summarised).
        rationale_panel = _model_rationale_table(focus_state, focus_result)

        # Prompt the blurb agent with the final-horizon values so the
        # narrative stays grounded in numbers that actually appear on the
        # chart.
        # Build the blurb prompt now (cheap), but defer the LLM call to
        # the second callback below so the chart appears as soon as the
        # bakeoff finishes rather than waiting on Ollama.
        final_values = {
            st: float(state_forecasts[st].model.predict(months)[-1])
            for st in all_states
            if st in state_forecasts
        }
        prompt = (
            f"{metric_label} forecast over {years_ahead} year(s):\n"
            + "\n".join(f"- {st}: {v:.2f}%" for st, v in final_values.items())
            + (
                f"\nRequirement: {'≥' if METRIC_THRESHOLD_DIRECTION[metric] == 'min' else '≤'} "
                f"{threshold:.2f}%"
                if threshold not in (None, "")
                else ""
            )
            + "\nGive a 3-4 sentence analysis aimed at a state workforce planner."
        )

        # The LFPR caveat only matters when LFPR is the chosen metric;
        # the unemployment-rate computation has no equivalent denominator
        # gotcha (BLS defines U-rate as Unemployment / Labor_Force, which
        # is exactly what we compute).
        caveat = (
            dcc.Markdown(
                LFPR_DENOMINATOR_NOTE,
                className="alert alert-warning small pi-method-caveat",
            )
            if metric == "LFPR"
            else None
        )

        # Inline placeholder where the deferred blurb callback will land.
        blurb_placeholder = html.Div(
            html.Em("Generating narrative analysis…", className="pi-muted small"),
            id="lfp-blurb-area",
        )

        body = [dcc.Graph(figure=fig)]
        if caveat is not None:
            body.append(caveat)
        body.extend(
            [
                html.Hr(),
                match_panel,
                trend_panel,
                rationale_panel,
                html.Hr(),
                blurb_placeholder,
                html.Hr(),
                methodology_panel(),
            ]
        )
        return html.Div(body), {"prompt": prompt, "n": int(n_clicks)}

    @app.callback(
        Output("lfp-blurb-area", "children"),
        Input("lfp-blurb-prompt", "data"),
        prevent_initial_call=True,
    )
    @error_boundary(fallback_id="lfp-blurb-area")
    def update_lfp_blurb(payload):
        """
        Deferred narrative pass — fires once the main callback has dropped
        a prompt into the store. Splitting this out lets the chart paint
        in ~30 s on a cold click instead of waiting for Ollama (~60 s
        round-trip on llama3.2:3b CPU).
        """
        if not payload or not payload.get("prompt"):
            raise PreventUpdate
        insight = generate_insight(payload["prompt"], active_tab="lfp")
        return html.Div(
            [
                dcc.Markdown(insight),
                dcc.Markdown("_Disclaimer: AI-generated; may contain inaccuracies._"),
            ]
        )

    # Chat lives in the global chat drawer now — registered in app.py.
