"""
Labor Force Participation forecast tab.

Runs a multi-model bakeoff (Naive / SeasonalNaive / ETS) per series
through expanding-window backtesting, picks the winner by minimum
out-of-sample RMSE, and displays the winning model's forecast
alongside a selection-rationale table and statistical diagnostics.
"""
from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from utils.constants import ALL_STATES, MONTH_MAP  # noqa: F401 — re-exported usage
from utils.data_pipeline import OUTPUT_JSON, ensure_data
from utils.forecasting import ForecastResult, select_forecaster
from utils.forecasting.trend import (
    TrendSummary,
    rolling_statistics,
    summarize_trend,
)
from utils.llm_utils import generate_insight

logger = logging.getLogger(__name__)

# Per-horizon cache. Value: {col_name: ForecastResult} so the winning
# model (refit on the full series) is reused across clicks, and the
# candidate metrics / diagnostics flow straight to the rationale panel.
lfp_model_cache: dict[int, dict[str, ForecastResult]] = {}

IA_COL = "IA_Labor_Force_Participation_Rate"
MIDWEST_COL = "Midwest_LFPR"


def _load_lfp_panel() -> pd.DataFrame:
    """Lazy-load the merged panel, derive Midwest_LFPR, dedupe by date."""
    ensure_data()  # no-op if cache is fresh
    df = pd.read_json(OUTPUT_JSON, orient="records")
    df["period"] = df["period"].map(MONTH_MAP).fillna(df["period"])
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["period"] + "-01",
        format="%Y-%B-%d",
        errors="coerce",
    )
    df.sort_values("date", inplace=True)
    lfp_cols = [c for c in df.columns if c.endswith("_Labor_Force_Participation_Rate")]
    df[MIDWEST_COL] = df[lfp_cols].mean(axis=1, skipna=True)
    # Collapse 12 state-rows/date into one — wide columns are identical
    # across states for a given date.
    df = df.drop_duplicates(subset="date")
    return df


def _get_or_train_lfp(years_ahead: int, df: pd.DataFrame) -> dict[str, ForecastResult]:
    """Return cached per-series ForecastResults for the horizon or run the bakeoff."""
    if years_ahead in lfp_model_cache:
        return lfp_model_cache[years_ahead]

    horizon = years_ahead * 12
    logger.info(f"[LFP] Cache miss horizon={years_ahead}y — running bakeoff")
    base = df[["date", IA_COL, MIDWEST_COL]].dropna()
    entry: dict[str, ForecastResult] = {}
    for col in (IA_COL, MIDWEST_COL):
        y = base[col].astype(float).to_numpy()
        dates = base["date"].to_numpy()
        result = select_forecaster(y, dates=dates, horizon=horizon, n_folds=3)
        logger.info(
            f"[LFP] {col}: winner={result.name} "
            f"rmse={result.metrics.rmse:.3f} mae={result.metrics.mae:.3f}"
        )
        entry[col] = result
    lfp_model_cache[years_ahead] = entry
    return entry


def _trend_summary_table(summaries: dict[str, TrendSummary | None]) -> html.Div:
    """
    Compact level/change/range/volatility table, one row per series.

    Only renders rows whose summary is non-None (empty / all-NaN series
    are skipped) so the table never shows blanks.
    """
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
                    html.Td(f"{s.current:.2f}%"),
                    html.Td(_pct(s.pct_1y)),
                    html.Td(_pct(s.pct_5y)),
                    html.Td(f"{s.all_time_low:.2f}%"),
                    html.Td(f"{s.all_time_high:.2f}%"),
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
    """Compact table showing every candidate's fold-averaged metrics."""
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
                [
                    html.Small(
                        f"Diagnostics — ADF p={_fmt_p(d.adf_pvalue)} · "
                        f"KPSS p={_fmt_p(d.kpss_pvalue)} · "
                        f"Ljung-Box p={_fmt_p(d.ljungbox_pvalue)} · "
                        f"Jarque-Bera p={_fmt_p(d.jarquebera_pvalue)} · "
                        f"Diebold-Mariano p={_fmt_p(d.dm_pvalue_vs_baseline)} "
                        "(vs naive; negative DM stat ⇒ winner beats baseline)",
                        className="text-muted",
                    ),
                ],
                className="mb-3",
            ),
        ]
    )


def render_layout():
    return html.Div(
        [
            html.H5("Labor Force Participation Forecast"),
            html.P(
                "Each run fits Naive, Seasonal-Naive, and Holt-Winters (ETS) "
                "through a 3-fold expanding-window backtest, then picks the "
                "model with the lowest out-of-sample RMSE and reports its "
                "stationarity and residual diagnostics alongside the chart.",
                className="text-muted small",
            ),
            html.Div(
                [
                    html.Label("Years Ahead:"),
                    dcc.Slider(
                        id="lfp-years-slider",
                        min=1,
                        max=5,
                        step=1,
                        marks={i: str(i) for i in range(1, 6)},
                        value=2,
                    ),
                    html.Button("Run Forecast", id="lfp-run", className="mt-2 btn btn-primary"),
                ],
                className="mb-3",
            ),
            dcc.Loading(id="loading-lfp", children=html.Div(id="lfp-output")),
            html.Hr(),
            html.H6("Ask the AI Assistant"),
            dcc.Input(
                id="lfp-chat-input",
                type="text",
                placeholder="Ask a question about the forecast...",
                style={"width": "80%"},
            ),
            html.Button(
                "Submit", id="lfp-chat-button", className="btn btn-outline-primary btn-sm ml-2"
            ),
            html.Div(id="lfp-chat-output", className="mt-3"),
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("lfp-output", "children"),
        Input("lfp-run", "n_clicks"),
        State("lfp-years-slider", "value"),
    )
    def update_lfp(n_clicks, years_ahead):
        if not n_clicks:
            raise PreventUpdate

        df = _load_lfp_panel()
        combo = df[["date", IA_COL, MIDWEST_COL]].dropna()
        combo[IA_COL] = combo[IA_COL].astype(float)
        combo[MIDWEST_COL] = combo[MIDWEST_COL].astype(float)

        entry = _get_or_train_lfp(years_ahead, df)
        ia_result = entry[IA_COL]
        mw_result = entry[MIDWEST_COL]

        months = years_ahead * 12
        preds_ia = ia_result.model.predict(months)
        preds_mw = mw_result.model.predict(months)

        last_date = combo["date"].max()
        dates = [last_date + pd.DateOffset(months=i + 1) for i in range(months)]

        # 12-month rolling mean overlays so the chart highlights the
        # trend under the monthly noise. Plotted as dashed lines to stay
        # visually subordinate to the raw series.
        ia_roll_mean, _ = rolling_statistics(combo[IA_COL].to_numpy(), window=12)
        mw_roll_mean, _ = rolling_statistics(combo[MIDWEST_COL].to_numpy(), window=12)

        fig = go.Figure(
            [
                go.Scatter(x=combo["date"], y=combo[IA_COL], mode="lines", name="Iowa Historic",
                           line=dict(width=1.5)),
                go.Scatter(x=combo["date"], y=ia_roll_mean, mode="lines",
                           name="Iowa 12-mo avg",
                           line=dict(dash="dash", color="rgba(31,119,180,0.55)", width=2)),
                go.Scatter(x=combo["date"], y=combo[MIDWEST_COL], mode="lines",
                           name="Midwest Historic", line=dict(width=1.5)),
                go.Scatter(x=combo["date"], y=mw_roll_mean, mode="lines",
                           name="Midwest 12-mo avg",
                           line=dict(dash="dash", color="rgba(255,127,14,0.55)", width=2)),
                go.Scatter(x=dates, y=preds_ia, mode="lines+markers",
                           name=f"Iowa Forecast ({ia_result.name})"),
                go.Scatter(x=dates, y=preds_mw, mode="lines+markers",
                           name=f"Midwest Forecast ({mw_result.name})"),
            ]
        )
        fig.update_layout(
            title=f"LFP Forecast (+{years_ahead} yrs)",
            xaxis_title="Date",
            yaxis_title="Labor Force Participation Rate",
            template="plotly_white",
        )

        prompt = (
            f"Iowa LFPR forecast: {preds_ia[-1]:.1f}% (model={ia_result.name}); "
            f"Midwest LFPR forecast: {preds_mw[-1]:.1f}% (model={mw_result.name}); "
            f"over {years_ahead} years. Provide 3–4 sentences of economic insight."
        )
        insight = generate_insight(prompt)

        # Trend summary — one row per series, displayed between the
        # chart and the model-rationale tables.
        trend_panel = _trend_summary_table(
            {
                "Iowa": summarize_trend(combo[IA_COL].to_numpy()),
                "Midwest mean": summarize_trend(combo[MIDWEST_COL].to_numpy()),
            }
        )

        return html.Div(
            [
                dcc.Graph(figure=fig),
                html.Hr(),
                trend_panel,
                _model_rationale_table("Iowa", ia_result),
                _model_rationale_table("Midwest mean", mw_result),
                html.Hr(),
                dcc.Markdown(insight),
                dcc.Markdown("_Disclaimer: AI-generated; may contain inaccuracies._"),
            ]
        )

    @app.callback(
        Output("lfp-chat-output", "children"),
        Input("lfp-chat-button", "n_clicks"),
        State("lfp-chat-input", "value"),
    )
    def update_lfp_chat(n_clicks, query):
        if not n_clicks or not query:
            raise PreventUpdate
        answer = generate_insight(query)
        return html.Div(
            [
                dcc.Markdown(answer),
                dcc.Markdown("_Disclaimer: AI-generated; may contain inaccuracies._"),
            ]
        )
