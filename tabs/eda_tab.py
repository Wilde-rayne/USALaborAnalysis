"""
EDA / Overview tab — interleaved layout with reactive AI explanations.

Day-3 refactor (matches the LFP pattern):

- A ``Refresh Data`` click triggers ``update_eda``, which (a) optionally
  forces a full-scope pull, (b) computes summary stats + four figures,
  (c) builds typed ``view_state`` dicts per panel, and (d) hands them
  to :mod:`utils.agents.blurb_async` so the panel narratives generate
  in a daemon thread instead of blocking the click.
- A ``dcc.Interval`` polls every 2 s and fills each panel's AI block
  as the background runner completes it. Progress strip + per-panel
  ``aria-live`` placeholders give SR-accessible feedback.
- Every run also writes the active panels into the global
  ``pi-active-view`` Store so the chat drawer's answers ground in
  what the user is currently looking at.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

import pandas as pd
import plotly.graph_objs as go
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate

from utils.agents import blurb_async
from utils.constants import ALL_STATES, END_YEAR, MONTH_MAP, START_YEAR, YEARS
from utils.data_pipeline import OUTPUT_JSON, ensure_data
from utils.ontology import ONTOLOGY
from tabs._components import (
    PANEL_BLURB_TYPE,
    error_boundary,
    figure_panel,
    progress_strip,
    render_blurb,
    tab_recap,
)
from tabs._methodology import chart_source_annotation

logger = logging.getLogger(__name__)


def _blurb_id(section: str) -> dict:
    """Pattern-matching ID convention shared with other tabs."""
    return {"type": PANEL_BLURB_TYPE, "tab": "eda", "section": section}


def render_layout():
    return html.Div(
        [
            html.H5("Exploratory Data Analysis / Overview"),

            html.Div(id="eda-metadata", className="mb-3"),

            html.Details(
                [
                    html.Summary("Definitions (CES, LAUS, LFPR, ARIMA / ETS / LSTM…)"),
                    html.Ul(
                        [
                            html.Li([html.B("CES"), " = Current Employment Statistics (BLS)"]),
                            html.Li([html.B("LAUS"), " = Local Area Unemployment Statistics (BLS)"]),
                            html.Li([html.B("LFPR"), " = Labor Force Participation Rate"]),
                            html.Li([html.B("ARIMA / ETS"), " = bake-off forecasting models"]),
                        ]
                    ),
                ],
                open=False,
                className="mb-4",
            ),

            html.Div(
                [
                    html.Label("Select States:", htmlFor="eda-state-selector"),
                    dcc.Dropdown(
                        id="eda-state-selector",
                        options=[{"label": s, "value": s} for s in ALL_STATES],
                        value=[ALL_STATES[0]],
                        multi=True,
                    ),
                    html.Br(),
                    html.Label("Select Year Range:", htmlFor="eda-year-range"),
                    dcc.RangeSlider(
                        id="eda-year-range",
                        min=YEARS[0],
                        max=YEARS[-1],
                        step=1,
                        marks={yr: str(yr) for yr in YEARS},
                        value=[YEARS[0], YEARS[-1] - 1],
                    ),
                    html.Br(),
                    html.Button(
                        "Refresh Data",
                        id="eda-refresh",
                        className="btn btn-primary",
                    ),
                ],
                className="mb-4",
            ),

            dcc.Loading(id="loading-eda", children=html.Div(id="eda-output")),
            # Cross-callback channel + polling tick (same pattern as LFP).
            dcc.Store(id="eda-blurb-payload", data=None),
            dcc.Interval(
                id="eda-progress-tick",
                interval=2_000,
                n_intervals=0,
                disabled=True,
            ),
        ],
        className="p-4",
    )


def register_callbacks(app):
    @app.callback(
        Output("eda-metadata", "children"),
        Input("eda-output", "children"),
    )
    def update_metadata(_):
        if not os.path.exists(OUTPUT_JSON):
            return ""
        df = pd.read_json(OUTPUT_JSON, orient="records")
        if df.empty:
            return ""
        df["period"] = df["period"].map(MONTH_MAP).fillna(df["period"])
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["period"] + "-01",
            errors="coerce",
        )
        total = len(df)
        start = df["date"].min().strftime("%b %Y")
        end = df["date"].max().strftime("%b %Y")
        refreshed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return html.Div(
            [
                html.P(f"Total records: {total:,}"),
                html.P(f"Date range: {start} – {end}"),
                html.P(f"Last refreshed: {refreshed}"),
            ],
            className="alert alert-info",
        )

    @app.callback(
        Output("eda-output", "children"),
        Output("eda-blurb-payload", "data"),
        Output("eda-progress-tick", "disabled"),
        Output("eda-progress-tick", "n_intervals"),
        Output("pi-active-view", "data", allow_duplicate=True),
        Input("eda-refresh", "n_clicks"),
        State("eda-state-selector", "value"),
        State("eda-year-range", "value"),
        prevent_initial_call=True,
    )
    @error_boundary(fallback_id="eda-output", extra_outputs=4)
    def update_eda(n_clicks, states, year_range):
        if not n_clicks:
            raise PreventUpdate

        # Force-refresh fetches the full project-wide scope so the LFP /
        # Super tabs aren't starved by an EDA-only filter; the display
        # below still honours the user's selection.
        ensure_data(ALL_STATES, START_YEAR, END_YEAR, force=True)

        if not os.path.exists(OUTPUT_JSON):
            return (
                html.Div("No data found. Please refresh.", className="text-danger"),
                None,
                True,
                0,
                None,
            )

        df = pd.read_json(OUTPUT_JSON, orient="records")
        df["period"] = df["period"].map(MONTH_MAP).fillna(df["period"])
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["period"] + "-01",
            format="%Y-%B-%d",
            errors="coerce",
        )
        df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
        df.sort_values("date", inplace=True)

        states = list(states or [])
        series_cols = sorted(
            c for c in df.columns if any(c.startswith(f"{st}_") for st in states)
        )

        # ----- Stats table + view_state -----
        stats = df[series_cols].describe().T[["mean", "50%", "min", "max", "std"]]
        stats.rename(columns={"50%": "median"}, inplace=True)
        stats_table = html.Table(
            [
                html.Thead(
                    html.Tr([html.Th(c) for c in ("Series", "Mean", "Median", "Min", "Max")])
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(idx),
                                html.Td(f"{row['mean']:.2f}"),
                                html.Td(f"{row['median']:.2f}"),
                                html.Td(f"{row['min']:.2f}"),
                                html.Td(f"{row['max']:.2f}"),
                            ]
                        )
                        for idx, row in stats.iterrows()
                    ]
                ),
            ],
            className="table table-sm table-striped mb-3",
        )

        stats_view = {
            "_kind": "eda_distribution",
            "title": "Summary statistics",
            "measures": _measures_in_columns(series_cols),
            "series": _stats_rows(stats),
        }

        # ----- Time-series figure + view_state -----
        fig_ts = go.Figure(
            [go.Scatter(x=df["date"], y=df[col], mode="lines", name=col) for col in series_cols]
        ).update_layout(
            title=f"Time series — {', '.join(states)}",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=70, r=30, t=60, b=90),
            annotations=[chart_source_annotation(x=0.0, y=-0.22)],
        )

        ts_view = {
            "_kind": "eda_timeseries",
            "title": "Multi-series time view",
            "states": states,
            "measures": _measures_in_columns(series_cols),
            "series": _series_first_last(df, series_cols),
        }

        # ----- Volatility (YoY + rolling) figure + view_state -----
        roll = df.set_index("date")[series_cols].rolling(12).mean()
        yoy = df.set_index("date")[series_cols].pct_change(12, fill_method=None).mul(100)
        vol_traces: list = []
        for col in series_cols:
            vol_traces.append(
                go.Scatter(x=yoy.index, y=yoy[col], mode="lines", name=f"{col} YoY %")
            )
        fig_vol = go.Figure(vol_traces).update_layout(
            title="Year-over-year % change",
            xaxis_title="Date",
            yaxis_title="Percent change",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=70, r=30, t=60, b=90),
            annotations=[chart_source_annotation(x=0.0, y=-0.22)],
        )

        vol_view = {
            "_kind": "eda_volatility",
            "title": "Year-over-year volatility",
            "rolling_window_months": 12,
            "series": _series_yoy_latest(yoy, series_cols),
        }

        # ----- Recap view (cross-panel synthesis) -----
        recap_view = {
            "_kind": "recap",
            "title": "Exploratory snapshot recap",
            "panels": [stats_view, ts_view, vol_view],
        }

        # ----- Spawn background blurb runner -----
        run_id = blurb_async.start_run({
            "stats":      stats_view,
            "timeseries": ts_view,
            "volatility": vol_view,
            "recap":      recap_view,
        })

        body = [
            progress_strip("eda-progress-status"),
            figure_panel(
                title="Summary statistics",
                figure=stats_table,
                caption=(
                    f"Mean / median / min / max / σ across {len(series_cols)} "
                    f"series in scope, {year_range[0]}–{year_range[1]}."
                ),
                blurb_id=_blurb_id("stats"),
            ),
            figure_panel(
                title="Time-series overlay",
                figure=dcc.Graph(figure=fig_ts),
                caption=(
                    "Each line is one column from the wide panel "
                    f"({', '.join(states)} × selected measures)."
                ),
                blurb_id=_blurb_id("timeseries"),
            ),
            figure_panel(
                title="Year-over-year volatility",
                figure=dcc.Graph(figure=fig_vol),
                caption=(
                    "Percent change vs the same month one year earlier. "
                    "Spikes near 2020 reflect pandemic shocks."
                ),
                blurb_id=_blurb_id("volatility"),
            ),
            tab_recap(
                title="Recap & deeper detail",
                blurb_id=_blurb_id("recap"),
            ),
        ]

        active_view = {
            "tab": "eda",
            "panels": [stats_view, ts_view, vol_view],
            "recap": recap_view,
        }
        payload = {"run_id": run_id, "n": int(n_clicks or 0)}

        return html.Div(body), payload, False, 0, active_view

    @app.callback(
        Output({"type": PANEL_BLURB_TYPE, "tab": "eda", "section": "stats"},      "children"),
        Output({"type": PANEL_BLURB_TYPE, "tab": "eda", "section": "timeseries"}, "children"),
        Output({"type": PANEL_BLURB_TYPE, "tab": "eda", "section": "volatility"}, "children"),
        Output({"type": PANEL_BLURB_TYPE, "tab": "eda", "section": "recap"},      "children"),
        Output("eda-progress-status", "children"),
        Output("eda-progress-tick", "disabled", allow_duplicate=True),
        Input("eda-progress-tick", "n_intervals"),
        State("eda-blurb-payload", "data"),
        prevent_initial_call=True,
    )
    @error_boundary(fallback_id="eda-progress-status", extra_outputs=5)
    def poll_eda_blurbs(_n, payload):
        if not payload or "run_id" not in payload:
            raise PreventUpdate
        snapshot = blurb_async.get_snapshot(payload["run_id"])
        if not snapshot:
            raise PreventUpdate

        order = ("stats", "timeseries", "volatility", "recap")
        outs: list = []
        for sec in order:
            text = snapshot.get(sec)
            if text is None:
                outs.append(html.Em(
                    f"Generating {sec}…", className="pi-muted small"
                ))
            elif text == "":
                outs.append(html.Em(
                    "no panel context — re-run the refresh",
                    className="pi-muted small",
                ))
            else:
                outs.append(render_blurb(text))

        completed = snapshot.get("completed", 0)
        total = snapshot.get("total", len(order))
        is_done = snapshot.get("status") == "done"
        progress_msg = (
            f"All {total} panel(s) ready."
            if is_done
            else f"Generated {completed} of {total} panel(s); generating next…"
        )
        return (*outs, progress_msg, is_done)

    # Chat lives in the global chat drawer now — registered in app.py.


# --------------------------------------------------------------------------
# View-state builders (kept module-private; no external callers)
# --------------------------------------------------------------------------
def _measures_in_columns(cols: list[str]) -> list[str]:
    """Pull the unique measure-key suffixes out of ``{state}_{measure}`` columns."""
    found: list[str] = []
    seen: set[str] = set()
    for c in cols:
        if "_" not in c:
            continue
        measure = c.split("_", 1)[1]
        if measure in ONTOLOGY.measures and measure not in seen:
            seen.add(measure)
            found.append(measure)
    return found


def _stats_rows(stats: pd.DataFrame) -> list[dict]:
    """Turn the describe-table into compact (state, measure, mean, std) rows."""
    out: list[dict] = []
    for label, row in stats.iterrows():
        if "_" not in label:
            continue
        state, measure = label.split("_", 1)
        if measure not in ONTOLOGY.measures:
            continue
        unit = ONTOLOGY.measures[measure].unit
        out.append({
            "state": state,
            "measure": measure,
            "mean": float(row.get("mean")) if pd.notna(row.get("mean")) else None,
            "std": float(row.get("std")) if pd.notna(row.get("std")) else None,
            "unit": unit,
        })
    return out[:24]


def _series_first_last(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """Per-column first / last observed values for the ``eda_timeseries`` view."""
    out: list[dict] = []
    for c in cols:
        if "_" not in c or c not in df.columns:
            continue
        state, measure = c.split("_", 1)
        if measure not in ONTOLOGY.measures:
            continue
        series = df[c].dropna()
        if series.empty:
            continue
        out.append({
            "state": state,
            "measure": measure,
            "first": float(series.iloc[0]),
            "last": float(series.iloc[-1]),
            "unit": ONTOLOGY.measures[measure].unit,
        })
    return out[:24]


def _series_yoy_latest(yoy: pd.DataFrame, cols: list[str]) -> list[dict]:
    """Per-column most-recent YoY pct change (skips NaN at the head)."""
    out: list[dict] = []
    for c in cols:
        if "_" not in c or c not in yoy.columns:
            continue
        state, measure = c.split("_", 1)
        if measure not in ONTOLOGY.measures:
            continue
        series = yoy[c].dropna()
        if series.empty:
            continue
        out.append({
            "state": state,
            "measure": measure,
            "yoy_pct": float(series.iloc[-1]),
        })
    return out[:24]
