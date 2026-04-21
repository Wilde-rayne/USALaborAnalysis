"""
Supersector employment forecast tab.

For each state × (sector, horizon) a bakeoff runs over Naive /
SeasonalNaive / ETS candidates. The winning model's forward
projection becomes the bar in the chart; the per-state selection
table underneath is the auditable receipt.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from utils.constants import ALL_STATES, MONTH_MAP, SUPERSECTORS
from utils.data_pipeline import OUTPUT_JSON, ensure_data
from utils.forecasting import ForecastResult, select_forecaster
from utils.llm_utils import generate_insight

logger = logging.getLogger(__name__)

# Cache keyed by (sector, years). Value: {state: ForecastResult}.
supersector_model_cache: dict[tuple[str, int], dict[str, ForecastResult]] = {}


def _load_super_panel() -> pd.DataFrame:
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
    return df


def _get_or_train_supersector(
    sector: str, years_ahead: int, df: pd.DataFrame
) -> dict[str, ForecastResult]:
    """Cached per-state ForecastResult for (sector, horizon)."""
    key = (sector, years_ahead)
    if key in supersector_model_cache:
        return supersector_model_cache[key]

    horizon = years_ahead * 12
    logger.info(f"[SUPER] Cache miss for {sector!r} +{years_ahead}y — bakeoff across states")
    entry: dict[str, ForecastResult] = {}
    for st in ALL_STATES:
        col = f"{st}_{sector}"
        if col not in df.columns:
            logger.warning(f"[SUPER] column {col} missing; skipping {st}")
            continue
        sub = df[["date", col]].dropna()
        if sub.empty:
            logger.warning(f"[SUPER] no data for {col}; skipping {st}")
            continue
        y = sub[col].astype(float).to_numpy()
        dates = sub["date"].to_numpy()
        try:
            result = select_forecaster(y, dates=dates, horizon=horizon, n_folds=3)
        except RuntimeError as exc:
            logger.warning(f"[SUPER] bakeoff failed for {col}: {exc}")
            continue
        entry[st] = result
    supersector_model_cache[key] = entry
    return entry


def _selection_summary(entry: dict[str, ForecastResult]) -> html.Div:
    """Per-state table: which model won for each state + its RMSE."""
    rows = []
    for st, result in sorted(entry.items()):
        rows.append(
            html.Tr(
                [
                    html.Td(st),
                    html.Td(result.name),
                    html.Td(f"{result.metrics.rmse:,.1f}"),
                    html.Td(f"{result.metrics.mae:,.1f}"),
                    html.Td(f"{result.metrics.bias:+,.1f}"),
                ]
            )
        )
    return html.Div(
        [
            html.H6("Per-state winners"),
            html.Table(
                [
                    html.Thead(
                        html.Tr([html.Th(c) for c in ("State", "Model", "RMSE", "MAE", "Bias")])
                    ),
                    html.Tbody(rows),
                ],
                className="table table-sm table-striped",
            ),
        ]
    )


def render_layout():
    return html.Div(
        [
            html.H5("Supersector Employment Forecast"),
            html.P(
                "For each state, a Naive/Seasonal-Naive/ETS bakeoff selects "
                "the model with the lowest expanding-window RMSE. Bars show "
                "the winning model's forward projection.",
                className="text-muted small",
            ),
            html.Div(
                [
                    html.Label("Select Supersector:"),
                    dcc.Dropdown(
                        id="supersector-dropdown",
                        options=[
                            {"label": s.replace("_", " "), "value": s} for s in SUPERSECTORS
                        ],
                        value=SUPERSECTORS[0],
                        clearable=False,
                    ),
                ],
                className="mb-2",
            ),
            html.Div(
                [
                    html.Label("Years Ahead:"),
                    dcc.Slider(
                        id="supersector-years-slider",
                        min=1,
                        max=5,
                        step=1,
                        marks={i: str(i) for i in range(1, 6)},
                        value=2,
                    ),
                    html.Button(
                        "Run Forecast", id="supersector-run", className="mt-2 btn btn-primary"
                    ),
                ],
                className="mb-3",
            ),
            dcc.Loading(id="loading-super", children=html.Div(id="super-output")),
            html.Hr(),
            html.H6("Ask the AI Assistant"),
            dcc.Input(
                id="super-chat-input",
                type="text",
                placeholder="Ask a question about this forecast...",
                style={"width": "80%"},
            ),
            html.Button(
                "Submit",
                id="super-chat-button",
                className="btn btn-outline-primary btn-sm ml-2",
            ),
            html.Div(id="super-chat-output", className="mt-3"),
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("super-output", "children"),
        Input("supersector-run", "n_clicks"),
        State("supersector-dropdown", "value"),
        State("supersector-years-slider", "value"),
    )
    def update_super(n_clicks, sector, years_ahead):
        if not n_clicks:
            raise PreventUpdate

        df = _load_super_panel()
        entry = _get_or_train_supersector(sector, years_ahead, df)
        if not entry:
            return html.Div(
                f"No {sector.replace('_', ' ')} data available for any Midwest state.",
                className="alert alert-warning",
            )

        # Final forecast value per state (end of horizon).
        forecasts: dict[str, float] = {}
        for st, result in entry.items():
            preds = result.model.predict(years_ahead * 12)
            forecasts[st] = float(preds[-1]) if len(preds) else 0.0

        vals = np.array(list(forecasts.values()), dtype=float)
        forecasts["Midwest Mean"] = float(vals.mean())
        forecasts["Midwest Median"] = float(np.median(vals))

        items = sorted(forecasts.items(), key=lambda x: x[1], reverse=True)
        labels, data = zip(*items)
        mn, mx = min(data), max(data)
        colors = [
            "blue"
            if lbl.startswith("Midwest")
            else f"rgb({int(255 * (1 - (v - mn) / (mx - mn + 1e-6)))},"
                 f"{int(255 * ((v - mn) / (mx - mn + 1e-6)))},0)"
            for lbl, v in items
        ]

        fig = go.Figure(
            [
                go.Bar(
                    x=list(labels),
                    y=list(data),
                    marker=dict(color=colors),
                    text=[f"{v:,.1f}" for v in data],
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title=f"{sector.replace('_', ' ')} Forecast (+{years_ahead} yrs)",
            xaxis_title="State",
            yaxis_title="Forecasted Employment",
            template="plotly_white",
        )

        # Narrative from the chat model, keyed on the winning-model mix.
        winner_counts: dict[str, int] = {}
        for r in entry.values():
            winner_counts[r.name] = winner_counts.get(r.name, 0) + 1
        winner_summary = ", ".join(f"{n} × {c}" for n, c in winner_counts.items())
        prompt = (
            f"From forecasts for '{sector}' (+{years_ahead} years): "
            + ", ".join(f"{lbl} {forecasts[lbl]:,.1f}" for lbl in labels)
            + f". Per-state model selection: {winner_summary}. "
            "Provide a concise 3-sentence analysis for regional planners."
        )
        insight = generate_insight(prompt)

        return html.Div(
            [
                dcc.Graph(figure=fig),
                html.Hr(),
                _selection_summary(entry),
                html.Hr(),
                dcc.Markdown(insight),
                dcc.Markdown("_Disclaimer: AI-generated; may contain inaccuracies._"),
            ]
        )

    @app.callback(
        Output("super-chat-output", "children"),
        Input("super-chat-button", "n_clicks"),
        State("super-chat-input", "value"),
    )
    def update_super_chat(n_clicks, query):
        if not n_clicks or not query:
            raise PreventUpdate
        answer = generate_insight(query)
        return html.Div(
            [
                dcc.Markdown(answer),
                dcc.Markdown("_Disclaimer: AI-generated; may contain inaccuracies._"),
            ]
        )
