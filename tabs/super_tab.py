import logging

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from utils.constants import ALL_STATES, MONTH_MAP, SUPERSECTORS
from utils.data_pipeline import OUTPUT_JSON, ensure_data
from utils.llm_utils import generate_insight
from utils.model_utils import forecast_with_model, train_test_rnn

logger = logging.getLogger(__name__)

# Cache keyed by (sector, horizon-in-years). On miss the callback trains
# per-state LSTMs on demand rather than raising PreventUpdate, so the tab
# is usable without the startup-time 540-model preload.
supersector_model_cache: dict[tuple[str, int], dict[str, dict]] = {}


def _load_super_panel() -> pd.DataFrame:
    """
    Lazy-load the merged panel.

    Wide CES columns (``{state}_{sector}``) repeat across the 12
    state-rows that share a date, so we collapse to one row per date
    before downstream time-series code touches it.
    """
    ensure_data()  # cache-hit if fresh
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


def _get_or_train_supersector(sector: str, years_ahead: int, df: pd.DataFrame) -> dict[str, dict]:
    """Return cached per-state models for (sector, horizon) or train them."""
    key = (sector, years_ahead)
    if key in supersector_model_cache:
        return supersector_model_cache[key]

    logger.info(f"[SUPER] Cache miss for {sector!r} +{years_ahead}y — training 12 models")
    entry: dict[str, dict] = {}
    for st in ALL_STATES:
        col = f"{st}_{sector}"
        if col not in df.columns:
            logger.warning(f"[SUPER] column {col} missing in panel; skipping {st}")
            continue
        sub = df[["date", col]].dropna()
        if sub.empty:
            logger.warning(f"[SUPER] no data for {col}; skipping {st}")
            continue
        model, metrics, last_window = train_test_rnn(sub, col)
        entry[st] = {"model": model, "metrics": metrics, "last_window": last_window}
    supersector_model_cache[key] = entry
    return entry


def render_layout():
    return html.Div([
        html.H5("Supersector Employment Forecast"),
        html.Div([
            html.Label("Select Supersector:"),
            dcc.Dropdown(
                id="supersector-dropdown",
                options=[{"label": s.replace("_", " "), "value": s} for s in SUPERSECTORS],
                value=SUPERSECTORS[0],
                clearable=False,
            ),
        ], className="mb-2"),

        html.Div([
            html.Label("Years Ahead:"),
            dcc.Slider(
                id="supersector-years-slider",
                min=1, max=5, step=1,
                marks={i: str(i) for i in range(1, 6)},
                value=2,
            ),
            html.Button("Run Forecast", id="supersector-run", className="mt-2 btn btn-primary"),
        ], className="mb-3"),

        html.Div(id="super-output"),

        html.Hr(),
        html.H6("Ask the AI Assistant"),
        dcc.Input(
            id="super-chat-input",
            type="text",
            placeholder="Ask a question about this forecast...",
            style={"width": "80%"}
        ),
        html.Button("Submit", id="super-chat-button", className="btn btn-outline-primary btn-sm ml-2"),
        html.Div(id="super-chat-output", className="mt-3"),
    ])


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

        # Generate forecasts for each state that has a trained model.
        forecasts: dict[str, float] = {}
        for st, info in entry.items():
            preds = forecast_with_model(info["model"], info["last_window"], years_ahead * 12)
            forecasts[st] = float(preds[-1]) if preds else 0.0

        # Compute Midwest aggregates
        vals = np.array(list(forecasts.values()))
        forecasts["Midwest Mean"]   = float(vals.mean())
        forecasts["Midwest Median"] = float(np.median(vals))

        # Prepare bar chart data, coloring Midwest bars differently
        items = sorted(forecasts.items(), key=lambda x: x[1], reverse=True)
        labels, data = zip(*items)
        mn, mx = min(data), max(data)
        colors = [
            "blue" if lbl.startswith("Midwest") else
            f"rgb({int(255*(1-(v-mn)/(mx-mn+1e-6)))},{int(255*((v-mn)/(mx-mn+1e-6)))},0)"
            for lbl, v in items
        ]

        fig = go.Figure([go.Bar(
            x=list(labels), y=list(data),
            marker=dict(color=colors),
            text=[f"{v:.1f}" for v in data],
            textposition="auto"
        )])
        fig.update_layout(
            title=f"{sector.replace('_',' ')} Forecast (+{years_ahead} yrs)",
            xaxis_title="State",
            yaxis_title="Forecasted Employment",
            template="plotly_white"
        )

        # Generate AI insight narrative
        prompt = (
            f"From forecasts for '{sector}' (+{years_ahead} years): " +
            ", ".join(f"{lbl} {forecasts[lbl]:.1f}" for lbl in labels) +
            ". Provide a concise 3-sentence analysis for regional planners."
        )
        insight = generate_insight(prompt)

        # Return graph and AI-generated insight
        return html.Div([
            dcc.Graph(figure=fig),
            html.Hr(),
            dcc.Markdown(insight),
            dcc.Markdown("_Disclaimer: AI-generated; may contain inaccuracies._"),
        ])

    @app.callback(
        Output("super-chat-output", "children"),
        Input("super-chat-button", "n_clicks"),
        State("super-chat-input", "value"),
    )
    def update_super_chat(n_clicks, query):
        if not n_clicks or not query:
            raise PreventUpdate
        answer = generate_insight(query)
        return html.Div([
            dcc.Markdown(answer),
            dcc.Markdown("_Disclaimer: AI-generated; may contain inaccuracies._"),
        ])
