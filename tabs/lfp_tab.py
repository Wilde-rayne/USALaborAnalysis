import logging

import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from utils.constants import ALL_STATES, MONTH_MAP  # noqa: F401 — re-exported usage
from utils.data_pipeline import OUTPUT_JSON, ensure_data
from utils.llm_utils import generate_insight
from utils.model_utils import forecast_with_model, train_test_rnn

logger = logging.getLogger(__name__)

# Cache keyed by horizon (years). On cache miss the callback trains lazily
# rather than raising PreventUpdate, so horizons that the startup preloader
# didn't touch still produce a forecast on first click.
lfp_model_cache: dict[int, dict[str, dict]] = {}

IA_COL = "IA_Labor_Force_Participation_Rate"
MIDWEST_COL = "Midwest_LFPR"


def _load_lfp_panel() -> pd.DataFrame:
    """Lazy-load the merged panel and derive Midwest_LFPR."""
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
    return df


def _get_or_train_lfp(years_ahead: int, df: pd.DataFrame) -> dict[str, dict]:
    """Return cached LFP models for the horizon or train and cache them."""
    if years_ahead in lfp_model_cache:
        return lfp_model_cache[years_ahead]

    logger.info(f"[LFP] Cache miss for horizon={years_ahead}y — training on demand")
    base = df[["date", IA_COL, MIDWEST_COL]].dropna()
    entry: dict[str, dict] = {}
    for col in (IA_COL, MIDWEST_COL):
        model, metrics, last_window = train_test_rnn(base[["date", col]], col)
        entry[col] = {"model": model, "metrics": metrics, "last_window": last_window}
    lfp_model_cache[years_ahead] = entry
    return entry


def render_layout():
    return html.Div([
        html.H5("Labor Force Participation Forecast"),
        html.Div([
            html.Label("Years Ahead:"),
            dcc.Slider(
                id="lfp-years-slider",
                min=1, max=5, step=1,
                marks={i: str(i) for i in range(1, 6)},
                value=2,
            ),
            html.Button("Run Forecast", id="lfp-run", className="mt-2 btn btn-primary"),
        ], className="mb-3"),
        html.Div(id="lfp-output"),

        html.Hr(),
        html.H6("Ask the AI Assistant"),
        dcc.Input(
            id="lfp-chat-input",
            type="text",
            placeholder="Ask a question about the forecast...",
            style={"width": "80%"}
        ),
        html.Button("Submit", id="lfp-chat-button", className="btn btn-outline-primary btn-sm ml-2"),
        html.Div(id="lfp-chat-output", className="mt-3"),
    ])


def register_callbacks(app):
    @app.callback(
        Output("lfp-output", "children"),
        Input("lfp-run", "n_clicks"),
        State("lfp-years-slider", "value"),
    )
    def update_lfp(n_clicks, years_ahead):
        # Train on demand if the cache doesn't already have this horizon.
        if not n_clicks:
            raise PreventUpdate

        df = _load_lfp_panel()
        combo = df[["date", IA_COL, MIDWEST_COL]].dropna()
        combo[IA_COL] = combo[IA_COL].astype(float)
        combo[MIDWEST_COL] = combo[MIDWEST_COL].astype(float)

        entry = _get_or_train_lfp(years_ahead, df)
        ia_m = entry[IA_COL]["model"]
        ia_w = entry[IA_COL]["last_window"]
        mw_m = entry[MIDWEST_COL]["model"]
        mw_w = entry[MIDWEST_COL]["last_window"]

        months = years_ahead * 12
        preds_ia = forecast_with_model(ia_m, ia_w, months)
        preds_mw = forecast_with_model(mw_m, mw_w, months)

        last_date = combo["date"].max()
        dates = [last_date + pd.DateOffset(months=i + 1) for i in range(months)]

        fig = go.Figure([
            go.Scatter(x=combo["date"], y=combo[IA_COL], mode="lines", name="Iowa Historic"),
            go.Scatter(x=combo["date"], y=combo[MIDWEST_COL], mode="lines", name="Midwest Historic"),
            go.Scatter(x=dates, y=preds_ia, mode="lines+markers", name="Iowa Forecast"),
            go.Scatter(x=dates, y=preds_mw, mode="lines+markers", name="Midwest Forecast"),
        ])
        fig.update_layout(
            title=f"LFP Forecast (+{years_ahead} yrs)",
            xaxis_title="Date",
            yaxis_title="Labor Force Participation Rate",
            template="plotly_white",
        )

        if preds_ia and preds_mw:
            ia_final, mw_final = preds_ia[-1], preds_mw[-1]
            prompt = (
                f"Iowa forecast: {ia_final:.1f}% vs. Midwest: {mw_final:.1f}% "
                f"over {years_ahead} years. Provide 3–4 sentences of economic insight."
            )
            insight = generate_insight(prompt)
        else:
            insight = "Not enough data to generate a forecast."

        return html.Div([
            dcc.Graph(figure=fig),
            html.Hr(),
            dcc.Markdown(insight),
            dcc.Markdown("_Disclaimer: AI-generated; may contain inaccuracies._"),
        ])

    @app.callback(
        Output("lfp-chat-output", "children"),
        Input("lfp-chat-button", "n_clicks"),
        State("lfp-chat-input", "value"),
    )
    def update_lfp_chat(n_clicks, query):
        if not n_clicks or not query:
            raise PreventUpdate
        answer = generate_insight(query)
        return html.Div([
            dcc.Markdown(answer),
            dcc.Markdown("_Disclaimer: AI-generated; may contain inaccuracies._"),
        ])
