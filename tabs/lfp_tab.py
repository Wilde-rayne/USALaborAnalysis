import pandas as pd
import plotly.graph_objs as go

from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate

from utils.constants     import ALL_STATES, MONTH_MAP
from utils.data_pipeline import OUTPUT_JSON
from utils.llm_utils     import generate_insight
from utils.model_utils   import forecast_with_model

lfp_model_cache: dict[int, dict[str, dict]] = {}


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
        # Only proceed when button is clicked and cache is populated
        if not n_clicks or years_ahead not in lfp_model_cache:
            raise PreventUpdate

        # Load and preprocess data
        df = pd.read_json(OUTPUT_JSON, orient="records")
        df["period"] = df["period"].map(MONTH_MAP).fillna(df["period"])
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["period"] + "-01",
            format="%Y-%B-%d", errors="coerce"
        )
        df.sort_values("date", inplace=True)

        # Prepare Iowa vs. Midwest Labor Force Participation
        ia_col = "IA_Labor_Force_Participation_Rate"
        lfp_cols = [c for c in df.columns if c.endswith("_Labor_Force_Participation_Rate")]
        df["Midwest_LFPR"] = df[lfp_cols].mean(axis=1, skipna=True)

        combo = df[["date", ia_col, "Midwest_LFPR"]].dropna()
        combo[ia_col] = combo[ia_col].astype(float)
        combo["Midwest_LFPR"] = combo["Midwest_LFPR"].astype(float)

        # Retrieve pre-trained models & windows from cache
        entry = lfp_model_cache[years_ahead]
        ia_m = entry[ia_col]["model"]
        ia_w = entry[ia_col]["last_window"]
        mw_m = entry["Midwest_LFPR"]["model"]
        mw_w = entry["Midwest_LFPR"]["last_window"]

        # Generate forecasts
        months = years_ahead * 12
        preds_ia = forecast_with_model(ia_m, ia_w, months)
        preds_mw = forecast_with_model(mw_m, mw_w, months)

        # Build timeline for forecast dates
        last_date = combo["date"].max()
        dates = [last_date + pd.DateOffset(months=i+1) for i in range(months)]

        # Construct the Plotly figure
        fig = go.Figure([
            go.Scatter(x=combo["date"], y=combo[ia_col],
                       mode="lines", name="Iowa Historic"),
            go.Scatter(x=combo["date"], y=combo["Midwest_LFPR"],
                       mode="lines", name="Midwest Historic"),
            go.Scatter(x=dates, y=preds_ia,
                       mode="lines+markers", name="Iowa Forecast"),
            go.Scatter(x=dates, y=preds_mw,
                       mode="lines+markers", name="Midwest Forecast"),
        ])
        fig.update_layout(
            title=f"LFP Forecast (+{years_ahead} yrs)",
            xaxis_title="Date",
            yaxis_title="Labor Force Participation Rate",
            template="plotly_white"
        )

        # Generate AI insight narrative
        if preds_ia and preds_mw:
            ia_final, mw_final = preds_ia[-1], preds_mw[-1]
            prompt = (
                f"Iowa forecast: {ia_final:.1f}% vs. Midwest: {mw_final:.1f}% "
                f"over {years_ahead} years. Provide 3–4 sentences of economic insight."
            )
            insight = generate_insight(prompt)
        else:
            insight = "Not enough data to generate a forecast."

        # Return graph and AI-generated insight
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
