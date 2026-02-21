from datetime import datetime
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
from utils.constants import ALL_STATES, START_YEAR, END_YEAR
from utils.llm_utils import generate_insight
from utils.preload_state import preload_completed_at

def render_layout():
    return html.Div(className="p-4", children=[
        html.H2("🗺 Prairie Insights Dashboard"),

        html.P(
            "This dashboard delivers clear, actionable insights into Midwest "
            "labor trends using BLS data (CES & LAUS), LSTM forecasts, and "
            "AI‐generated narratives."
        ),

        html.H4("Data Coverage"),
        html.Ul([
            html.Li(f"States covered: {len(ALL_STATES)} ({', '.join(ALL_STATES)})"),
            html.Li(f"Years: {START_YEAR} – {END_YEAR}"),
            html.Li(f"Last full ETL & model warm-up: {preload_completed_at or 'loading...'}"),
        ]),

        html.H4("How to Use"),
        html.Ul([
            html.Li([html.B("EDA / Overview"), ": Explore historical charts & summary stats."]),
            html.Li([html.B("LFP Forecast"), ": Forecast Iowa vs. Midwest LFPR with LSTM."]),
            html.Li([html.B("Supersector Forecast"), ": Forecast sector employment."]),
            html.Li([html.B("About"), ": This page & AI assistant."]),
        ]),

        html.Details([
            html.Summary("Under the Hood"),
            html.Ul([
                html.Li("Data pipeline pulls CES & LAUS via BLS APIs, merges annually."),
                html.Li("Embeddings built with Spark for AI context retrieval."),
                html.Li("LSTM models trained once on launch (12-month windows)."),
                html.Li("Local LLaMA-2 chat model preloaded & warmed in background."),
            ]),
        ], open=False, className="mb-4"),

        html.Hr(),

        html.H6("Ask the AI Assistant"),
        dcc.Input(
            id="about-chat-input",
            type="text",
            placeholder="Ask me about the data or models…",
            style={"width": "80%"}
        ),
        html.Button(
            "Submit", id="about-chat-button",
            className="btn btn-outline-primary btn-sm ml-2"
        ),
        html.Div(id="about-chat-output", className="mt-3"),
    ])


def register_callbacks(app):
    @app.callback(
        Output("about-chat-output", "children"),
        Input("about-chat-button", "n_clicks"),
        State("about-chat-input", "value"),
    )
    def update_about_chat(n_clicks, query):
        if not n_clicks or not query:
            raise PreventUpdate
        answer = generate_insight(query)
        return html.Div([
            dcc.Markdown(answer),
            dcc.Markdown("_Disclaimer: AI-generated; may contain inaccuracies._")
        ])
