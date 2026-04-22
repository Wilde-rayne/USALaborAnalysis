"""
About tab — project background, architecture, and data-source index.

Content here is the canonical "what does this thing actually do" doc
for visitors hitting the dashboard cold. Keep it in sync with the
README and the `utils/ontology.py` source list — drift here costs
trust, because it's the only narrative description a UI-first
visitor will read.
"""
from __future__ import annotations

from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from utils import preload_state
from utils.constants import ALL_STATES, END_YEAR, START_YEAR
from utils.llm_utils import generate_insight
from utils.ontology import ONTOLOGY


def _source_table() -> html.Table:
    """Render the ontology's DataSource catalog as a small reference table."""
    header = html.Thead(
        html.Tr(
            [html.Th(c) for c in ("Source", "API", "What it provides")]
        )
    )
    body_rows = []
    for src in ONTOLOGY.sources.values():
        body_rows.append(
            html.Tr(
                [
                    html.Td([html.A(src.name, href=src.url, target="_blank")]),
                    html.Td("✓" if src.api_available else "—"),
                    html.Td(src.description),
                ]
            )
        )
    return html.Table(
        [header, html.Tbody(body_rows)],
        className="table table-sm table-striped",
    )


def render_layout():
    preload_msg = preload_state.preload_completed_at or "still warming up…"
    return html.Div(
        className="p-4",
        children=[
            html.H2("Prairie Insights — US Labor Analysis"),
            html.P(
                "An interactive data-science platform for US state-level labor "
                "and economic indicators. Every forecast runs a multi-model "
                "bakeoff on historical data, picks the winner by out-of-sample "
                "RMSE, and shows the statistical diagnostics (stationarity, "
                "residual whiteness, Diebold-Mariano vs naive baseline) under "
                "the chart.",
                className="lead",
            ),
            html.Hr(),
            html.H4("Data coverage"),
            html.Ul(
                [
                    html.Li(
                        f"Currently active: {len(ALL_STATES)} "
                        f"jurisdiction(s) — {', '.join(ALL_STATES)}"
                    ),
                    html.Li(
                        f"Architecturally supported: all 50 states + DC + "
                        f"5 territories ({len(ONTOLOGY.states)} total). "
                        "Switch via STATE_SET=all_states env."
                    ),
                    html.Li(f"Monthly history: {START_YEAR} – {END_YEAR}"),
                    html.Li(f"Last data refresh + warm-up: {preload_msg}"),
                ]
            ),
            html.H4("Data sources"),
            _source_table(),
            html.H4("Tabs"),
            html.Ul(
                [
                    html.Li(
                        [
                            html.B("EDA / Overview"),
                            ": time-series, rolling average, YoY change, "
                            "summary stats. AI-written narrative below each view.",
                        ]
                    ),
                    html.Li(
                        [
                            html.B("LFP Forecast"),
                            ": Iowa vs. Midwest labor-force participation "
                            "rate. 12-mo rolling overlays, trend summary, "
                            "model-selection rationale, residual diagnostics.",
                        ]
                    ),
                    html.Li(
                        [
                            html.B("Supersector Forecast"),
                            ": sector-by-state employment forecast with a "
                            "site-selection recommendation card, a "
                            "relative-to-median threshold filter, and a "
                            "per-state winner table.",
                        ]
                    ),
                    html.Li(
                        [html.B("About"), ": this page and the assistant."]
                    ),
                ]
            ),
            html.Details(
                [
                    html.Summary("Under the hood"),
                    html.Ul(
                        [
                            html.Li(
                                "Forecasting — ``utils.forecasting`` runs an "
                                "expanding-window backtest across a candidate "
                                "set (Naive / Seasonal-Naive / Holt-Winters "
                                "ETS by default; LSTM available as an "
                                "opt-in), picks the minimum-RMSE model per "
                                "series, refits it on the full history."
                            ),
                            html.Li(
                                "Diagnostics — Augmented Dickey-Fuller & KPSS "
                                "for stationarity, Ljung-Box for residual "
                                "autocorrelation, Jarque-Bera for normality, "
                                "Diebold-Mariano with the Harvey-Leybourne-"
                                "Newbold small-sample correction vs a Naive "
                                "baseline."
                            ),
                            html.Li(
                                "Chat & blurbs — LangChain ChatOllama against "
                                "llama3.2:3b (user-facing) with a retrieval "
                                "layer that surfaces ontology-enriched "
                                "sentences instead of raw table rows."
                            ),
                            html.Li(
                                "Data-refresh agent — LangChain DeepAgents "
                                "harness over phi3 binds the CES, LAUS, "
                                "Census, BEA, and FRED fetch tools so it can "
                                "orchestrate a multi-source refresh from a "
                                "natural-language prompt."
                            ),
                            html.Li(
                                "Embeddings — sentence-transformers "
                                "``intfloat/e5-small-v2`` in-process (no "
                                "Spark); the corpus is generated by "
                                "``SentenceRAGBuilder`` from three layers: "
                                "per-(state, year, metric) facts, "
                                "cross-state rankings, and annual trend "
                                "summaries."
                            ),
                            html.Li(
                                "Reactive startup — on first request the app "
                                "verifies the cache is fresh; stale or "
                                "missing triggers a background pipeline run "
                                "via the fetcher toolbelt, otherwise the "
                                "dashboard binds in sub-second time."
                            ),
                        ]
                    ),
                ],
                open=False,
                className="mb-4",
            ),
            html.Hr(),
            html.H6("Ask the AI Assistant"),
            html.Label(
                "Your question",
                htmlFor="about-chat-input",
                className="visually-hidden",
            ),
            dcc.Input(
                id="about-chat-input",
                type="text",
                placeholder="Ask about the data, forecasts, or architecture…",
                style={"width": "80%"},
            ),
            html.Button(
                "Submit",
                id="about-chat-button",
                className="btn btn-outline-primary btn-sm ml-2",
                **{"aria-label": "Submit chat question"},
            ),
            html.Div(id="about-chat-output", className="mt-3"),
        ],
    )


def register_callbacks(app):
    @app.callback(
        Output("about-chat-output", "children"),
        Input("about-chat-button", "n_clicks"),
        State("about-chat-input", "value"),
    )
    def update_about_chat(n_clicks, query):
        if not n_clicks or not query:
            raise PreventUpdate
        answer = generate_insight(query, active_tab="about")
        return html.Div(
            [
                dcc.Markdown(answer),
                dcc.Markdown("_Disclaimer: AI-generated; may contain inaccuracies._"),
            ]
        )
