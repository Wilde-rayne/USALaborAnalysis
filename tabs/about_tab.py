"""
About tab — project background, architecture, and data-source index.

Content here is the canonical "what does this thing actually do" doc
for visitors hitting the dashboard cold. Keep it in sync with the
README and the `utils/ontology.py` source list — drift here costs
trust, because it's the only narrative description a UI-first
visitor will read.
"""
from __future__ import annotations

from dash import html

from utils import preload_state
from utils.constants import ALL_STATES, END_YEAR, START_YEAR
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
            html.H5("Attribution & acknowledgments"),
            html.Ul(
                [
                    html.Li(
                        [
                            html.B("Built with Llama."),
                            " The chat and blurb layer runs Meta's "
                            "Llama 3.2 (3B parameter chat variant) locally "
                            "via Ollama. Use of the model is governed by the ",
                            html.A(
                                "Llama 3.2 Community License",
                                href="https://www.llama.com/llama3_2/license/",
                                target="_blank",
                                rel="noopener",
                            ),
                            ".",
                        ]
                    ),
                    html.Li(
                        [
                            html.B("Data attribution."),
                            " Source: U.S. Bureau of Labor Statistics "
                            "(CES, LAUS, JOLTS, QCEW, CPI); U.S. Census "
                            "Bureau (ACS, PEP); U.S. Bureau of Economic "
                            "Analysis (SAINC1); Federal Reserve Bank of "
                            "St. Louis (FRED); Federal Housing Finance "
                            "Agency (HPI). All data are in the public "
                            "domain (17 USC §105) and are reproduced here "
                            "with the standard agency citation framing.",
                        ]
                    ),
                    html.Li(
                        [
                            html.B("Software stack."),
                            " Embeddings: intfloat/e5-small-v2 "
                            "(sentence-transformers, Apache-2.0). "
                            "Forecasting: statsmodels (BSD-3); TensorFlow "
                            "(Apache-2.0, opt-in LSTM only). UI: Dash + "
                            "Plotly + dash-bootstrap-components. Agent "
                            "harness: LangChain + DeepAgents (MIT).",
                        ]
                    ),
                    html.Li(
                        [
                            html.B("AI assistance."),
                            " Portions of this codebase were developed "
                            "with assistance from Anthropic's Claude "
                            "(Claude Code). All design decisions, "
                            "scientific claims, dataset choices, and "
                            "the final code state are author-authored "
                            "and author-verified.",
                        ]
                    ),
                ],
                className="small text-muted",
            ),
        ],
    )


def register_callbacks(app):
    # Chat lives in the global chat drawer now — registered in app.py.
    pass
