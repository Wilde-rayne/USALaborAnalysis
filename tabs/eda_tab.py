import os
from datetime import datetime

import pandas as pd
import plotly.graph_objs as go
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate

from utils.data_pipeline import OUTPUT_JSON, refresh_all
from utils.llm_utils import generate_insight
from utils.constants import ALL_STATES, YEARS, MONTH_MAP, DEFAULT_TIMEOUT


def render_layout():
    return html.Div(
        [
            html.H5("Exploratory Data Analysis / Overview"),

            html.Div(id="eda-metadata", className="mb-3"),

            html.Details(
                [
                    html.Summary("Definitions (CES, LAUS, LFPR, LSTM…)") ,
                    html.Ul(
                        [
                            html.Li([html.B("CES"), " = Current Employment Statistics"]),
                            html.Li([html.B("LAUS"), " = Local Area Unemployment Statistics"]),
                            html.Li([html.B("LFPR"), " = Labor Force Participation Rate"]),
                            html.Li([html.B("LSTM"), " = Long Short-Term Memory RNN"]),
                        ]
                    ),
                ],
                open=False,
                className="mb-4",
            ),

            html.Div(
                [
                    html.Label("Select States:"),
                    dcc.Dropdown(
                        id="eda-state-selector",
                        options=[{"label": s, "value": s} for s in ALL_STATES],
                        value=[ALL_STATES[0]],
                        multi=True,
                    ),
                    html.Br(),
                    html.Label("Select Year Range:"),
                    dcc.RangeSlider(
                        id="eda-year-range",
                        min=YEARS[0],
                        max=YEARS[-1],
                        step=1,
                        marks={yr: str(yr) for yr in YEARS},
                        value=[YEARS[0], YEARS[-1] - 1],
                    ),
                    html.Br(),
                    html.Button("Refresh Data", id="eda-refresh", className="btn btn-primary"),
                ],
                className="mb-4",
            ),

            dcc.Loading(id="loading-eda", children=html.Div(id="eda-output")),

            html.Hr(),
            html.H6("Ask the AI Assistant"),
            dcc.Input(
                id="eda-chat-input",
                type="text",
                placeholder="Ask a question about this data...",
                style={"width": "80%"}
            ),
            html.Button("Submit", id="eda-chat-button", className="btn btn-outline-primary btn-sm ml-2"),
            html.Div(id="eda-chat-output", className="mt-3")
        ],
        className="p-4",
    )


def register_callbacks(app):
    @app.callback(
        Output("eda-metadata", "children"),
        Input("eda-output", "children")
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
            errors="coerce"
        )
        total = len(df)
        start = df["date"].min().strftime("%b %Y")
        end = df["date"].max().strftime("%b %Y")
        refreshed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return html.Div(
            [
                html.P(f"Total records: {total}"),
                html.P(f"Date range: {start} – {end}"),
                html.P(f"Last refreshed: {refreshed}"),
            ],
            className="alert alert-info",
        )

    @app.callback(
        Output("eda-output", "children"),
        Input("tabs", "active_tab"),
        Input("eda-refresh", "n_clicks"),
        State("eda-state-selector", "value"),
        State("eda-year-range", "value"),
    )
    def update_eda(active_tab, n_clicks, states, year_range):
        if active_tab != "eda":
            raise PreventUpdate

        if n_clicks:
            start_year, end_year = year_range
            refresh_all(states, start_year, end_year)

        if not os.path.exists(OUTPUT_JSON):
            return html.Div("No data found. Please refresh.", className="text-danger")

        df = pd.read_json(OUTPUT_JSON, orient="records")
        df["period"] = df["period"].map(MONTH_MAP).fillna(df["period"])
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["period"] + "-01",
            format="%Y-%B-%d",
            errors="coerce",
        )

        if n_clicks:
            df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
        df.sort_values("date", inplace=True)

        series_cols = sorted(
            c for c in df.columns if any(c.startswith(f"{st}_") for st in states)
        )

        stats = df[series_cols].describe().T[["mean", "50%", "min", "max"]]
        stats.rename(columns={"50%": "median"}, inplace=True)
        stats_table = html.Table(
            [
                html.Thead(html.Tr([html.Th(c) for c in ["Series", "Mean", "Median", "Min", "Max"]])),
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
            className="table table-sm table-striped mb-4",
        )

        ts_traces = [go.Scatter(x=df["date"], y=df[col], mode="lines", name=col) for col in series_cols]
        fig_ts = go.Figure(ts_traces).update_layout(
            title=f"Time Series: {', '.join(states)}",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
        )

        hist_traces = [go.Histogram(x=df[col].dropna(), name=col, opacity=0.75) for col in series_cols]
        fig_hist = go.Figure(hist_traces).update_layout(
            title="Value Distribution", barmode="overlay", template="plotly_white"
        )

        roll = df.set_index("date")[series_cols].rolling(12).mean().reset_index()
        roll_traces = [go.Scatter(x=roll["date"], y=roll[col], mode="lines", name=col) for col in series_cols]
        fig_roll = go.Figure(roll_traces).update_layout(
            title="12-Month Rolling Average", xaxis_title="Date", yaxis_title="Value", template="plotly_white"
        )

        yoy = df.set_index("date")[series_cols].pct_change(12).mul(100).reset_index()
        yoy_traces = [go.Scatter(x=yoy["date"], y=yoy[col], mode="lines", name=col) for col in series_cols]
        fig_yoy = go.Figure(yoy_traces).update_layout(
            title="Year-over-Year % Change", xaxis_title="Date", yaxis_title="Percent Change", template="plotly_white"
        )

        summary_prompt = (
            "Write four short paragraphs (≤ 40 words each):\n"
            f"1. Descriptive-stat highlights for {', '.join(states)} {year_range[0]}-{year_range[1]}.\n"
            "2. Main trends in the time-series plot.\n"
            "3. Insights from the histogram and 12-month rolling average.\n"
            "4. Explanation of notable year-over-year swings."
        )
        summary_insight = generate_insight(
            summary_prompt,
            timeout=DEFAULT_TIMEOUT
        )

        return html.Div(
            [
                html.H6("Summary Statistics"),
                stats_table,
                html.Hr(),
                dcc.Graph(figure=fig_ts),
                html.Hr(),
                dcc.Graph(figure=fig_hist),
                html.Hr(),
                dcc.Graph(figure=fig_roll),
                html.Hr(),
                dcc.Graph(figure=fig_yoy),
                html.Hr(),
                dcc.Markdown(summary_insight),
                dcc.Markdown("_Disclaimer: AI-generated; may contain inaccuracies._"),
            ]
        )

    @app.callback(
        Output("eda-chat-output", "children"),
        Input("eda-chat-button", "n_clicks"),
        State("eda-chat-input", "value")
    )
    def update_eda_chat(n_clicks, query):
        if not n_clicks or not query:
            raise PreventUpdate
        answer = generate_insight(query, timeout=DEFAULT_TIMEOUT)
        return html.Div([
            dcc.Markdown(answer),
            dcc.Markdown("_Disclaimer: AI-generated; may contain inaccuracies._")
        ])
