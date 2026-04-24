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
from utils.forecasting.models import default_candidates
from utils.llm_utils import generate_insight
from utils.ontology import ONTOLOGY
from tabs._components import error_boundary
from tabs._methodology import methodology_panel

# Region options offered in the filter. Territories are grouped under
# "Pacific" + "Caribbean" per the ontology; "All" shows whatever
# ``ALL_STATES`` resolves to (env-controlled).
REGION_OPTIONS: list[dict] = [
    {"label": "All active", "value": "__all__"},
    {"label": "Midwest",    "value": "Midwest"},
    {"label": "Northeast",  "value": "Northeast"},
    {"label": "South",      "value": "South"},
    {"label": "West",       "value": "West"},
    {"label": "Caribbean (territories)", "value": "Caribbean"},
    {"label": "Pacific (territories)",   "value": "Pacific"},
]

SORT_OPTIONS: list[dict] = [
    {"label": "Forecast value (high → low)", "value": "value_desc"},
    {"label": "Forecast value (low → high)", "value": "value_asc"},
    {"label": "% growth vs. last known",     "value": "growth_desc"},
    {"label": "CI width (most uncertain)",   "value": "ci_desc"},
    {"label": "CI width (most certain)",     "value": "ci_asc"},
]

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
            # Skip ARIMA on Super — this loop runs 12+ bakeoffs and the
            # ARIMA grid search would push cold-click latency past a
            # minute per sector. LFP keeps ARIMA (only 2 series).
            result = select_forecaster(
                y,
                dates=dates,
                horizon=horizon,
                n_folds=3,
                candidates=default_candidates(include_arima=False),
            )
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


def _recommendation_panel(
    forecasts: dict[str, float],
    *,
    sector: str,
    years_ahead: int,
    top_n: int = 3,
) -> html.Div:
    """
    "Where's the best place to site this sector?" card.

    Ranks states by forecasted employment, reports the top-N and
    bottom-N relative to the median, and annotates each with its
    percent delta so a reader can eyeball the spread.
    """
    # Separate per-state forecasts from the aggregate rows the caller
    # also sticks into ``forecasts``.
    states_only = {k: v for k, v in forecasts.items() if not k.startswith("Midwest")}
    if len(states_only) < 2:
        return html.Div()

    sorted_items = sorted(states_only.items(), key=lambda p: p[1], reverse=True)
    median = float(np.median([v for _, v in sorted_items]))

    def _row(rank: int, state: str, value: float) -> html.Tr:
        pct = ((value - median) / median * 100.0) if median else 0.0
        arrow = "▲" if pct > 0 else ("▼" if pct < 0 else "–")
        return html.Tr(
            [
                html.Td(f"#{rank}"),
                html.Td(state),
                html.Td(f"{value:,.1f}"),
                html.Td(f"{arrow} {pct:+.1f}% vs median"),
            ]
        )

    top_rows = [
        _row(i + 1, st, v) for i, (st, v) in enumerate(sorted_items[:top_n])
    ]
    bot_rows = [
        _row(len(sorted_items) - (top_n - 1 - i), st, v)
        for i, (st, v) in enumerate(reversed(sorted_items[-top_n:]))
    ]

    readable_sector = sector.replace("_", " ")
    return html.Div(
        [
            html.H6(
                f"Where to site {readable_sector} (+{years_ahead} yr forecast)"
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Small("Top 3 — strongest forecast",
                                       className="text-success fw-bold"),
                            html.Table(
                                [html.Tbody(top_rows)],
                                className="table table-sm mb-3",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Small("Bottom 3 — weakest forecast",
                                       className="text-danger fw-bold"),
                            html.Table(
                                [html.Tbody(bot_rows)],
                                className="table table-sm",
                            ),
                        ]
                    ),
                ]
            ),
            html.Small(
                f"Median forecast across states: {median:,.1f}",
                className="text-muted",
            ),
        ],
        className="alert alert-light border mb-3",
    )


def _apply_threshold(
    forecasts: dict[str, float], threshold_pct: float | None
) -> dict[str, float]:
    """
    Drop per-state entries whose forecast is below ``threshold_pct``% of
    the median. Aggregate rows ("Midwest Mean / Median") always pass.
    A ``None`` or non-positive threshold disables the filter.
    """
    if not threshold_pct or threshold_pct <= 0:
        return forecasts
    states_only = {k: v for k, v in forecasts.items() if not k.startswith("Midwest")}
    if not states_only:
        return forecasts
    median = float(np.median(list(states_only.values())))
    cutoff = median * (threshold_pct / 100.0)
    kept = {k: v for k, v in forecasts.items() if k.startswith("Midwest") or v >= cutoff}
    return kept


def render_layout():
    return html.Div(
        [
            html.H5("Supersector Employment Forecast"),
            html.P(
                "For each state, a Naive / Seasonal-Naive / ETS bakeoff "
                "selects the model with the lowest expanding-window RMSE. "
                "Bars show the winning model's forward projection with "
                "95 % prediction intervals as error bars.",
                className="text-muted small",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Supersector", htmlFor="supersector-dropdown"),
                            dcc.Dropdown(
                                id="supersector-dropdown",
                                options=[
                                    {"label": s.replace("_", " "), "value": s}
                                    for s in SUPERSECTORS
                                ],
                                value=SUPERSECTORS[0],
                                clearable=False,
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Label("Region filter", htmlFor="supersector-region"),
                            dcc.Dropdown(
                                id="supersector-region",
                                options=REGION_OPTIONS,
                                value="__all__",
                                clearable=False,
                            ),
                            html.Small(
                                "Restrict the chart and the recommendation card "
                                "to states in one Census region.",
                                className="pi-muted",
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Label("Sort by", htmlFor="supersector-sort"),
                            dcc.Dropdown(
                                id="supersector-sort",
                                options=SORT_OPTIONS,
                                value="value_desc",
                                clearable=False,
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Label("Years ahead", htmlFor="supersector-years-slider"),
                            dcc.Slider(
                                id="supersector-years-slider",
                                min=1,
                                max=5,
                                step=1,
                                marks={i: str(i) for i in range(1, 6)},
                                value=2,
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Hide states under this % of median (0 = show all)",
                                htmlFor="supersector-threshold",
                            ),
                            dcc.Slider(
                                id="supersector-threshold",
                                min=0,
                                max=80,
                                step=10,
                                marks={0: "off", 25: "25%", 50: "50%", 75: "75%"},
                                value=0,
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Run Forecast",
                                id="supersector-run",
                                className="btn btn-primary",
                            ),
                        ],
                        className="mb-3",
                    ),
                ],
                className="pi-selector-grid",
            ),
            dcc.Loading(id="loading-super", children=html.Div(id="super-output")),
        ]
    )


def _apply_region_filter(entry: dict[str, ForecastResult], region: str) -> dict[str, ForecastResult]:
    if region in (None, "__all__"):
        return entry
    allowed = {s.code for s in ONTOLOGY.states_in(region)}
    return {code: result for code, result in entry.items() if code in allowed}


def _sort_states(
    state_values: dict[str, dict],
    mode: str,
) -> list[str]:
    """
    ``state_values`` maps state code → {"value", "growth", "ci_width"}.
    Returns the state codes in the order requested.
    """
    if mode == "value_asc":
        key = lambda c: state_values[c]["value"]; reverse = False  # noqa: E731
    elif mode == "growth_desc":
        key = lambda c: state_values[c]["growth"]; reverse = True  # noqa: E731
    elif mode == "ci_desc":
        key = lambda c: state_values[c]["ci_width"]; reverse = True  # noqa: E731
    elif mode == "ci_asc":
        key = lambda c: state_values[c]["ci_width"]; reverse = False  # noqa: E731
    else:
        key = lambda c: state_values[c]["value"]; reverse = True  # noqa: E731 — value_desc default
    return sorted(state_values.keys(), key=key, reverse=reverse)


def register_callbacks(app):
    @app.callback(
        Output("super-output", "children"),
        Input("supersector-run", "n_clicks"),
        State("supersector-dropdown", "value"),
        State("supersector-region", "value"),
        State("supersector-sort", "value"),
        State("supersector-years-slider", "value"),
        State("supersector-threshold", "value"),
    )
    @error_boundary(fallback_id="super-output")
    def update_super(n_clicks, sector, region, sort_mode, years_ahead, threshold):
        if not n_clicks:
            raise PreventUpdate

        df = _load_super_panel()
        entry = _get_or_train_supersector(sector, years_ahead, df)
        entry = _apply_region_filter(entry, region)
        if not entry:
            return html.Div(
                f"No {sector.replace('_', ' ')} data available for the "
                f"selected region.",
                className="alert alert-warning",
            )

        # Per-state summary: forecast value at end of horizon, growth
        # vs. last observed value, and 95% CI width — feeds the chart
        # error bars and the sort selector.
        months = years_ahead * 12
        state_summary: dict[str, dict] = {}
        for st, result in entry.items():
            interval = result.model.predict_interval(months, alpha=0.05)
            if interval is None:
                preds = result.model.predict(months)
                lower, upper = preds, preds
            else:
                preds, lower, upper = interval
            col = f"{st}_{sector}"
            last_known = float(df[col].dropna().iloc[-1]) if col in df.columns and not df[col].dropna().empty else 0.0
            forecast_val = float(preds[-1]) if len(preds) else 0.0
            growth = ((forecast_val - last_known) / last_known * 100.0) if last_known else 0.0
            state_summary[st] = {
                "value":    forecast_val,
                "lower":    float(lower[-1]) if len(lower) else forecast_val,
                "upper":    float(upper[-1]) if len(upper) else forecast_val,
                "ci_width": float(upper[-1] - lower[-1]) if len(lower) else 0.0,
                "growth":   growth,
                "last":     last_known,
                "model":    result.name,
                "rmse":     result.metrics.rmse,
            }

        # Aggregate reference rows (always shown; don't get filtered by
        # region — they're descriptive for whatever set is on-screen).
        vals = np.array([s["value"] for s in state_summary.values()], dtype=float)
        mean_val = float(vals.mean()) if vals.size else 0.0
        median_val = float(np.median(vals)) if vals.size else 0.0

        # Threshold filter — applied after growth etc. are computed so
        # the recommendation panel can still see the full set.
        display_states = _sort_states(state_summary, sort_mode)
        flat_forecasts = {c: state_summary[c]["value"] for c in display_states}
        flat_forecasts["Region Mean"] = mean_val
        flat_forecasts["Region Median"] = median_val
        all_forecasts = dict(flat_forecasts)
        flat_forecasts = _apply_threshold(flat_forecasts, threshold)

        # Rebuild ordered state list after threshold.
        display_states = [c for c in display_states if c in flat_forecasts]

        # Plotly bars + error bars for 95% CI.
        labels = display_states + ["Region Mean", "Region Median"]
        data = [state_summary[c]["value"] for c in display_states] + [mean_val, median_val]
        error_up = (
            [state_summary[c]["upper"] - state_summary[c]["value"] for c in display_states]
            + [0.0, 0.0]
        )
        error_dn = (
            [state_summary[c]["value"] - state_summary[c]["lower"] for c in display_states]
            + [0.0, 0.0]
        )
        hover = (
            [
                f"<b>{c}</b><br>forecast {state_summary[c]['value']:,.1f}"
                f"<br>95% CI [{state_summary[c]['lower']:,.1f}, "
                f"{state_summary[c]['upper']:,.1f}]"
                f"<br>growth vs last {state_summary[c]['growth']:+.1f}%"
                f"<br>model {state_summary[c]['model']} (RMSE {state_summary[c]['rmse']:.2f})"
                for c in display_states
            ]
            + [f"<b>Region Mean</b><br>{mean_val:,.1f}",
               f"<b>Region Median</b><br>{median_val:,.1f}"]
        )
        mn, mx = (min(data), max(data)) if data else (0.0, 1.0)
        colors = [
            "rgb(43,79,129)" if lbl.startswith("Region")
            else f"rgb({int(255 * (1 - (v - mn) / (mx - mn + 1e-6)))},"
                 f"{int(255 * ((v - mn) / (mx - mn + 1e-6)))},0)"
            for lbl, v in zip(labels, data)
        ]

        fig = go.Figure(
            [
                go.Bar(
                    x=labels,
                    y=data,
                    marker=dict(color=colors),
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=error_up,
                        arrayminus=error_dn,
                        color="rgba(60, 70, 90, 0.55)",
                        thickness=1.5,
                        width=6,
                    ),
                    text=[f"{v:,.0f}" for v in data],
                    textposition="auto",
                    hovertext=hover,
                    hoverinfo="text",
                )
            ]
        )
        fig.update_layout(
            title=(
                f"{sector.replace('_', ' ')} Forecast (+{years_ahead} yrs) "
                f"— {('All active' if region in (None, '__all__') else region)}"
            ),
            xaxis_title="State",
            yaxis_title="Forecasted Employment",
            template="plotly_white",
        )

        winner_counts: dict[str, int] = {}
        for r in entry.values():
            winner_counts[r.name] = winner_counts.get(r.name, 0) + 1
        winner_summary = ", ".join(f"{n} × {c}" for n, c in winner_counts.items())
        prompt = (
            f"From forecasts for '{sector}' (+{years_ahead} years, "
            f"region={region}, sort={sort_mode}): "
            + ", ".join(f"{lbl} {flat_forecasts[lbl]:,.1f}" for lbl in labels if lbl in flat_forecasts)
            + f". Per-state model selection: {winner_summary}. "
            "Provide a concise 3-sentence analysis for regional planners."
        )
        insight = generate_insight(prompt)

        return html.Div(
            [
                dcc.Graph(figure=fig),
                html.Hr(),
                _recommendation_panel(
                    all_forecasts, sector=sector, years_ahead=years_ahead
                ),
                _selection_summary(entry),
                html.Hr(),
                dcc.Markdown(insight),
                dcc.Markdown("_Disclaimer: AI-generated; may contain inaccuracies._"),
                html.Hr(),
                methodology_panel(),
            ]
        )

    # Chat lives in the global chat drawer now — registered in app.py.
