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

from utils.agents import blurb_async
from utils.constants import ALL_STATES, SUPERSECTORS
from utils.data_pipeline import ensure_data
from utils.forecasting import ForecastResult, select_forecaster
from utils.forecasting.models import default_candidates
from utils.ontology import ONTOLOGY
from tabs._components import (
    PANEL_BLURB_TYPE,
    error_boundary,
    figure_panel,
    progress_strip,
    render_blurb,
    tab_recap,
)
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

#: Reserved labels used for aggregate rows in the forecast dict. State
#: codes never collide with these strings, so a set-membership filter is
#: the safe way to separate per-state forecasts from the reference rows
#: the layout also keeps in the same dict.
AGGREGATE_KEYS: frozenset[str] = frozenset({"Region Mean", "Region Median"})

# Cache keyed by (sector, years). Value: {state: ForecastResult}. The
# bounded LRU wrapper guarantees the worker process can't grow without
# limit if a user explores many sector × horizon combinations.
from collections import OrderedDict


class _LRUCache(OrderedDict):
    """Bounded OrderedDict — newest insertion stays at the right end."""

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.maxsize:
            self.popitem(last=False)


supersector_model_cache: _LRUCache = _LRUCache(maxsize=64)


def _load_super_panel() -> pd.DataFrame:
    from utils.data_pipeline import load_panel_df

    ensure_data()
    return load_panel_df()


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
    states_only = {k: v for k, v in forecasts.items() if k not in AGGREGATE_KEYS}
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
    the median. Aggregate rows (``AGGREGATE_KEYS``) always pass. A
    ``None`` or non-positive threshold disables the filter.
    """
    if not threshold_pct or threshold_pct <= 0:
        return forecasts
    states_only = {k: v for k, v in forecasts.items() if k not in AGGREGATE_KEYS}
    if not states_only:
        return forecasts
    median = float(np.median(list(states_only.values())))
    cutoff = median * (threshold_pct / 100.0)
    kept = {k: v for k, v in forecasts.items() if k in AGGREGATE_KEYS or v >= cutoff}
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
            # Day-3: same threaded-runner + polling pattern as the LFP
            # tab. ``update_super`` spawns a daemon-thread blurb runner,
            # the polling callback below ticks every 2 s and fills each
            # panel as its individual Ollama call returns.
            dcc.Store(id="super-blurb-payload", data=None),
            dcc.Interval(
                id="super-progress-tick",
                interval=2_000,
                n_intervals=0,
                disabled=True,
            ),
        ]
    )


def _blurb_id(section: str) -> dict:
    """Pattern-matching ID convention shared with other tabs."""
    return {"type": PANEL_BLURB_TYPE, "tab": "super", "section": section}


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
        Output("super-blurb-payload", "data"),
        Output("super-progress-tick", "disabled"),
        Output("super-progress-tick", "n_intervals"),
        Output("pi-active-view", "data", allow_duplicate=True),
        Input("supersector-run", "n_clicks"),
        State("supersector-dropdown", "value"),
        State("supersector-region", "value"),
        State("supersector-sort", "value"),
        State("supersector-years-slider", "value"),
        State("supersector-threshold", "value"),
        prevent_initial_call=True,
    )
    @error_boundary(fallback_id="super-output", extra_outputs=4)
    def update_super(n_clicks, sector, region, sort_mode, years_ahead, threshold):
        if not n_clicks:
            raise PreventUpdate

        df = _load_super_panel()
        entry = _get_or_train_supersector(sector, years_ahead, df)
        entry = _apply_region_filter(entry, region)
        if not entry:
            return (
                html.Div(
                    f"No {sector.replace('_', ' ')} data available for the "
                    f"selected region.",
                    className="alert alert-warning",
                ),
                None,
                True,
                0,
                None,
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

        # Threshold filter — applied to per-state forecasts before the
        # aggregate reference rows are computed, so "Region Mean" and
        # "Region Median" summarize the set the user can actually see
        # in the chart. See combined-review.md Track B row 5.
        display_states = _sort_states(state_summary, sort_mode)
        flat_forecasts = {c: state_summary[c]["value"] for c in display_states}
        flat_forecasts = _apply_threshold(flat_forecasts, threshold)
        # Rebuild ordered state list after threshold.
        display_states = [c for c in display_states if c in flat_forecasts]

        # Aggregate reference rows are computed on the post-filter set
        # so the labels in the chart legend match the numerical content.
        vals = np.array(
            [state_summary[c]["value"] for c in display_states], dtype=float
        )
        mean_val = float(vals.mean()) if vals.size else 0.0
        median_val = float(np.median(vals)) if vals.size else 0.0

        # Re-attach the aggregate rows to the flat-forecast dict for
        # downstream consumers (recommendation panel + view payload).
        flat_forecasts["Region Mean"] = mean_val
        flat_forecasts["Region Median"] = median_val
        all_forecasts = dict(flat_forecasts)

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

        # ----- Build typed view_state payloads -----
        sector_label = sector.replace("_", " ")
        region_label = "All states" if region in (None, "__all__") else region

        forecast_view = {
            "_kind": "supersector_forecast",
            "title": f"{sector_label} +{years_ahead}-yr forecast",
            "sector": sector,
            "sector_label": sector_label,
            "region_label": region_label,
            "horizon_years": years_ahead,
            "sort_mode": sort_mode,
            "states": [
                {
                    "code":      c,
                    "forecast":  round(state_summary[c]["value"], 1),
                    "lower_ci":  round(state_summary[c]["lower"], 1),
                    "upper_ci":  round(state_summary[c]["upper"], 1),
                    "growth":    round(state_summary[c]["growth"], 2),
                    "model":     state_summary[c]["model"],
                    "rmse":      round(state_summary[c]["rmse"], 2),
                }
                for c in display_states
            ],
            "region_mean":   round(mean_val, 1),
            "region_median": round(median_val, 1),
        }

        # Top / bottom relative to median for the recommendation view —
        # restricted to the post-threshold display_states so the panel
        # matches what the chart shows.
        sorted_by_value = sorted(
            ((c, state_summary[c]["value"]) for c in display_states),
            key=lambda kv: kv[1],
            reverse=True,
        )
        top = [
            {"code": c, "value": round(v, 1)}
            for c, v in sorted_by_value[:3]
        ]
        bottom = [
            {"code": c, "value": round(v, 1)}
            for c, v in sorted_by_value[-3:]
        ]
        recommendation_view = {
            "_kind": "supersector_recommendation",
            "title": f"Where to site {sector_label} (+{years_ahead}-yr horizon)",
            "sector": sector,
            "sector_label": sector_label,
            "horizon_years": years_ahead,
            "median": round(median_val, 1),
            "top": top,
            "bottom": bottom,
        }

        models_view = {
            "_kind": "supersector_models",
            "title": "Per-state bake-off winners",
            "sector": sector,
            "sector_label": sector_label,
            "winners": [
                {
                    "code":  c,
                    "model": state_summary[c]["model"],
                    "rmse":  round(state_summary[c]["rmse"], 2),
                    "mae":   round(entry[c].metrics.mae, 2),
                    "bias":  round(entry[c].metrics.bias, 2),
                }
                for c in display_states
            ],
        }

        recap_view = {
            "_kind": "recap",
            "title": f"{sector_label} ({region_label}) +{years_ahead}-yr outlook recap",
            "panels": [forecast_view, recommendation_view, models_view],
        }

        # ----- Spawn background blurb runner -----
        run_id = blurb_async.start_run({
            "forecast":       forecast_view,
            "recommendation": recommendation_view,
            "models":         models_view,
            "recap":          recap_view,
        })

        # ----- Interleaved body -----
        body = [
            progress_strip("super-progress-status"),
            figure_panel(
                title=forecast_view["title"],
                figure=dcc.Graph(figure=fig),
                caption=(
                    f"Bars: per-state forecast at +{years_ahead} years. "
                    f"Error bars: 95 % prediction interval from each "
                    f"state's winning model."
                ),
                blurb_id=_blurb_id("forecast"),
            ),
            figure_panel(
                title=recommendation_view["title"],
                figure=_recommendation_panel(
                    all_forecasts, sector=sector, years_ahead=years_ahead
                ),
                caption=(
                    f"Top / bottom 3 vs the across-state median "
                    f"({median_val:,.1f}). Use as a regional siting prior."
                ),
                blurb_id=_blurb_id("recommendation"),
            ),
            figure_panel(
                title=models_view["title"],
                figure=_selection_summary(entry),
                caption=(
                    "Naive / Seasonal-Naive / ETS bake-off winners per "
                    "state. Mixed picks signal methodological uncertainty; "
                    "uniform picks signal stable signal in the data."
                ),
                blurb_id=_blurb_id("models"),
            ),
            tab_recap(
                title="Recap & deeper detail",
                blurb_id=_blurb_id("recap"),
            ),
            html.Hr(),
            methodology_panel(),
        ]

        active_view = {
            "tab": "super",
            "panels": [forecast_view, recommendation_view, models_view],
            "recap": recap_view,
        }
        payload = {"run_id": run_id, "n": int(n_clicks)}
        return (
            html.Div(body),
            payload,
            False,
            0,
            active_view,
        )

    @app.callback(
        Output({"type": PANEL_BLURB_TYPE, "tab": "super", "section": "forecast"},       "children"),
        Output({"type": PANEL_BLURB_TYPE, "tab": "super", "section": "recommendation"}, "children"),
        Output({"type": PANEL_BLURB_TYPE, "tab": "super", "section": "models"},         "children"),
        Output({"type": PANEL_BLURB_TYPE, "tab": "super", "section": "recap"},          "children"),
        Output("super-progress-status", "children"),
        Output("super-progress-tick", "disabled", allow_duplicate=True),
        Input("super-progress-tick", "n_intervals"),
        State("super-blurb-payload", "data"),
        prevent_initial_call=True,
    )
    @error_boundary(fallback_id="super-progress-status", extra_outputs=5)
    def poll_super_blurbs(_n, payload):
        """Tick-driven polling — same shape as the LFP / EDA pollers."""
        if not payload or "run_id" not in payload:
            raise PreventUpdate
        snapshot = blurb_async.get_snapshot(payload["run_id"])
        if not snapshot:
            raise PreventUpdate

        order = ("forecast", "recommendation", "models", "recap")
        outs: list = []
        for sec in order:
            text = snapshot.get(sec)
            if text is None:
                outs.append(html.Em(
                    f"Generating {sec}…", className="pi-muted small"
                ))
            elif text == "":
                outs.append(html.Em(
                    "no panel context — re-run the forecast",
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
        return (*outs, progress_msg, is_done
        )

    # Chat lives in the global chat drawer now — registered in app.py.
