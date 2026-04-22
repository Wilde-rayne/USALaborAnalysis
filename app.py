# app.py
import logging
import os
import threading
from datetime import datetime

import requests
from dotenv import load_dotenv

from utils.constants import (
    ALL_STATES,
    DEFAULT_TIMEOUT,
    END_YEAR,
    MONTH_MAP,
    OLLAMA_API_PATH,
    OLLAMA_MODEL,
    OLLAMA_URL,
    START_YEAR,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from utils import preload_state  # module so we can mutate preload_completed_at

# --------------------------------------------------------------------------
# Reactive startup
# --------------------------------------------------------------------------
# PRELOAD_SCOPE controls what the background thread warms up before the UI
# becomes responsive. Tabs lazily populate the rest on first interaction.
#
#   none         — pure reactive. Data + embeddings + models arrive on demand.
#   data         — DEFAULT. Ensure all_data.json exists (cache-hit if fresh).
#   embeddings   — + warm the embeddings cache for RAG chat.
#   lfp          — + train the 10 LFP LSTM models (~1 min).
#   full         — + train 540 supersector × horizon models (~45 min).
#
# Override via environment; dev-friendly default is "data".
PRELOAD_SCOPE = os.getenv("PRELOAD_SCOPE", "data").lower()
VALID_SCOPES = {"none", "data", "embeddings", "lfp", "full"}
if PRELOAD_SCOPE not in VALID_SCOPES:
    logger.warning(
        f"[PRELOAD] Unknown PRELOAD_SCOPE={PRELOAD_SCOPE!r}; falling back to 'data'."
    )
    PRELOAD_SCOPE = "data"


def _warm_ollama() -> None:
    """Fire-and-forget warm-up so the first user chat isn't cold."""
    try:
        requests.post(
            f"{OLLAMA_URL.rstrip('/')}{OLLAMA_API_PATH}",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "system", "content": "warm up"}],
            },
            timeout=DEFAULT_TIMEOUT * 2,
        )
    except Exception as exc:  # noqa: BLE001 — intentional best-effort
        logger.info(f"[PRELOAD] Ollama warm-up skipped: {exc}")


def _load_panel_df():
    """Load the merged panel JSON into a date-indexed DataFrame."""
    import pandas as pd

    from utils.data_pipeline import OUTPUT_JSON

    df = pd.read_json(OUTPUT_JSON, orient="records")
    df["period"] = df["period"].map(MONTH_MAP).fillna(df["period"])
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["period"] + "-01",
        format="%Y-%B-%d",
        errors="coerce",
    )
    df.sort_values("date", inplace=True)
    # Collapse (state, year, month) rows to one per date — wide columns
    # are identical across the 12 state-rows that share a date.
    df = df.drop_duplicates(subset="date")
    return df


def _preload_lfp_models(df) -> None:
    """
    Run the LFP bakeoff over the 5 horizons × 2 series (IA + Midwest).

    Populates ``tabs.lfp_tab.lfp_model_cache`` with ``ForecastResult``
    objects keyed by horizon so the first click after startup hits a
    fully-fit winner instead of running the backtest inline.
    """
    from tabs.lfp_tab import IA_COL, MIDWEST_COL, lfp_model_cache
    from utils.forecasting import select_forecaster

    lfp_cols = [c for c in df.columns if c.endswith("_Labor_Force_Participation_Rate")]
    df[MIDWEST_COL] = df[lfp_cols].mean(axis=1, skipna=True)
    base = df[["date", IA_COL, MIDWEST_COL]].dropna()

    for years in range(1, 6):
        entry: dict = {}
        for col in (IA_COL, MIDWEST_COL):
            y = base[col].astype(float).to_numpy()
            dates = base["date"].to_numpy()
            entry[col] = select_forecaster(y, dates=dates, horizon=years * 12, n_folds=3)
        lfp_model_cache[years] = entry


def _preload_supersector_models(df) -> None:
    """
    Run the supersector bakeoff over every (sector, horizon, state) cell.

    With the default Naive/SeasonalNaive/ETS candidate set each cell
    takes ~3s; 9 sectors × 5 horizons × 12 states × 3s ≈ 27 min on
    one CPU. Only runs under PRELOAD_SCOPE=full; otherwise the tabs
    fill the cache lazily on click.
    """
    from tabs.super_tab import SUPERSECTORS, supersector_model_cache
    from utils.forecasting import select_forecaster

    for sector in SUPERSECTORS:
        for years in range(1, 6):
            entry: dict = {}
            for st in ALL_STATES:
                col = f"{st}_{sector}"
                if col not in df.columns:
                    continue
                sub = df[["date", col]].dropna()
                if sub.empty:
                    continue
                y = sub[col].astype(float).to_numpy()
                dates = sub["date"].to_numpy()
                try:
                    entry[st] = select_forecaster(
                        y, dates=dates, horizon=years * 12, n_folds=3
                    )
                except RuntimeError:
                    continue  # every candidate failed — skip this state
            supersector_model_cache[(sector, years)] = entry


def _background_preload() -> None:
    """
    Warm up only what PRELOAD_SCOPE asks for. Tabs handle their own
    lazy-loading for anything skipped here.
    """
    started = datetime.now()
    logger.info(f"[PRELOAD] scope={PRELOAD_SCOPE} — starting warm-up")

    if PRELOAD_SCOPE == "none":
        preload_state.preload_completed_at = (
            f"{datetime.now():%Y-%m-%d %H:%M:%S} (scope=none)"
        )
        return

    # Always cache-aware: fetches from BLS/Census only if data/all_data.json
    # is missing or older than CACHE_MAX_AGE_SECONDS.
    from utils.data_pipeline import ensure_data

    ensure_data(ALL_STATES, START_YEAR, END_YEAR)

    if PRELOAD_SCOPE in {"embeddings", "lfp", "full"}:
        from utils.embeddings import load_embeddings

        load_embeddings()

    if PRELOAD_SCOPE in {"lfp", "full"}:
        _warm_ollama()
        df = _load_panel_df()
        _preload_lfp_models(df)

    if PRELOAD_SCOPE == "full":
        df = _load_panel_df()  # reload in case lfp mutated it
        _preload_supersector_models(df)

    elapsed = datetime.now() - started
    preload_state.preload_completed_at = (
        f"{datetime.now():%Y-%m-%d %H:%M:%S} (scope={PRELOAD_SCOPE}, took {elapsed})"
    )
    logger.info(f"[PRELOAD] done: {preload_state.preload_completed_at}")


threading.Thread(target=_background_preload, daemon=True).start()


import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output

tabs = {
    'eda':   __import__('tabs.eda_tab',   fromlist=['render_layout','register_callbacks']),
    'lfp':   __import__('tabs.lfp_tab',   fromlist=['render_layout','register_callbacks']),
    'super': __import__('tabs.super_tab', fromlist=['render_layout','register_callbacks']),
    'about': __import__('tabs.about_tab', fromlist=['render_layout','register_callbacks']),
}

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    # Standard meta tags for small-screen legibility and browser
    # dark-mode preference pickup.
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "color-scheme", "content": "light dark"},
    ],
)
server = app.server


# Real health endpoint for the docker-compose healthcheck.
# Previously the compose file hit /health and relied on `|| exit 1` to
# mask the 404 from a missing route, which meant an unhealthy process
# could still look healthy. Returning a 200 with preload status lets
# external monitors (load balancers, CI smoke) actually sample liveness.
@server.route("/health")
def _health():  # noqa: D401 — Flask handler
    """Return preload status and a 200/503 based on readiness."""
    from utils import preload_state as _ps  # noqa: PLC0415

    ready = _ps.preload_completed_at is not None
    body = {
        "status": "healthy" if ready else "starting",
        "preload_completed_at": _ps.preload_completed_at,
    }
    return body, (200 if ready else 503)

from tabs._components import status_pill  # noqa: E402

_STATUS_INTERVAL_MS = 5_000


def _status_strip() -> html.Div:
    """Header strip below the navbar — data freshness + state coverage."""
    return html.Div(
        id="pi-status-strip",
        className="pi-status-strip",
        children=_status_children(),
    )


def _status_children() -> list:
    """Rebuilt every few seconds from a lightweight interval poll."""
    ready = preload_state.preload_completed_at is not None
    return [
        html.Span("preload", className="pi-status-label"),
        status_pill(
            preload_state.preload_completed_at or "warming up…",
            tone="ok" if ready else "warn",
        ),
        html.Span("states", className="pi-status-label"),
        status_pill(
            f"{len(ALL_STATES)} active",
            tone="default",
            title=", ".join(ALL_STATES),
        ),
        html.Span("coverage", className="pi-status-label"),
        status_pill(f"{START_YEAR}–{END_YEAR}", tone="default"),
        html.Span("chat model", className="pi-status-label"),
        status_pill(OLLAMA_MODEL, tone="default"),
    ]


app.layout = html.Div(
    [
        dbc.NavbarSimple(
            "Prairie Insights — US Labor Analysis",
            color="dark",
            dark=True,
        ),
        _status_strip(),
        dcc.Interval(id="pi-status-tick", interval=_STATUS_INTERVAL_MS, n_intervals=0),
        html.Div(
            [
                dbc.Tabs(
                    id="tabs",
                    active_tab="eda",
                    children=[
                        dbc.Tab(label="EDA / Overview",       tab_id="eda"),
                        dbc.Tab(label="LFP Forecast",         tab_id="lfp"),
                        dbc.Tab(label="Supersector Forecast", tab_id="super"),
                        dbc.Tab(label="About",                tab_id="about"),
                    ],
                ),
                html.Div(id="tab-content"),
            ],
            className="pi-page",
        ),
    ]
)


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
)
def display_tab(active_tab):
    return tabs.get(active_tab, tabs["eda"]).render_layout()


@app.callback(
    Output("pi-status-strip", "children"),
    Input("pi-status-tick", "n_intervals"),
)
def _refresh_status(_n):
    return _status_children()

def register_all_callbacks(app):
    for m in tabs.values():
        m.register_callbacks(app)

register_all_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
