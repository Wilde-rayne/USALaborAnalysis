# app.py
import logging
import os
import secrets
import threading
from datetime import datetime

import requests
from dotenv import load_dotenv

from utils.constants import (
    ALL_STATES,
    DEFAULT_TIMEOUT,
    END_YEAR,
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
    """Thin wrapper around ``utils.data_pipeline.load_panel_df`` for
    backward-compatibility with callers inside this module."""
    from utils.data_pipeline import load_panel_df

    return load_panel_df()


def _preload_lfp_models(df) -> None:
    """
    Pre-train the LFP bakeoff for the default focus state across the
    five horizons. Cache key matches the I7 rewrite —
    ``(metric, state_code, years_ahead)`` — so the first user click
    after a ``PRELOAD_SCOPE=lfp`` boot hits a warm cache.

    The 'Midwest_LFPR' aggregate that this function used to compute is
    no longer a thing on the LFP tab — that tab now does focus-state +
    peer-states matching, with peers chosen by the user. Pre-training
    every (metric × state × horizon) combination is too expensive
    (~50 states × 2 metrics × 5 horizons × ~10 s/bakeoff = ~80 min);
    the lazy on-click path covers that case fine. Here we only warm
    the headline series so the demo's first click is instant.
    """
    from tabs.lfp_tab import _column_for, lfp_model_cache
    from utils.constants import ALL_STATES
    from utils.forecasting import select_forecaster

    headline_state = "IA" if "IA" in ALL_STATES else ALL_STATES[0]
    for metric in ("LFPR", "Unemployment_Rate"):
        col = _column_for(metric, headline_state)
        if col not in df.columns:
            continue
        series = df[["date", col]].dropna()
        if len(series) < 24:
            continue
        y = series[col].astype(float).to_numpy()
        dates = series["date"].to_numpy()
        for years in range(1, 6):
            try:
                lfp_model_cache[(metric, headline_state, years)] = select_forecaster(
                    y, dates=dates, horizon=years * 12, n_folds=3
                )
            except RuntimeError:
                continue


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

# Stable Flask SECRET_KEY so signed session cookies survive restarts
# and a multi-worker deployment shares one signing key. Production
# must set FLASK_SECRET_KEY (see .env.example); dev falls back to a
# per-process random key so no default secret ever lands in git.
server.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
server.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    # Opt-in once TLS termination is in front; flipping it on under
    # plain-HTTP local docker-compose breaks dev sessions.
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "").lower()
    in {"1", "true", "yes"},
)


# Health endpoint for the compose healthcheck and external probes.
# Body is intentionally minimal — exposing version strings, model
# names, or preload timestamps would let an unauthenticated probe
# fingerprint the deployment (OWASP API A05:2023).
@server.route("/health")
def _health():  # noqa: D401 — Flask handler
    """Return ``{"status": ...}`` with HTTP 200 if ready, 503 if warming up."""
    ready = preload_state.preload_completed_at is not None
    return {"status": "healthy" if ready else "starting"}, (200 if ready else 503)

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


from tabs._chat_drawer import render_drawer as _render_chat_drawer  # noqa: E402
from tabs._chat_drawer import register_callbacks as _register_chat_drawer  # noqa: E402


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
        # One-shot chat drawer — replaces the per-tab chat strips.
        _render_chat_drawer(),
        # Global "what is the user looking at" payload — each tab's
        # main callback writes its current view_state here so the
        # chat drawer can ground its answers in the panels visible on
        # screen instead of just the user's question text.
        dcc.Store(id="pi-active-view", storage_type="session", data=None),
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
    _register_chat_drawer(app)


register_all_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
