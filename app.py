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
    """Train the 10 LFP LSTM models (IA + Midwest × 5 horizons). ~1 min."""
    from tabs.lfp_tab import lfp_model_cache
    from utils.model_utils import train_test_rnn

    ia_col = "IA_Labor_Force_Participation_Rate"
    lfp_cols = [c for c in df.columns if c.endswith("_Labor_Force_Participation_Rate")]
    df["Midwest_LFPR"] = df[lfp_cols].mean(axis=1, skipna=True)
    base = df[["date", ia_col, "Midwest_LFPR"]].dropna()

    for years in range(1, 6):
        entry = {}
        for col in [ia_col, "Midwest_LFPR"]:
            model, metrics, last_window = train_test_rnn(base[["date", col]], col)
            entry[col] = {"model": model, "metrics": metrics, "last_window": last_window}
        lfp_model_cache[years] = entry


def _preload_supersector_models(df) -> None:
    """Train 9 × 5 × 12 = 540 LSTM models. Slow (~45 min). 'full' scope only."""
    from tabs.super_tab import SUPERSECTORS, supersector_model_cache
    from utils.model_utils import train_test_rnn

    for sector in SUPERSECTORS:
        for years in range(1, 6):
            entry = {}
            for st in ALL_STATES:
                col = f"{st}_{sector}"
                sub = df[["date", col]].dropna()
                model, metrics, last_window = train_test_rnn(sub, col)
                entry[st] = {
                    "model": model,
                    "metrics": metrics,
                    "last_window": last_window,
                }
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
    suppress_callback_exceptions=True
)
server = app.server

app.layout = html.Div([
    dbc.NavbarSimple("Prairie Insights: Midwest Labor Dashboard",
                     color="dark", dark=True, className="mb-4"),
    dbc.Tabs(
        id="tabs", active_tab="eda",
        children=[
            dbc.Tab(label="EDA / Overview",       tab_id="eda"),
            dbc.Tab(label="LFP Forecast",         tab_id="lfp"),
            dbc.Tab(label="Supersector Forecast", tab_id="super"),
            dbc.Tab(label="About",                tab_id="about"),
        ],
    ),
    html.Div(id="tab-content", className="p-4"),
])

@app.callback(Output("tab-content","children"),
              Input("tabs","active_tab"))
def display_tab(active_tab):
    return tabs.get(active_tab, tabs['eda']).render_layout()

def register_all_callbacks(app):
    for m in tabs.values():
        m.register_callbacks(app)

register_all_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
