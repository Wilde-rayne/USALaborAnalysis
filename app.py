# app.py
import os
from dotenv import load_dotenv
import threading
import requests
import pandas as pd
from datetime import datetime
from utils.model_utils import train_test_rnn
from utils.data_pipeline import OUTPUT_JSON
from utils.constants import (
    ALL_STATES, MONTH_MAP, START_YEAR, END_YEAR,
    OLLAMA_URL, OLLAMA_API_PATH, OLLAMA_MODEL, DEFAULT_TIMEOUT
)

load_dotenv()

from utils.preload_state import preload_completed_at

def _background_preload():
    global preload_completed_at

    from utils.data_pipeline import refresh_all
    refresh_all(ALL_STATES, 2012, END_YEAR)

    from utils.embeddings import load_embeddings
    load_embeddings()

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role":"system","content":"warm up"}]
        }
        requests.post(
            f"{OLLAMA_URL.rstrip('/')}{OLLAMA_API_PATH}",
            json=payload,
            timeout=DEFAULT_TIMEOUT * 2
        )
    except Exception:
        pass

    df = pd.read_json(OUTPUT_JSON, orient="records")
    df["period"] = df["period"].map(MONTH_MAP).fillna(df["period"])
    df["date"]   = pd.to_datetime(
        df["year"].astype(str) + "-" + df["period"] + "-01",
        format="%Y-%B-%d", errors="coerce"
    )
    df.sort_values("date", inplace=True)

    from tabs.lfp_tab import lfp_model_cache
    ia_col = "IA_Labor_Force_Participation_Rate"
    lfp_cols = [c for c in df if c.endswith("_Labor_Force_Participation_Rate")]
    df["Midwest_LFPR"] = df[lfp_cols].mean(axis=1, skipna=True)
    base = df[["date", ia_col, "Midwest_LFPR"]].dropna()

    for years in range(1, 6):
        entry = {}
        for col in [ia_col, "Midwest_LFPR"]:
            model, metrics, last_window = train_test_rnn(base[["date", col]], col)
            entry[col] = {"model": model, "metrics": metrics, "last_window": last_window}
        lfp_model_cache[years] = entry

    from tabs.super_tab import supersector_model_cache, SUPERSECTORS
    for sector in SUPERSECTORS:
        for years in range(1, 6):
            key = (sector, years)
            entry = {}
            for st in ALL_STATES:
                col = f"{st}_{sector}"
                sub = df[["date", col]].dropna()
                model, metrics, last_window = train_test_rnn(sub, col)
                entry[st] = {"model": model, "metrics": metrics, "last_window": last_window}
            supersector_model_cache[key] = entry

    preload_completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
