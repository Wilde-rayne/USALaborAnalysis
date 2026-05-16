# Midwest Labor-Market Dashboard – Technical Report

*DS 4010 (2025) • Author – Rayne Wilde (rayne.k.wilde@gmail.com)*

**Abstract.**
We present an end-to-end solution for ingesting, cleaning, modeling, and visualizing U.S. labor-market data for Iowa and eleven peer Midwestern states. Our pipeline leverages BLS APIs and a modular Python codebase to produce a compact Parquet/JSON dataset. The forecasting layer runs a multi-model bakeoff (Naive / Seasonal-Naive / Holt-Winters ETS / ARIMA with an opt-in LSTM) and picks per-series winners by out-of-sample RMSE; a Dash/Plotly web app surfaces the forecasts, residual diagnostics, and AI-written narrative summaries to policy makers and students. The chat and blurb layer is **built with Llama** — Meta's Llama 3.2 (3B) running locally via Ollama under the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/).

---

## 1  Project Goal & Audience  

Our goal is to deliver a transparent, reproducible **Midwest Labor-Market Dashboard** that transforms raw Bureau of Labor Statistics (BLS) data into actionable forecasts and visualizations. By packaging all ETL, modeling, and frontend code in Docker containers, we empower policy makers seeking near-term labor-market projections, regional planners comparing state performance and sectoral drivers, and DS 4010 students as a teaching example of full-stack data science.

---

## 2  Data Pipeline ( ~ 75 %)  

### 2.1  Data Collection (ETL)  

We ingest three primary sources:

| Source                      | URL                                                                                         | Format      | Period    | Raw Rows |
| --------------------------- | ------------------------------------------------------------------------------------------- | ----------- | --------- | -------- |
| **CES (Employment)**        | `https://api.bls.gov/publicAPI/v2/timeseries/data/SMS38000006500000001`                     | JSON (API)  | 1996–2024 | ~36 000  |
| **LAUS (Unemployment)**     | `https://api.bls.gov/publicAPI/v2/timeseries/data/LASST190000000000006`                     | JSON (API)  | 1996–2024 | ~24 000  |
| **Population (Census)**     | `https://download.bls.gov/pub/time.series/la/la.population`                                  | CSV         | 1996–2024 | ~350     |

Fetch scripts in `fetch_ces_data.py`, `fetch_laus_data.py`, and `fetch_population_data.py` write raw files to `data/raw/`, totaling about 120 MB.

### 2.2  Cleaning & Transformation  

We load the merged JSON, pivot series, drop the M13 pseudo-month, forward-fill missing values, and save compressed Parquet:

```python
import pandas as pd

df = pd.read_json('data/all_data.json')
df_wide = (
    df
    .assign(series=lambda d: d['series_id'].map(decode_series))
    .drop(columns=['series_id', 'footnotes'])
    .pivot(index=['year', 'period'], columns='series', values='value')
    .reset_index()
)
df_wide = df_wide[df_wide.period != 'M13']
df_wide.sort_values(['year', 'period'], inplace=True)
df_wide.fillna(method='ffill', inplace=True)
df_wide.to_parquet('data/all_data.parquet', index=False)
```

The resulting Parquet file is 18 MB, an 85 % reduction.

### 2.3  Row Counts & Storage  

| Stage                   | Rows   | Columns | Size   |
| ----------------------- | ------ | ------- | ------ |
| Raw CES text            | 36 000 | 4       | 90 MB  |
| Raw LAUS text           | 24 000 | 4       | 30 MB  |
| Census CSV              | 350    | 3       | 0.3 MB |
| **Merged Parquet/JSON** | 68 350 | 200+    | 18 MB  |

### 2.4  Investigated but Not Used  

| Idea                | Rationale                   | Dropped Because                |
| ------------------- | --------------------------- | ------------------------------ |
| AWS Glue ETL        | Serverless scheduling       | Free-tier limits; local run    |
| DuckDB storage      | In-process OLAP performance | Parquet already performant     |
| Prophet forecasting | Seasonality modeling        | LSTM gave lower MAE            |
| K‑12 data           | Enrichment                  | Coverage < 40 %                |

---

## 3  Modeling  

### 3.1  LFPR Forecast  

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(60, 1)),
    LSTM(32),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32)
```

LSTM achieves MAE ~ 0.35 percentage points versus ARIMA’s 0.41 pp.

### 3.2  Sectoral Employment  

We train the same LSTM architecture separately for each BLS super-sector.

### 3.3  Historical Analogues  

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def closest_month(target, history):
    data = np.vstack([target, history])
    scaled = StandardScaler().fit_transform(data)
    sims = scaled[1:] @ scaled[0] / (
        np.linalg.norm(scaled[1:], axis=1) * np.linalg.norm(scaled[0])
    )
    return history[np.argmax(sims)]
```

Analogues provide interpretable context.

---

## 4  Dashboard & UI ( ~ 25 %)  

```python
import dash
from dash import html, dcc

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='EDA', children=[...]),
        dcc.Tab(label='LFPR Forecast', children=[...]),
        dcc.Tab(label='Sector Forecast', children=[...]),
        dcc.Tab(label='About', children=[...]),
    ])
])
```

Inputs include a state selector, year-range slider, and forecast horizon. Outputs are interactive line charts, heatmaps, and text summaries.

### 4.1  Illustrations

*Figure 1: Iowa Unemployment Rate, 1996–2024.*
(Generated interactively via the dashboard — see `tabs/lfp_tab.py`
and `tabs/eda_tab.py` to reproduce against the bundled panel.)

*Figure 2: Iowa Labor-Force Participation Rate Forecast.*
(Generated interactively via the dashboard — see `tabs/lfp_tab.py`
to reproduce against the bundled panel.)

---

## 5  Application & Discussion  

**Question:** _How do Iowa’s labor-market dynamics align with the Midwestern average?_ Iowa’s LFPR remains within ±0.8 pp of the region since 2010. Sector analysis shows construction led early 2020s recovery.

---

## 6  Next Steps  

Integrate CPI and housing indices into Temporal Fusion Transformers, add “shock scenario” sliders, and automate nightly refresh via CI/CD.

---

## 7  Availability  

| Artifact     | Location                                         |
| ------------ | ------------------------------------------------ |
| Code         | `git.las.iastate.edu/kbouwman/ds4010`            |
| Dash App     | Docker Hub: `rayne/ds4010:latest`                |
| Report       | `REPORT_FINAL.md`                                |

---

## 8  Acknowledgments

**Built with Llama.** Chat and blurb output is generated by Meta's
[Llama 3.2](https://www.llama.com/llama3_2/license/) (3B chat variant)
running locally via Ollama; the Llama 3.2 Community License governs
this service's use of the model.

**Data attribution.** Source: U.S. Bureau of Labor Statistics (CES,
LAUS, JOLTS, QCEW, CPI); U.S. Census Bureau (ACS, PEP); U.S. Bureau
of Economic Analysis (SAINC1); Federal Reserve Bank of St. Louis
(FRED); Federal Housing Finance Agency (HPI). Underlying data are in
the public domain (17 USC §105).

**Software stack.** Forecasting and statistical diagnostics use
`statsmodels` (BSD-3) and `numpy` / `pandas` / `scipy`; the opt-in
LSTM uses TensorFlow (Apache-2.0). The embedding layer uses
`intfloat/e5-small-v2` via `sentence-transformers` (Apache-2.0). The
agent harness uses LangChain and DeepAgents (MIT). The full citation
registry is at `utils/citations.py` and is rendered in the
dashboard's methodology panel.

**AI assistance.** Portions of this codebase were developed with
assistance from Anthropic's Claude (via Claude Code) for code
authoring, refactoring, methodology critique, and citation curation.
All design decisions, scientific claims, dataset and methodology
choices, and the final code state are author-authored and
author-verified. This disclosure follows emerging academic and
industry norms for AI-assisted authorship.

---

## Appendix: All Series Plots  

```{=include}
img_gallery.md
```  
