# Prairie Insights: U.S. Labor Market Dashboard

An interactive dashboard analyzing labor and employment trends across all 51
U.S. jurisdictions (50 states + DC), leveraging BLS data (CES, LAUS, JOLTS,
QCEW, CPI), Census ACS/PEP, BEA, and FRED, alongside multi-model forecasts
and AI-generated narratives. Iowa and the broader Midwest serve as the
default narrative entry point but the panel and tabs cover the full country.

---

## 🚀 Project Overview

- **Data sources (multi-source fetch agents, all free with a key)**
  - **BLS CES** — state-level employment by supersector (SM series)
  - **BLS LAUS** — labor force, employment, unemployment, unemployment rate
  - **US Census** — ACS + PEP population estimates
  - **BEA** — state annual personal income (SAINC1)
  - **FRED** — state macro indicators (UR, PI, NGSP, …) via `{ST}{IND}` naming
- **Forecasting — multi-model bakeoff, not a single architecture**
  - Always-on candidate set: Naive, Seasonal-Naive, Holt-Winters ETS,
    and ARIMA (with an AIC-selected order grid). LSTM is available as an
    opt-in fifth candidate but is *not* the production default.
  - An expanding-window backtest fits each candidate on the leading
    window and scores it on the held-out fold; the per-series winner is
    selected by minimum out-of-sample RMSE (see
    `utils/forecasting/selection.py`).
  - 95 % prediction intervals come from each model's native variance
    (`get_forecast` for ETS / ARIMA) or a residual-bootstrap band for the
    baselines.
  - Per-forecast diagnostics: ADF / KPSS / Ljung-Box / Jarque-Bera /
    Diebold-Mariano (with HLN small-sample correction) vs. Naive baseline.
- **LLM layer — LangChain + Ollama**
  - Chat / blurbs: `llama3.2:3b` via `ChatOllama`
  - Background agents: `phi3` via `deepagents.create_deep_agent`
  - Sentence-RAG corpus built deterministically from an ontology of
    states × supersectors × measures — no tabular blobs in embeddings
  - Embeddings: `intfloat/e5-small-v2` (in-process PyTorch, no Spark)  

---

## 📁 Directory Structure

```
USALaborAnalysis/
├── app.py                            ← Dash entry + /health + preload
├── assets/                           ← CSS, design tokens
├── data/                             ← cached output (.gitignored except configs)
│   ├── ces_state_sms_codes.json
│   ├── laus_state_codes.json
│   └── raw/                          ← per-source BLS/Census/BEA/FRED TXTs
├── tabs/
│   ├── about_tab.py                  ← this page, plus source catalog
│   ├── eda_tab.py                    ← exploratory data analysis
│   ├── lfp_tab.py                    ← LFP forecast (trend-heavy)
│   └── super_tab.py                  ← supersector forecast + site selection
├── utils/
│   ├── agents/                       ← LangChain + DeepAgents harness
│   │   ├── base.py                   ← LaborAgent + LaborAgentConfig
│   │   ├── blurb.py                  ← BlurbAgent — narrative blurbs
│   │   ├── deep.py                   ← DeepAgents data-refresh agent (phi3)
│   │   ├── ollama.py                 ← chat_agent / worker_agent factories
│   │   ├── sentence_rag.py           ← ontology-enriched RAG corpus
│   │   └── tools.py                  ← @tool fetch wrappers for the agent
│   ├── forecasting/                  ← multi-model bakeoff framework
│   │   ├── base.py                   ← BaseForecaster + metric/diagnostic types
│   │   ├── models.py                 ← Naive / SeasonalNaive / ETS / LSTM
│   │   ├── selection.py              ← expanding-window backtest + selector
│   │   ├── diagnostics.py            ← ADF / KPSS / Ljung-Box / JB / DM
│   │   └── trend.py                  ← trend summaries + rolling stats
│   ├── constants.py                  ← env-driven STATE_SET resolution
│   ├── data_pipeline.py              ← ensure_data cache + refresh_all
│   ├── embeddings.py                 ← e5-small-v2 in-process
│   ├── fetch_{bea,ces,laus,population,fred}_data.py
│   ├── llm_utils.py                  ← BlurbAgent-backed generate_insight
│   ├── merge_all_data.py             ← long + wide panel construction
│   └── ontology.py                   ← states × supersectors × measures
├── tests/
│   ├── test_smoke.py
│   └── utils/                        ← 15 test files, 230+ cases
├── docker-compose.yml                ← ollama + dashboard services
├── Dockerfile.dashboard
├── deploy.sh                         ← --full / --dev modes
└── requirements.txt / requirements-test.txt / pyproject.toml
```

---

## 🛠️ Features & Usage

### Tabs & Navigation

- **EDA / Overview**  
  Historical charts and summary statistics  
- **LFP Forecast**  
  Labor Force Participation Rate forecasts for IA vs. Midwest  
- **Supersector Forecast**  
  Sector‐level employment forecasts across states  
- **About**  
  Context, definitions, data & model details, and team credits  

### Definitions

- **CES** = Current Employment Statistics  
- **LAUS** = Local Area Unemployment Statistics  
- **LFPR** = Labor Force Participation Rate  
- **LSTM** = Long Short-Term Memory RNN  

---

## ⚙️ Local Deployment

### 1. Prerequisites

- Docker & Docker Compose (v2+)
- (Optional) Apache Spark locally or Docker image, but default is `local[*]`.

### 2. Configure secrets

API credentials live in a git-ignored `.env` at the repo root. Create it from
the template and fill it in:

```bash
cp .env.example .env
# edit .env and set CENSUS_API_KEY (required) and BLS_API_KEY (optional)
```

- **BLS** — optional; unregistered calls are rate-limited. Free key:
  https://data.bls.gov/registrationEngine/
- **Census** — required for ACS/PEP population data. Free key:
  https://api.census.gov/data/key_signup.html

docker-compose will refuse to start with a clear error if `CENSUS_API_KEY`
is missing.

### 3. Automated (`deploy.sh`)

Make the script executable and run:

```bash
chmod +x deploy.sh
./deploy.sh          # build dashboard only
./deploy.sh --full   # rebuild both services from scratch
./deploy.sh --dev    # hot-reload dashboard (re-uses existing Ollama container)
```

- **Dashboard** → http://localhost:8050  
- **Ollama API** → http://localhost:11434  

---

### 4. Manual via Docker Compose

**Production** (cache volumes + one‐time build):

```bash
docker-compose up -d --build
```

**Development** (hot-reload your code):

1. Edit `docker-compose.yml` to ensure your code folder is bind-mounted under `dashboard`:

   ```yaml
   services:
     dashboard:
       volumes:
         - ./:/app:cached
       command: >
         gunicorn app:server --reload --bind 0.0.0.0:8050 --workers 1            --threads 8 --worker-class gthread --timeout 300
   ```

2. Bring up Ollama first (if not already):

   ```bash
   docker-compose up -d ollama
   ```

3. Then:

   ```bash
   docker-compose up -d dashboard
   ```

Any changes in `./` will trigger Gunicorn’s `--reload` and Spark/embeddings will only rebuild when the container restarts.

---

## 🔧 Troubleshooting

- View logs:

  ```bash
  docker-compose logs -f dashboard
  docker-compose logs -f ollama
  ```

- Clear embeddings cache:

  ```bash
  docker exec -it ds4010-dashboard-1 rm /app/embeddings_cache.npz
  ```

- Ensure your BLS data JSONs are in `data/ces_state_sms_codes.json` and `data/laus_state_codes.json`.

---

## 📄 About This Dashboard

This dashboard provides clear, actionable insights into U.S. labor-market
trends across all 51 jurisdictions (50 states + DC), leveraging BLS data
(CES, LAUS, JOLTS, QCEW, CPI), Census ACS/PEP, BEA, and FRED, alongside
multi-model forecasts and AI-generated narratives. Iowa and Midwest peer
comparisons are featured as recurring examples in the LFP and Super tabs.

**Purpose & Audience**  
Enable exploration of historical employment/unemployment metrics and forecast future trends at state and sector levels. Targeted at new graduates, policymakers, and labor analysts.

**Data & Models**  
- **Data**: CES = industry counts; LAUS = unemployment & LFPR; JOLTS,
  QCEW, CPI, Census ACS/PEP, BEA personal income, FRED state macro
- **Forecast**: per-series bakeoff across Naive, Seasonal-Naive,
  Holt-Winters ETS, and ARIMA (LSTM available behind an opt-in flag)
- **AI Insights**: Local LLMs via Ollama — `llama3.2:3b` for chat, `phi3` for the DeepAgents harness  

**Project & Team**  
Prairie Insights — regional labor market analyses  
- **Rayne Wilde** (GitHub: @kbouwman)  

_Last updated: 2025-05-01_

---

## ⚖️ License & Acknowledgments

Code is released under the **MIT License** (see `LICENSE`). Third-party
dependency licenses are aggregated in `NOTICE` — required by the
Apache-2.0 deps (TensorFlow, Plotly, LangChain family, DeepAgents)
before any binary redistribution.

### Built with Llama

The chat and blurb layer runs Meta's **Llama 3.2** (3B parameter chat
variant) locally via Ollama. Use of the model is governed by the
[Llama 3.2 Community License](https://www.llama.com/llama3_2/license/).
The agent-refresh helper additionally runs Microsoft's **Phi-3** (MIT).
Neither model's weights are redistributed in this repo — Ollama pulls
them at runtime.

### Data attribution

Source: **U.S. Bureau of Labor Statistics** (CES, LAUS, JOLTS, QCEW,
CPI); **U.S. Census Bureau** (ACS, PEP); **U.S. Bureau of Economic
Analysis** (SAINC1 personal income); **Federal Reserve Bank of
St. Louis** (FRED); **Federal Housing Finance Agency** (HPI). All
underlying data are in the public domain (17 USC §105) and are
reproduced here with the standard agency citation framing.

### Software stack

- **Embeddings** — `intfloat/e5-small-v2` (sentence-transformers,
  Apache-2.0). In-process PyTorch; the prior Spark cluster has been
  retired.
- **Forecasting** — `statsmodels` (BSD-3) for ETS / ARIMA / ADF /
  KPSS / Ljung-Box / Jarque-Bera / Diebold-Mariano; TensorFlow
  (Apache-2.0) for the opt-in LSTM only.
- **UI** — Dash + Plotly + dash-bootstrap-components.
- **Agent harness** — LangChain + LangChain-Ollama + DeepAgents (MIT).
- **Numerics** — NumPy, pandas, SciPy.

The methodology panel on each forecast tab renders the corresponding
peer-reviewed citation for every model, statistical test, and software
library; the canonical registry is in `utils/citations.py`.

### AI assistance

Portions of this codebase were developed with the assistance of
Anthropic's **Claude** (via Claude Code) for code authoring,
refactoring, methodology critique, and citation curation. All design
decisions, scientific claims, and dataset/methodology choices are
human-authored and human-verified. This disclosure follows emerging
academic and industry norms for AI-assisted authorship.