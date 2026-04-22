# Prairie Insights: Midwest Labor Dashboard

An interactive dashboard analyzing labor and employment trends in Iowa and the broader Midwest, leveraging BLS data (CES & LAUS), machine learning forecasts, and AI‐generated narratives.

---

## 🚀 Project Overview

- **Data sources (multi-source fetch agents, all free with a key)**
  - **BLS CES** — state-level employment by supersector (SM series)
  - **BLS LAUS** — labor force, employment, unemployment, unemployment rate
  - **US Census** — ACS + PEP population estimates
  - **BEA** — state annual personal income (SAINC1)
  - **FRED** — state macro indicators (UR, PI, NGSP, …) via `{ST}{IND}` naming
- **Forecasting — multi-model bakeoff, not a single architecture**
  - Candidate set: Naive, Seasonal-Naive, Holt-Winters ETS (LSTM opt-in)
  - Expanding-window backtest picks the lowest-RMSE model per series
  - Per-forecast diagnostics: ADF / KPSS / Ljung-Box / Jarque-Bera /
    Diebold-Mariano (with HLN small-sample correction) vs. Naive baseline
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

This dashboard provides clear, actionable insights into labor trends in the Midwest, leveraging BLS data (CES & LAUS), machine learning forecasts, and AI-generated narratives.

**Purpose & Audience**  
Enable exploration of historical employment/unemployment metrics and forecast future trends at state and sector levels. Targeted at new graduates, policymakers, and labor analysts.

**Data & Models**  
- **Data**: CES = industry counts; LAUS = unemployment & LFPR  
- **Forecast**: LSTM RNN with 12-month windows  
- **AI Insights**: Local LLM (`llama2:chat`) based narratives  

**Project & Team**  
Prairie Insights — regional labor market analyses  
- **Rayne Wilde** (GitHub: @kbouwman)  

_Last updated: 2025-05-01_

---

## ⚖️ License & Acknowledgments

- Data from **BLS API**  
- Embeddings via **Sentence-Transformers**  
- Forecasting built on **TensorFlow** & **Spark**  
- UI built with **Dash** & **Bootstrap**