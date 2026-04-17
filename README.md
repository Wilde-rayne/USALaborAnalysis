# Prairie Insights: Midwest Labor Dashboard

An interactive dashboard analyzing labor and employment trends in Iowa and the broader Midwest, leveraging BLS data (CES & LAUS), machine learning forecasts, and AI‐generated narratives.

---

## 🚀 Project Overview

- **Data sources**  
  - **CES** (Current Employment Statistics): industry‐level employment counts  
  - **LAUS** (Local Area Unemployment Statistics): unemployment rates & labor force metrics  
- **Forecasting**  
  - LSTM RNN models (12-month windows, 75/25 split, 10 epochs, batch size 16)  
- **Embeddings & AI**  
  - Contextual snippets embedded via **Sentence-Transformers** (`intfloat/e5-small-v2`)  
  - **Apache Spark** parallelizes embedding of ~3k snippets at container startup  
  - Local LLM (Ollama “llama2:chat”) generates narrative insights  

---

## 📁 Directory Structure

```
ds4010/
├── assets/  
│   └── bootstrap.min.css  
├── img/ ← contains reference images for .md files
├── data/  
│   ├── raw/  
│   │   ├── ces/  
│   │   ├── laus/  
│   │   ├── METADATA.md  
│   │   └── README.md  
│   ├── all_data.json  
│   ├── METADATA.md  
│   └── README.md  
├── tabs/  
│   ├── about_tab.py  
│   ├── eda_tab.py  
│   ├── lfp_tab.py  
│   └── super_tab.py  
├── utils/  
│   ├── constants.py  
│   ├── data_pipeline.py  
│   ├── embeddings.py  
│   ├── fetch_ces_data.py  
│   ├── fetch_laus_data.py
│   ├── fetch_population_data.py  
│   ├── graphics.py    
│   ├── merge_all_data.py  
│   ├── llm_utils.py  
│   └── model_utils.py  
├── .gitignore  
├── app.py  
├── deploy.sh                ← supports `--full` and `--dev` (hot reload) 
├── build.sh                 ← Makes .md into a pdf 
├── docker-compose.yml       ← prod & dev volumes + hot-reload bind mounts  
├── Dockerfile.dashboard  
├── Dockerfile.ollama  
├── MILESTONE.md  
├── README.md 
├── REPORT.md  
└── requirements.txt  
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