# Expansion roadmap: data, ontology, models, time-awareness

A forward-looking review of `USALaborAnalysis` at commit `5133949`
(`reviewable/loving-shtern-800851`). The platform already runs an
operational multi-source pipeline, a multi-model forecast bakeoff with
prediction intervals, an LLM orchestration layer with three reviewer
tiers, and a 20-paper citation registry. This document recommends the
next layer: net-new data sources, a temporally aware knowledge graph,
the model and evaluation upgrades that take the system from "Holt-
Winters with diagnostics" to "research-engineer credible," and the
retrofits that make every published forecast auditable a year later.

The user has restricted implementation to a Phase 1 cleanup pass, so
nothing below ships in this branch — it is a roadmap, not a plan of
record.

---

## 1. State-of-the-platform inventory

The dashboard is **further along than a typical exploratory project**.
A reader walking into the code today finds:

- **Eight live data fetchers** under `utils/fetch_*.py`: BLS CES, BLS
  LAUS, BLS JOLTS (national preset + experimental state-level slot),
  BLS QCEW state totals (employment level, total quarterly wages,
  average weekly wage, establishment count), BLS regional CPI-U (4
  Census regions + US city average), BEA SAINC1 personal income, FRED
  state-indicator family (UR / PI / NGSP / STHPI / MHI), Census ACS
  working-age population (B23025_001E), and basic Census PEP
  population. All are wrapped as `@tool` LangChain capabilities in
  `utils/agents/tools.py` with state-code normalization, year-range
  validation (1948-floor, ≤30 year span), and OWASP A04 per-call
  ceilings.
- **An ontology singleton** (`utils/ontology.py`) covering 50 states +
  DC + 5 territories, 13 BLS supersectors with NAICS rough mappings, 7
  measures with LAUS/CES suffix decoding, and 9 data-source records.
  `parse_series_id` round-trips LASST and SMS series ids; `describe`
  renders them as prose for the embedding layer.
- **A forecasting framework** (`utils/forecasting/`) with five
  forecasters (Naive, Seasonal-Naive, ETS, ARIMA grid-search by AIC,
  optional LSTM), expanding-window backtest, RMSE-by-default scoring,
  per-fold-fresh instantiation, residual-bootstrap and statsmodels-
  analytical prediction intervals, four invariant checks
  (`point_within_historical_envelope`, `ci_contains_point`,
  `ci_width_reasonable`, `rmse_against_baseline`), and five statistical
  diagnostics (ADF, KPSS, Ljung-Box, Jarque-Bera, Diebold-Mariano with
  HLN small-sample correction).
- **A nested-harness LLM pipeline** (`utils/agents/`): a base
  `LaborAgent` over `ChatOllama`, a `BlurbAgent` with a
  `default_blurb_agent` factory, a `BlurbOrchestrator` running
  parallel section specialists with three reviewer tiers (pure-Python
  `strip_preamble` + `quick_check`, optional LLM `evaluate`, optional
  cross-section `HolisticReviewer`), a `SentenceRAGBuilder` that
  renders ontology-aware sentences before embedding, a `DeepAgents`-
  backed `data_refresh_agent` for orchestrated background fetches, and
  a per-process result cache keyed on `(section, JSON view_state,
  model)`.
- **A citations registry** (`utils/citations.py`) with 20 entries
  spanning forecasting (Holt 1957, Winters 1960, Box-Jenkins 1970,
  Hyndman & Athanasopoulos 2018), residual tests (Dickey-Fuller,
  KPSS, Ljung-Box, Jarque-Bera, Diebold-Mariano, Harvey-Leybourne-
  Newbold), backtesting (Tashman 2000), and data-source handbooks
  (CES, LAUS, JOLTS, QCEW, CPI, ACS, BEA, FRED, FHFA HPI). Both
  `render_inline` ("Box & Jenkins, 1970") and `render_full` forms ship.
- **A 51-jurisdiction panel** (Phase D): `STATE_SET=all_jurisdictions`
  flips on the full 50+DC+territories slate; the ontology-driven
  `bootstrap_state_codes.py` keeps `ces_state_sms_codes.json` and
  `laus_state_codes.json` aligned without hand-editing.
- **A four-tab Dash UI** (EDA, LFP Forecast, Supersector Forecast,
  About) with prediction intervals, peer-state multi-state forecasts,
  region filters, error bars, sort modes, recommendation thresholds, a
  floating chat drawer, a methodology + references panel, an
  interleaved figure→AI→figure→AI→recap layout on LFP and EDA, view-
  state grounding through sentence-RAG, and progressive rendering so
  the chart lands before the LLM blurb.
- **~20 test files** under `tests/utils/` and `tests/tabs/` exercising
  the agent harness, fetchers, ontology, forecasting core, statsmodels
  bindings, diagnostics, the LFPR-ACS denominator path, and tab-module
  smoke.
- **Two methodology docs** under `docs/methodology/`: `agent_pipeline.md`
  (the three-tier reviewer architecture, prompt-engineering decisions,
  parallelism rationale, cache key, and verification cadence) and
  `lfpr_denominator.md` (the ACS B23025_001E switch that brought
  residual error from ~2 pp to <0.5 pp).

That is a meaningful inventory. The remainder of this document
assumes it as a baseline and proposes additions that complement, not
duplicate, the existing capability.

---

## 2. Data source expansion table (NEW sources)

The eight-source baseline already covers wages, prices, headcount,
labor force, and tightness. The next layer should add **occupational
detail, firm dynamics, real-time labor demand, vintage-aware backtests,
and the crosswalks required to fuse those into the existing state-by-
supersector frame**. Every row below is checked against the existing
`utils/fetch_*.py` and `utils/ontology.SOURCES` so nothing here
re-recommends something already shipped.

| # | Source | Agency | URL | Granularity | Cadence | License | Unlocks | Effort |
|---|---|---|---|---|---|---|---|---|
| 1 | **OEWS** (Occupational Employment & Wage Statistics) | BLS | https://www.bls.gov/oes/ | State, MSA, NAICS×SOC | Annual (May) | Public domain | Wage distributions by occupation × state; occupation-mix shifts | M |
| 2 | **O\*NET** Database | DOL/ETA | https://www.onetcenter.org/database.html | SOC occupation | Quarterly | CC-BY 4.0 | Skill / task / knowledge / ability vectors; THE source for skills-based reasoning | M |
| 3 | **BDS** (Business Dynamics Statistics) | Census | https://www.census.gov/programs-surveys/bds.html | State, MSA, sector | Annual | Public domain | Firm entry / exit / job creation / job destruction; resilience signals | S |
| 4 | **LEHD/LODES** (Longitudinal Employer-Household Dynamics) | Census | https://lehd.ces.census.gov/data/ | Census block, OD pairs | Annual | Public domain | Commute / origin-destination jobs; sub-state labor sheds | L |
| 5 | **ALFRED** (Archival FRED) | St. Louis Fed | https://alfred.stlouisfed.org/ | National + state | Real-time vintages | Public domain | Vintage-aware backtest data; "what did we know on date X" | M |
| 6 | **Employment Projections** | BLS | https://www.bls.gov/emp/ | National + state | Biennial | Public domain | 10-year occupation / industry projections; long-horizon anchor | S |
| 7 | **Productivity & Costs** | BLS | https://www.bls.gov/lpc/ | National + sector | Quarterly | Public domain | Output per hour, unit labor costs; productivity narrative | S |
| 8 | **ECI** (Employment Cost Index) | BLS | https://www.bls.gov/ncs/ect/ | Region × industry × occupation | Quarterly | Public domain | Compensation-growth signal independent of mix; deflator companion | S |
| 9 | **WARN Notices** | State labor depts (aggregator: LayoffTracker.com, ARC Center) | per-state, e.g. https://www.dol.gov/agencies/eta/layoffs | State, county | Event-driven | Public records | Early layoff warnings ahead of CES revisions | L |
| 10 | **IRS SOI Migration** | IRS | https://www.irs.gov/statistics/soi-tax-stats-migration-data | County → county | Annual | Public domain | Working-age migration flows; labor-supply leading indicator | M |
| 11 | **Federal Reserve Beige Book** | Federal Reserve | https://www.federalreserve.gov/monetarypolicy/beigebook.htm | 12 Districts | 8×/year | Public domain | Narrative-text RAG source for qualitative regional context | M |
| 12 | **BEA GDP by Metro & Personal Income components** | BEA | https://apps.bea.gov/regional/ | State, MSA, county | Annual (some Q) | Public domain | Beyond SAINC1: GDP-by-industry (SAGDP), per-capita components | S |
| 13 | **BEA Regional Price Parities** (RPP) | BEA | https://www.bea.gov/data/prices-inflation/regional-price-parities-state-and-metro-area | State, MSA | Annual | Public domain | Cost-of-living-adjusted real wages; missing piece in QCEW story | S |
| 14 | **IPEDS Completions** | NCES | https://nces.ed.gov/ipeds/ | Institution × CIP | Annual | Public domain | Education-program supply pipeline; the CIP side of CIP→SOC | M |
| 15 | **NAICS↔SOC, SOC↔O\*NET, CIP↔SOC crosswalks** | BLS / Census / NCES | https://www.bls.gov/emp/documentation/crosswalks.htm | Code map | Versioned | Public domain | Fuses industries / occupations / programs into one graph | S |
| 16 | **Treasury Yield Curve + Recession indicators** | Federal Reserve / FRED | https://fred.stlouisfed.org/ (DGS10, DGS2, USREC, T10Y2Y, NFCI) | National | Daily / monthly | Public domain | Macro-cycle covariates for a structural VAR | S |
| 17 | **CPS Public-Use Microdata** | BLS/Census | https://www.census.gov/data/datasets/time-series/demo/cps/cps-basic.html | Person, monthly | Monthly | Public domain | Flow rates, demographic decomposition of LFP | L |
| 18 | **ACS PUMS** (Public-Use Microdata Sample) | Census | https://www.census.gov/programs-surveys/acs/microdata.html | Person, PUMA | Annual | Public domain | Worker characteristic distributions; supports occupation × demographic crosstabs | L |
| 19 | **DOL/CareerOneStop + WIOA participant data** | DOL/ETA | https://www.dol.gov/agencies/eta/performance/wioa/data | State, program | Quarterly | Public domain | Workforce-development pipeline; complements IPEDS | M |
| 20 | **MSA / CBSA delineation files** | OMB / Census | https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html | MSA definition | Periodic (~5 yr) | Public domain | Vintage MSA boundaries; required for honest comparisons over time | S |

Effort key: **S** ≈ ≤1 dev-day, **M** ≈ 2-5 dev-days, **L** ≈ 1-2 weeks
including modeling integration.

### Commentary on the top eight picks

**#2 O\*NET (the foundational add)** — This is the single highest-
leverage addition. The existing platform answers "how many jobs are
there in supersector X in state Y?". With O\*NET it can answer "what
skills, tasks, and knowledge bundles are gaining or losing share in
state Y as its industry mix shifts?". O\*NET ships Skill, Knowledge,
Ability, Work-Activity, and Tools-and-Technology vectors per SOC
occupation; combined with OEWS (#1) it produces a state × skill
matrix derivable as a sumproduct of OEWS occupation employment and
O\*NET skill importance scores. That matrix is the substrate for every
"skills-based" narrative the LLM layer should be able to produce. The
CC-BY 4.0 license requires attribution but is otherwise unrestricted.

**#1 OEWS** — OEWS converts the supersector frame into an occupational
one. The 22 major SOC groups roll up cleanly, and the publication ships
state × MSA × occupation × wage-percentile (mean / 10 / 25 / median /
75 / 90), letting the dashboard surface wage *distributions* rather
than only averages — which is the difference between "wages rose" and
"the median wage rose but the 10th-percentile wage stagnated."

**#5 ALFRED** — The least visible but most academically important add.
Every forecast the platform currently publishes is fit on the *current*
vintage of CES / LAUS / FRED data. Six months later the same data has
been revised, often materially (CES annual benchmark revisions
routinely move payroll counts by 200-500 k jobs nationally). A
backtest run on revised data can make a 2022 forecast look better
than it actually was at the time. ALFRED stores every prior vintage —
so a 2024-M03 forecast can be evaluated against the data as it stood
on 2024-M03, not as it stands today. This is the prerequisite for
*honest* retrospective accuracy claims.

**#3 BDS** — BDS publishes job creation, job destruction, firm
entry / exit, and establishment-age distributions. It is the lens the
existing JOLTS pull lacks: JOLTS shows aggregate openings / hires /
quits, BDS shows the *firm-level dynamics* behind them. Pairs naturally
with QCEW (level + wages) to produce a four-quadrant view of any
state-by-sector cell: growing-firm hiring, growing-firm quits, dying-
firm separations, new-entrant hiring.

**#4 LEHD/LODES** — LODES (Origin-Destination Employment Statistics)
publishes Census-block-level worker-flow tabulations: where do the
people who work in this block live? This unlocks **labor-shed**
analysis — the cluster of residential locations from which an MSA
draws its workers. For a portfolio piece aimed at research-engineer
audiences, this is the highest-impact geographic add. Effort is "L"
because the block-level files are large (gigabytes per year per
state) and require aggregation tooling.

**#11 Beige Book** — The Beige Book is a regional, narrative,
qualitative complement to the existing quantitative pipeline. Each of
the 12 Federal Reserve Districts publishes ~5-8 page summaries 8×/year
of "what we're hearing from contacts" — anecdotes about hiring
difficulty, wage pressures, sector-specific conditions. Run through
the existing embedding pipeline (`utils/embeddings.py`, e5-small-v2),
the Beige Book becomes a retrieval source the chat drawer can cite:
"the Chicago Fed in 2024-M03 reported softening manufacturing demand
in Iowa and Indiana." This is the kind of detail the LLM layer cannot
fabricate from numeric series alone.

**#6 Employment Projections** — BLS publishes 10-year occupation and
industry employment projections (the 2022-32 vintage is current). They
provide a long-horizon anchor for the existing 12-24-month forecasts:
a model that predicts manufacturing growth over 10 years when BLS
projects decline should be flagged as inconsistent with the official
view, and the dashboard should *say so*.

**#13 BEA Regional Price Parities** — RPP converts nominal wages to
real wages by state and MSA. Without it, the dashboard's QCEW
"average weekly wage" comparisons across states are an apples-to-
oranges mistake: $1,200/week in San Francisco is materially less
than $1,200/week in Des Moines. RPP fixes that with a single per-
state index. Tiny code lift; large interpretive payoff.

Picks **#7 (Productivity & Costs)**, **#8 (ECI)**, **#10 (IRS SOI
migration)**, and **#16 (Treasury / recession indicators)** are
"obvious complements" — small adds that round out the structural-VAR
covariate set described in §4. **#9 (WARN notices)**, **#15
(crosswalks)**, and **#20 (MSA delineation files)** are infrastructural:
they don't show up directly in a tab but make every other source
honest about geography and timing. **#14 (IPEDS)** and **#19 (WIOA)**
close the supply-side loop — how many people are *trained for* each
occupation each year — which O\*NET and OEWS leave dangling.

---

## 3. Knowledge graph / ontology (temporally aware)

The existing `Ontology` dataclass is excellent for what it does:
state-supersector-measure decoding of BLS series ids and prose
description for the embedding layer. To support the data expansion in
§2 — particularly the occupational, geographic, and skills additions —
it needs to grow into a **temporally aware knowledge graph** with
explicit entity types, typed relations, and bi-temporal facts.

### Entity types

| Entity | Examples / id space | Notes |
|---|---|---|
| **State** | FIPS 2-digit + USPS 2-letter | already in ontology |
| **Region** | "Midwest", "Plains", BEA region | existing label needs broadening: BLS, Census, BEA, and Federal Reserve all use different region groupings |
| **MSA / CBSA** | OMB CBSA 5-digit code | vintaged — CBSA definitions change roughly every 5 years |
| **County** | FIPS 5-digit | natural sub-state aggregation |
| **Industry** | NAICS 6→4→3→2 hierarchy | with vintage (NAICS 2017, 2022) |
| **Supersector** | BLS 2-digit | already in ontology |
| **Occupation** | SOC 6-digit + 2017 / 2018 vintages | rolls up SOC 6 → 5 → 3 → 2 |
| **Skill / Knowledge / Ability** | O\*NET element id | O\*NET Content Model hierarchy |
| **Task** | O\*NET task id | mapped to SOC |
| **Education program** | CIP 6-digit | rolls up to CIP 4 → 2 |
| **Series** | source-specific id (LASST…, SM…, JTS…, OEW…) | with metadata: cadence, seasonal, unit, vintage |
| **Demographic group** | sex × age × race × ethnicity tuple | from ACS, CPS |
| **Vintage / Release** | (source, release_date) | which version of the data this row is from |
| **Geography crosswalk** | (geo_a, geo_b, vintage_a, vintage_b, weight) | many-to-many; weighted by population share |

### Relations (typed edges)

| Relation | From | To | Cardinality | Carries |
|---|---|---|---|---|
| `contained_in` | County | State | N:1 | — |
| `contained_in` | MSA | State (list) | N:M | crosses state lines |
| `rolls_up_to` | NAICS 6 → 4 → 3 → 2 | NAICS | N:1 each | — |
| `rolls_up_to` | SOC 6 → 5 → 3 → 2 | SOC | N:1 each | — |
| `supersector_of` | NAICS 2-digit | Supersector | N:1 | BLS-defined |
| `measured_by` | Series | Measure | N:1 | unit |
| `measured_at` | Series | Entity | N:1 | the geography / industry the series describes |
| `requires_skill` | Occupation | Skill | N:M | O\*NET importance score + level |
| `performs_task` | Occupation | Task | N:M | — |
| `trains_for` | CIP program | Occupation | N:M | NCES → SOC crosswalk |
| `superseded_by` | Series | Series | 1:0..1 | when BLS retires / reissues a series |
| `peer_of` | State | State | N:M | computed similarity; not an authoritative crosswalk |
| `has_observation` | Series | (time, value) | 1:N | a fact (see bi-temporal model) |
| `has_forecast` | Series, Model, ForecastDate | (target_time, point, lo, hi) | 1:N | a forecast journal entry (see §5) |

### Bi-temporal model

Every fact carries **two time dimensions**:

- **Valid time** `(valid_from, valid_to)` — when the world was in this
  state. For a CES employment observation, valid time is the reference
  month: 2024-M03 has `valid_from = 2024-03-01, valid_to = 2024-04-01`.
- **Transaction time** `(recorded_at, superseded_at)` — when this row
  was added to our store, and when (if ever) it was replaced. A 2024-M03
  CES preliminary release recorded on 2024-04-19 sits in the store
  alongside the 2024-05-17 revised release for the same reference month.
  Same `valid_time`, different `recorded_at`, with the older row's
  `superseded_at` set to the newer row's `recorded_at`.

This pattern is standard in temporal-database theory (the SQL:2011
"system-versioned tables" feature implements it). Its load-bearing
contribution to **this** project is that it makes the following
queries trivially answerable:

1. *"As of 2024-M03 the day after release, what was the 2-year-ahead
   forecast for Iowa LFPR?"* → query forecasts with `recorded_at ≤
   2024-03-09` for target `2026-03-01`.
2. *"Re-run our 2024-M03 forecast bakeoff using only data that was
   available on 2024-M03."* → query observations with `recorded_at ≤
   2024-03-09`. Without bi-temporality, the bakeoff silently consumes
   post-revision data and overstates accuracy. This is the heart of
   honest forecast evaluation (see Croushore & Stark, *Real-Time Data
   Set for Macroeconomists*, J. of Econometrics 2001).
3. *"Show me only the most recent revision for each (series,
   reference period)."* → `valid_time` join with `recorded_at`
   max-aggregation; the default presentation query.

The ALFRED data source (#5 in §2) is the upstream feed that makes
this possible — it publishes every vintage of every FRED series so the
store can be populated retroactively.

### Crosswalks (the boring infrastructure that everything depends on)

| Crosswalk | Source | Maintained by | Vintage cadence |
|---|---|---|---|
| **NAICS ↔ SOC** (which occupations are in which industry?) | BLS National Industry-Occupation Matrix | BLS | Biennial |
| **SOC ↔ O\*NET-SOC** (BLS SOC ↔ O\*NET's slightly-finer "O\*NET-SOC" extension) | O\*NET | DOL | With each O\*NET release |
| **NAICS supersector ↔ NAICS sector** | BLS CES technical note | BLS | At every NAICS revision |
| **FIPS ↔ BLS area code** (LAUS uses 13-character area codes; QCEW uses FIPS) | BLS | BLS | Stable |
| **FIPS ↔ Census GEOID** (PEP, ACS, BDS all use slightly different GEOID conventions) | Census | Census | Stable |
| **CBSA definitions across vintages** (OMB 2013 vs 2018 vs 2023) | OMB delineation files | OMB | Every ~5 years |
| **CIP ↔ SOC** (which occupations does this education program train for?) | NCES + BLS | BLS | With each CIP revision |
| **OEWS NAICS group ↔ QCEW NAICS** (OEWS aggregates more aggressively for confidentiality) | BLS OEWS technical note | BLS | Annual |
| **NAICS 2017 ↔ NAICS 2022** (BLS rolled the panel forward to NAICS 2022 in 2023; older QCEW is NAICS 2017) | BLS NAICS revision documentation | BLS | At each revision |

These tables are typically published as XLSX / CSV under a few hundred
KB each. The recommendation is to vendor them into `data/crosswalks/`
with a vintage suffix (`naics_soc_2018.csv`, `naics_soc_2020.csv`),
load them through an `OntologyCrosswalk` helper, and key every
multi-source query through the crosswalk rather than letting
join-on-string-equality go feral. Several pipelines in the wild have
broken silently because they joined NAICS-2017 industry codes from
QCEW against NAICS-2022 codes from CES.

### Implementation sketch: DuckDB over Parquet

The current pipeline stores everything as JSON / CSV on disk and lifts
into pandas at read-time. That works for the 17 k-row panel reported
in the Phase D handoff. With the additions above, the row count grows
by roughly:

| Add | Rows | Order of magnitude |
|---|---|---|
| OEWS state × occupation × year × percentile | 51 × 800 × 25 × 6 | 6 M |
| O\*NET occupation × element | 800 × 250 | 0.2 M |
| BDS state × sector × year | 51 × 18 × 45 | 0.04 M |
| LODES block-level OD | varies — start at MSA-aggregated tabulations | 1-100 M |
| QCEW already in panel | extends to NAICS-4 / NAICS-6 | 50-500 M |
| ALFRED vintage-multiplier | each existing series ×10-50 vintages | 5-25 M |

This is the size at which JSON-in-pandas starts to hurt. The
recommendation is to **adopt DuckDB over Parquet for the analytical
store**:

- Parquet for at-rest storage: columnar, compressed, partitioned by
  source / state / year. Every fetcher writes to `data/parquet/`
  alongside the existing `data/raw/` TXT files; the TXTs become a
  reproducibility audit trail, not the working store.
- DuckDB as the query engine: embedded, no server, reads Parquet
  directly, SQL is the lingua franca every analyst (and most LLMs)
  speaks. DuckDB's `read_parquet('data/parquet/**/*.parquet')` over the
  full corpus runs on a developer laptop.
- A *schema* — not a normalized relational schema, a star schema in
  the dimensional-modeling tradition:

```text
dim_entity         (entity_id PK, entity_type, code, name, attrs JSON)
dim_geography      (geo_id PK, geo_type, fips, msa_code, vintage, parent_geo_id)
dim_time           (date PK, year, quarter, month, fiscal_year, …)
dim_series         (series_id PK, source, cadence, unit, seasonal_adj,
                    measure_id, geo_id, industry_id, occupation_id, demographic_id,
                    superseded_by_series_id, valid_from, valid_to)
dim_vintage        (vintage_id PK, source, release_date, comment)
dim_model          (model_id PK, family, params JSON, code_commit_sha)

fact_observation   (series_id, date, value,
                    vintage_id, recorded_at, superseded_at)
fact_forecast      (forecast_id PK, series_id, target_date,
                    model_id, made_at,
                    point, lo_95, hi_95, lo_80, hi_80,
                    data_vintage_id, model_version)
fact_diagnostic    (forecast_id, test, statistic, pvalue)
```

The "don't build it" caveat from the prompt: this section is a
**proposed shape**, not a migration plan. The right time to act is
when adding O\*NET / OEWS / ALFRED forces the issue. Until then, the
current JSON/CSV-in-pandas store is adequate, and the only concrete
near-term work is **vendoring the crosswalk tables** so the existing
sources stop silently mis-joining.

---

## 4. Models & variables

ARIMA already shipped in the bakeoff (`utils/forecasting/models.py:218`,
commit `407c206`). The selection harness, expanding-window backtest,
and Diebold-Mariano vs Naive baseline are operational. Recommended
next-layer additions, in priority order:

### Probabilistic forecasting

- **Conformal prediction** — A non-parametric, model-agnostic wrapper
  that produces calibrated prediction intervals from any point
  forecaster's hold-out residuals. The current ETS / ARIMA intervals
  rely on the parametric Gaussian assumption baked into statsmodels.
  Conformal intervals are *guaranteed* to achieve at least the nominal
  coverage in finite samples, under exchangeability assumptions that
  hold for the existing expanding-window backtest. Implementation:
  ~150 lines on top of the existing `BaseForecaster.predict_interval`,
  re-using the fold residuals already computed in
  `selection.py:_fold_metrics`. Cite Shafer & Vovk 2008 and Stankeviciute,
  Alaa, & van der Schaar 2021 (conformal time series).

- **Zero-shot foundation models** — Chronos (Amazon, 2023), TimesFM
  (Google, 2024), and Lag-Llama (ServiceNow, 2024) are pretrained
  transformer / diffusion forecasters released under permissive
  licenses. They produce probabilistic forecasts from a series with
  zero fine-tuning. For a labor-portfolio piece this is genuinely
  novel: most BLS-data dashboards run ARIMA. The risk is that they
  can hallucinate seasonal shapes that don't match BLS data. Mitigate
  by including them as bakeoff candidates and letting the expanding-
  window backtest decide.

### Hierarchical reconciliation

- **MinT (Minimum Trace) reconciliation** — The existing platform
  forecasts `Total_Nonfarm` and each of 12 supersectors *independently*.
  By accounting identity the supersectors must sum to total nonfarm,
  but the independent forecasts almost certainly don't. MinT (Wickramasuriya,
  Athanasopoulos, & Hyndman 2019, JASA) is the canonical reconciliation
  method: it produces a single set of base-level forecasts that respect
  the hierarchy and minimizes trace of the forecast-error variance
  matrix. Direct upgrade path for the Super tab.

- **Bottom-up + top-down hybrid** — Simpler than MinT, ships in the
  R package `hts` and Python's `hierarchicalforecast`. Useful as a
  baseline for the MinT comparison.

### Spatial / cross-sectional

- **Spatial-VAR** — A vector autoregression over the state panel,
  with a spatial-weights matrix derived from inter-state migration
  (IRS SOI, source #10 in §2) or contiguity. Captures the "neighbors
  drag each other" effect — Iowa's labor market doesn't move
  independently of Illinois's. Implementation: PySAL's `spreg` or
  statsmodels' `VAR` with a manually-constructed weights matrix.

- **Graph Neural Network (GNN) on state adjacency** — More flexible
  than spatial-VAR but heavier in dependencies and harder to interpret.
  Worth a research-engineer-portfolio bullet, but only after the
  spatial-VAR baseline exists to beat.

### Causal-leaning

- **Structural VAR with sign restrictions** — A VAR identified by
  the *sign* of impulse responses rather than ordering. Lets the
  analyst ask: "if oil prices rise, what's the response of Iowa
  manufacturing employment six months out?" Cite Uhlig 2005, Rubio-
  Ramirez et al 2010.

- **Synthetic control** — For "what would have happened if event X
  hadn't occurred?" counterfactuals. Useful for the COVID-19 trough
  question every labor dashboard eventually has to answer. Cite
  Abadie, Diamond, & Hainmueller 2010.

### Skills-based reasoning (requires O\*NET)

- **Occupation → skill graph traversal** — Given OEWS state-by-
  occupation employment and O\*NET skill-importance vectors, compute
  state-by-skill employment weights. Forecast each state's
  employment-weighted skill exposure 2 years out from the supersector
  forecast plus the BLS Employment Projections occupation shares
  (source #6). This is the headline narrative: "Iowa's
  manufacturing-heavy mix exposes it to 2× the national average
  Operation-Monitoring requirement, but at 0.6× the national
  Programming requirement — making it sensitive to automation in the
  former and insulated from software-job shocks in the latter."

### Evaluation upgrade

- **Rolling-origin CV with multiple horizons** — The existing expanding-
  window backtest produces one accuracy number per (series, model).
  Rolling-origin cross-validation (Tashman 2000, already cited) at each
  horizon h ∈ {1, 3, 6, 12, 24} months reveals *where* a model
  degrades. ETS routinely beats ARIMA at h=1 and loses at h=24.

- **MASE** (Mean Absolute Scaled Error, Hyndman & Koehler 2006) —
  Scale-free, interpretable as "ratio of our error to the seasonal-
  naive error." A MASE < 1 means we beat seasonal-naive; > 1 means we
  don't. MAPE is misleading on near-zero series; MASE isn't.

- **CRPS** (Continuous Ranked Probability Score) — A proper scoring
  rule for probabilistic forecasts. Without it, the prediction-
  interval upgrade in §2 is unmeasured. Cite Gneiting & Raftery 2007.

- **Pinball loss** (Quantile loss) — Per-quantile scoring; lets us
  say "our 80th-percentile forecasts are well-calibrated, our 95th
  are too wide."

- **PIT (Probability Integral Transform) histograms** — Visualization
  of calibration. A well-calibrated probabilistic forecast produces a
  uniform PIT histogram; over-confidence produces a U-shape, under-
  confidence a hump.

- **Per-horizon decomposition** — Decompose accuracy into trend,
  seasonality, and noise contributions to identify *where* error
  enters. Already partially implemented in `trend.py`; needs lifting
  into the evaluation report.

- **Seasonal-naive baseline as the comparator everywhere** — The
  current `_baseline_residuals` uses `NaiveForecaster` (persistence).
  Seasonal-naive is a stronger baseline for monthly labor data and
  should be the comparator for the Diebold-Mariano test.

### Variable inventory — top ~25 candidates ranked by likely lift

Numbered roughly by expected explanatory value for state-level labor
forecasts at 12-24 month horizons.

| Rank | Variable | Source | Hypothesis |
|---|---|---|---|
| 1 | State unemployment rate (own lag) | LAUS (existing) | already used; baseline |
| 2 | State labor force participation rate (own lag) | LAUS / ACS (existing) | already used |
| 3 | National unemployment rate | LAUS (existing) | states co-move with national cycle |
| 4 | State QCEW total employment (own lag) | QCEW (existing) | already used |
| 5 | State average weekly wage | QCEW (existing) | wage growth lags labor-market tightness |
| 6 | JOLTS quits rate | JOLTS (existing) | leading indicator of tightness |
| 7 | JOLTS openings-per-unemployed (V/U ratio) | JOLTS (existing) | THE tightness indicator |
| 8 | State personal income | BEA SAINC1 (existing) / FRED | demand-side covariate |
| 9 | State FHFA HPI growth | FRED STHPI (existing) | housing → construction employment leading |
| 10 | State median household income | FRED MHI (existing) | distributional check |
| 11 | Regional CPI growth | BLS CPI (existing) | nominal-to-real deflator |
| 12 | OEWS state-by-occupation mix shift | OEWS (NEW) | composition effects |
| 13 | BDS firm entry / exit net | BDS (NEW) | structural change indicator |
| 14 | BDS job creation rate | BDS (NEW) | gross flows |
| 15 | LODES inflow-to-outflow ratio | LODES (NEW) | net commuting → labor sheds |
| 16 | IRS SOI net migration | IRS SOI (NEW) | labor supply leading |
| 17 | 10-year Treasury yield | FRED (NEW) | macro-cycle covariate |
| 18 | 2y-10y Treasury spread | FRED (NEW) | recession-leading |
| 19 | NFCI (Chicago Fed National Financial Conditions Index) | FRED (NEW) | credit conditions |
| 20 | Beige Book sentiment | Beige Book (NEW) | qualitative narrative |
| 21 | BLS productivity growth (sector) | BLS LPC (NEW) | wage-pressure release valve |
| 22 | ECI compensation growth | BLS ECI (NEW) | wage pressure independent of mix |
| 23 | BEA Regional Price Parities | BEA RPP (NEW) | real-wage adjustment |
| 24 | IPEDS completions in relevant CIPs | IPEDS (NEW) | supply pipeline |
| 25 | CPS labor-flow rates (E→U, U→E, N→E) | CPS basic (NEW) | demographic decomposition |

The first 11 already exist in the pipeline; the model layer just isn't
exploiting them yet (the current forecasters are univariate). The
multivariate work — adding exogenous covariates to ETS via ARIMAX, or
moving up to VAR / VECM — is the natural next step after the
hierarchical-reconciliation slice.

---

## 5. Temporal awareness retrofits

The forecasting layer currently has *no concept of data vintage*.
Every retrofit below is small, but together they are the difference
between a dashboard and an auditable forecasting system.

### Vintage-aware caching

The data pipeline caches at `data/all_data.json` with a 7-day TTL
(`utils/data_pipeline.py:33`). The cache key should grow from "is the
file fresh?" to **`(data_vintage_hash, model_version)`**:

- `data_vintage_hash` — a hash of `(source, series_id, latest_period,
  release_date)` tuples for every series the cache includes. When BLS
  releases a CES revision, the hash changes, the cache invalidates,
  and downstream forecasts recompute. When nothing has changed
  upstream, the cache stays hot.
- `model_version` — `code_commit_sha` of the forecasting module. When
  the bakeoff candidate set changes (or a bug fix lands in an
  estimator), every cached forecast invalidates. Right now a stale
  forecast from a buggy ETS implementation can sit in cache forever.

Implementation: a `_cache_key` helper next to `cache_is_fresh`. The
existing 7-day TTL stays as a *floor* for cases when nothing's
changed upstream and the cache should be refreshed anyway (e.g. to
pick up a new model).

### Forecast journal

Every forecast the orchestrator produces should land in
`data/forecast_journal.parquet` (or a SQLite / DuckDB table) with the
schema:

```text
forecast_id           — uuid
forecast_made_at      — wall clock UTC
target_series         — series_id
target_horizon        — months ahead
model_family          — "ets", "arima", "naive", …
model_version         — code commit sha
data_vintage_hash     — what data did we have when we made this?
view_state_hash       — what filters / peers / horizon settings?
point                 — point forecast
lo_95, hi_95          — 95% CI bounds
lo_80, hi_80          — 80% CI bounds
backtest_rmse         — selector's hold-out RMSE for this candidate
backtest_mase         — MASE if available
diagnostics_json      — ADF / KPSS / Ljung-Box / JB / DM p-values
invariant_failures    — list of any failed invariants from §4
```

After 12 months of operation, the journal becomes the **input to a
retrospective accuracy report**. The reviewer agent that today catches
"this forecast violates the historical envelope" gains a new check:
"the same model on this same series produced forecasts whose realized
accuracy at h=12 was MASE 1.4 in the prior year — flag low confidence."

This is the single highest-impact retrofit in the document. It costs
~200 lines and three new tests, and it turns the dashboard from "we
show forecasts" into "we show forecasts *and we know how good ours
have been*."

### Embedding cache invalidation rules

The sentence-RAG builder embeds prose like "Iowa's 2024 LFPR was
64.5%." into a vector cache at `/app/.cache/embeddings_cache.npz`
(see commit `1db84f6`). Today the cache is invalidated on any data
refresh. With vintage-aware tracking it should invalidate at a
**per-sentence granularity**: when a 2024 LFPR revision lands, only
the sentences mentioning 2024 LFPR are recomputed.

Implementation: sentence hash → embedding mapping in the cache; the
sentence-generation pass emits both sentences and their vintage hashes,
and the cache only re-embeds sentences whose hash has changed. Cost is
slight code-complexity growth; benefit is that monthly data refreshes
no longer trigger a full ~20-minute embedding rebuild.

### "Freshness" indicator in the dashboard

Every chart and every blurb should carry a small, unobtrusive
**freshness badge**: "Data: CES 2024-M03 release (2024-04-19); LAUS
2024-M02 release (2024-03-15)." The forecast journal supplies this
trivially.

For the LLM layer, freshness is a **grounding fact**: the orchestrator
prepends "as of <release_date>, the most recent observation for
<series> is <value>" to every panel's facts block. This is the cheap
fix for the "AI confidently states 2023 data when 2024 has already
released" failure mode.

### Vintage-aware backtesting (the longest retrofit)

The full payoff of bi-temporality (§3) is **vintage-aware backtesting**.
Implementation requires the ALFRED data source (§2 #5) for FRED-side
series and historical revision archives for BLS-side series (the BLS
publishes these in their `archive` endpoints; they are not the same
shape as the current pull and a Phase-3+ effort).

The recommendation is to **stage this**: do the cheap retrofits
(journal, freshness, vintage-keyed cache) first. Phase 4+ adds ALFRED
pulls. Phase 5+ rebuilds the backtest harness to consume vintage data.

---

## 6. Prioritized roadmap

| Item | Phase | Effort | Expected payoff |
|---|---|---|---|
| Crosswalk vendoring (NAICS↔SOC, FIPS↔BLS, CBSA, NAICS17↔22) | 2 | S | unblocks every multi-source join; prevents silent miscodes |
| Seasonal-naive as the DM baseline (replace plain naive) | 2 | S | tougher comparator; existing tests need updating |
| Forecast journal (write every forecast to Parquet) | 2 | S | enables retrospective accuracy reporting at 12 mo |
| Vintage-keyed cache (data_vintage_hash + model_version) | 2 | S | invalidates correctly on revisions or code changes |
| Freshness badge + grounding fact in blurb prompts | 2 | S | kills the "AI cites old data" failure mode |
| MASE + CRPS + pinball loss in `ForecastMetrics` | 2 | S | proper scoring rules; one-shot upgrade |
| Conformal prediction wrapper around any forecaster | 3 | M | calibrated CIs without parametric assumptions |
| O\*NET fetcher + skill-importance ingestion | 3 | M | unlocks every skills-based narrative |
| OEWS fetcher (state × occupation × wage percentile) | 3 | M | wage-distribution analysis, not just averages |
| BEA RPP (Regional Price Parities) fetcher | 3 | S | real-wage adjustment; trivial to add |
| BDS fetcher (firm entry / exit / job creation) | 3 | S | structural-change signals beyond JOLTS |
| Seasonal hierarchical reconciliation (MinT / bottom-up) | 3 | M | supersectors sum to total nonfarm |
| Spatial-VAR over states with IRS-SOI weights | 3 | M | inter-state spillover modelling |
| Beige Book ingestion + RAG corpus extension | 3 | M | qualitative regional narrative; novel for portfolio |
| ALFRED fetcher + vintage table population | 4 | M | foundation for vintage-aware backtest |
| DuckDB/Parquet analytical store migration | 4 | M | needed when O\*NET + OEWS + ALFRED push row counts up |
| Knowledge graph: dim/fact schema + sql interface | 4 | L | clean substrate for cross-source queries |
| BLS Employment Projections ingestion + long-horizon anchor | 4 | S | inconsistency check on 2-yr forecasts |
| LEHD/LODES MSA-level OD aggregation | 4 | L | labor-shed analysis — visually striking |
| Chronos / TimesFM / Lag-Llama as bakeoff candidates | 4 | M | zero-shot foundation forecasters; novel narrative |
| Structural VAR with sign restrictions | 5 | L | causal-leaning "what-if" capability |
| Synthetic control for event studies | 5 | M | counterfactual narratives (COVID, etc.) |
| GNN on state adjacency for cross-state forecast | 5 | L | research-engineer portfolio bullet |
| IPEDS + WIOA supply-pipeline integration | 5 | M | closes the labor-supply loop |
| Vintage-aware backtest harness rebuild | 5 | L | honest retrospective accuracy claims |
| CPS / ACS PUMS person-level integration | 6 | L | demographic decomposition of LFP |
| WARN-notice scraper / aggregator | 6 | L | real-time layoff signals |

Phases referenced are the user's existing six-phase plan (clean base
→ foundation → agents/ontology → data expansion → UI polish → public
release). The table places every recommendation in the earliest phase
where its prerequisites are reasonable.

---

## 7. Anti-recommendations

Five things the project should explicitly **not** do, with reasons.

1. **Do not rewrite to React + FastAPI.** The user's memory file
   already records this constraint, but it bears repeating. The Dash
   UI is operational, accessible, and being progressively elevated
   (commits `092a895`, `b52fa7d`, `b9fb9c1`, `8affe8e`). The right
   investment is in the substrate behind it, not the frontend layer.
   A React rewrite is months of engineering for zero data-science
   payoff.

2. **Do not migrate to PySpark for analytical compute.** The
   methodology doc (`docs/methodology/agent_pipeline.md`) already
   records that an earlier PySpark attempt was rolled back because
   pickling overhead dominated the per-task work for ≤10 LLM calls.
   The same logic applies to the analytical store: at 17 k rows
   today and ~5 M rows after OEWS, DuckDB on a developer laptop
   outperforms Spark by an order of magnitude. The first time the
   project genuinely needs Spark is at the LEHD/LODES integration
   (~100 M rows for full block-level OD across all states), and even
   then the right answer is probably "compute MSA-level aggregates
   in DuckDB and store the result," not "stand up a Spark cluster."

3. **Do not add a sixth model class before MinT reconciliation
   lands.** The existing bakeoff has Naive, Seasonal-Naive, ETS,
   ARIMA, and (optional) LSTM. Adding Prophet, NeuralProphet, or
   another statsmodels variant before the *hierarchy* problem is
   solved produces marginally more candidates and zero new insight.
   The supersector forecasts not summing to total nonfarm is a
   first-order correctness issue. The lack of a sixth forecaster is
   not.

4. **Do not let the citation registry grow without a corresponding
   use.** The `citations.py` registry has 20 entries and every one
   is referenced by either the methodology panel, an invariant
   docstring, or an agent prompt. The temptation when reading a new
   paper is to add it speculatively. Resist this: a citation that
   isn't grounded in code or surfaced in a panel is a maintenance
   tax with no reader.

5. **Do not try to scrape WARN notices state-by-state in the next
   phase.** Each state's labor department publishes WARN data in a
   different format, on a different cadence, often as PDFs with
   inconsistent structure. The aggregation effort is real and
   thankless. If WARN data is wanted, **buy or pull from an
   existing aggregator** (Investors-Hub, Layoffs.fyi, ARC Center,
   or DOL's centralized list where available) rather than re-doing
   the scraping at portfolio scale. Phase 6 placement in the
   roadmap above reflects this.

---

## Closing thought

The platform's research-engineer-portfolio thesis is most credible
when the **transparency of the methodology** is as visible as the
forecasts themselves. The five highest-leverage retrofits in this
document — the forecast journal, vintage-keyed caching, MASE/CRPS
scoring, conformal intervals, and ALFRED vintage data — all serve
that thesis. The data-source additions in §2 expand the analytical
surface; the model upgrades in §4 sharpen what can be said about
that surface; the ontology / knowledge-graph work in §3 connects
them. Together, they describe the next 12 months of work without
re-treading anything the team has already shipped.
