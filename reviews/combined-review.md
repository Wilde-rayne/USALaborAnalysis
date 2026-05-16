# Combined Review — Prioritized Action List

*Target: `reviewable/loving-shtern-800851` at commit 5133949 (104 files)*
*Synthesized from 01-code, 02-methodology, 03-attribution, 04-expansion + meta-review spot-checks.*

This document is the action source-of-truth for Phase-1 remediation. Every entry has been (a) verified against the cited file, (b) deduplicated across the four source reviews, and (c) had its severity re-justified by the meta-review.

Severity convention (combined): **C** = critical (security / wrong numbers); **H** = high (broken behavior or misleading public claim); **M** = medium (tech debt, drift, dead code); **L** = low (nits, polish).

---

## The five things that matter most

1. **Wire the Phase E/F fetchers into the merge pipeline.** `utils/fetch_{qcew,jolts,cpi,fred,bea}_data.py` write to `data/raw/{source}/` but `utils/merge_all_data.py` does not read any of them. Every Phase-E commit landed as a dead-end. Without this fix, the QCEW/JOLTS/CPI/FRED/BEA data exists on disk and is reachable only through the agent-tool path — not the main panel that powers EDA/LFP/Super tabs. **Fix:** add `read_qcew`, `read_jolts`, `read_cpi`, `read_fred`, `read_bea` helpers in `merge_all_data.py` and join them onto the panel by `(state, year, month)` with appropriate column-name prefixes. **Effort: L.** **Source: 01-code top-10 #4.**

2. **Fix the aggregate-row filter in `tabs/super_tab.py`.** Lines 164, 240, 245 filter using `k.startswith("Midwest")`, but the aggregate rows are constructed at lines 472–473 with keys `"Region Mean"` / `"Region Median"`. The aggregates therefore leak into both the recommendation panel (ranked as if they were states) and the threshold filter (cutoff applied to them). **Fix:** introduce a module-level `AGGREGATE_KEYS = {"Region Mean", "Region Median"}` constant and filter via `k not in AGGREGATE_KEYS` in all three sites. **Effort: S.** **Source: 01-code S1 + 02-methodology M0-8.**

3. **Stop using `np.load(..., allow_pickle=True)` on the embeddings cache and add the model name to the cache key.** `utils/embeddings.py:275` loads a pickle from `HF_HOME` (defaults to `~/.cache`); on any shared/CI host this is an RCE sink. Cache key (`_combined_input_hash` at 147–161) hashes the data file contents but omits `LOCAL_EMBED_MODEL` — switching the model leaves stale 384-dim vectors that look fresh. **Fix:** persist chunks as JSON sidecar + a `.npz` of the float matrix; load with `allow_pickle=False`; include `LOCAL_EMBED_MODEL` in the hash. **Effort: S.** **Source: 01-code S0 + 02-methodology M3-5.**

4. **Apply e5 prefixes and switch retrieval to cosine against the cached embeddings.** `utils/embeddings.py:237-252` calls `model.encode(texts)` with no `query:`/`passage:` prefix; `retrieve_context` (lines 311–355) ignores the cached embeddings and falls back to lexical Jaccard. The model card explicitly requires the prefixes and the dashboard's RAG quality is degraded as a result. **Fix:** prepend `passage:` when building corpus, `query:` at query time; switch retrieve_context to cosine using the existing `_embs` matrix (the existing code path was disabled due to "torch-vs-tensorflow allocator contention" — easy to gate behind an env flag rather than removed). **Effort: S.** **Source: 02-methodology M1-7 + M1-8.**

5. **Add `timeout=30` to `utils/fetch_ces_data.py:78`.** Every other fetcher has a timeout; this one is the outlier. A hung BLS connection pins a gunicorn gthread worker indefinitely. **Fix:** five-character change. **Effort: S.** **Source: 01-code S0.**

---

## Phase 1 remediation queue (act now)

### Track A — Code hygiene & security

| Sev | File:line | Problem | Fix | Effort |
|---|---|---|---|---|
| C | `utils/fetch_ces_data.py:78` | `requests.post` with no `timeout=` — hung BLS pins a worker | Add `timeout=30` | S |
| C | `utils/embeddings.py:275` | `np.load(..., allow_pickle=True)` reads from shared `HF_HOME` — arbitrary-code-execution sink | Store chunks as JSON sidecar; load with `allow_pickle=False` | S |
| H | `app.py:23` + `utils/{data_pipeline,fetch_ces_data,fetch_laus_data,fetch_population_data,merge_all_data}.py` | Six modules call `logging.basicConfig` at import — library modules silently override test/app loggers | Keep `basicConfig` only in `app.py`; library modules use `logger = logging.getLogger(__name__)` only | S |
| H | `utils/fetch_population_data.py:35,48` | Two bare `except:` swallow `KeyboardInterrupt`/`SystemExit` | Narrow to `except Exception:` and log the suppression | S |
| H | `utils/fetch_laus_data.py:34` | `sys.exit(1)` but `sys` never imported — `NameError` on missing codes file | Add `import sys`, or raise `FileNotFoundError` | S |
| H | `tabs/super_tab.py:164,240,245` | Aggregate-row filter uses `"Midwest"` prefix but keys are `"Region Mean"` / `"Region Median"` | Introduce `AGGREGATE_KEYS = {"Region Mean", "Region Median"}`; filter by set membership in all three sites | S |
| H | `tests/tabs/test_tab_smoke.py:71-94` | Asserts the LFP / Super / EDA tabs register only one callback each; each now also registers a `poll_*_blurbs` callback. Tests are stale and will fail | Update expected sets, or relax to `>=` membership | S |
| H | `tabs/_components.py:161-179` | `loading_skeleton(lines=N)` slices `(100,90,70)[:lines]` — for `lines>3` silently truncates instead of repeating widths | `widths = ((100, 90, 70) * ((lines // 3) + 1))[:lines]` or cap with `ValueError` | S |
| H | `utils/forecasting/selection.py:288-303` | `_baseline_residuals` has two branches producing identical output — dead conditional | Drop the `if any(isinstance(...))` and always fit fresh baseline | S |
| H | `utils/agents/blurb_orchestra.py:218-220` | `if width > last_val * 0.5:` flags CI > 50% of last value but message says `"width is X% of the last observed value"` — comparison is absolute vs fractional | `if last_val > 0 and width / last_val > 0.5` | S |
| H | `deploy.sh:30,53` | `sudo apt-get install -y` without confirmation — root-level mutation on a shared dev host | Print plan, require `--yes-install`, or document prior consent | S |
| M | `utils/agents/image_writer.py` | Entire module dead — `ImageWriterAgent`/`FigureSpec` referenced only in other modules' docstrings | Delete file; remove docstring references in `blurb_orchestra.py:8,21` and `reviewer.py:404,439` | S |
| M | `utils/data_pipeline.py:97-143` + `:166` | `validate_lfpr_data` is defined, never exported, only call site is commented out | Delete function + comment | S |
| M | `utils/embeddings.py:179-194` | `preprocess_for_embedding` only used by tests; the "thousand jobs" heuristic is unit-blind anyway | Delete function + dependent tests | S |
| M | `utils/model_utils.py`, `utils/graphics.py` | Defined, only test-suite usage; `graphics.py` imports matplotlib which isn't in `requirements.txt` | Delete both, drop tests | S |
| M | `utils/bootstrap_state_codes.py` | Standalone CLI, never imported as a library | Move under `scripts/` (or leave in `utils/` if you prefer — semantic-only) | S |
| M | `tabs/about_tab.py:12-17` | Six dead imports (`Input, Output, State, dcc, PreventUpdate, generate_insight`); `register_callbacks` body is `pass` | Delete dead imports | S |
| M | `tabs/lfp_tab.py:43` | `from utils.llm_utils import AI_FAILURE_MESSAGE, explain_view` — neither used | Delete | S |
| M | `utils/agents/blurb_async.py:35` | `explain_view` imported, never used | Drop the import (keep `AI_FAILURE_MESSAGE`) | S |
| M | `utils/agents/blurb_orchestra.py:71, 77, 88` | `lru_cache`, `AGENT_MODEL_ENV`, `DEFAULT_TIMEOUT` imported, never used in the module body | Delete dead imports | S |
| M | `utils/fetch_jolts_data.py:28`, `utils/fetch_cpi_data.py:21` | `import json` declared, never used | Delete | S |
| M | `utils/agents/sentence_rag.py:298-330` | `_render_eda_overview` registered for `_kind="eda_overview"` but no caller emits that kind | Delete renderer + dispatch entry | S |
| M | `utils/agents/sentence_rag.py:256-277` | `agent_polish` is a test-only path; `polish=True` never enabled in production | Either gate behind env var with a doc note, or delete | S |
| M | `tabs/_components.py:99,152,161` | `empty_state`, `section`, `loading_skeleton` are tested but no production caller invokes them | Delete (or wire into the tabs that should use them) | S |
| M | `utils/constants.py:18,40` | `OLLAMA_EMBED_PATH` defined but never used; `MODEL_NAME = OLLAMA_MODEL` is an unused alias | Delete | S |
| M | `requirements.txt:10,11,38,43` + `Dockerfile.dashboard:27` | `pyspark`, `scikit-learn`, `tqdm`, `tenacity` not imported anywhere in `utils/` or `tabs/`. `pyspark` alone pulls in openjdk-21 (~200 MB) | Drop these four; remove `openjdk-21-jre` install; remove `PYSPARK_PYTHON` env in `docker-compose.yml:68-69` | M |
| M | `docker-compose.yml:17,23` | `OLLAMA_NUM_PARALLEL: "4"` + `OLLAMA_MAX_LOADED_MODELS: "1"` are inconsistent | Pick a coherent pair (likely `NUM_PARALLEL=1` since `BlurbOrchestrator` serializes per the comment in `blurb_orchestra.py:240-247`) | S |
| M | `utils/fetch_population_data.py:15` + `utils/merge_all_data.py:28` | Two hard-coded `STATE_FIPS` / `fips_map` dictionaries duplicate `ONTOLOGY.states_by_fips` | Replace both with `ONTOLOGY.states_by_fips` lookups | S |
| M | `app.py:65-82`, `tabs/lfp_tab.py:101-114`, `tabs/super_tab.py:62-73` | Three near-identical implementations of "load OUTPUT_JSON, map period via MONTH_MAP, build date, drop_duplicates" | Extract `utils.data_pipeline.load_panel_df()` helper, call from all three | S |
| M | `tabs/lfp_tab.py:61`, `tabs/super_tab.py:59` | Module-level `dict` caches with no eviction grow unbounded across worker lifetime | Wrap in `cachetools.LRUCache` or `functools.lru_cache` | S |
| M | `tabs/lfp_tab.py:642` | `gap_anchor = pd.Timestamp(year=datetime.now().year + 1, ...)` — non-deterministic test at year-boundaries | Inject a clock arg or read a `NOW` constant; freezegun-compatible | S |
| M | `utils/agents/reviewer.py:55-82` | `_PREAMBLE_MARKERS` are matched in declaration order; longer prefixes need to come first | Sort markers by length descending at module load | S |
| L | `utils/constants.py:18-19` | `OLLAMA_API_PATH = OLLAMA_CHAT_PATH` is a double-alias | Pick one name | S |
| L | `utils/merge_all_data.py:7-9` | `logging.basicConfig` inside `if not logger.handlers:` is a half-fix | Drop entirely (the basicConfig hygiene fix above subsumes it) | S |
| L | `utils/agents/blurb_orchestra.py:298-322` | `ThreadPoolExecutor` only torn down via `teardown(wait=True)` / `__exit__` — leaked threads on panicking caller | Add `__del__` with `teardown(wait=False)` + a warning | S |

### Track B — Methodology correctness (Phase-1-safe subset)

| Sev | File:line | Problem | Fix | Effort |
|---|---|---|---|---|
| H | `utils/embeddings.py:237-252` | e5 model requires `query:`/`passage:` prefixes, never applied — known 5-15% retrieval-quality loss | Prepend `passage:` when building corpus, `query:` at retrieve | S |
| H | `utils/embeddings.py:311-355` | `retrieve_context` uses lexical Jaccard, ignoring the cached e5 embeddings (`_embs`) | Switch to cosine against `_embs` after applying the `query:` prefix; gate the old lexical path behind an env flag for the torch/tf-coexistence fallback | S |
| H | `utils/merge_all_data.py:* (read functions)` | Phase E/F fetchers (QCEW, JOLTS, CPI, FRED, BEA) write to `data/raw/` but `merge_all_data` does not read them | Add `read_qcew`, `read_jolts`, `read_cpi`, `read_fred`, `read_bea` and join into the panel by `(state, year, month)` with prefix conventions matching CES (e.g. `{ST}_QCEW_AverageWeeklyWage`) | L |
| H | `utils/merge_all_data.py:37` | `measure_map["009"] = "Population"` is dead and misleading — LAUS doesn't publish per-state population at the suffix 009 | Delete the entry; add a comment that population comes from Census PEP/ACS | S |
| H | `tabs/super_tab.py:466-475` | The displayed "median forecast across states" is computed before the threshold filter, but states shown are post-filter — label is misleading | Re-label as "median across all in-scope states", or recompute median on the post-filter set | S |
| M | `utils/constants.py:90-100` | `SUPERSECTORS` enum has 9 entries; `data/ces_state_sms_codes.json` has 13 (Total_Nonfarm, Total_Private, Leisure_Hospitality, Other_Services are fetched but unreachable from UI) | Sync `SUPERSECTORS` to the 13 in JSON (or derive it from the ontology to remove the drift surface entirely) | S |
| M | `utils/data_pipeline.py:28` (`OUTPUT_JSON`) | Hard-coded relative path; resolves against process CWD | Resolve to repo root via `Path(__file__).resolve().parents[1] / "data" / "all_data.json"` | S |
| M | `tabs/_methodology.py:51-54` + `utils/forecasting/models.py:233-238` | Methodology bullet renders the ARIMA grid as a string; the model class hardcodes the same tuple — drift risk | Render the methodology bullet from `ARIMAForecaster.DEFAULT_GRID` at module load | S |

### Track C — Attribution & licensing

| Sev | Locus | Problem | Fix | Effort |
|---|---|---|---|---|
| H | About tab + `README.md` + `REPORT.md` | Llama 3.2 Community License requires "Built with Llama" attribution when output is offered as a service. The dashboard is exactly that. Currently no attribution exists | Add `Built with Llama` + link to the license in About tab footer and README "License & Acknowledgments" | S |
| H | `README.md:215-220` | Acknowledgments stale: "TensorFlow & Spark" is wrong (Spark retired, TF is opt-in LSTM only); "BLS API" omits Census/BEA/FRED/FHFA/JOLTS/QCEW | Rewrite to match `utils/citations.py` content; the methodology panel already gets this right | S |
| H | `REPORT.md:3` | Author email `kbouwman@iastate.edu` conflicts with `pyproject.toml:12` `rayne.k.wilde@gmail.com` | Reconcile to a single canonical email | S |
| H | `tabs/_methodology.py:121-123` | `LFPR_DENOMINATOR_NOTE` links to literal `https://github.com/` placeholder | Point to actual repo file URL or the public release URL (deferrable until repo is public; for now use `docs/methodology/lfpr_denominator.md` relative path) | S |
| H | `REPORT.md:136,139` | `img/IA_Unemployment_Rate.png` and `img/IA_Labor_Force_Participation_Rate.png` referenced but `img/` directory doesn't exist | Either add the images (with a "Author-generated via Plotly" caption) or remove the figure markdown | S |
| H | `utils/constants.py:21` + `deploy.sh:14` + `.env.example:52` | Default `OLLAMA_MODEL=llama2:chat` differs from production `llama3.2:3b`. Llama 2 has a *different* license posture (700M-MAU consent) and is not what's actually used | Update default to `llama3.2:3b` across all three locations | S |
| H | `utils/citations.py:144-167` | `bls_laus_handbook`, `bls_jolts_handbook`, `bls_qcew_handbook` have `venue="BLS Handbook of Methods"` but URLs point at the *technical-documentation* pages, not `/opub/hom/<topic>/home.htm` | Update each URL to `/opub/hom/lau/home.htm`, `/opub/hom/jlt/home.htm`, `/opub/hom/cew/home.htm` (or change the venue text) | S |
| H | README + REPORT acknowledgments | No AI-assistance disclosure despite `docs/handoff/day_*.md` documenting agent-pair authorship over multiple days | Add a one-paragraph acknowledgment per the 03-attribution suggested text | S |
| H | `utils/citations.py` | Missing software/model citations: `wang_e5_2022`, `reimers_gurevych_2019`, `seabold_perktold_2010`, `pedregosa_sklearn_2011`, `harris_numpy_2020`, `mckinney_pandas_2010`, `abadi_tensorflow_2016`, `tukey_1977`, `akaike_1974`, `efron_1979`, `llama3_license_2024`. Optional: `shinn_reflexion_2023`, `minsky_1986` | Paste the 03-attribution recipe block into `_CITATIONS_LIST` and add a `SOFTWARE_ATTRIBUTION_KEYS` tuple in `tabs/_methodology.py` | S |
| M | Repo root | No `NOTICE` / `THIRD_PARTY_LICENSES.md` for Apache-2.0 deps (tensorflow, plotly, langchain*, deepagents) | Add a `NOTICE` listing each dep + license — required before any binary redistribution (Docker Hub flip) | S |
| M | About tab + chart footers | BLS / Census / BEA / FRED / FHFA each have a citation policy expecting "Source: U.S. ..." attribution; not currently surfaced on charts | Add a `data-source` footer to chart layouts: "Source: BLS LAUS / Census ACS / ..." | S |

### Track D — Documentation truth

| Sev | Locus | Problem | Fix | Effort |
|---|---|---|---|---|
| H | `README.md:1` | Title still "Midwest Labor Dashboard"; project ships 51-jurisdiction code (per `phase_d_results.md`) | Rename to "USA Labor Dashboard" or similar; same in `README.md:3` opening paragraph | S |
| H | `README.md:202-205` | LSTM is described as the headline forecast architecture; it's opt-in and not the production default | Rewrite to describe the Naive / Seasonal-Naive / ETS / ARIMA bakeoff with LSTM as opt-in | S |
| H | `REPORT.md:90` | Claim "LSTM achieves MAE ~ 0.35 pp versus ARIMA's 0.41 pp" is not reproducible — no saved fold experiment | Either reproduce + cite the experiment, or remove the specific number | S |
| H | `REPORT.md:99-109` | `closest_month` analogue analysis described but not implemented anywhere in the codebase | Either implement (with tests) or remove from REPORT | S |
| H | `REPORT.md:146` | "Iowa's LFPR remains within ±0.8 pp of the region since 2010" — not reproducible | Either back with a validation test or remove the number | S |
| H | `tabs/_methodology.py` LFPR caveat | "Within ~2 pp of published BLS state LFPR" is supported by the doc only for ~IA-like states; UT/ID are ~3 pp per the doc itself | Reword to "approximately ±2 pp on average; ~3 pp at the extremes (UT/ID/ME/FL)" | S |
| H | `docs/methodology/lfpr_denominator.md:60` | "±2 pp" claim glosses over the per-state spread the same doc admits at lines 59–61 | Already partially documented; tighten the language and link to the planned ACS-default switch | S |
| M | `docs/methodology/agent_pipeline.md:255-256` | Cites Hyndman & Athanasopoulos 3rd ed.; `utils/citations.py` registers the 2nd ed. — inconsistent across the codebase | Decide on 3rd ed. (2021) as canonical; update `utils/citations.py:` hyndman entry and the OTexts URL to `/fpp3/` | S |
| M | `tabs/_methodology.py` docstring | Claims "20+ references" but the registry has 21 (and the 03-attribution recipe adds ~11 more — pushing past 30) | Update docstring after applying the recipe | S |
| L | `MILESTONE.md` | Lists team membership but says "sole member of the team" downstream — minor inconsistency that confuses solo-vs-team framing | Standardize to "sole-author solo team Prairie Insights" | S |

---

## Deferred (Phase 2+) — captured but not acted on now

Roadmap items, model-architecture changes, schema migrations, data-source additions, and bigger methodology lifts that exceed Phase 1 scope. None of these should be touched by current remediation agents.

- **Methodology (Phase 2):**
  - SARIMA candidate in the bakeoff with `seasonal_order=(P,D,Q,12)` (02-methodology M1-3)
  - Diebold-Mariano run on fold-level out-of-sample residuals, not in-sample (M1-6)
  - MASE / CRPS / pinball loss in `ForecastMetrics` (04-expansion §4)
  - Seasonal-naive as the DM comparator everywhere
  - Conformal prediction wrapper over `BaseForecaster.predict_interval` (04-expansion §4)
  - Multiple-testing correction (Bonferroni / FDR) on the requirements pass/fail panel (M1-15)
  - Mann-Kendall or linear-trend significance test in `trend.py` (M1-14)
  - ACS-backed LFPR denominator becomes default rather than fallback (M0-4)
  - LSTM polish: validation split, early stopping, removing `PYTHONHASHSEED` no-op (M1-2, M2-5) — *or* remove LSTM entirely

- **Data sources (Phase 3+):**
  - O*NET skill vectors (04-expansion #2)
  - OEWS occupation × wage percentiles (04-expansion #1)
  - BEA RPP for real-wage adjustment (04-expansion #13)
  - BDS for firm dynamics (04-expansion #3)
  - Beige Book qualitative RAG corpus (04-expansion #11)
  - ALFRED vintage data for honest backtesting (04-expansion #5)
  - LEHD/LODES labor-shed analysis (04-expansion #4)
  - Treasury yield + recession indicators (04-expansion #16)
  - BLS Employment Projections (04-expansion #6)
  - NAICS↔SOC, CIP↔SOC, FIPS↔BLS, CBSA, NAICS17↔22 crosswalks vendored (04-expansion §3)

- **Schema / metadata (Phase 3+):**
  - `seasonal` and `vintage` columns in every raw fetcher output (M3-1, M3-2)
  - Forecast journal at `data/forecast_journal.parquet` (04-expansion §5)
  - Vintage-keyed cache: `(data_vintage_hash, model_version)` (04-expansion §5)
  - Freshness badge in the dashboard + grounding fact in blurb prompts (04-expansion §5)
  - CPI deflator applied to FRED nominal series (M3-3)
  - Per-sentence-granular embedding cache invalidation (04-expansion §5)
  - DuckDB/Parquet analytical store when row counts grow past ~5M (04-expansion §3)

- **Model architecture (Phase 4+):**
  - MinT hierarchical reconciliation (supersectors → total nonfarm) (04-expansion §4, M1-13)
  - Spatial-VAR with IRS-SOI migration weights (04-expansion §4)
  - Structural VAR with sign restrictions (04-expansion §4)
  - Chronos / TimesFM / Lag-Llama foundation-model candidates in the bakeoff (04-expansion §4)
  - GNN on state adjacency (04-expansion §4)
  - Demographic stratification (sex × age × race) via ACS B23001 / CPS basic (M2-6)

- **UI / UX (Phase 5+):**
  - 50-state default flip (M2-1) with regionalized naming
  - `STATE_SET` cleanup
  - Synthetic control event-study capability (04-expansion §4)

- **Public release (Phase 6):**
  - Replace `assets/bootstrap.min.css` with CDN-only `dbc.themes.BOOTSTRAP` (01-code S2)
  - Docker Hub flip requires the NOTICE file to be present (Track C above)
  - WARN-notice ingestion via an aggregator (04-expansion §7 anti-recommendation)
  - IPEDS / WIOA supply-pipeline integration

---

## Quality notes from meta-review

Remediation agents should know the following before they start work:

1. **DO NOT act on 02-methodology M0-1.** It claims a supersector-code mismatch between `utils/ontology.py` and `data/ces_state_sms_codes.json`. The claim is wrong — the bootstrap formula `f"SM{seasonal}{st.fips}00000{ss.code}000000{datatype}"` produces strings byte-for-byte identical to the JSON. `sid[10:12]` of `SMS19000003000000001` is `"30"`, not `"03"`. The test `test_parses_ces_manufacturing` passes correctly. The original reviewer made a character-counting error and presented it as the codebase's #1 critical bug. **If you find yourself "fixing" supersector codes to make them match a misread JSON, stop immediately.**

2. **02-methodology M0-4 (LFPR error magnitude) is partially overstated.** The doc itself admits ±3 pp at the extremes; the reviewer escalates this to ±5–6 pp without independent verification. Use the doc's own number when updating methodology language; do not introduce new claims without computing them.

3. **01-code S0 logging.basicConfig finding is over-severitized.** It's an operational hygiene issue, not a security/data-loss risk. Treat as H (high), not C (critical), and de-prioritize against the actual S0 timeout + pickle issues.

4. **Citation count.** Registry has 21 entries (not 20 as 03-attribution A3-6 claims). Update the methodology docstring after applying the additional citations.

5. **Several findings cross-reinforce.** When acting on the Midwest aggregate-row filter, the e5 prefix fix, or the embedding pickle fix, expect to touch test files too — the smoke tests already need updates (Track A row 7) and the embedding tests will need updates after the cache-format change (Track A row 2).

6. **Don't touch the ontology data model during cleanup.** The `Ontology` singleton at `utils/ontology.py` is load-bearing for the Phase-3+ knowledge-graph work described in 04-expansion §3. Bug fixes inside the existing dataclass methods are fine; adding/removing fields or changing the supersector code convention is *not* a Phase-1 change.

7. **The Phase E/F fetcher wiring (Track A row 3 / The five things #1) is the single most leveraged Phase-1 fix.** It moves the codebase from "we fetched data but didn't use it" to "the panel actually contains what the README says it does". Sequence-wise, expect this work to take the longest of all Phase-1 items.

8. **Documentation truth (Track D) and attribution (Track C) cannot be deferred even if they feel less urgent than code changes.** The "Built with Llama" attribution is a license requirement, not a polish item; the AI-assistance disclosure is a hiring-norms requirement for a portfolio piece; the stale README acknowledgments will be the first thing a reviewer notices. Address Track C and D in parallel with Tracks A and B.
