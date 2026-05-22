# Phase-1 Remediation Summary

*Branch: `remediation/phase-1` ← chain of `track-a-code` → `track-b-methodology` → `track-c-attribution` → `track-d-docs`*
*Base: `reviewable/loving-shtern-800851 @ 58e5ac7` (carries the six review artifacts in `reviews/`).*

This branch closes 61 Phase-1 items surfaced by the multi-track critical review on 2026-05-15. Diff vs. base:

```
43 files changed, 1,738 insertions(+), 950 deletions(-)
9 atomic commits across 4 tracks
Tests: 282 passed, 5 skipped (heavy deps absent in CI lite env)
```

## How this came about

1. **Four parallel critical reviews** at `reviews/01-code.md`, `reviews/02-methodology.md`, `reviews/03-attribution.md`, `reviews/04-expansion.md`. ~155 distinct findings total.
2. **Meta-review** at `reviews/meta-review.md` spot-checked ≥4 findings per review against the actual code. Caught one critical false positive (`02-methodology M0-1`: claimed ontology supersector code mismatch was a char-counting error — *do not act on it*).
3. **Combined synthesis** at `reviews/combined-review.md` produced a deduplicated, severity-recalibrated, prioritized action list partitioned into four remediation tracks.
4. **Sequenced remediation** in four branches, each chained from the previous so later tracks see earlier edits.

## What's in this branch — track by track

### Track A — Code hygiene & security (`c796825`)
32 items, 31 files, +323/-805 (net cleanup).

**Security (C):**
- `utils/fetch_ces_data.py:78` — added `timeout=30` to the BLS POST (only fetcher without one; hung connections were pinning gunicorn workers).
- `utils/embeddings.py:275` — killed `np.load(..., allow_pickle=True)`. Cache now: JSON sidecar for chunks + `.npz` for the float matrix, loaded with `allow_pickle=False`. `LOCAL_EMBED_MODEL` mixed into the cache key so model switches invalidate stale 384-dim vectors.
- `deploy.sh` — added `--yes-install` gate; no more silent `sudo apt-get install`.
- `utils/fetch_population_data.py` — two bare `except:` narrowed to `except Exception` with warning logs (no longer swallows `KeyboardInterrupt`/`SystemExit`).
- `utils/fetch_laus_data.py:34` — added missing `import sys` (`sys.exit` was `NameError`).

**Behavior bugs (H):**
- `tabs/super_tab.py` — `AGGREGATE_KEYS = frozenset({"Region Mean", "Region Median"})`; aggregate-row filter now matches actual keys instead of the stale `"Midwest"` prefix. Fixes the recommendation panel ranking regional aggregates as states.
- `tabs/_components.py` — `loading_skeleton(lines>3)` now repeats the `(100, 90, 70)` pattern; `ValueError` for `lines<1`.
- `utils/forecasting/selection.py` — dropped dead conditional in `_baseline_residuals` (both branches produced identical output).
- `utils/agents/blurb_orchestra.py` — CI-width comparison switched to `width / last_val > 0.5` so the threshold matches the "% of last observed value" message.
- `tests/tabs/test_tab_smoke.py` — relaxed callback-name asserts; the `poll_*_blurbs` callbacks no longer fail smoke.

**Logging hygiene (H):**
- Removed `logging.basicConfig` from 5 library modules; kept it in `app.py` only. Library modules now use `getLogger(__name__)`.

**Dead code & dependency sweep (M):**
- Deleted `utils/agents/image_writer.py`, `utils/model_utils.py`, `utils/graphics.py`, dependent tests.
- Moved `utils/bootstrap_state_codes.py` → `scripts/bootstrap_state_codes.py`.
- Removed `validate_lfpr_data`, `preprocess_for_embedding`, `_render_eda_overview`, `agent_polish`, `empty_state`, `section`, and ~10 unused imports across the agent and tab modules.
- `requirements.txt` — dropped `pyspark`, `scikit-learn`, `tqdm`, `tenacity` (none imported anywhere). `Dockerfile.dashboard` — removed `openjdk-21-jre-headless` and `PYSPARK_*` envs. ~500 MB off the image.
- `docker-compose.yml` — `OLLAMA_NUM_PARALLEL` aligned with `OLLAMA_MAX_LOADED_MODELS`.

**Refactor (M):**
- FIPS lookups dedup'd to `ONTOLOGY.states_by_fips`.
- Extracted `utils.data_pipeline.load_panel_df()`; `app.py` and both tabs now share it (was three near-identical copies).
- Wrapped module-level model caches in a small in-file LRU (`OrderedDict`, maxsize 64).
- `lfp_tab.py` `gap_anchor` made year-boundary-safe via `_current_year()` helper.
- `utils/agents/reviewer.py` `_PREAMBLE_MARKERS` sorted longest-first.

### Track B — Methodology correctness (`d6a2a1f`, `59b5dac`)
8 items, 8 files, +783/-79. **Includes the single most leveraged Phase-1 fix:**

- **`d6a2a1f` Phase E/F fetcher wiring** — `read_qcew`, `read_jolts`, `read_cpi`, `read_fred`, `read_bea` helpers added to `utils/merge_all_data.py`. Each returns long-format `(state, year, month, metric, value)` with documented broadcast semantics:
  - QCEW quarterly → 3 months per quarter (constant)
  - JOLTS national-only → broadcast to every state with caveat
  - CPI region → mapped to member states
  - FRED quarterly → 3 months; annual → 12 months
  - BEA annual → 12 months
  - `_join_long_phase_ef` pivots to `{ST}_{METRIC}` columns and left-joins on `(year, month)`. Each reader wrapped in try/except so a bad source can't sink the merge.
  - Also removed dead `"009": "Population"` mapping from `measure_map`.
  - 6 new tests in `tests/utils/test_merge_all_data.py`.
- **`59b5dac` Methodology fixes:**
  - `utils/embeddings.py` — e5 `query:`/`passage:` prefixes applied; `retrieve_context` switched to cosine over `_embs` (was silently lexical Jaccard). Lexical preserved as `USE_LEXICAL_RETRIEVAL=true` fallback. Cache version bumped to invalidate prefix-less caches.
  - `tabs/super_tab.py` — mean / median bars and recommendation top/bottom now compute on the post-threshold `display_states` set.
  - `utils/constants.py` `SUPERSECTORS` — derived from `utils.ontology.SUPERSECTORS` at module load. Eliminates the 9-vs-13 enum drift.
  - `utils/data_pipeline.py` `OUTPUT_JSON` — resolved to absolute path via `Path(__file__).resolve().parents[1]`.
  - `tabs/_methodology.py` ARIMA grid — rendered from `ARIMAForecaster.DEFAULT_GRID` at module load. Single source of truth = model class.

### Track C — Attribution & licensing (`6f29139`, `37b605a`, `2f93ac4`)
11 items, 13 files + new `NOTICE`, +575/-28.

- **`6f29139`** — "Built with Llama" notice (required by Llama 3.2 Community License when surfacing LLM output as a service); README & REPORT "License & Acknowledgments" sections rewritten to match the actual stack (Spark removed, TF opt-in only, all data sources listed); REPORT.md author email reconciled; broken `img/IA_*.png` references replaced with dashboard-generated notes; AI-assistance disclosure paragraph added to README + REPORT.
- **`37b605a`** — BLS Handbook URLs corrected (LAUS, JOLTS, QCEW all now point at `/opub/hom/<topic>/home.htm`); citation registry grew **21 → 34** with verified entries for Wang 2022 (e5), Reimers-Gurevych 2019 (SBERT), Seabold-Perktold 2010 (statsmodels), Pedregosa 2011 (sklearn), Harris 2020 (NumPy), McKinney 2010 (pandas), Abadi 2016 (TF), Tukey 1977, Akaike 1974, Efron 1979, Llama 3.2 License 2024, Shinn 2023 (Reflexion), Minsky 1986 (Society of Mind); methodology panel renders them in a dedicated "Software & model attribution" subsection. Dead `https://github.com/` placeholder link replaced with relative path to the LFPR methodology doc.
- **`2f93ac4`** — `OLLAMA_MODEL` default updated from `llama2:chat` to `llama3.2:3b` in `constants.py`, `deploy.sh`, `.env.example` (production already used 3.2; defaults disagreed); `NOTICE` file created at repo root for Apache-2.0 deps; `chart_source_annotation()` helper wired into LFP, Super, EDA chart layouts so every published chart carries the BLS / Census / BEA / FRED / FHFA source line.

### Track D — Documentation truth (`5d41950`, `145f54b`, `0fe0e3c`)
10 items, 6 files, +64/-45.

- **`5d41950`** — Title and scope: "Midwest Labor Dashboard" → "Prairie Insights: U.S. Labor Market Dashboard"; Forecasting prose rewritten to describe the Naive / Seasonal-Naive / ETS / ARIMA bakeoff (LSTM opt-in, not headline); MILESTONE team framing standardized to "sole-author team".
- **`145f54b`** — Unreproducible specific numbers removed from REPORT.md: the "MAE ~0.35 pp vs 0.41 pp" claim in §3.1, the entire §3.3 "Historical Analogues" section (closest_month wasn't implemented), the "±0.8 pp since 2010" Iowa claim in §5. Replaced with honest qualitative descriptions.
- **`0fe0e3c`** — LFPR caveat reworded to "±2 pp on average; up to ~3 pp at extremes (UT, ID, ME, FL)"; lfpr_denominator.md summary tightened with the per-state spread admitted up front; Hyndman 2nd ed (2018) → 3rd ed (2021) with otexts URL `/fpp2/` → `/fpp3/` (registry key kept stable); "20+ references" docstring claim replaced with drift-proof phrasing (registry is now 34 and growing).

## What was deliberately NOT touched

- **`02-methodology M0-1`** (ontology supersector code mismatch) — meta-review confirmed this is a char-counting false positive. The bootstrap formula produces strings byte-for-byte identical to the JSON. Do not act on it.
- Phase 2+ items — see `reviews/combined-review.md:114-167` "Deferred" section: SARIMA candidate, fold-level Diebold-Mariano, MASE/CRPS/pinball metrics, conformal prediction, MinT hierarchical reconciliation, ACS-default LFPR denominator, demographic stratification, vintage capture, bi-temporal store, etc.
- Phase 3+ data sources and ontology expansion — see `reviews/04-expansion.md` for the full roadmap (O*NET, OEWS, LEHD/LODES, BDS, BEA RPP, Beige Book, ALFRED, etc.).

## Verification

- `python -m pytest tests/ --tb=short -q` passes 282 / skips 5 on each track (skips are missing heavy deps: `dash`, `sentence_transformers`, `langchain_core`, `deepagents` — same skips on the base branch).
- 6 new tests in `tests/utils/test_merge_all_data.py` cover the Phase E/F readers and the end-to-end merge.
- 3 new tests in `tests/tabs/test_methodology.py` cover the software attribution and chart source annotation (dash-gated; will run in the Docker/full env).
- `python -c "from utils import constants, data_pipeline, fetch_ces_data, fetch_laus_data, fetch_population_data, merge_all_data; from utils.agents import sentence_rag"` imports clean.

## Branch chain

```
reviewable/loving-shtern-800851 @ 58e5ac7  (reviews/* committed)
└── remediation/track-a-code @ c796825
    └── remediation/track-b-methodology @ 59b5dac  (d6a2a1f, 59b5dac)
        └── remediation/track-c-attribution @ 2f93ac4  (6f29139, 37b605a, 2f93ac4)
            └── remediation/track-d-docs @ 0fe0e3c  (5d41950, 145f54b, 0fe0e3c)
                └── remediation/phase-1 @ HEAD  (this branch + this file)
```

Each track branch is independently reviewable and cherry-pickable. `remediation/phase-1` is the integration tip.

## Recommended next steps

1. Open one PR against `main` from `remediation/phase-1` (or four PRs, one per track).
2. After merge, archive the per-track branches (`git branch -D remediation/track-{a,b,c,d}-*`).
3. The previous `claude/reverent-noyce-a643f3` branch is now superseded; either close any open PR off it or merge it as a no-op given `remediation/phase-1` contains all of its work plus this remediation pass.
4. Schedule Phase 2 work from `reviews/combined-review.md` deferred section.
