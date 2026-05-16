# Critical Methodology & Data-Correctness Review

*Target: `reviewable/loving-shtern-800851` at commit 5133949*

## Severity guide

- **M0** — produces wrong numbers / invalid public claim
- **M1** — methodologically dubious or unsupported claim
- **M2** — avoidable bias or weak design
- **M3** — metadata/reproducibility gap

---

## Findings (each cites file:line)

### M0 — Wrong numbers / invalid public claims

**M0-1. Ontology supersector codes and bootstrap-generated CES SMS series IDs do NOT match the data/ces_state_sms_codes.json file.**

- `utils/ontology.py:150-167` defines BLS supersector codes as `"10"` (Mining/Logging), `"20"` (Construction), `"30"` (Manufacturing), `"40"` (TTU), etc.
- `data/ces_state_sms_codes.json:333` has IA Manufacturing = `"SMS19000003000000001"` (note characters 10–11 = `"03"`, not `"30"`).
- `utils/ontology.py:466` `ces_series_id` is `f"SM{seasonal}{st.fips}00000{ss.code}000000{datatype}"`. With ss.code=`"30"` the bootstrap output is `SMS190000300000000_1` (chars 10–11 = `"30"`), which does NOT equal the JSON's `"SMS19000003000000001"` (chars 10–11 = `"03"`).
- Therefore running `python -m utils.bootstrap_state_codes` will REPLACE the working JSON with a JSON whose series IDs no longer match anything BLS publishes. Subsequent fetches will silently return empty `series` lists; the panel will quietly miss every CES series; LFPR will still compute but supersector tab will be empty for every state.
- `utils/ontology.py:380` `parse_series_id` reads `sid[10:12]` and looks it up in `supersectors_by_code`. For the JSON's actual string (`"SMS19000003000000001"`) this yields `"03"`, which is absent from the ontology; the parser returns `SeriesSpec(supersector=None)`.
- `tests/utils/test_ontology.py:119-125` (`test_parses_ces_manufacturing`) explicitly asserts `spec.supersector is not None` for input `"SMS19000003000000001"`. Either this test is silently failing in CI, or the parser disagrees with reality. Either way, the ontology's supersector codes are wrong vs. the BLS SMS encoding used in the production JSON.
- **Impact:** any agent / tool that calls `ces_series_id` (e.g. for territories / non-Midwest states the JSON doesn't pre-populate) will request the wrong series. The dashboard ships data that *happened* to work because the JSON was hand-curated, but the methodology is broken; re-generating from the ontology produces wrong IDs.

**M0-2. LAUS labor-force series suffix `006` is "Labor Force Level," not "Civilian Labor Force"; the `Population` measure suffix `009` does not exist in BLS LAUS.**

- `utils/merge_all_data.py:37` `measure_map = {"006": "Labor_Force", "005": "Employment", "004": "Unemployment", "009": "Population"}`.
- BLS LAUS state series have measure-codes 003 (unemp rate), 004 (unemp level), 005 (emp level), 006 (labor force level). Suffix `009` is **not a published LAUS measure** at the statewide level.
- The merger searches `RAW_DIR_LAUS/LASST…009.txt`; nothing in the data pipeline ever creates such a file (the BLS fetcher only pulls codes listed in `laus_state_codes.json`, none of which end in 009). The Population column in the panel is therefore populated EXCLUSIVELY from `POP_<ST>.txt` files (Census PEP), never from LAUS.
- Per LAUS API docs the closest "Population" measure is `00000000000003` (rate) — there is no per-state population measure with `_009`. The presence of `009` is dead code with a misleading mapping that suggests LAUS provides population (it does not).
- **Impact:** the `Population` denominator in LFPR is silently 100% Census PEP-derived (or its ACS fallback), with no LAUS cross-check despite the code implying otherwise.

**M0-3. The LFP tab's "forecast value at year+N" omits the data-lag gap from the verdict for the requirements/AI views, but the chart shows the year-N value AFTER the gap — these are different points.**

- `tabs/lfp_tab.py:954-963` calls `_compute_forecast_points(all_states, state_forecasts, total_horizon)` with `total_horizon = gap_months + months`. The "point" stored is `preds[-1]` from a prediction of length `total_horizon`.
- However, `tabs/lfp_tab.py:692-695` slices `preds = preds_full[gap_months:]` for the chart display.
- For the focus state, both the chart endpoint and the verdict use `preds[-1]` of the full-horizon prediction, so they agree (the last point is the same). But peer points are derived independently (`tabs/lfp_tab.py:756`: `peer_preds = peer_result.model.predict(total_horizon)[gap_months:]`) and the peer median annotated on the chart at line 924 uses `p_preds[-1]` from `predict(total_horizon)` — same endpoint.
- Verdict: the points themselves agree, but the displayed "forecast end year" `vs._kind=forecast_panel.forecast_end_year = gap_anchor.year + years_ahead - 1` (`lfp_tab.py:1011`) is off-by-one vs the actual prediction endpoint, which is `last_date + (gap_months + horizon) months ahead`. For `last_date=Dec 2024`, `years_ahead=2`, `gap_anchor=Jan 2026`: `forecast_end_year=2027`. The actual last predicted month is `Dec 2024 + 36 mo = Dec 2027` — so end-year = 2027 is technically correct, but the AI prompt says forecast covers "2027" (start year + 0–1 years), implying a 1-year window where the data is a 2-year window. The grammar of the prompt is inconsistent with what's plotted.

**M0-4. `LFPR_WORKING_AGE_FRACTION = 0.78` correction is documented as US average; per-state error can be 5+ pp, not the claimed ~2 pp.**

- `docs/methodology/lfpr_denominator.md:60` claims residual error of ~±2 pp on average. The doc also states UT/ID sit ~3 pp high and ME/FL ~2 pp low. State-specific CNI16+/total-pop shares range from ~73% (UT) to ~82% (ME, FL) per ACS B23025 (verified externally); applying a fixed 0.78 introduces a ±5–6 pp swing at the extremes.
- The ACS-backed path (`utils/merge_all_data.py:240-256`) is opt-in and only engages when `WAP_<state>.txt` files are present. Out of the box (per `phase_d_results.md`'s 51-state run notes), most users hit the uniform-0.78 fallback. The user-facing dashboard prominently shows LFPR values without flagging the 5–6 pp ceiling on confidence for HI, UT, FL etc.
- The methodology doc itself uses Iowa as the only worked example (where the 0.78 correction is approximately right). The "±2 pp" claim is not supported for the full 50-state cohort.

**M0-5. `read_working_age_population` linearly interpolates and forward-fills CNI16+ across years with NO error margin propagated; produces high-confidence numbers from low-confidence proxies.**

- `utils/merge_all_data.py:147-164` interpolates ACS B23025_001E across years and uses `limit_direction="both"` for forward + backward fill from a single known year.
- `tests/utils/test_lfpr_acs_denominator.py:155-159` even tests that a single known year for 2010 gets propagated to 1996 AND 2024 with no warning. ACS B23025 starts 2005; pre-2005 years use the 2005 value extrapolated backward by ~9 years.
- Pre-2005 ACS coverage is non-existent, so the "ACS-corrected LFPR" for 1996–2004 is constructed by extrapolation from 2005, then divided into LAUS labor force. This is presented to the user identically to a 2010 LFPR that uses contemporaneous ACS data.
- **Impact:** the panel reports more precision than the underlying data supports for any year before 2005 or after the latest ACS release. No vintage/confidence band is preserved; the `LFPR` column has no error metadata.

**M0-6. Population PEP fetcher interpolates linearly between non-adjacent years and substitutes for unfetched years; no logging of how many rows are interpolated.**

- `utils/fetch_population_data.py:90-107` fills missing years via linear interpolation between known bookends (or via ffill / bfill from the nearest known year). No record of which years were genuinely fetched vs interpolated lands in the output file or in any metadata.
- `data/raw/laus/POP_<ST>.txt` is then consumed identically in `read_population` as if every value were authoritative.
- LFPR computation uses these `Population` values without flagging whether the row was an interpolation. The user cannot see whether 1996 IA population is the real fetched value or extrapolated.

**M0-7. The Wide-format `{state}_Labor_Force_Participation_Rate` is renamed from `LFPR` but the LFPR_RAW column also gets widened — but the per-supersector CES columns get added LATER, after deduplication, leaving the panel with **per-state `IA_LFPR_RAW`** while CES rows get *aggregated* across states.**

- `utils/merge_all_data.py:319-327` builds `ces_rows` indexed by `state` and pivots into `ces_wide`. The pivot key is `(year, month)` only, so for any year+month tuple where multiple states have CES data, `aggfunc="first"` discards all but one state's data per column. But the columns are `{st}_{sector}` so they're per-state column-distinct — so this isn't a fan-out, but the `pivot_table.aggfunc="first"` is sensitive to row ordering and silently drops duplicates rather than warning.
- More importantly: `ces_rows` is a list of dicts where each dict has only one `{st}_{sector}` column populated. The `pivot_table` fills NaN for every other state. Then `pd.merge(panel, ces_wide, on=["year","month"], how="left")` joins. For 51 states × 13 supersectors × 348 months = ~230,000 rows in `ces_rows`, the pivot is a wide table where most entries are NaN. The merge is correct but inefficient (O(N²) memory).

**M0-8. The `super_tab.py` `_apply_threshold` and `_recommendation_panel` reference "Midwest" hard-coded.**

- `tabs/super_tab.py:163-166` `states_only = {k: v for k, v in forecasts.items() if not k.startswith("Midwest")}`.
- `tabs/super_tab.py:239-246` filters using `k.startswith("Midwest")`.
- In a 50-state rollout the aggregate keys are `"Region Mean"` / `"Region Median"` per line 472–473 — these do NOT start with `"Midwest"`, so the filter incorrectly INCLUDES the aggregate rows in the per-state ranking and median computation.
- The dropdown label in line 222 says "Median forecast across states: {median}" but the median was computed across states + 2 aggregate rows + state values. The displayed median is biased toward the mean (because mean+median rows pull it inward).
- **Impact:** the recommendation panel's "top 3" and "bottom 3" can include `"Region Mean"` or `"Region Median"` as if they were states. Visually obvious only when those rows make the top-3 cut.

**M0-9. The `peer_median_forecast` value in the LFP forecast view-state is the median of *peer states only*, but the chart's "peer median forecast" annotation includes the focus state — they will not match.**

- `tabs/lfp_tab.py:994-999` `peer_points = [forecast_points[p]["point"] for p in peer_states if forecast_points.get(p) is not None]` then `peer_median = float(np.median(peer_points))` — excludes focus state. View-state has peer-only median.
- `tabs/lfp_tab.py:909-924` builds `peer_point_values` from `peer_states` only (loop iterates over peers, not all_states), so chart annotation also excludes focus state. They DO agree.
- However, the inline statistical context the AI sees says `"Peer-state median forecast at the same horizon is X"` — and the visual annotation also says `"peer median forecast (X%)"` — so user could think focus is included.
- This is more an M1 framing issue: the "peer-state median" excludes the focus state, which is the standard convention but isn't called out in the AI-facing prompt or the chart label.

---

### M1 — Dubious methodology / unsupported claim

**M1-1. Forecast model selection uses 3-fold expanding-window backtest with `n_eval` rows used in metric computation, NOT per-fold standardized — bias from very short folds is averaged out without significance weighting.**

- `utils/forecasting/selection.py:165-177` averages `mae`, `rmse`, `mape`, `smape`, `bias` straight across folds. If fold 1 has 36 test points and fold 3 has 36 test points (always equal in `_expanding_window_folds`), this is unbiased. But the `aic`/`bic` use `per_fold[-1]` (last fold), which has the largest train set — fine as a heuristic but the lack of fold-level standard error / confidence interval means winners can be tied within sampling noise.
- The `n_folds=3` default at every callsite (`lfp_tab.py:151`, `super_tab.py:106`) is the minimum reliable number — no documentation of why 3 was chosen vs. 5 or 10.

**M1-2. LSTM `LSTMForecaster` has only `epochs=10`, `batch_size=16`, `units=32`, and a single LSTM layer; no early stopping, no validation split, no regularization, with ~300 obs and ~5,000 parameters.**

- `utils/forecasting/models.py:348-407` defines the LSTM. With 32 units + a 12-step lookback + dense output, parameter count is ~5–6K. Training set is ~300 obs (sliding windows yield ~288). Train MSE will overfit; no holdout validation per epoch; no early stopping.
- Activation is `"relu"` on the LSTM layer (`models.py:392`) which is non-standard — LSTM uses `tanh` for cell state and `sigmoid` for gates by default. Setting `activation="relu"` overrides the OUTPUT activation only (Keras semantics), but is a smell.
- The model.fit call `model.fit(X, Y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)` does not pass `validation_data` — there's no way to monitor for overfitting. This is fine for opt-in / experimental use, but the LSTM should NEVER be the production winner without validation.

**M1-3. ARIMA grid `{(1,1,1), (2,1,1), (1,1,2), (2,1,2)}` is too narrow for monthly seasonal data; no seasonal ARIMA component.**

- `utils/forecasting/models.py:233-238` defines the default grid. None of these orders include `seasonal_order` parameters, so the fitted ARIMA is non-seasonal.
- For monthly labor data with strong annual seasonality, SARIMA `(p,d,q)(P,D,Q,s=12)` is the standard. Non-seasonal ARIMA on monthly seasonal data overfits the deterministic seasonal component as a non-stationary error structure, producing wider CIs than needed.
- The Hyndman-Athanasopoulos reference cited in `agent_pipeline.md` line 256 covers SARIMA in section 9.7; the implementation omits this.

**M1-4. ETS / Holt-Winters fallback to "non-seasonal Holt" when series length < 2 × seasonal_periods silently changes the underlying model without flagging.**

- `utils/forecasting/models.py:141-148` falls back to non-seasonal when `y.size < seasonal_periods * 2 = 24`.
- For a 24-obs series this happens silently. The metric output `ForecastMetrics(model="ets", ...)` does not distinguish whether the fitted model was seasonal or not; the diagnostics panel renders "Holt-Winters" regardless.

**M1-5. CI computation for LSTM uses default `predict_interval` from `BaseForecaster` — residual-bootstrap with `sqrt(step)` variance growth, applied to in-sample residuals.**

- `utils/forecasting/base.py:117-145` builds the band from `np.std(resid, ddof=0)` and `sqrt(step)` scaling.
- In-sample LSTM residuals systematically UNDERESTIMATE out-of-sample variance (overfitting is endemic). The resulting CIs are too narrow for LSTM forecasts, but no caveat is shown to the user.
- For ARIMA, `predict_interval` uses statsmodels' `get_forecast` analytical bounds (correct). For ETS, `predict_interval` uses the parametric bootstrap via `.simulate()` (1000 paths) — fine. The discrepancy: LSTM CIs are uncalibrated.

**M1-6. Diebold-Mariano test compares the WINNER's residuals to a Naive baseline — but the winner's residuals are IN-SAMPLE residuals from the full-series refit, while the Naive baseline residuals are also full-series in-sample. DM is then a test on in-sample fit, not out-of-sample forecast accuracy.**

- `utils/forecasting/selection.py:297-303` `_baseline_residuals` fits a Naive on the full series and returns `baseline.residuals` (in-sample, i.e. first-differences).
- `utils/forecasting/diagnostics.py:226` passes those alongside `final.residuals` (winner's full-series in-sample residuals).
- DM-test theory is about *forecast* losses; using in-sample residuals biases the comparison toward complex models that fit better in-sample but don't necessarily forecast better.
- The fold-level RMSE comparison already does out-of-sample evaluation; if the DM test were run on **fold-level forecast errors** it would be properly calibrated. As is, DM gives misleading p-values that the dashboard surfaces in the model rationale table (`tabs/lfp_tab.py:260`).

**M1-7. Embeddings model is e5-small-v2 but the required `query: ` and `passage: ` prefixes from the model card are NEVER applied.**

- `utils/embeddings.py:237-252` (`_embed_texts`) calls `model.encode(texts)` directly. No prefix. The e5 model card (intfloat/e5-small-v2) explicitly requires `query: ` prefix for queries and `passage: ` prefix for documents — without these, retrieval quality drops 5–15% on benchmark tasks.
- `utils/embeddings.py:311-355` `retrieve_context` doesn't apply the prefix either (and falls back to lexical-only retrieval anyway). The cached embeddings are wasted because retrieval doesn't use them.

**M1-8. Retrieval is "tokenizer-based lexical overlap" (Jaccard-ish), NOT cosine similarity against the pre-computed embeddings.**

- `utils/embeddings.py:311-355` `retrieve_context` uses `_chunk_tokens` (a set-of-tokens index built at load time) and computes `overlap / sqrt(|chunk|)`. The pre-computed e5-small embeddings are loaded but unused for retrieval.
- The function header lists this as a workaround for "torch-vs-tensorflow native allocator contention" (`embeddings.py:317-323`).
- **Impact:** the dashboard claims "embeddings: e5-small-v2 in-process PyTorch" but actually uses lexical Jaccard. The README's "Sentence-RAG corpus built deterministically from an ontology of states × supersectors × measures — no tabular blobs in embeddings" implies semantic retrieval; the implementation degrades to keyword matching.

**M1-9. `_safe_mape` filter drops rows where `|y_true| < 1e-12`, but bias and other metrics are computed on the full array — these metrics are then averaged across folds with different `n_eval` counts.**

- `utils/forecasting/selection.py:32-41` filters MAPE to rows where `|y_true| > 1e-12`; `_smape` does the same. Other metrics use the full array.
- For series that pass through zero (rare but possible — e.g. monthly net flow data), the MAPE could be NaN'd while RMSE is fine.
- Cross-fold averaging in `selection.py:163-177` treats MAPE-NaN folds specially (`mapes_ok = [...]`), but the overall `n_eval` count adds up the test-set sizes including the NaN ones. Inconsistent denominators.

**M1-10. `summarize_trend.delta_5y` checks `arr.size <= 5 * periods_per_year` (= 60 for monthly), but the comparison is `arr.size > step`, requiring 61 observations — this is fine, BUT `delta_1y` for monthly data only needs 13 obs which means at month-13 the comparison is against month-1 — a single early-month reading vs latest, which may straddle a recession.**

- `utils/forecasting/trend.py:55-66` — delta_1y at month 13 compares `arr[-1] - arr[-13]`. For a series starting Jan 2020 ending Feb 2021, this would be Feb-2021 minus Jan-2020 — exactly 13 months apart, straddling the 2020 pandemic. The "1-year delta" semantic is met but the result has high noise.
- This is a UI/communication issue: a user reading "1-year change: -2 pp" might interpret it as "stable normal year" not "pandemic shock vs pre-pandemic".

**M1-11. The "0.78 working-age fraction" lift produces LFPR values that should be compared against the BLS-published series (`LASST{FIPS}0000000000005` or similar), but the codebase doesn't fetch BLS-published LFPR for cross-validation.**

- No LAUS measure with `laus_suffix="009"` exists in BLS (per M0-2); the published LFPR by state is computed from CPS at the BLS end and isn't published as a per-state monthly series.
- The codebase derives LFPR from the LAUS labor-force level / Census population. Comparing against BLS-published annual state LFPR would require a separate fetcher. Without it, the "~2 pp residual error" claim in `docs/methodology/lfpr_denominator.md:60` is untestable in CI.

**M1-12. The `closest_month` historical-analogue code shown in `REPORT.md:99-109` is dead — no implementation exists in the codebase.**

- Grepped: no `closest_month` function exists in `utils/`. The REPORT claims this functionality but it's a paper-claim only.

**M1-13. Forecasting bake-off is per-state and per-(metric, horizon) independent — no spatial / hierarchical reconciliation.**

- States that geographically share a labor market (e.g. KS/MO around Kansas City, MD/DC/VA in DMV) are forecast independently. No hierarchical bottom-up / top-down reconciliation. No spillover model.
- For a "regional planner" audience this is a real methodological gap; the README's claim "regional planners comparing state performance" implies hierarchical analysis that doesn't exist.

**M1-14. Trend detection in `test_trend.py` covers level/delta/range/volatility — no statistical test for "is this a real trend or sampling noise?".**

- `utils/forecasting/trend.py` and its tests compute deltas and volatility but never run a Mann-Kendall or linear-trend significance test. The `delta_5y` is presented as a number without a p-value or confidence interval.
- The dashboard then displays this in a "trend summary" table where the user infers significance from magnitude alone.

**M1-15. Multiple-testing correction is absent across state-level forecasts.**

- The LFP requirements panel evaluates threshold pass/fail per state independently. The user is shown ✓/✗ per state with no Bonferroni / FDR correction. With 51 jurisdictions and a 95% PI, the family-wise error rate is ~92% — at least one state will appear "marginal" by chance.

---

### M2 — Bias / weak design

**M2-1. Default `STATE_SET=midwest` ships 12 hand-picked states; the bias is inherited by everything downstream including the embedding corpus, the AI specialists, and the model bake-off priors.**

- `utils/constants.py:65-73` defaults to Midwest. The portfolio framing claims a "USALaborAnalysis" project but ships a Midwest tool out of the box.
- The README's "About This Dashboard" sells "Midwest Labor Dashboard" while `MEMORY.md:phase_d_results.md` notes the 51-state version is run but "Midwest naming needs regionalizing before a public flip." Code mostly catches up — `super_tab.py:38-46` lists all 5 Census regions in the dropdown — but the legacy default biases initial views.

**M2-2. "Supersector aggregation" presents `Region Mean` / `Region Median` of state-level CES SA series as if additive.**

- `tabs/super_tab.py:464-466` `mean_val`, `median_val` are computed on `vals` (state-level forecasts).
- CES seasonally-adjusted series are NOT additive across states (BLS publishes SA at each level independently with different decomposition fits). Mean/median of state SA series ≠ aggregate-level SA series.
- The chart and recommendation panel present these aggregates as if they were neutral statistics. They're not — they're biased estimates of the true regional aggregate that BLS would publish if computed centrally.

**M2-3. ACS B23025_001E uses 1-year estimates (≥65,000 pop areas only); states with low population have either no estimate or much wider sampling MoE not exposed.**

- `utils/fetch_working_age_population.py:39-42` calls ACS 1-year (`/acs/acs1`). For PR/VI/GU/AS/MP territories, 1-year estimates are not published (Wyoming is borderline). The fetcher logs and skips — but the LFPR for these jurisdictions then ALWAYS hits the uniform-0.78 fallback.
- No MoE (margin of error) from `B23025_001M` is fetched, so the uncertainty in the denominator can't be propagated into the LFPR confidence interval.

**M2-4. ARIMA grid search is unbounded by stationarity tests — ADF/KPSS are computed AFTER the model is chosen, not used to inform `d` parameter selection.**

- `utils/forecasting/models.py:233-238` always uses `d=1` (first differencing) in the grid. If the series is already stationary, `d=1` is over-differencing; if it has a unit root + trend, `d=1` may be under-differencing.
- The ADF/KPSS results in `diagnostics.py` are displayed AFTER fit — they don't feed back into the model choice.

**M2-5. Random seed for LSTM is fixed at 0 by default but each LSTM uses `tf.random.set_seed(self.seed)` (`models.py:376`), which is process-wide global state. Concurrent LSTM fits in a `ThreadPoolExecutor` will clobber each other's seeds.**

- The orchestrator runs panels in parallel threads (`blurb_orchestra.py`). The LSTM is opt-in, but if enabled, two LSTMs fit concurrently would share/overwrite the global TF seed.
- More importantly: `os.environ["PYTHONHASHSEED"] = str(self.seed)` (`models.py:371`) is mutated AT FIT TIME. Python only reads `PYTHONHASHSEED` at startup; setting it post-import is a no-op. The "deterministic" claim is overstated.

**M2-6. No demographic stratification (sex × age × race) despite ACS providing it; the LFPR analysis aggregates over all demographics.**

- The user-facing claim (`README.md:1-9`) mentions "labor market dynamics" but the demographic axes are unused. ACS B23001 (Labor Force, sex by age) is straightforward to add.
- A serious labor analysis would stratify; this dashboard does not.

**M2-7. The `_apply_threshold` filter in `super_tab.py:230-246` HIDES low-forecast states from the display but they still drive the median / mean computation.**

- The display median is computed BEFORE the threshold filter (`super_tab.py:466`), but states shown to the user are POST-filter (`super_tab.py:478`). The "median forecast across states" text shown in the recommendation card uses the pre-filter median.
- A user setting threshold=50% of median would see only states above that bar, with a "median forecast across states: X" label that reflects ALL states including the dropped ones. Misleading.

---

### M3 — Metadata / reproducibility

**M3-1. Data vintage / fetch date is NOT recorded in any output file.**

- `data/raw/laus/LASST….txt`, `data/raw/ces/SMS….txt`, `data/raw/bea/*`, `data/raw/fred/*` all have schema `series_id,year,period,value` with no `fetch_date`, `revision`, or `vintage` column.
- BLS publishes revised data monthly; running the pipeline today and a year from now produces different numbers for the same (year, period) without any way to detect drift.
- `data/all_data.json` likewise has no metadata. The README claims reproducibility via Docker; this is reproducibility of code, not of data.

**M3-2. No `seasonal` flag is preserved per series in the panel.**

- The CES `bootstrap_state_codes.py` hard-codes `seasonal="S"` (SA). LAUS series IDs in `data/laus_state_codes.json` use the `LASST` prefix (which is the "state" LAUS pattern; the `S` in LASST does not necessarily indicate seasonal adjustment — `LAUST`/`LASBS` patterns exist for non-SA).
- The merged panel labels columns `IA_Manufacturing` without indicating whether the series is seasonally adjusted. A user comparing `IA_Manufacturing` (SA) to a year-over-year change interpretation would not see the SA assumption.

**M3-3. FRED price/income series (FRED MHI, STHPI) are NOMINAL by default; CPI is fetched but never used to deflate them.**

- `utils/fetch_fred_data.py:39-45` `FRED_INDICATORS` includes `PI`, `NGSP`, `MHI`, `STHPI` — all dollar-denominated. The CPI fetcher (`utils/fetch_cpi_data.py`) downloads regional CPI but there is no code path that deflates the FRED nominal series to real terms before they enter the panel.
- The dashboard's "comparing 1996 vs 2024" claim in the CPI fetcher docstring (`fetch_cpi_data.py:7-9`) implies real-terms comparison; the implementation never deflates.

**M3-4. FHFA HPI (STHPI) is fetched as the FHFA All-Transactions index — base year 1980Q1 = 100. No deflator and no documentation of this fact.**

- `utils/fetch_fred_data.py:43` `"STHPI": ("FHFA state house price index", "quarterly", "{st}STHPI")`.
- FHFA's All-Transactions index is purchase + appraisals; the Purchase-Only index would be more appropriate for transaction-volume analysis. No documentation of the choice.

**M3-5. Cache keys do NOT include the data vintage.**

- `utils/embeddings.py:147-161` `_combined_input_hash` hashes the data file contents (good), but the SentenceTransformer model name is NOT in the cache key — if `LOCAL_EMBED_MODEL` changes from `e5-small-v2` to `roberta-base` the cache key stays the same, yielding stale 384-dim vectors for the wrong model.
- `utils/agents/blurb_orchestra.py` `_section_result_cache` keys on `(section, json view_state, model)` — but the panel `view_state` includes float-precision values; tiny changes in `last_actual_value` produce cache misses. There's no rounding before key generation, so the cache hit rate is artificially low.

**M3-6. Forecast `ForecastResult.candidates` lists per-candidate metrics but does NOT save the *predicted values* for each fold — debugging "why did model X win?" requires re-running the bake-off.**

- `utils/forecasting/base.py:69-85` `ForecastResult` carries `metrics` + `diagnostics` but no `fold_predictions` array.
- This is fixable but currently every audit of a winner choice is a re-run.

**M3-7. `OUTPUT_JSON = "data/all_data.json"` is a hard-coded relative path that resolves against the process CWD; tests run from project root work, but importing `utils.data_pipeline` from any other directory writes to that other directory's `data/all_data.json`.**

- `utils/data_pipeline.py:28`.
- Container runtime sets CWD = /app, so the prod path is fine. Researchers running scripts locally must `cd` to the repo root first.

**M3-8. The `Total_Nonfarm` and `Total_Private` supersectors are listed in the ontology + JSON but not in the dashboard's `SUPERSECTORS` enum.**

- `utils/constants.py:90-100` `SUPERSECTORS` enum has 9 entries: omits `Total_Nonfarm`, `Total_Private`, `Leisure_Hospitality`, `Other_Services`.
- The CES JSON has all 13. `super_tab.py:269` consumes `SUPERSECTORS` from constants — so 4 supersectors are FETCHED but UNREACHABLE from the UI. Methodology: code-vs-UI drift.

**M3-9. Citations registry has 20 entries but several are 2018 / 2024 textbook references with no DOI; methodology_panel renders them as bibliography but the `bls_ces_handbook` etc. are URLs that may rot.**

- `utils/citations.py:34-208`. The 2024 BLS handbook entries are useful but not stable identifiers. The Hyndman-Athanasopoulos reference cites the 2nd ed. (2018) but the methodology doc (`docs/methodology/agent_pipeline.md:255`) cites the 3rd ed. (2021) — inconsistent across the codebase.

**M3-10. Forecast invariants (`utils/forecasting/invariants.py:32`) check `point ∈ mean ± 3σ` of history — but "history" is the full series, including the forecast period if the function is called incorrectly.**

- `audit_forecast(history=..., point=..., ...)` (`invariants.py:136-160`) trusts the caller to pass observed history only. There is no guard against passing the full-series array. A documentation comment in `forecasting/invariants.py:13` says it's pure but doesn't warn about history scope.

---

## Top 10 highest-leverage methodology fixes

1. **Fix the ontology-vs-data supersector code mismatch (M0-1).** Either change `Ontology.SUPERSECTORS` codes to match the JSON ("01"/"02"/"03"/…/"90"), or change the format string in `ces_series_id` to put `{ss.code}` at chars 11-12, or — best — make the bootstrap script the single source of truth and regenerate the JSON to match. Add a CI test that builds the bootstrap JSON and asserts it equals the checked-in JSON byte-for-byte. **This is the only finding that breaks code that hasn't been manually shielded.**

2. **Remove the dead `"009": "Population"` mapping in `merge_all_data.py:37` (M0-2)** and add a comment that LAUS does NOT publish per-state population; population comes only from Census PEP/ACS via `read_population` / `read_working_age_population`.

3. **Wire `query: ` / `passage: ` prefixes into e5 embedding (M1-7)** and switch retrieval from lexical Jaccard to cosine against the cached embeddings (M1-8). The current pipeline pays the cost of computing embeddings but throws away their semantic value. Even a 10% retrieval quality improvement compounds across every chat / blurb call.

4. **Add a `seasonal` and `vintage` column to every raw output file (M3-1, M3-2)** and propagate into the merged panel. This is one column per file but enables real downstream auditing.

5. **Replace the uniform `0.78` LFPR correction with the ACS-backed path by default (M0-4)**, falling back to per-region 78%/74%/82% bands rather than the uniform constant. Pre-fetch the WAP files for all 51 jurisdictions during the cold-data refresh so the per-state denominator is the production path, not the fallback.

6. **Run Diebold-Mariano on fold-level OUT-OF-SAMPLE forecast errors (M1-6)**, not on full-series in-sample residuals. Each fold's predictions are already computed by `_fold_metrics`; persisting them in `ForecastResult.fold_predictions` would enable correct DM evaluation.

7. **Add SARIMA to the candidate set (M1-3)** with `seasonal_order=(1,1,1,12)` as a baseline. The non-seasonal ARIMA on monthly data is a methodology smell that an academic reviewer would catch in the first paragraph.

8. **Replace the dashboard's hand-rolled "Region Mean / Region Median" rows (M2-2) with an explicit aggregate-level CES fetch (Supersector code `00` + area `00000`)** for the US total. Adding state-level SA series doesn't yield the BLS-published aggregate.

9. **Fix the `_apply_threshold` ordering in `super_tab.py:230-246` (M2-7)** so the displayed median is recomputed on the post-filter set, or label the displayed median as "across all in-scope states."

10. **Add multiple-testing correction (Bonferroni or BH-FDR) to the requirements pass/fail panel (M1-15)**, displaying both the per-state verdict and the family-wise / FDR-adjusted verdict so users can see the joint risk of false positives.

---

## Claims table

| Source | Claim | Code evidence | Verdict |
|---|---|---|---|
| `README.md:18-19` | "ADF / KPSS / Ljung-Box / Jarque-Bera / Diebold-Mariano (with HLN small-sample correction) vs. Naive baseline" | `utils/forecasting/diagnostics.py:148-212` | ✅ supported (DM-HLN at line 197); ⚠ but DM uses in-sample residuals (M1-6) |
| `README.md:21-23` | "Chat / blurbs: `llama3.2:3b` via `ChatOllama`" | `utils/constants.py:21`, `utils/agents/base.py:53` | ✅ supported |
| `README.md:25` | "Embeddings: `intfloat/e5-small-v2` (in-process PyTorch, no Spark)" | `utils/embeddings.py:78`, `utils/embeddings.py:111` | ⚠ embedded but never used for retrieval (M1-7, M1-8); claim misleads |
| `README.md:202-205` | "Forecast: LSTM RNN with 12-month windows" | `utils/forecasting/models.py:348-407` | ⚠ LSTM is opt-in, not the default (`models.py:434, 461`); README is stale |
| `REPORT.md:90` | "LSTM achieves MAE ~ 0.35 percentage points versus ARIMA's 0.41 pp" | not reproducible — no saved experiment | ❌ unsupported (no fold results saved) |
| `REPORT.md:99-109` | `closest_month` analogue analysis | grep finds nothing | ❌ no implementation (M1-12) |
| `REPORT.md:117-130` | "4 tabs incl. EDA, LFPR Forecast, Sector Forecast, About" | `app.py` / `tabs/*.py` | ✅ supported |
| `REPORT.md:146` | "Iowa's LFPR remains within ±0.8 pp of the region since 2010" | not reproducible — no validation test | ❌ unsupported numeric claim |
| `docs/methodology/lfpr_denominator.md:60` | "residual error of ±2 pp on average" | `tests/utils/test_lfpr_denominator.py:43` asserts 60 ≤ corrected ≤ 75 (loose) | ⚠ partly supported — but ±2 pp claim only valid for ~IA-like states (M0-4) |
| `docs/methodology/lfpr_denominator.md:69-71` | "ACS B23025_001E for every state and year" | `utils/fetch_working_age_population.py:39-42` (only 1-year ACS post-2005, gap 2020) | ⚠ partly — pre-2005 and territories are gaps; linear interp fills (M0-5) |
| `docs/methodology/agent_pipeline.md:138-141` | "rmse_against_baseline: winning model RMSE < naive RMSE — Diebold & Mariano 1995" | `utils/forecasting/invariants.py:110-130` | ⚠ supported as caveat check but not as DM test |
| `docs/methodology/agent_pipeline.md:235-236` | "DM stat / p-value populated when residuals + baseline residuals present" | `utils/forecasting/diagnostics.py:225-238` | ⚠ supported in code but methodologically wrong (M1-6) |
| `tabs/_methodology.py:39-43` | "winning model chosen by minimum out-of-sample RMSE over 3-fold expanding-window backtest" | `utils/forecasting/selection.py:195-285` | ✅ supported |
| `tabs/_methodology.py:51-54` | "ARIMA orders are AIC-selected from {(1,1,1),(2,1,1),(1,1,2),(2,1,2)}" | `utils/forecasting/models.py:233-238` | ✅ supported; ❌ but no seasonal component (M1-3) |
| `tabs/_methodology.py:60-66` | "95% prediction intervals from statsmodels' `get_forecast` for ETS + ARIMA, Brownian residual bootstrap for baselines" | `utils/forecasting/models.py:171-198, 296-316`, `base.py:117-145` | ✅ supported; ⚠ LSTM CI uncalibrated (M1-5) |
| `tabs/_methodology.py:111-124` | "0.78 working-age civilian-noninstitutional correction (US average per BLS Handbook of Methods, ch. 1)" | `utils/constants.py:20`, `utils/merge_all_data.py:258` | ✅ supported value; ⚠ claim of "within ~2 pp of published BLS state LFPR" overstated for high/low-CNI states (M0-4) |
| `README.md:9-14` | "Data sources (multi-source fetch agents): BLS CES / LAUS / Census ACS+PEP / BEA / FRED" | all fetchers exist | ✅ supported |
| `README.md:14` | "FRED — state macro indicators (UR, PI, NGSP, …) via `{ST}{IND}` naming" | `utils/fetch_fred_data.py:39-45` | ⚠ partial — MHI uses different `MEHOINUS{ST}A646N` template, not pure `{ST}{IND}` |
| `tests/utils/test_ontology.py:119-125` | parse_series_id("SMS19000003000000001") returns Manufacturing | `utils/ontology.py:380, 384` look up sid[10:12]="03" which is NOT in supersectors_by_code | ❌ test likely fails or M0-1 hides it |

---

## What's already done well

- **Test coverage is genuinely strong.** `tests/utils/test_forecasting_core.py` uses Hypothesis for property-based testing of the selector. `test_forecasting_statsmodels.py` cleanly skips when optional deps are missing. `test_lfpr_acs_denominator.py` covers both the ACS-path and the 0.78-fallback path with golden arithmetic.
- **The 4-tier forecast bake-off (Naive / Seasonal-Naive / ETS / ARIMA) with deterministic seeds, AIC-based ARIMA grid search, and the option to ship LSTM behind a flag is exactly the right architecture for a portfolio piece.** The decision NOT to default to LSTM on ~300-obs series is correct.
- **`utils/forecasting/invariants.py` is an unusually thoughtful piece of code** — the four invariants are appropriate for the data type, the failure messages are machine-readable, and the design (return `(ok, msg)`) composes well.
- **The Diebold-Mariano implementation in `utils/forecasting/diagnostics.py:148-212` is sound on the test theory** (HAC variance, HLN small-sample correction, t-distribution fallback to normal). The methodological flaw is in WHERE residuals come from (M1-6), not the test code itself.
- **`utils/ontology.py` as a single source of truth is well-designed** — frozen dataclasses, lookup facade, sort-stable accessors. The ONE issue is the supersector code semantics (M0-1).
- **The `LFPR_RAW` column is preserved alongside the corrected `LFPR`** so transparency is maintained for anyone wanting to audit.
- **`utils/forecasting/selection.py:93-117` `_expanding_window_folds` is mathematically clean** with proper error handling for too-short series.
- **`utils/citations.py` is exactly the kind of register a serious research codebase needs.** 20 entries cover all the methodology claims.
- **Sentence-RAG (`utils/agents/sentence_rag.py`) generates prose facts rather than feeding key-value JSON to embeddings** — the design principle is correct (M1-7, M1-8 are implementation gaps, not design flaws).
- **The "data lag" visualization in `tabs/lfp_tab.py:838-876` (gray band for months elapsed but not yet published)** is a genuinely careful piece of UI methodology — most dashboards silently extend forecasts to "today" and conflate data lag with forecast.
- **The methodology panel (`tabs/_methodology.py`)** surfaces references inline with text, with both inline-citation and full-bibliography rendering. Above-average for a portfolio dashboard.
