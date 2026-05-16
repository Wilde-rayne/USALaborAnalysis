# Attribution, Citation & Plagiarism Review

*Target: `reviewable/loving-shtern-800851` at commit 5133949*

## Severity guide
A0 = legal/license risk; A1 = missing required citation; A2 = under-credited / wrong-citation; A3 = polish

## Scope summary

- `LICENSE` present (MIT, 2026 Rayne Wilde) — see A0-1.
- `utils/citations.py` carries 20 hand-curated entries. All 20 papers/handbooks are **real publications** (verified against my training corpus; web access blocked in this environment). Every paper is also actually surfaced — `tabs/_methodology.py` cites all 12 forecasting/diagnostic keys via `FORECAST_NOTES` + `DIAGNOSTIC_NOTES`, and all 9 data-source keys via `DATA_SOURCE_KEYS`. No orphan citations and no obvious hallucinations.
- The big gaps are (a) **software/model attribution** that should exist alongside the academic registry (e5, Sentence-Transformers, statsmodels, deepagents, the llama3.2/phi3 community licenses), (b) **data-source legal posture** for BLS/Census/FRED/BEA/FHFA copyright-and-use language, (c) **AI-assistance disclosure** given that the docs/handoff/* trail makes the multi-day Claude/agent-assisted development unambiguous, and (d) some BLS handbook URLs that don't actually point at the "Handbook of Methods" venue they claim.

## Findings

### A0 — License/legal risk

**A0-1. MIT LICENSE present but no NOTICE / third-party license bundle.** `LICENSE` is a clean MIT (Copyright 2026 Rayne Wilde). However, the project ships zero downstream-license aggregation:
- Bootstrap 5.1.3 (`assets/bootstrap.min.css`) — MIT; the copyright banner is preserved at the top of the minified file (lines 1–5). OK.
- Bundled Python deps (sentence-transformers, transformers, torch, tensorflow, statsmodels, scipy, pandas, numpy, dash, plotly, dash-bootstrap-components, scikit-learn, gunicorn, requests, tenacity, hypothesis, pytest, ruff, langchain, langchain-core, langchain-ollama, deepagents, pyspark, tqdm, python-dotenv, pyyaml) — all permissive (Apache 2.0 / BSD / MIT), but none of their LICENSE files are aggregated in the repo. For an MIT release that **does not redistribute the binaries**, this is fine; if the docker image is ever distributed as a release artifact (and `MILESTONE.md` line 161 / `REPORT.md` line 161 mention `rayne/ds4010:latest` on Docker Hub), Apache-2.0 deps require a `NOTICE` accompanying the binary. **Action**: add a `NOTICE` (or `THIRD_PARTY_LICENSES.md`) listing each dep + license, especially before flipping the Docker image public.
- `pyproject.toml` declares `license = { text = "MIT" }`. Aligns with `LICENSE`. OK.

**A0-2. llama3.2 / phi3 model use vs. their community licenses.** `docker-compose.yml` defaults to `OLLAMA_MODEL=llama3.2:3b` and `OLLAMA_AGENT_MODEL=phi3`. The legacy `OLLAMA_MODEL=llama2:chat` references (still present in `utils/constants.py:21` default, `deploy.sh:14`, `docker-compose.yml` comments l.5 + l.64, `.env.example:52`, `utils/embeddings.py:6`) document an earlier configuration; they should either be (a) deleted as historical, or (b) updated to current. The codebase does **not redistribute model weights** — Ollama pulls them at runtime. That means:
  - **Llama 3.2 Community License** (Meta) applies to the runtime use. The license requires (i) a "Built with Llama" attribution **if the output is offered as a service**, and (ii) inclusion of the license / AUP if the weights are redistributed. The dashboard *is* a service whose UI surfaces LLM output. **Action**: add "Built with Llama" + a link to <https://www.llama.com/llama3_2/license/> to the About tab and to README's License & Acknowledgments section.
  - **Phi-3 license** is MIT (microsoft/Phi-3). No additional attribution required, but it should still appear in the dependency / model-attribution list.
  - No bundled weights → llama2/llama3 redistribution rules do not directly bite, but the service-attribution requirement does.

**A0-3. Inconsistent llama2 reference in `utils/constants.py:21`.** `OLLAMA_MODEL = get_env("OLLAMA_MODEL", "llama2:chat")` is the default if the env var is unset. The docker-compose unconditionally sets `OLLAMA_MODEL=llama3.2:3b`, so the default never fires in container deployment, but a local-run developer that imports `constants.MODEL_NAME` will see `llama2:chat`. Llama 2 has *different* license terms (Llama 2 Community License) — service use requires Meta's 700M-MAU consent clause. Either delete the legacy default or change to `llama3.2:3b` to match production. Same fix needed in `deploy.sh:14` (`OLLAMA_MODELS=("llama2:chat")`) and `.env.example:52`.

**A0-4. CI third-party actions.** `.github/workflows/ci.yml` uses `actions/checkout@v4`, `actions/setup-python@v5`, `actions/upload-artifact@v4`. All three are official GitHub actions under MIT. OK — no attribution required beyond what's already in the workflow.

### A1 — Missing required citations

**A1-1. `intfloat/e5-small-v2` embedding model — Wang et al. 2022 missing.** The model is used in `utils/embeddings.py:78`, `Dockerfile.dashboard:49`, and `tabs/about_tab.py:147`. The correct citation is:

> Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., & Wei, F. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training*. arXiv:2212.03533. <https://arxiv.org/abs/2212.03533>

Add as a new `wang_e5_2022` citation entry and surface in About / Methodology.

**A1-2. Sentence-Transformers library — Reimers & Gurevych 2019 missing.** `sentence-transformers` is used wherever embeddings happen. Correct citation:

> Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019. <https://arxiv.org/abs/1908.10084>

Add as `reimers_gurevych_2019`.

**A1-3. statsmodels — Seabold & Perktold 2010 missing.** ETS (`utils/forecasting/models.py:138`), ARIMA (`models.py:255`), ADF/KPSS (`diagnostics.py:59,82`), Ljung-Box (`diagnostics.py:111`), and the Diebold-Mariano helper all run through `statsmodels`. The library's canonical citation is:

> Seabold, S., & Perktold, J. (2010). *Statsmodels: Econometric and Statistical Modeling with Python*. Proceedings of the 9th Python in Science Conference (SciPy 2010). <https://conference.scipy.org/proceedings/scipy2010/seabold.html>

Add as `seabold_perktold_2010`.

**A1-4. scikit-learn — Pedregosa et al. 2011 missing.** `sklearn.preprocessing.StandardScaler`, `sklearn.linear_model.LinearRegression`, and `sklearn.metrics.confusion_matrix` are exemplar code (REPORT.md §3.3, `docs/agents/chart_library_examples.md` §4–5). Even though current usage is light, the dep is in `requirements.txt:10`. Add:

> Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825–2830.

**A1-5. Plotly / Dash / NumPy / pandas — no software attribution layer at all.** REPORT.md §4 and the README "License & Acknowledgments" name-drop "Dash & Bootstrap" and "TensorFlow & Spark" but cite none. The README's Acknowledgments line "Forecasting built on TensorFlow & Spark" is also factually misleading at the current snapshot — TensorFlow is only used for the opt-in LSTM (`utils/forecasting/models.py:336`), and Spark has been retired (`docker-compose.yml:42-53` documents the removal, and `utils/embeddings.py:5-9` says "The prior Spark `local[*]` implementation hung … dropping the cluster removes the moving part entirely"). Add a software-stack section with citations or DOIs for: pandas (McKinney 2010), NumPy (Harris et al. 2020, *Nature*), Plotly (Plotly Technologies Inc., 2015), Dash, TensorFlow (Abadi et al. 2016), and Tukey 1977 *EDA* (the methodology doc already invokes "Tukey fences" — `utils/forecasting/invariants.py:43`).

**A1-6. AI-assistance disclosure.** `docs/handoff/day_1_resume.md`, `day_5_resume.md`, and `day_5_review_ready.md` document a multi-day Claude-Code-assisted development pass; commit messages on the active branch (e.g. `b9fb9c1 feat(ui): extend interleave + reactivity + view-grounding to EDA + Super tabs (Day-3)`) follow the same cadence. For a portfolio showcase aimed at research-engineer hiring, the README + REPORT should disclose that significant authorship is AI-assisted (per emerging academic and industry norms — e.g., NeurIPS / ICLR 2024 policy, Anthropic's Acceptable Use Policy). Suggested text in README & REPORT acknowledgments:

> *Portions of this codebase were developed with the assistance of Anthropic's Claude (Claude Code). All design decisions, data interpretations, and the final code state are the author's own.*

**A1-7. Iowa State / DS 4010 / instructor / cohort credit.** REPORT.md is labeled "DS 4010 (2025) • Author – Rayne Wilde (kbouwman@iastate.edu)" but no instructor or course context is given. The MILESTONE.md is a verbatim assignment log from the spring 2025 course. If the project will be public-facing, add a one-line acknowledgment of (a) DS 4010 course instructor(s) and (b) any peer reviewers from the cohort, or explicitly state "sole-author solo team Prairie Insights" (which MILESTONE.md does corroborate — "Team Members include Rayne Wilde", "As the sole member of the team…").

### A2 — Under-credited / incorrect citations

**A2-1. `bls_laus_handbook` URL points at the wrong page.** `utils/citations.py:144-151` cites "Local Area Unemployment Statistics — Technical Documentation" with venue "BLS Handbook of Methods" but URL <https://www.bls.gov/lau/laumthd.htm>. That URL is the standalone "Estimation Methodology" page. The actual BLS *Handbook of Methods — LAUS* chapter is at <https://www.bls.gov/opub/hom/lau/home.htm>. Either:
  - keep current URL and change the `venue` from "BLS Handbook of Methods" to "BLS Technical Documentation", or
  - keep `venue="BLS Handbook of Methods"` and change the URL to the `/opub/hom/lau/` URL.

**A2-2. `bls_jolts_handbook` URL is the state-data page, not the handbook.** `utils/citations.py:152-159` says venue "BLS Handbook of Methods" but URL <https://www.bls.gov/jlt/jlt_statedata.htm>. Actual BLS HoM JOLTS chapter is <https://www.bls.gov/opub/hom/jlt/home.htm>. Same fix as A2-1.

**A2-3. `bls_qcew_handbook` URL is the QCEW overview, not the handbook.** `utils/citations.py:160-167` URL <https://www.bls.gov/cew/overview.htm> is the overview page. HoM chapter: <https://www.bls.gov/opub/hom/cew/home.htm>. Same fix.

These three are minor but matter for an auditable "every claim traces to a published methodology" stance — `tabs/_methodology.py:201-219` literally renders the URL as a clickable handbook reference, so a reviewer clicking through gets the wrong document.

**A2-4. README "License & Acknowledgments" is stale and mis-attributes.** README.md:215-220 says:
```
- Data from **BLS API**
- Embeddings via **Sentence-Transformers**
- Forecasting built on **TensorFlow** & **Spark**
- UI built with **Dash** & **Bootstrap**
```
Issues:
  - "Data from BLS API" obscures Census, BEA, FRED, FHFA, JOLTS, QCEW (all loaded by `utils/fetch_*.py`) — under-credits four data sources.
  - "Forecasting built on TensorFlow & Spark" is wrong at this commit: TensorFlow is opt-in LSTM only; Spark has been retired (see A1-5).
  - "Sentence-Transformers" without specifying e5-small-v2 (the actual model in use).

Rewrite to match `utils/citations.py` content. The methodology panel already does this correctly — it's just the README intro that's stale.

**A2-5. `hyndman_athanasopoulos_2018` cites the 2nd edition, but `docs/methodology/agent_pipeline.md:255-256` cites the 3rd ed. (2021).** Same authors and book, different edition. Either:
  - keep the 2nd-ed citation and revise the agent_pipeline.md reference to match (the 2nd ed. has equivalent §5.5 and §4.5 sections), or
  - update the citation registry to the 3rd-ed (2021) and bump the linked OTexts URL to `https://otexts.com/fpp3/`. The 3rd ed. is the maintained reference now; bumping is cleaner.

**A2-6. LangChain / DeepAgents not attributed.** `utils/agents/base.py`, `utils/agents/deep.py`, `utils/agents/tools.py`, `utils/agents/blurb.py`, and `utils/llm_utils.py` all rely on LangChain (`langchain_ollama.ChatOllama`, `langchain_core.tools.tool`, `langchain_core.prompts.ChatPromptTemplate`) and DeepAgents (`deepagents.create_deep_agent`). LangChain is MIT-licensed (langchain-ai/langchain); DeepAgents is MIT (langchain-ai/deepagents). Add a one-line credit in the README + About tab. These don't have canonical academic citations but should at least appear in the dependency-attribution layer.

**A2-7. ARIMA citation is correct but incomplete.** `tabs/_methodology.py:54` cites Box & Jenkins 1970 for the ARIMA grid (1,1,1)…(2,1,2). The classical Box-Jenkins reference is correct. However, the implementation uses statsmodels' `ARIMA(...).fit()` which is its own contribution and AIC-grid search descends from Akaike 1974 ("A new look at the statistical model identification", IEEE T Automatic Control 19(6), 716–723). Add Akaike 1974 if you want to be exhaustive about the AIC criterion.

**A2-8. Forecasting CI methodology uses parametric bootstrap (ETS `.simulate()`).** `utils/forecasting/models.py:171-198` runs 1000 simulated paths through statsmodels' state-space simulate and takes empirical quantiles. The technique descends from Efron 1979 ("Bootstrap methods: another look at the jackknife", Ann. Statist. 7(1), 1–26) and is described in Hyndman & Athanasopoulos §11.5 for ETS. Citing only `hyndman_athanasopoulos_2018` (via `tabs/_methodology.py:65`) is enough for an undergraduate audience but a research-engineer audience expects Efron 1979 too.

### A3 — Polish

**A3-1. `LFPR_DENOMINATOR_NOTE` markdown link is a dead anchor.** `tabs/_methodology.py:121-123` renders the LFPR caveat with `[docs/methodology/lfpr_denominator.md](https://github.com/)` — the href is literally `https://github.com/`. Likely meant to point at the file on the public repo once it's published. Fix to the actual GitHub URL or strip the link entirely.

**A3-2. REPORT.md figures reference paths that don't exist.** REPORT.md:136, :139 reference `img/IA_Unemployment_Rate.png` and `img/IA_Labor_Force_Participation_Rate.png`. There is no `img/` directory in the repo (verified by `Glob`). Either add the images (and credit the generator — "Author-generated using Plotly, June 2025" or similar), or remove the broken figure refs. Not an attribution issue, but a credibility issue for an auditable report.

**A3-3. `_chunks_from_records` description is outdated.** `utils/embeddings.py:209` calls `SentenceRAGBuilder.build_corpus(records)` which now produces three layers (facts + rankings + trends). The comment block "Layer 3 — optional LLM polish (phi3 by default)" still talks about phi3 polishing on the default path. In `utils/agents/sentence_rag.py:262-277`, `agent_polish` requires an attached `agent` and is off by default — comment in `utils/embeddings.py:212` correctly notes `polish=False`, but the prior sentence's "polish=True flag routes each sentence through the worker agent (phi3 by default)" is true only if a phi3 agent is explicitly attached. Tighten the wording.

**A3-4. Tashman 2000 venue text "International Journal of Forecasting" — confirm volume.** Volume 16, issue 4, pages 437–450. The citation entry says "16(4), 437–450" which matches. OK.

**A3-5. KPSS 1992 venue says "Journal of Econometrics, 54(1–3), 159–178" — confirm pagination.** The correct page range is 159–178, and the issue spans 1–3 of vol. 54. Cited correctly. OK.

**A3-6. The README claims "20+ references" in `tabs/_methodology.py` docstring (l.13), but the registry currently holds exactly 20.** Either change the docstring to "20 references" or grow the registry to 21+ (which the A1 additions would do).

**A3-7. The reviewer / holistic-auditor / tile-reviewer pattern (`utils/agents/reviewer.py`, `utils/agents/blurb_orchestra.py`) has obvious parentage in Reflexion (Shinn et al. 2023) and Society of Mind (Minsky 1986) but no inline citation.** The architecture diagram in `docs/methodology/agent_pipeline.md:11-20` and `docs/handoff/day_5_review_ready.md:48-58` would be more credible with explicit acknowledgment of antecedents:

> Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., & Yao, S. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023. <https://arxiv.org/abs/2303.11366>
>
> Minsky, M. (1986). *The Society of Mind*. Simon & Schuster.
>
> (Optional) Yao, S., et al. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*. arXiv:2305.10601.

This is a research-engineer-portfolio piece; not citing the obvious antecedents reads as either unaware or unwilling-to-acknowledge.

**A3-8. Brownian residual-bootstrap CI for baselines.** `tabs/_methodology.py:64-65` says "or from a Brownian residual-bootstrap band for the baselines". The technique is the standard Hyndman & Athanasopoulos §3.5 residual-bootstrap, not strictly "Brownian". If you want to be precise, cite or rephrase.

**A3-9. JOLTS state-data series IDs are labeled "experimental" by BLS.** `utils/fetch_jolts_data.py:13-16` documents this caveat in the docstring; the data-source citation `bls_jolts_handbook` does NOT note the experimental status. For a published-data audit, the citation note should reflect that the state-level series are still designated experimental by BLS.

**A3-10. Bootstrap MIT banner preserved in `assets/bootstrap.min.css`.** Lines 1-5 carry the upstream copyright + "Licensed under MIT" notice. OK, no action.

**A3-11. `prairie.css` (`assets/prairie.css`) is author-written from comments and structure.** No external attribution required. OK.

## Citation registry audit

Verified against my training-corpus knowledge of canonical references; **web access blocked** in this environment, so URLs / DOIs were not live-checked but are syntactically and semantically consistent with known publications.

| Key | Cited as | Real paper? | Method in code? | Verdict | Fix |
|---|---|---|---|---|---|
| `hyndman_athanasopoulos_2018` | Forecasting: Principles and Practice (2nd ed., OTexts) | Yes (real, OTexts ed.) | Yes — overall forecasting framework (`tabs/_methodology.py:42,48,65`, agent_pipeline.md) | OK with caveat | Methodology doc cites 3rd ed. 2021; either align registry to 3rd ed. + `https://otexts.com/fpp3/` or update methodology doc to 2nd ed. (A2-5) |
| `holt_1957` | Holt 1957 ONR Memorandum 52 | Yes (real — also reprinted in IJF 2004) | Yes — Holt-Winters in `models.py` ETSForecaster | OK | None |
| `winters_1960` | Winters 1960 Management Science 6(3) | Yes — DOI valid | Yes — Holt-Winters seasonal in `ETSForecaster` | OK | None |
| `box_jenkins_1970` | Box & Jenkins 1970 Holden-Day | Yes — canonical | Yes — ARIMA grid in `ARIMAForecaster` | OK | Optionally add Akaike 1974 for AIC selection (A2-7) |
| `hochreiter_schmidhuber_1997` | LSTM, Neural Computation 9(8) | Yes — DOI valid | Yes — opt-in LSTM in `LSTMForecaster` and REPORT.md §3.1 | OK | None |
| `dickey_fuller_1979` | Dickey-Fuller, JASA 74(366) | Yes — DOI valid | Yes — `diagnostics.adf_pvalue` | OK | None |
| `kpss_1992` | KPSS, J. Econometrics 54(1-3) | Yes — DOI valid | Yes — `diagnostics.kpss_pvalue` | OK | None |
| `ljung_box_1978` | Ljung-Box, Biometrika 65(2) | Yes — DOI valid | Yes — `diagnostics.ljungbox_pvalue` | OK | None |
| `jarque_bera_1980` | Jarque-Bera, Economics Letters 6(3) | Yes — DOI valid | Yes — `diagnostics.jarquebera_pvalue` | OK | None |
| `diebold_mariano_1995` | Diebold-Mariano, JBES 13(3) | Yes — DOI valid | Yes — `diagnostics.diebold_mariano` + `invariants.rmse_against_baseline` | OK | None |
| `harvey_leybourne_newbold_1997` | HLN small-sample correction, IJF 13(2) | Yes — DOI valid | Yes — applied in `diagnostics.diebold_mariano:196-201` | OK | None |
| `tashman_2000` | Out-of-sample tests, IJF 16(4) | Yes — DOI valid | Yes — referenced for expanding-window backtest in `_methodology.py:42` | OK | None |
| `bls_ces_handbook` | BLS HoM CES | Yes — real BLS chapter | Yes — `fetch_ces_data.py` | OK | URL correctly points at HoM |
| `bls_laus_handbook` | BLS HoM LAUS | Yes — real BLS chapter | Yes — `fetch_laus_data.py` | venue/URL mismatch | Either update URL to `/opub/hom/lau/home.htm` or change venue from "Handbook of Methods" to "Technical Documentation" (A2-1) |
| `bls_jolts_handbook` | BLS HoM JOLTS | Yes — real BLS chapter | Yes — `fetch_jolts_data.py` | venue/URL mismatch | Same as A2-1 — update URL or change venue (A2-2) |
| `bls_qcew_handbook` | BLS HoM QCEW | Yes — real BLS chapter | Yes — `fetch_qcew_data.py` | venue/URL mismatch | Same fix (A2-3) |
| `bls_cpi_handbook` | BLS HoM CPI | Yes — real BLS chapter | Yes — `fetch_cpi_data.py` | OK | None |
| `census_acs_handbook` | Census ACS Design & Methodology | Yes — real Census document | Yes — `fetch_working_age_population.py`, `fetch_population_data.py` | OK | None |
| `bea_regional_handbook` | BEA Regional Methodology | Yes — real BEA document | Yes — `fetch_bea_data.py` | OK | None |
| `fred_api` | FRED API | Yes — real Fed reference | Yes — `fetch_fred_data.py` | OK | Citation should ideally also cover FRED's series-level "Suggested Citation" — see License posture |
| `fhfa_hpi_handbook` | FHFA HPI Technical Description | Yes — real FHFA document | Yes — FHFA HPI via FRED in `fetch_fred_data.py` (`STHPI` indicator) | OK | None |

**Summary: 20/20 entries verify as real. 3/20 have a URL-vs-venue mismatch (LAUS, JOLTS, QCEW point at product/technical pages while claiming the "Handbook of Methods" venue). No hallucinated entries.**

## Citation recipe (paste-ready additions)

Append to `_CITATIONS_LIST` in `utils/citations.py`:

```python
    # ----- Software / model attribution ---------------------------------
    Citation(
        key="reimers_gurevych_2019",
        authors="Reimers, N., and Gurevych, I.",
        year=2019,
        title="Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        venue="Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
        url="https://arxiv.org/abs/1908.10084",
    ),
    Citation(
        key="wang_e5_2022",
        authors="Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., and Wei, F.",
        year=2022,
        title="Text Embeddings by Weakly-Supervised Contrastive Pre-training",
        venue="arXiv:2212.03533",
        url="https://arxiv.org/abs/2212.03533",
    ),
    Citation(
        key="seabold_perktold_2010",
        authors="Seabold, S., and Perktold, J.",
        year=2010,
        title="Statsmodels: Econometric and Statistical Modeling with Python",
        venue="Proceedings of the 9th Python in Science Conference",
        url="https://conference.scipy.org/proceedings/scipy2010/seabold.html",
    ),
    Citation(
        key="pedregosa_sklearn_2011",
        authors="Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E.",
        year=2011,
        title="Scikit-learn: Machine Learning in Python",
        venue="Journal of Machine Learning Research, 12, 2825–2830",
        url="https://www.jmlr.org/papers/v12/pedregosa11a.html",
    ),
    Citation(
        key="harris_numpy_2020",
        authors="Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., et al.",
        year=2020,
        title="Array programming with NumPy",
        venue="Nature, 585(7825), 357–362",
        url="https://doi.org/10.1038/s41586-020-2649-2",
    ),
    Citation(
        key="mckinney_pandas_2010",
        authors="McKinney, W.",
        year=2010,
        title="Data Structures for Statistical Computing in Python",
        venue="Proceedings of the 9th Python in Science Conference, 56–61",
        url="https://conference.scipy.org/proceedings/scipy2010/mckinney.html",
    ),
    Citation(
        key="abadi_tensorflow_2016",
        authors="Abadi, M., Barham, P., Chen, J., et al.",
        year=2016,
        title="TensorFlow: A System for Large-Scale Machine Learning",
        venue="12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)",
        url="https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi",
    ),
    Citation(
        key="tukey_1977",
        authors="Tukey, J. W.",
        year=1977,
        title="Exploratory Data Analysis",
        venue="Addison-Wesley",
    ),
    Citation(
        key="akaike_1974",
        authors="Akaike, H.",
        year=1974,
        title="A new look at the statistical model identification",
        venue="IEEE Transactions on Automatic Control, 19(6), 716–723",
        url="https://doi.org/10.1109/TAC.1974.1100705",
    ),
    Citation(
        key="efron_1979",
        authors="Efron, B.",
        year=1979,
        title="Bootstrap Methods: Another Look at the Jackknife",
        venue="The Annals of Statistics, 7(1), 1–26",
        url="https://doi.org/10.1214/aos/1176344552",
    ),
    # ----- Agent architecture antecedents -------------------------------
    Citation(
        key="shinn_reflexion_2023",
        authors="Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., and Yao, S.",
        year=2023,
        title="Reflexion: Language Agents with Verbal Reinforcement Learning",
        venue="NeurIPS 2023",
        url="https://arxiv.org/abs/2303.11366",
    ),
    Citation(
        key="minsky_1986",
        authors="Minsky, M.",
        year=1986,
        title="The Society of Mind",
        venue="Simon & Schuster",
    ),
    # ----- Model licenses (non-academic but worth surfacing) ------------
    Citation(
        key="llama3_license_2024",
        authors="Meta Platforms, Inc.",
        year=2024,
        title="Llama 3.2 Community License Agreement",
        venue="Meta",
        url="https://www.llama.com/llama3_2/license/",
    ),
)
```

Also add `wang_e5_2022`, `reimers_gurevych_2019`, and `llama3_license_2024` to a new `SOFTWARE_ATTRIBUTION_KEYS` tuple in `tabs/_methodology.py` and render them alongside the data-source list.

## License posture recommendation

1. **Keep MIT for the code.** It is compatible with every Python dep listed in `requirements.txt`.
2. **Add a `NOTICE` (or `THIRD_PARTY_LICENSES.md`)** before any Docker Hub / public-release flip. List every Python dep with its license. The Apache-2.0 deps (tensorflow, plotly, langchain, langchain-ollama, langchain-core, deepagents) all require the NOTICE clause when redistributed in binary form.
3. **Llama 3.2 Community License attribution.** Add "Built with Llama" + the license link to the About tab and README acknowledgments. This is a hard requirement for any service whose output is offered to users via a Llama-family model. The `OLLAMA_MODEL` env-default in `utils/constants.py:21` should be bumped from `llama2:chat` to `llama3.2:3b` for consistency with docker-compose; either remove `llama2:chat` from the codebase entirely or document why both eras coexist.
4. **BLS / Census / BEA / FRED / FHFA — public-domain, but attribution per their citation policies is expected.**
   - **BLS**: 17 USC §105 puts BLS publications in the public domain; the BLS Linking and Copyright Information page asks for source attribution ("Source: U.S. Bureau of Labor Statistics") on derived content. The README + About tab name BLS but don't include the canonical "Source:" line. Add to README + About.
   - **Census**: <https://www.census.gov/data/developers/about/terms-of-service.html> — attribution required as "Source: U.S. Census Bureau, [survey name and year]". Methodology panel includes a Census link; add "Source: U.S. Census Bureau" framing on any rendered figure (Plotly fig.update_layout titles or footers).
   - **FRED**: St. Louis Fed's *Citation Requirements* (<https://research.stlouisfed.org/docs/api/terms_of_use.html>) require: "FRED® data, Federal Reserve Bank of St. Louis" plus the series ID. Currently absent.
   - **BEA**: <https://www.bea.gov/help/guidelines-for-citing-bea> requires "Source: U.S. Bureau of Economic Analysis" + the table number. The fetcher's docstring records SAINC1 line 3 but the dashboard does not surface this in the rendered chart.
   - **FHFA HPI**: <https://www.fhfa.gov/data/hpi> — public-domain but FHFA requests attribution + the series version (e.g. "FHFA House Price Index, all-transactions, NSA, [vintage]").
5. **Bootstrap CSS** — MIT attribution is preserved in the file banner. No further action needed.
6. **Sentence-Transformers / e5-small-v2** — Both are Apache-2.0; their NOTICE files cover the requirement. The model itself isn't redistributed in the repo; HuggingFace pulls it at build time (`Dockerfile.dashboard:48-49`). The model card requires citation of Wang et al. 2022 (A1-1) when used in research outputs.
7. **DeepAgents / LangChain** — MIT. Mention in dependency-attribution layer (A2-6) but no special posture required.
8. **AI-assistance disclosure (A1-6)** — strongly recommended for the portfolio framing. Modern academic publication norms (NeurIPS, ICLR, ACL) require explicit AI-assistance disclosure; for hiring portfolios this signals professional norms-awareness.

**Bottom line: license itself is clean. The gap is everything around it — NOTICE, model attribution, data-source attribution per each agency's citation policy, and AI-assistance disclosure. None of these is a legal blocker, but together they're the difference between "MIT released, license compliant" and "publishable in a research-engineer portfolio".**
