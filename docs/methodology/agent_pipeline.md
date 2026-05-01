# Agent pipeline methodology

This document records the architecture, prompt-engineering choices,
statistical-rigor invariants, and testing notes for the dashboard's
nested-harness AI pipeline. It complements the chart-library examples
in `docs/agents/chart_library_examples.md` and the LFPR-denominator
methodology in `docs/methodology/lfpr_denominator.md`.

## Architecture (current — landed `340930a`)

```
base ──→ total context ──→ page context
                                ├──→ specialist (parallel) → strip_preamble → quick_check (free) → ship
                                │           ↓ optional (off by default)
                                │       LLM reviewer → revise → tile reviewer
                                │           ↓ optional (off by default)
                                │       holistic auditor → targeted revise
                                │
                                └──→ image writer / FigureSpec      (stub — next slice)
```

Every layer is a transient, short-lived agent: created at click time,
invoked, and discarded. State that persists across clicks lives in
the orchestrator's per-process result cache (key = `(section,
JSON view_state, model)`) and in the panel `view_state` dicts the
tabs hand to `BlurbOrchestrator.run_pipeline`.

## Layer responsibilities

### Specialist agents (`utils/agents/blurb_orchestra._make_specialist`)

One per panel section. Each carries a section-specific system
prompt (see `_SECTION_PROMPTS`) — `forecast`, `requirements`, `trend`,
`stats`, `timeseries`, `volatility`, `recommendation`, `models`,
`recap`. The system prompt fixes the agent's role; the user prompt
provides the panel facts.

All specialists use the same model (`OLLAMA_MODEL`,
default `llama3.2:3b`). The earlier two-model split (phi3 panels +
llama3.2:3b recap) tripped the "llama runner process has terminated"
OOM under the 7.5 GB container; one model = one runner = no
second-model load to fail.

### `strip_preamble` (`utils/agents/reviewer.py`)

Pure-Python — runs after every specialist call. Removes leading
clauses like "Based on the provided facts,", "Here is a 3-sentence
executive summary:", "Panel: <title>", "Headline Finding:". Loops up
to 3 passes for nested preambles. Refuses to cut at decimal points
inside numbers (an earlier draft ate "remain stable at 66.3%" by
breaking at the period inside `66.3`).

### `ReviewerAgent.quick_check`

Pure-Python heuristic. Catches the most common failure modes (empty
output, suspiciously short < 80 chars, known preamble markers, error
sentinels from `LaborAgent.invoke`'s exception handler, fewer than
3 numeric digits — likely not grounded). No LLM call. Runs on every
candidate.

### `ReviewerAgent.evaluate` *(off by default)*

LLM critic. Two-line response: `PASS|REVISE` + one-sentence reason.
Only fires when `quick_check` flags the candidate AND
`enable_llm_review=True`. Adds ~30 s per failed candidate; gated
behind the flag so the reactive default path isn't burdened.

### `TileReviewer` *(stub — activates with image-writer slice)*

Audits a `(figure spec, blurb)` tile pair for cross-modal
consistency. Will catch chart-text contradictions ("the narrative
says LFPR is rising but the chart shows a decline"). Currently a
no-op pass because `ImageWriterAgent.write_spec` returns `None`;
auto-engages when the figure-builder lands.

### `HolisticReviewer` *(off by default)*

Cross-section auditor. Reads every panel narrative + the recap
together. Issues `REVISE_<section>` directives when:
- the recap fails to weave panels into a synthesis
- panels contradict each other
- a non-obvious cross-panel insight is missing

Followed by `_holistic_revise` for one targeted rewrite. Adds 1–2
LLM calls per click; gated behind `enable_holistic_audit=True`.

## Prompt-engineering notes

### Why the system prompt is short and section-specific

We want the model's first sentence to land directly on the analysis,
not on a "Sure, here's an explanation" preamble. Long generic system
prompts encourage exposition; short role-setting prompts produce
crisper output:

> "You are a labor-market forecast analyst. In 2 sentences, interpret
> the panel facts in terms of forward expectations and model
> uncertainty (CI width, RMSE, peer comparison). Cite the specific
> numbers; do not invent any."

### Why we render facts as ontology-aware sentences instead of JSON

The user's explicit guidance: "we don't want to feed raw numbers
… we want it turned into ontology-aware embeddings the AI can
learn on". `SentenceRAGBuilder.render_view_sentences` produces prose
like:

> "Iowa's last published labor force participation rate was 64.5 %
> in 2024."

instead of:

> `{"state": "IA", "measure": "LFPR", "year": 2024, "value": 64.5}`

The model parrots the input format — prose facts produce prose
analysis; key/value JSON produces key/value listings.

### Why the recap waits for reviewed panels

`BlurbOrchestrator.run_pipeline` runs panels in parallel via
`as_completed`; the recap only fires once every panel has settled.
The recap specialist's prompt then receives the *reviewed* panel
narratives as additional facts (via the `reviewed_panels` field on
the recap view_state), so the synthesis weaves the corrected
sentences rather than re-deriving from raw view_state data.

## Statistical-rigor invariants

`utils/forecasting/invariants.py` carries the academic-rigor
checks referenced in every forecast panel's grounding facts.

### Invariants

| Function | Check | Citation |
|---|---|---|
| `point_within_historical_envelope` | forecast ∈ mean ± 3σ of history | Tukey 1977 (extreme-outlier fences) |
| `ci_contains_point` | model's 95 % PI brackets its own point estimate | Hyndman & Athanasopoulos 2021 §5.5 |
| `ci_width_reasonable` | CI width ≤ 8σ of history | Hyndman & Athanasopoulos 2021 §4.5 |
| `rmse_against_baseline` | winning model RMSE < naive RMSE | Diebold & Mariano 1995 |

### How invariants reach the AI

`_invariant_failures_for(view_state)` runs at the top of
`BlurbOrchestrator._review_and_revise`. Failures are appended to
the facts block under a "Statistical caveats" header so the
specialist's prompt looks like:

```
Topic: IA Labor Force Participation Rate forecast (starts 2027)

Facts:
- The Iowa labor force participation rate forecast covers 2027
  using the ARIMA model selected from a Naive / Seasonal-Naive /
  Holt-Winters / ARIMA bake-off (out-of-sample RMSE 1.00).
- The +2-year point forecast is 66.3 %, with a 95 % confidence
  interval of 64.0 %–68.6 %.
- Historical context: Iowa LFPR averaged 65.4 % across the
  observed window with σ 1.4 %; deviations larger than ~2σ are
  noteworthy.
- Peer-state median forecast at the same horizon is 65.9 %;
  Iowa's point estimate is above the peer median by 0.4 %.

Statistical caveats (cite these — the model uncertainty is genuine):
- 95 % prediction interval is 4.6 (7% of the last observed
  value) — model uncertainty is moderate.

Write the analysis directly — no header line, no "Panel:" prefix,
no "Headline:" prefix. Cite the numbers above; do not invent any.
```

### Future invariants (next slice)

- **Diebold-Mariano significance** on the winning-vs-naive comparison
  (we have the RMSE check; the formal test gives a p-value).
- **Ljung-Box** residual autocorrelation (does the model leave
  meaningful structure in residuals?).
- **Jarque-Bera** residual normality (relevant to the parametric CI
  width).

## Performance notes

### Why specialists run in parallel via threads, not Spark

The orchestrator's `ThreadPoolExecutor(max_workers=4)` dispatches
section specialists in parallel. We tried PySpark earlier in the
project; the pickling overhead dominated the per-task work for
≤10 LLM calls. Threads + `as_completed` give us the same parallelism
shape with zero serialization cost.

Spark is the right answer when blurb generation needs to fan across
multiple machines (production scale). At single-host scale it
actively hurts.

### Why specialist timeout is 120 s

`SPECIALIST_TIMEOUT_SECONDS=120` in `blurb_orchestra.py`. The
env-driven `OLLAMA_TIMEOUT` (default 600 s, used by the chat drawer)
is generous because a chat user can wait. The orchestrator's polling
callback updates the page every ~2 s; if a single specialist hangs
for >120 s on a single CPU it's almost certainly stuck behind a
queue, and the user is better served by `AI_FAILURE_MESSAGE` showing
in that one panel than by the whole runner blocking on a hung future.

### Why we cache by (section, JSON view_state, model)

Re-clicks of the same Run-Forecast button (same focus state, same
peers, same horizon, same threshold) produce identical view_states.
The orchestrator's `_section_result_cache` returns the prior text
in <10 ms instead of paying another ~30 s LLM round-trip. Cache
size is bounded at `_CACHE_MAX_ENTRIES=256`; insertion order is the
proxy for staleness when we need to evict.

## Testing notes

### What's covered

- `utils/forecasting/invariants.py` — atomic invariants tested in
  the module's `__main__` smoke (envelope / CI / RMSE caveats).
- `utils/agents/reviewer.strip_preamble` — 5 hand-curated cases
  from real model outputs (`Based on the provided facts,`,
  `Here is a 3-sentence executive summary:`, `Panel: …`, plus a
  no-preamble negative case).
- `utils/agents/sentence_rag` — view_state → sentence rendering,
  smoke-tested in `tests/utils/test_sentence_rag.py` (existing).

### What's not yet covered

- `BlurbOrchestrator.run_pipeline` end-to-end — there's no test
  that exercises the parallel-specialist + recap-with-reviewed-
  panels flow against a stubbed `LaborAgent`. Worth adding before
  the image-writer slice activates more LLM round-trips.
- Cross-tab pi-active-view propagation — covered by the chat
  drawer's view-state ingestion but no explicit test.

### Verification cadence

The handoff doc (`docs/handoff/day_5_review_ready.md`) records the
last preview-tool timing pass:

| Tab | Body+chart | All 4 panels |
|---|---|---|
| LFP Forecast | 6.4 s | 87 s |
| EDA / Overview | 7.8 s | 61 s |
| Supersector Forecast | 7.2 s | 85.5 s |

After today's caching + enrichment slice, the timing target on a
re-click of identical params is **<5 s for body+chart and <2 s for
all blurbs (cache hit)**, with first-click times at or below the
above numbers. The screenshot pass in this slice's verification step
records the new timings + visual quality.

## References

- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting:
  Principles and Practice* (3rd ed.). OTexts.
- Diebold, F. X. & Mariano, R. S. (1995). "Comparing Predictive
  Accuracy." *Journal of Business & Economic Statistics*, 13(3),
  253–263.
- Ljung, G. M. & Box, G. E. P. (1978). "On a measure of lack of
  fit in time series models." *Biometrika*, 65(2), 297–303.
- Jarque, C. M. & Bera, A. K. (1980). "Efficient tests for
  normality, homoscedasticity and serial independence of regression
  residuals." *Economics Letters*, 6(3), 255–259.
