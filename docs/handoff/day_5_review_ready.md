# Day 5 — preview-verified, awaiting human review

> **Status**: Branch `claude/reverent-noyce-a643f3` @ `340930a`. Tree
> clean. All three tabs verified in preview to render full
> orchestrator output (chart + 3 figure tiles + recap + progress
> strip + AI prose grounded in the actual numbers) within 60–90 s.
> User is doing visual review next.

## TL;DR

| Tab | Body+chart | All 4 panels | Status |
|---|---|---|---|
| **LFP Forecast** | 6.4 s | **87 s** | "All 4 panel(s) ready." |
| **EDA / Overview** | 7.8 s | **61 s** | "All 4 panel(s) ready." |
| **Supersector Forecast** | 7.2 s | **85.5 s** | "All 4 panel(s) ready." |

Every panel filled with real prose citing rendered numbers (66.3 %
point forecast, 95 % CI 63.4–69.2 %, ARIMA model, IL / MN peer rates,
Mining and Logging +2-yr forecast, North Dakota leader). Phase-J
chrome on LFP (`STARTS 2027`, range selector, hatched data-lag band)
all rendering correctly.

## What landed since the previous handoff

`340930a` is the reactive-defaults pass that fixed two blockers
surfaced during my preview test:

1. **LLM-review + holistic-audit overhead** — `b4982c5`'s nested
   pipeline was firing 9–14 LLM calls per click. On single-CPU
   Ollama that exceeded the 600 s timeout on later panels and
   produced a 7-minute hang with zero panels rendered. Default-
   disabled both layers via new `enable_llm_review` /
   `enable_holistic_audit` constructor flags. The free heuristic
   ``ReviewerAgent.quick_check`` still runs.
2. **Phi3 runner OOM** — `OLLAMA_MAX_LOADED_MODELS=2` tried to keep
   phi3 + llama3.2:3b resident together; the 7.5 GB container
   couldn't hold both runners. Switched every specialist to
   `llama3.2:3b` uniform; dropped MAX_LOADED to 1.
3. **Preamble noise** — added `strip_preamble` in
   `utils/agents/reviewer.py`, plus changed the orchestrator's
   prompt template to use `Topic:` instead of `Panel: <title>` and
   explicitly tell the model "no header line, no Panel: prefix, no
   Headline: prefix". Loops up to 3 passes for nested preambles
   ("Panel: …\nHeadline Finding: …"). Refuses to cut at decimal
   points inside numbers.

## Architecture (reactive defaults; rigorous-mode flags reserved)

```
base ──→ total context ──→ page context
                                ├──→ specialist (parallel) → quick_check (free) → ship
                                │           ↓ optional (off by default)
                                │       LLM reviewer → revise → tile reviewer
                                │           ↓ optional (off by default)
                                │       holistic auditor → targeted revise
                                │
                                └──→ image writer / FigureSpec      (stub — next slice)
```

Reactive defaults shipped in `340930a`:

- `enable_llm_review = False` — only the free heuristic fires
- `enable_holistic_audit = False`
- `max_retries = 0` — no revision loop on the hot path
- `SPECIALIST_TIMEOUT_SECONDS = 120` — fail-fast per call
- `OLLAMA_MAX_LOADED_MODELS = 1`
- All specialists on `OLLAMA_MODEL` (llama3.2:3b)

The deeper review trio (per-section LLM critic, tile auditor,
holistic auditor + targeted revise) stays in code, gated behind
`enable_*=True` for offline / batch / "rigorous mode" callers
where wall time matters less than rigour.

## Known quality gaps (likely to surface in human review)

1. **Hallucinated threshold on requirements panel** — clicking LFP
   without a threshold occasionally produces "all three states meet
   or exceed the policy threshold of 65 %" (the 65 % is invented).
   Fix path: enable the per-section LLM reviewer for the
   requirements section, or surface a "rigorous mode" toggle.
2. **`Panel:` echo** — the strip catches every variant we've seen,
   but if a fresh one shows up, add the marker to
   `utils/agents/reviewer.py:_PREAMBLE_MARKERS`.

## File index for this slice

| File | Purpose |
|---|---|
| `utils/agents/blurb_orchestra.py` | `BlurbOrchestrator` (parallel specialists + holistic + recap), `_review_and_revise`, `_make_specialist`, section-specific system prompts |
| `utils/agents/blurb_async.py` | Daemon-thread runner, calls `BlurbOrchestrator.run_pipeline` |
| `utils/agents/reviewer.py` | `ReviewerAgent` (heuristic+LLM), `TileReviewer`, `HolisticReviewer`, `strip_preamble` + `_PREAMBLE_MARKERS` |
| `utils/agents/image_writer.py` | `ImageWriterAgent` skeleton + `FigureSpec` dataclass — next slice activates |
| `utils/forecasting/invariants.py` | `audit_forecast` + envelope / CI / RMSE-vs-baseline checks |
| `docs/agents/chart_library_examples.md` | Seed corpus for the image-writer (Plotly / matplotlib / seaborn / sklearn / keras templates) |
| `docs/handoff/day_5_resume.md` | Earlier resume doc (Docker died mid-verify) |
| `docs/handoff/day_5_review_ready.md` | This doc — preview-verified state |

## Recovery if Docker dies again

The `dockerInference` socket bug keeps recurring. Same fix as
prior sessions::

```powershell
Get-Process | Where-Object { $_.Name -like "*docker*" -or
    $_.Name -like "com.docker*" } | Stop-Process -Force
$stamp = Get-Date -Format yyyyMMddHHmmss
Rename-Item "$env:LOCALAPPDATA\Docker\run" "run.stuck-$stamp"
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

After Docker recovers, bring stack back up::

```bash
cd "C:/Users/wilde/Documents/gits/USALaborAnalysis/.claude/worktrees/reverent-noyce-a643f3"
docker compose --env-file ../../../.env down
# Then in the agent: mcp__Claude_Preview__preview_start { name: "dashboard" }
until [ "$(docker inspect -f '{{.State.Health.Status}}' \
  reverent-noyce-a643f3-dashboard-1 2>/dev/null)" = "healthy" ]; do
  sleep 3
done
```

The image is already built with every commit through `340930a` —
no `--build` flag needed.

## Live target

`http://localhost:8050/` — Docker is up, preview server running,
container healthy. Defaults to Iowa focus / IL+MN peers / LFPR /
+2 yrs / no threshold.

## Next slice (after human review approves)

1. **Wire `audit_forecast` failures** into the reviewer's critique
   prompt so the AI acknowledges model uncertainty when invariants
   flag a forecast (currently the invariants module is built and
   tested but not consumed by the runtime path).
2. **Activate `ImageWriterAgent.write_spec`** — implement the LLM
   round-trip + JSON parse + validation against `FigureSpec`, then
   add `tabs/_charts.py` builder so tabs call the writer instead of
   constructing figures inline.
3. **Activate `TileReviewer`** — auto-fires once `write_spec`
   returns non-`None`.
4. **Surface a "rigorous mode" toggle** — flip `enable_llm_review`
   + `enable_holistic_audit` for offline / batch runs.
5. **Inject chart-library examples** from
   `docs/agents/chart_library_examples.md` into the writer's
   grounding context.
6. **Academic-rigor stat tests** beyond the four invariants:
   Diebold-Mariano, Ljung-Box, Jarque-Bera. Compose into
   `audit_forecast`.

## Commit history

```
340930a fix(agents): reactive orchestrator defaults + preamble strip + single-model
b47a663 docs(handoff): Day-5 resume — orchestrator landed, Docker died mid-verify
b4982c5 feat(agents): nested-harness orchestrator with three reviewer layers + Day-4 fixes
b9fb9c1 feat(ui): Day-3 extend interleave to EDA + Super
8affe8e feat(ui): Day-2 reactivity + a11y + view-aware chat drawer
e66dab8 feat(ui): Day-1 interleave on LFP, view-grounded
1db84f6 fix(perf): embeddings_cache → /app/.cache
c4ae737 feat(security): API hardening + simplify cleanup
f89ab45 fix(deps): langchain 1.x bump
fe82923 perf(ui): progressive render (I9a)
```
