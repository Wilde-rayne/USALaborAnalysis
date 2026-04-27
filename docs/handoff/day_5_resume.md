# Day 5 — nested-harness orchestrator resume

> **Status at handoff**: all code committed at `b4982c5`. Working tree
> clean. The reactivity test in preview was blocked by Docker Desktop's
> recurring `dockerInference` socket bug — recover Docker, re-bring up
> the stack (no rebuild needed), and run the verification plan below.

## TL;DR

- Branch `claude/reverent-noyce-a643f3` @ `b4982c5`
- **Tree clean** — all Day-4 fixes + the nested-harness orchestrator are
  committed.
- The next step is **my reactivity test in preview**, after which I
  hand back to you for human verification before the next slice
  (image-writer activation + invariants-into-prompts).

## What's in `b4982c5`

The user's full architecture sketch is wired:

```
base ──→ total context ──→ page context
                                ├──→ review → writer → review        (panels: parallel)
                                │              ↓
                                │         tile reviewer              (per-tile a11y; stub)
                                │              ↓
                                │       reviewed panel texts ─→ recap writer
                                │                                    ↓
                                │                            holistic auditor
                                │                                    ↓
                                │                        targeted REVISE_<section>
                                └──→ image writer / FigureSpec        (stub — Day-6)
```

| Piece | Module | Status |
|---|---|---|
| Specialist agents (per section, parallel) | `utils/agents/blurb_orchestra.py` | wired (`ThreadPoolExecutor(max_workers=4)`, freed after click) |
| Per-section reviewer (heuristic + LLM) | `utils/agents/reviewer.py` `ReviewerAgent` | wired |
| Per-tile reviewer (figure ↔ blurb) | `utils/agents/reviewer.py` `TileReviewer` | passes through until image-writer non-None |
| Recap with reviewed panels in context | `BlurbOrchestrator.run_pipeline` Phase 2 | wired |
| Holistic auditor (cross-section) | `HolisticReviewer` + `_holistic_revise` | wired with targeted revise |
| Image writer / `FigureSpec` | `utils/agents/image_writer.py` | skeleton |
| Statistical-rigor invariants | `utils/forecasting/invariants.py` | tested; consumption hook documented |
| Chart-library examples corpus | `docs/agents/chart_library_examples.md` | written — Plotly / matplotlib / seaborn / sklearn / keras |

Plus the Day-4 visible fixes (`OLLAMA_TIMEOUT` 600 s, Phase J gap viz
on LFP, dark-mode legibility, chart range-selector, chat FAB hit-target,
prompt rewrites) and Ollama parallelism bumps in `docker-compose.yml`.

## Recovery sequence (Docker is finicky)

The same `dockerInference` socket bug hit again:

```
starting services: initializing Inference manager: listening on
unix://C:\Users\wilde\AppData\Local\Docker\run\dockerInference: remove
... The file cannot be accessed by the system.
```

Run, in elevated PowerShell if WslService is wedged:

```powershell
# 1. Kill all docker procs (clean state)
Get-Process | Where-Object { $_.Name -like "*docker*" -or
    $_.Name -like "com.docker*" } | Stop-Process -Force

# 2. Bounce WSL (skip if `wsl --shutdown` hangs without admin)
wsl --shutdown

# 3. Rename the stuck run/ directory aside — this is the actual fix.
#    EnableDockerAI=false does NOT prevent the inference manager
#    from trying to bind; the rename-aside does.
$stamp = Get-Date -Format yyyyMMddHHmmss
Rename-Item "$env:LOCALAPPDATA\Docker\run" "run.stuck-$stamp"

# 4. Relaunch Docker Desktop
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

If WslService refuses to stop without admin, last resort is a
**Windows reboot** — fixes everything.

After Docker recovers:

```bash
cd "C:/Users/wilde/Documents/gits/USALaborAnalysis/.claude/worktrees/reverent-noyce-a643f3"
docker compose --env-file ../../../.env up -d
until [ "$(docker inspect -f '{{.State.Health.Status}}' \
  reverent-noyce-a643f3-dashboard-1 2>/dev/null)" = "healthy" ]; do
  sleep 3
done
```

**No `--build` needed** — the image already has every commit through
`b4982c5` baked in. Bringing the stack back is just a container
recreate, ~15 s.

## Verification plan I was running

1. Open preview at `http://localhost:8050/`.
2. Click **LFP Forecast → Run Forecast** with defaults
   (Iowa focus, IL/MN peers, LFPR, +2 yrs, no threshold).
3. Time-stamp:
   - Body appears (target ≤ 30 s)
   - First panel filled (target ≤ 90 s — parallel dispatch)
   - All 4 panels + recap + holistic audit complete (target ≤ 5 min)
4. Verify each panel narrative:
   - At least 2 numeric digits cited
   - No "Here's a plain-English explanation" preamble
   - Interprets, doesn't just describe (statistical observation OR
     workforce-planner implication)
5. Repeat for **EDA** (Refresh Data) and **Super** (Run Forecast).
6. Re-click with same params → panels return ~instantly (LRU hit).
7. Hand back to user.

## Known infra fragilities

- Docker Desktop's named-pipe API drops 500s after sustained Ollama
  load (~10-30 min). Recovery is the kill + relaunch above.
- The `dockerInference` reparse-point in `%LOCALAPPDATA%\Docker\run\`
  gets stuck on every failed boot; rename the parent dir aside and
  Docker recreates it cleanly.
- `EnableDockerAI=false` in `%APPDATA%\Docker\settings-store.json`
  does **not** prevent the inference manager from trying to bind.
- Ollama on CPU: each LLM call ~30-60 s. The `OLLAMA_NUM_PARALLEL=4`
  bump in compose lets the 4 panel specialists overlap rather than
  serialize, but on a single CPU "parallel" means token-level
  interleaving — total wall time ≈ 1.5-2× a single call rather than
  4×. Better than serial, not magic.

## Commit history (most recent first)

```
b4982c5 feat(agents): nested-harness orchestrator + Day-4 fixes        ← Day 5
b9fb9c1 feat(ui): Day-3 extend interleave to EDA + Super
8affe8e feat(ui): Day-2 reactivity + a11y + view-aware chat drawer
e66dab8 feat(ui): Day-1 interleave on LFP, view-grounded
1db84f6 fix(perf): embeddings_cache path → /app/.cache
c4ae737 feat(security): API hardening + simplify cleanup
f89ab45 fix(deps): langchain 1.x bump
fe82923 perf(ui): progressive render (I9a)
```

## Next slice (after human verification passes)

User's open requests, in priority order:

1. **Wire `audit_forecast` failures** from
   `utils/forecasting/invariants.py` into the reviewer's critique
   prompt so the AI acknowledges model uncertainty when invariants
   flag a forecast.
2. **Activate `ImageWriterAgent.write_spec`** — implement the LLM
   round-trip + JSON parse + validation against `FigureSpec`, then
   add `tabs/_charts.py` builder so the tabs call the writer instead
   of constructing figures inline.
3. **Activate `TileReviewer`** — auto-fires once `write_spec`
   returns non-None.
4. **Inject good chart examples** from
   `docs/agents/chart_library_examples.md` into the writer's
   grounding context.
5. **Add academic-rigor stat tests** beyond the four invariants:
   Diebold-Mariano significance, Ljung-Box residual autocorrelation,
   Jarque-Bera normality on residuals. Compose into `audit_forecast`.
