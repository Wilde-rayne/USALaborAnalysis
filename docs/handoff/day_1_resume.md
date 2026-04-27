# Day 1 — interleave slice resume

> **Status at handoff**: code complete, in-process smoke green, browser
> smoke is the only unverified step. Docker Desktop's API kept dying so
> the user is rebooting; this document is the resume point.

## TL;DR

- Branch `claude/reverent-noyce-a643f3` @ `c4ae737`
- 7 files modified, **two commits to make once browser smoke passes**:
  1. `fix(perf): route embeddings_cache to /app/.cache so non-root container can persist it`
  2. `feat(ui): interleaved figure→AI→figure→AI→recap on LFP, view-grounded via sentence-RAG (J1)`
- The Day-1 brief was a **vertical slice on LFP only** — prove the
  pattern before spreading to EDA / Super.

## Files modified

| File | Δ | Purpose |
|---|---|---|
| `utils/embeddings.py` | +13/-1 | **Regression fix.** Cache path was relative → root-owned `/app/`; non-root user couldn't write it; every chat call re-embedded 3 593 chunks (~27 s). Now under `$HF_HOME` (`/app/.cache/`, app-owned). Verified: cold 27.1 s → warm 1.0 s |
| `utils/agents/sentence_rag.py` | +216 | `render_view_sentences(view_state)` + 4 per-kind renderers (`forecast_panel`, `requirements_panel`, `trend_panel`, `recap`) + `default_rag_builder()` singleton |
| `utils/agents/blurb.py` | +57 | `BlurbAgent.explain_view(view_state, mode='panel'\|'recap')` — feeds **sentences** (not raw numbers) to the LLM |
| `utils/llm_utils.py` | +43 | `explain_view()` public facade with LRU cache keyed on (JSON view_state, mode, model). Same shape as the existing `generate_insight` / `_cached_invoke` |
| `tabs/_components.py` | +89 | `figure_panel(figure, caption, blurb_id)`, `tab_recap(blurb_id)`, `render_blurb(text)`, `PANEL_BLURB_TYPE` constant |
| `tabs/lfp_tab.py` | +341/-89 | Interleaved layout, single 4-output `fill_lfp_blurbs` callback (sequential blurbs in one HTTP fetch), shared `_compute_forecast_points()` so AI verdict can never disagree with the rendered table |
| `assets/prairie.css` | +68 | `.pi-panel-tile` / `.pi-recap-section` / `.pi-panel-blurb` styles + `--pi-accent` and `--pi-surface-subtle` vars in `:root` and dark-mode |

## What the LLM now sees (the key correction)

The user explicitly said *"don't feed raw numbers — render to ontology-
aware sentences for the AI to learn on"*. Sample of the new prompt
content for the forecast panel:

> The Iowa labor force participation rate forecast over the next 2
> year(s) using the ETS model selected from a Naive / Seasonal-Naive /
> Holt-Winters / ARIMA bake-off (out-of-sample RMSE 0.45). The last
> published Iowa labor force participation rate was 64.5 % in 2024.
> The +2-year point forecast is 65.2 %, with a 95 % confidence interval
> of 64.1 %–66.3 %. Peer states under comparison: Illinois, Minnesota,
> Wisconsin (Midwest census region).

— ontology-tagged (state names, region, measure name, model name).
Never `forecast_point: 65.2`, `metric_key: LFPR`, `winning_model: ets`.

## What's verified end-to-end (post-reboot, before this handoff)

- Branch + working-tree state intact
- Docker engine 28.4.0 alive
- Container rebuild fast (~30 s), container healthy in ~21 s
- HTTP: `/` 200 in 13 ms, `/health` 200 in 7 ms, body slim
- Ollama serving `llama3.2:3b` and `phi3:latest`
- All 7 new symbols import in container
- Singleton works
- Sentence renderer produces 4 sentences per panel; recap rolls up to 11
- Layout `Div` renders without raising
- 2 callbacks register, including the MATCH pattern Dash validates at
  registration time:
  `{"section":["MATCH"],"tab":"lfp","type":"pi-panel-blurb"}.children`
- Embedding cache file lands at `/app/.cache/embeddings_cache.npz`
  (5 155 796 bytes), 27 × speedup proven

## What's NOT verified (post-restart goal)

- **Live browser interaction**: page load → click Run Forecast → chart
  paints first → 4 AI panels populate in sequence via MATCH → recap
  renders last
- Visual rhythm: tile density, inset-card border-left accent, captions
- Per-panel prose grounded in the rendered numbers (not invented)
- Re-run with same params → all 4 panels return instantly (LRU hit)

## Resume runbook

```bash
cd "C:/Users/wilde/Documents/gits/USALaborAnalysis/.claude/worktrees/reverent-noyce-a643f3"

# 1. Confirm state
git status --short          # 7 modified files expected
git log --oneline -3        # c4ae737 at HEAD

# 2. Bring stack up (cached layers, only COPY rebuilds)
docker compose --env-file ../../../.env up -d --build

# 3. Wait for healthy
until [ "$(docker inspect -f '{{.State.Health.Status}}' \
  reverent-noyce-a643f3-dashboard-1 2>/dev/null)" = "healthy" ]; do
  sleep 3
done

# 4. Sanity probes
curl -sS -m 5 -w "\n/ → HTTP %{http_code} in %{time_total}s\n" \
  -o /dev/null http://localhost:8050/
curl -sS -m 5 http://localhost:8050/health

# 5. Run preview tool to drive the page (.claude/launch.json already
# has autoPort: false). Bring the manual stack DOWN first so the
# preview tool can manage the lifecycle:
docker compose --env-file ../../../.env down
# Then in the agent:
#   mcp__Claude_Preview__preview_start { name: "dashboard" }
#   mcp__Claude_Preview__preview_screenshot
#   mcp__Claude_Preview__preview_click { selector: '[id=lfp]' }   # or whatever the LFP tab nav is
#   mcp__Claude_Preview__preview_click { selector: '#lfp-run' }
#   ... wait for chart, then per-panel blurbs ...
#   mcp__Claude_Preview__preview_screenshot                       # confirm interleave
```

## Phase plan (where Day 1 sits in the bigger picture)

```
J  Forecast gap + current-year blanking         pending  (smallest-scope next move)
K  Tab rename — Overview / Forecast / Industry  pending  (blocks L's full rollout)
L  Figure quality + interleave                  Day 1 sliced 1 tab — extends after K
M  Relational topological ontology              pending  (foundation for N + RAG)
N  AI awareness of view                         Day 1 MVP done via explain_view
O  Performance                                  pending — cache fix already a chunk
P  Industry granularity (NAICS-3 via QCEW)      pending — after M
Q  Storage adapter — Mongo + SQL split          confirmed: SQL=panel, Mongo=RAG
R  FIPS hardening (a/b/c)                       choice deferred — needs user pick
S  Public container + GitHub release            last
```

## Known infra fragility — Docker Desktop

After ~10-30 min of activity (especially heavy Ollama + embedder
work), Docker Desktop's named-pipe API starts returning HTTP 500 on
the docker CLI even though container processes stay alive:

```
request returned 500 Internal Server Error for API route and version
http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.51/...
```

Recovery ladder (least to most invasive):
1. Stop-Process all `docker*` / `com.docker.*` procs → relaunch from
   `Program Files\Docker\Docker\Docker Desktop.exe`
2. If WSL is also wedged: `wsl --shutdown` (may need elevated PS:
   `Restart-Service WslService -Force`)
3. If the `dockerInference` reparse-point is stuck: rename
   `%LOCALAPPDATA%\Docker\run\` aside; Docker recreates it on next
   boot
4. **Machine reboot** — fixes everything else
