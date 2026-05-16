"""
Threaded background blurb runner for the interleaved tab layout.

The Day-1 single-callback architecture worked (a single 4-output Dash
callback that loops through the panels) but blocked the user for the
full ~120 s while every panel cooked sequentially behind one HTTP
fetch. Browser preview surfaced the issue: a 2-minute "dead zone"
where nothing changes on screen except a loading spinner.

This module decouples the panel generation from the request lifecycle:

1. The main click callback assembles the per-panel ``view_state``
   dicts and spawns a daemon thread that fills a shared in-memory
   state slot keyed by ``run_id`` (UUID). The callback returns
   immediately with placeholders.
2. A ``dcc.Interval`` ticks every ~2 s while the run is active.
3. A polling callback reads ``BLURB_STATE[run_id]`` and updates only
   the panels that have completed since the last tick — every panel
   pops in as soon as its individual Ollama call returns rather than
   waiting for the slowest one.

State is per-process (in-memory dict guarded by a lock) — fine for
``gunicorn --workers 1 --threads 8``; if we ever scale to multiple
workers we'll need diskcache / Redis. Old runs get GC'd after ten
minutes via the cutoff in :func:`start_run` so the dict stays bounded.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Mapping

from utils.llm_utils import AI_FAILURE_MESSAGE

logger = logging.getLogger(__name__)


#: LFP-tab section order. Kept as a public constant for back-compat
#: even though every tab now passes its own section names through
#: :func:`start_run`. Polling callbacks return outputs in their own
#: hand-rolled order, so this is purely informational.
PANEL_SECTIONS: tuple[str, ...] = ("forecast", "requirements", "trend", "recap")

#: Drop runs older than this (seconds since they finished). Bounds
#: the in-memory dict for long-lived gunicorn workers.
_RUN_RETENTION_SEC = 600


#: ``{run_id: {<section>: text|None, "status": str, "started_at": float, "done_at": float|None}}``
#: Guarded by ``_state_lock``. Never read or mutate without the lock.
_state_lock = threading.Lock()
_run_state: dict[str, dict] = {}


def start_run(view_states: Mapping[str, dict | None]) -> str:
    """
    Allocate a run, kick off a daemon thread to fill its blurbs, and
    return the run id. ``view_states`` is a tab-defined mapping from
    section name to the typed view_state dict for that section — the
    runner uses the caller's keys, so each tab can name its panels
    however it wants ("forecast"/"trend" for LFP, "stats"/"timeseries"
    for EDA, etc.). A ``None`` value marks a section that has nothing
    to render (the polling layer treats that as "no panel context").
    """
    run_id = uuid.uuid4().hex
    sections = list(view_states.keys())
    initial: dict = {sec: None for sec in sections}
    initial["status"] = "starting"
    initial["started_at"] = time.time()
    initial["done_at"] = None
    initial["completed"] = 0
    initial["total"] = sum(1 for v in view_states.values() if v)
    # Snapshot the view_states so the thread can't see the caller's
    # later mutations and so JSON-cache keys are stable.
    snapshot = dict(view_states)

    with _state_lock:
        _run_state[run_id] = initial
        _gc_old_runs_locked()

    thread = threading.Thread(
        target=_fill_blurbs,
        args=(run_id, snapshot),
        name=f"blurb-runner-{run_id[:8]}",
        daemon=True,
    )
    thread.start()
    return run_id


def get_snapshot(run_id: str) -> dict | None:
    """Lock-protected copy of a run's current state. ``None`` if unknown."""
    with _state_lock:
        state = _run_state.get(run_id)
        return dict(state) if state else None


def is_done(run_id: str) -> bool:
    """True once every panel has either filled or failed for this run."""
    snap = get_snapshot(run_id)
    return bool(snap and snap.get("status") == "done")


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------
def _fill_blurbs(run_id: str, view_states: Mapping[str, dict | None]) -> None:
    """
    Daemon thread body. Splits ``view_states`` into "panels" (run in
    parallel via :class:`BlurbOrchestrator`) and "recap" (run after
    panels settle so it can weave the reviewed panel narratives).
    Each section's final text lands in ``_run_state[run_id][section]``
    via the orchestrator's ``on_section_done`` callback so the polling
    callback in the tab can show panels arriving as they're ready.
    """
    # Lazy import — keeps the orchestrator's executor + reviewer out
    # of the module-load critical path for callers that just want the
    # state dict (e.g. tests).
    from utils.agents.blurb_orchestra import BlurbOrchestrator  # noqa: PLC0415

    panel_views = {k: v for k, v in view_states.items() if k != "recap"}
    recap_view = view_states.get("recap")
    total_panels = sum(1 for v in panel_views.values() if v)
    total_with_recap = total_panels + (1 if recap_view else 0)
    completed = [0]  # mutable cell for the closure below

    def _on_section_done(section: str, text: str) -> None:
        completed[0] += 1
        with _state_lock:
            state = _run_state.get(run_id)
            if state is None:
                return
            state[section] = text
            state["completed"] = completed[0]
            if completed[0] < total_with_recap:
                state["status"] = (
                    f"reviewed {completed[0]} of {total_with_recap} "
                    f"section(s); next: {'recap' if completed[0] >= total_panels else '…'}"
                )
            else:
                state["status"] = "done"

    # Mark sections we'll never produce (caller passed None) up front
    # so the polling callback's placeholder logic can tell "nothing to
    # render" apart from "still cooking".
    with _state_lock:
        state = _run_state.get(run_id)
        if state is not None:
            for k, v in view_states.items():
                if not v:
                    state[k] = ""
            state["status"] = (
                f"dispatching {total_panels} panel specialist(s) in parallel"
            )

    try:
        with BlurbOrchestrator(max_workers=4, max_retries=1) as orch:
            orch.run_pipeline(
                panel_views=panel_views,
                recap_view=recap_view,
                on_section_done=_on_section_done,
            )
    except Exception as exc:  # noqa: BLE001 — never let the runner kill the worker
        logger.warning(
            "[blurb-async] orchestrator failed for run %s: %s: %s",
            run_id, type(exc).__name__, exc,
            exc_info=True,
        )
        with _state_lock:
            state = _run_state.get(run_id)
            if state is not None:
                # Ensure any sections that didn't land get a fallback so
                # the polling callback can stop spinning on them.
                for k in view_states:
                    if state.get(k) is None:
                        state[k] = f"_{AI_FAILURE_MESSAGE}_"

    with _state_lock:
        state = _run_state.get(run_id)
        if state is not None:
            state["status"] = "done"
            state["done_at"] = time.time()


def _gc_old_runs_locked() -> None:
    """Drop runs older than the retention window. Caller holds the lock."""
    cutoff = time.time() - _RUN_RETENTION_SEC
    stale = [
        rid for rid, s in _run_state.items()
        if (s.get("done_at") or s.get("started_at", 0)) < cutoff
    ]
    for rid in stale:
        del _run_state[rid]
