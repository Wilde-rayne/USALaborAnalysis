"""
Per-click blurb orchestrator with specialist subagents.

Architecture sketch (matches the user's "nested harness" design)::

    base ──→ total context ──→ page context
                                     ├──→ review → writer → review   (text track, this module)
                                     └──→ image writer / creator     (utils.agents.image_writer)

- **base** (system-wide intent): defined as the LaborAgent's
  ``system_prompt`` for each specialist — what role the agent plays.
- **total context** (across-tab corpus): the ontology-aware sentences
  the SentenceRAGBuilder produces from the merged labor panel. Cached
  at module level inside the embeddings pipeline.
- **page context** (per-tab, per-click): the panel ``view_state``
  dicts the tab passes into ``run_pipeline``. Rendered into prose by
  ``SentenceRAGBuilder.render_view_sentences``.
- **review → writer → review**: implemented here as
  :class:`ReviewerAgent` + :class:`HolisticReviewer` looped through
  per-section specialists.
- **image writer**: stubbed in :mod:`utils.agents.image_writer` —
  the parallel track that will own figure spec generation.

The Day-2/Day-3 sequential runner sent every panel through the same
generic chat agent, so each panel paid the same per-call latency and
shared the same generic system prompt. The user asked for a faster +
more precise architecture: cache a context bundle once per click,
fan out specialist subagents in parallel, free them when the work
is done.

This module is that orchestrator. It is intentionally light-weight:

- ``BlurbOrchestrator`` holds a ``ThreadPoolExecutor`` and a snapshot
  of the per-click context bundle (the ontology-aware sentences a
  ``SentenceRAGBuilder`` rendered out of the panel ``view_state``
  dicts). Threads are the parallelism mechanism; ``ThreadPoolExecutor``
  is sufficient for ~4-10 concurrent calls and avoids PySpark's
  pickling + driver-overhead tax. Spark is the scale-out path when we
  need to fan blurbs across multiple machines — at single-host scale
  it actively hurts.

- ``_make_specialist`` builds a small :class:`utils.agents.base.LaborAgent`
  with a section-specific ``system_prompt``. That's the "very specific
  agents" the user asked for: a forecast specialist for the forecast
  panel, a requirements specialist for the threshold panel, a trend
  specialist for the historic-shift panel, etc. Each agent is created,
  invoked, and discarded per click — no module-level singletons that
  would carry stale context between sessions.

- The recap section uses the larger chat model (``OLLAMA_MODEL``,
  default ``llama3.2:3b``) for its synthesising pass; the panel
  specialists default to the smaller / faster ``OLLAMA_AGENT_MODEL``
  (default ``phi3``) because they only need to interpret 3-5 sentences
  of context. With ``OLLAMA_NUM_PARALLEL > 1`` and
  ``OLLAMA_MAX_LOADED_MODELS >= 2`` (set in docker-compose.yml) Ollama
  can actually serve both models concurrently — that's where the
  reactivity win lands.

- ``BlurbOrchestrator.fan_out`` returns a ``{section: Future}`` dict so
  callers can poll completions with :func:`concurrent.futures.as_completed`
  for the "panels arrive as they become ready" UX. ``teardown`` shuts
  the executor down; the agents themselves are GC'd when the
  orchestrator goes out of scope.
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Callable, Mapping

from utils.agents.base import LaborAgent, LaborAgentConfig
from utils.agents.ollama import (
    AGENT_MODEL_ENV,
    CHAT_MODEL_ENV,
    OLLAMA_BASE_URL,
)
from utils.agents.reviewer import (
    HolisticReviewer,
    ReviewerAgent,
    TileReviewer,
    strip_preamble,
)
from utils.agents.sentence_rag import default_rag_builder
from utils.constants import DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


#: Section-specific specialist prompts. Each one is its own micro-
#: persona — the model produces tighter output than a general chat
#: prompt would. Keep them short (the system prompt counts against
#: every request's token budget).
_SECTION_PROMPTS: dict[str, str] = {
    # LFP tab
    "forecast": (
        "You are a labor-market forecast analyst. In 2 sentences, "
        "interpret the panel facts in terms of forward expectations "
        "and model uncertainty (CI width, RMSE, peer comparison). "
        "Cite the specific numbers; do not invent any."
    ),
    "requirements": (
        "You are a state workforce planner. In 2 sentences, explain "
        "which states clear or miss the policy threshold and what "
        "that implies for resource allocation or hiring policy. "
        "Cite the specific numbers; do not invent any."
    ),
    "trend": (
        "You are a labor economist. In 2 sentences, interpret the "
        "historic trajectory: direction, magnitude, and how it "
        "compares to volatility or peer states. Cite the specific "
        "numbers; do not invent any."
    ),
    # EDA tab
    "stats": (
        "You are a data analyst describing dataset shape. In 2 "
        "sentences, characterise central tendency and spread for "
        "the selected series. Cite the specific numbers."
    ),
    "timeseries": (
        "You are a time-series analyst. In 2 sentences, describe "
        "how the selected series have moved across the window. "
        "Cite the specific numbers."
    ),
    "volatility": (
        "You are a volatility analyst. In 2 sentences, describe how "
        "stable or unstable the year-over-year changes are right "
        "now and what that implies. Cite the specific numbers."
    ),
    # Super tab
    "recommendation": (
        "You are a regional siting advisor. In 2 sentences, identify "
        "which states stand out at the top and bottom and what that "
        "implies for siting / expansion decisions. Cite numbers."
    ),
    "models": (
        "You are a forecast diagnostics analyst. In 2 sentences, "
        "summarise the bake-off model mix across the in-scope states "
        "and what its diversity implies about confidence. Cite numbers."
    ),
    # End-of-tab synthesis (uses the larger chat model)
    "recap": (
        "You are a senior planner producing an executive summary. In "
        "3 sentences across the panels: lead with the headline "
        "finding, weave one statistical comparison (vs peers or "
        "historical norm), close with one actionable next step. "
        "Cite numbers; do not invent any."
    ),
}


#: Per-specialist call timeout. The orchestrator-side default is
#: deliberately tighter than the env-driven ``DEFAULT_TIMEOUT`` (which
#: governs the chat drawer + tools): a panel specialist that doesn't
#: respond in ~2 minutes on CPU Ollama is almost certainly stuck
#: behind a queue; failing fast surfaces the issue to the user via
#: the AI_FAILURE_MESSAGE rather than blocking the polling callback
#: from making forward progress on the other panels.
SPECIALIST_TIMEOUT_SECONDS: int = 120


def _make_specialist(
    section: str,
    *,
    model: str | None = None,
    timeout: int = SPECIALIST_TIMEOUT_SECONDS,
) -> LaborAgent:
    """
    Build a short-lived specialist for a single panel section.

    All specialists default to the chat model (``OLLAMA_MODEL``).
    The earlier two-model split (panels on phi3, recap on
    llama3.2:3b) tripped the ``llama runner process has terminated``
    error in single-CPU + 7.5 GB RAM containers — phi3's runner
    OOM'd when llama3.2:3b was already resident, even with
    ``OLLAMA_MAX_LOADED_MODELS=2``. One model means one runner, no
    second-model load to fail; the chat model is also strictly the
    higher-quality one for narrative work.
    """
    if model is None:
        model = os.getenv(CHAT_MODEL_ENV, "llama3.2:3b")
    return LaborAgent(
        LaborAgentConfig(
            model=model,
            system_prompt=_SECTION_PROMPTS.get(section, _SECTION_PROMPTS["forecast"]),
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
            timeout=timeout,
        )
    )


class BlurbOrchestrator:
    """
    Per-click parallel specialist dispatch + reviewer loop.

    Usage::

        orch = BlurbOrchestrator(max_workers=4)
        try:
            orch.run_pipeline(
                panel_views={"forecast": fv, "requirements": rv, "trend": tv},
                recap_view=recap_view,
                on_section_done=lambda sec, text: ...,  # update progress
            )
        finally:
            orch.teardown()

    Pipeline shape:

    1. **Panel specialists run in parallel** — one short-lived
       ``LaborAgent`` per section, threaded through the executor.
    2. **Each specialist's output runs through** :class:`ReviewerAgent`
       (cheap heuristic first, LLM critic only if the heuristic
       flags weak output). One revision pass on failure.
    3. **Recap runs after** the panels settle. It receives the
       *reviewed* panel texts as part of its grounding context, so
       the synthesis weaves the corrected sentences rather than the
       raw specialist drafts. Recap is reviewed too; one revision
       pass.
    4. ``on_section_done(section, text)`` fires as each section's
       final text becomes available — the polling callback uses that
       to update the page progressively.
    """

    def __init__(
        self,
        *,
        max_workers: int = 4,
        max_retries: int = 0,
        enable_llm_review: bool = False,
        enable_holistic_audit: bool = False,
    ):
        """
        ``max_retries`` / ``enable_llm_review`` / ``enable_holistic_audit``
        default OFF: a first click only pays the cost of N specialist
        calls + 1 recap, which keeps wall-time reactive on single-CPU
        Ollama. Pass ``enable_*=True`` for the full nested-harness
        review pass when latency matters less than rigour (e.g. an
        offline batch run, or a future "rigorous mode" UI toggle).

        The reviewer's free heuristic quick-check still runs in either
        mode and is what catches the most common preamble / no-numbers
        failure modes; only the LLM-backed evaluation + targeted
        rewrite are gated behind the flag.
        """
        # ``thread_name_prefix`` makes the threads visible in
        # ``ps -L`` / debuggers as ``blurb-orch-N`` so it's obvious
        # which work is in flight when the dashboard is busy.
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="blurb-orch",
        )
        self._reviewer = ReviewerAgent()
        self._tile_reviewer = TileReviewer()
        self._holistic = HolisticReviewer()
        self._max_retries = max_retries
        self._enable_llm_review = enable_llm_review
        self._enable_holistic_audit = enable_holistic_audit
        self._lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fan_out(
        self,
        view_states: Mapping[str, dict | None],
    ) -> dict[str, Future]:
        """
        Submit one *reviewed* specialist per non-empty section. Returns
        a ``{section: Future}`` dict so callers can poll completions
        in arrival order via :func:`as_completed`. Each future resolves
        to the final reviewed text for that section.
        """
        if self._closed:
            raise RuntimeError("BlurbOrchestrator already torn down")
        futures: dict[str, Future] = {}
        for section, vs in view_states.items():
            if not vs:
                continue
            futures[section] = self._executor.submit(
                self._review_and_revise, section, vs
            )
            logger.debug("[orch] dispatched review-loop for section=%s", section)
        return futures

    def run_pipeline(
        self,
        *,
        panel_views: Mapping[str, dict | None],
        recap_view: dict | None,
        on_section_done: Callable[[str, str], None] | None = None,
    ) -> dict[str, str]:
        """
        Top-level pipeline: panels in parallel, then recap with the
        reviewed panel texts in context. Calls ``on_section_done`` as
        each section's final text becomes available so the UI can
        progressively reveal panels rather than waiting for the whole
        run to finish.
        """
        on_section_done = on_section_done or (lambda *_: None)

        # ----- Phase 1: panels (parallel) -----
        panel_futures = self.fan_out(panel_views)
        panel_results: dict[str, str] = {}
        for fut in as_completed(panel_futures.values()):
            section = next(s for s, f in panel_futures.items() if f is fut)
            try:
                text = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[orch] section=%s failed: %s", section, exc)
                from utils.llm_utils import AI_FAILURE_MESSAGE  # noqa: PLC0415

                text = f"_{AI_FAILURE_MESSAGE}_"
            panel_results[section] = text
            on_section_done(section, text)

        # ----- Phase 2: recap (uses reviewed panels) -----
        if recap_view:
            enriched_recap = dict(recap_view)
            # Surface the reviewed panel texts so the recap can weave
            # the corrected synthesis rather than re-deriving from
            # the raw view-state facts.
            enriched_recap["reviewed_panels"] = panel_results
            try:
                recap_text = self._review_and_revise("recap", enriched_recap)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[orch] recap failed: %s", exc)
                from utils.llm_utils import AI_FAILURE_MESSAGE  # noqa: PLC0415

                recap_text = f"_{AI_FAILURE_MESSAGE}_"
            panel_results["recap"] = recap_text
            on_section_done("recap", recap_text)

        # ----- Phase 3: holistic audit (cross-section coherence) -----
        # OFF by default — the auditor is two extra LLM calls per
        # click (audit + targeted revise) which on single-CPU Ollama
        # extends wall time past the "reactive" budget. Enable for
        # offline / batch runs where latency matters less than rigour.
        if not self._enable_holistic_audit:
            return panel_results

        try:
            section_facts = self._collect_facts(panel_views, recap_view)
            audit = self._holistic.audit(panel_results, section_facts)
            if not audit.get("passed"):
                target = audit.get("target") or "recap"
                feedback = audit.get("feedback", "")
                logger.info(
                    "[orch] holistic audit asks revise of %s: %s",
                    target, feedback[:160],
                )
                target_view = (
                    recap_view if target == "recap"
                    else (panel_views.get(target) or {})
                )
                if target_view:
                    revised = self._holistic_revise(
                        target, target_view, panel_results.get(target, ""), feedback,
                        sibling_panels=panel_results,
                    )
                    if revised:
                        panel_results[target] = revised
                        on_section_done(target, revised)
        except Exception as exc:  # noqa: BLE001 — never let auditor block ship
            logger.warning("[orch] holistic audit raised: %s", exc)

        return panel_results

    def teardown(self, wait: bool = True) -> None:
        """Release the thread pool. Called from the runner once all
        futures have settled (or after a hard timeout)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # ``cancel_futures=False`` so any specialist that's already
        # mid-flight gets a chance to land its result; ``wait=True``
        # blocks until all in-flight calls complete.
        self._executor.shutdown(wait=wait, cancel_futures=False)
        logger.debug("[orch] torn down")

    # Context-manager support so callers can ``with BlurbOrchestrator()``.
    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, *exc) -> None:
        self.teardown(wait=True)

    # ------------------------------------------------------------------
    # Internal — single-section pipeline
    # ------------------------------------------------------------------
    def _review_and_revise(self, section: str, view_state: dict) -> str:
        """
        Specialist → reviewer → maybe revise. Loops at most
        ``max_retries`` times on REVISE verdicts before accepting the
        latest candidate as-is. Every failure path returns *some*
        text — the AI panel is never left empty.
        """
        builder = default_rag_builder()
        sentences = builder.render_view_sentences(view_state)
        facts = "\n".join(f"- {s}" for s in sentences)

        # Recap also gets the reviewed panel texts (if the caller
        # supplied them) as additional grounding so the synthesis
        # weaves the corrected drafts.
        reviewed_panels = view_state.get("reviewed_panels") or {}
        if reviewed_panels:
            reviewed_block = "\n".join(
                f"### {sec.title()} panel narrative\n{txt}"
                for sec, txt in reviewed_panels.items()
                if txt
            )
            facts = f"{facts}\n\nReviewed panel narratives:\n{reviewed_block}"

        title = view_state.get("title") or section
        specialist = _make_specialist(section)

        def _initial_prompt() -> str:
            # Don't include a "Panel: <title>" header in the prompt —
            # the model parrots it back as the first line ("Panel: IA
            # Labor Force Participation Rate outlook…"), which then
            # has to be stripped post-hoc. The system prompt already
            # conveys what role the agent plays; the facts block is
            # all the model needs.
            return (
                f"Topic: {title}\n\n"
                f"Facts:\n{facts}\n\n"
                f"Write the analysis directly — no header line, no "
                f"\"Panel:\" prefix, no \"Headline:\" prefix. Cite "
                f"the numbers above; do not invent any."
            )

        prompt = _initial_prompt()
        candidate = specialist.invoke(prompt)
        # Strip any leading preamble clause the model emitted ("Based
        # on the provided facts, …", "Here is a 3-sentence executive
        # summary:", "Panel: Executive Summary", etc.). Single string
        # operation — no LLM round-trip — so it's safe in the reactive
        # default path and consistently improves first-line quality.
        candidate = strip_preamble(candidate) or candidate

        # Free heuristic check — runs even with LLM review disabled
        # because it's pure-Python (preamble / digit count / sentinels).
        ok, reason = self._reviewer.quick_check(candidate)
        if ok or not self._enable_llm_review:
            logger.debug(
                "[orch] section=%s shipped after specialist call "
                "(quick_check=%s, llm_review_enabled=%s)",
                section, ok, self._enable_llm_review,
            )
            return candidate

        # Optional LLM-review + revision loop (off by default — the
        # extra round-trips multiply latency on single-CPU Ollama).
        for attempt in range(self._max_retries):
            review = self._reviewer.review(candidate, facts)
            if review["passed"]:
                tile = self._tile_reviewer.review(
                    section=section,
                    blurb=candidate,
                    figure_spec=None,  # image-writer stub
                    facts=facts,
                )
                if tile["passed"]:
                    return candidate
                review = tile
            logger.info(
                "[orch] section=%s revising (stage=%s): %s",
                section, review.get("stage"), review.get("feedback", "")[:120],
            )
            prompt = (
                f"{_initial_prompt()}\n\n"
                f"Earlier draft:\n{candidate}\n\n"
                f"Reviewer asked you to revise — {review['feedback']}\n"
                f"Rewrite the analysis. Keep it short. Cite specific "
                f"numbers from the facts."
            )
            candidate = specialist.invoke(prompt)
        return candidate

    def _collect_facts(
        self,
        panel_views: Mapping[str, dict | None],
        recap_view: dict | None,
    ) -> dict[str, str]:
        """Render the grounding facts for every section so the
        holistic auditor can compare narrative against source."""
        builder = default_rag_builder()
        out: dict[str, str] = {}
        for section, vs in panel_views.items():
            if not vs:
                continue
            out[section] = "\n".join(
                f"- {s}" for s in builder.render_view_sentences(vs)
            )
        if recap_view:
            out["recap"] = "\n".join(
                f"- {s}" for s in builder.render_view_sentences(recap_view)
            )
        return out

    def _holistic_revise(
        self,
        section: str,
        view_state: dict,
        prior_text: str,
        auditor_feedback: str,
        *,
        sibling_panels: Mapping[str, str],
    ) -> str:
        """
        One-shot rewrite driven by the holistic auditor's feedback.

        Unlike :meth:`_review_and_revise` (which loops on a per-
        section reviewer), this is a single deterministic retry — the
        auditor has already done the deep critique and produced
        actionable feedback, we just need to apply it. The specialist
        sees the full sibling-panel context so the rewrite stays
        coherent with the rest of the tab.
        """
        builder = default_rag_builder()
        sentences = builder.render_view_sentences(view_state)
        facts = "\n".join(f"- {s}" for s in sentences)
        siblings_block = "\n".join(
            f"### {sec.title()} narrative (already finalised)\n{txt}"
            for sec, txt in sibling_panels.items()
            if sec != section and txt
        )
        title = view_state.get("title") or section
        prompt = (
            f"Panel: {title}\n"
            f"Facts:\n{facts}\n\n"
            f"Sibling panel narratives the user will see alongside "
            f"this one:\n{siblings_block}\n\n"
            f"Earlier draft of THIS section:\n{prior_text}\n\n"
            f"The senior auditor asked you to revise — {auditor_feedback}\n"
            f"Rewrite this section. Keep it short, cite specific "
            f"numbers, weave in the auditor's directive. Do not "
            f"contradict the sibling narratives."
        )
        try:
            specialist = _make_specialist(section)
            return specialist.invoke(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[orch] holistic-revise %s failed: %s", section, exc)
            return ""
