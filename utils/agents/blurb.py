"""
BlurbAgent — narrative blurb generator for the tab UIs.

Sits one layer above :class:`utils.agents.base.LaborAgent`: callers
pass structured arguments ("measure = LFPR, horizon = 2 years,
historic = [...], forecast = [...], winning model = ets"), the agent
turns them into a focused 3-5 sentence narrative.

Why a class rather than free functions? Each BlurbAgent instance
holds the underlying LaborAgent, so callers can inject a mocked
agent in tests or swap the chat model for a specific tab without
touching the callers.
"""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

from utils.agents.base import LaborAgent
from utils.agents.ollama import chat_agent
from utils.ontology import ONTOLOGY

logger = logging.getLogger(__name__)


class BlurbAgent:
    """Produces short narrative insights for the dashboard tabs."""

    def __init__(self, agent: LaborAgent | None = None) -> None:
        self.agent = agent or chat_agent()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _summarize_series(self, values: Sequence[float]) -> str:
        """Short numeric summary used in forecast prompts."""
        if not values:
            return "no data"
        first, last = values[0], values[-1]
        lo, hi = min(values), max(values)
        return (
            f"first={first:,.2f}, last={last:,.2f}, "
            f"range=[{lo:,.2f}, {hi:,.2f}], n={len(values)}"
        )

    # ------------------------------------------------------------------
    # Public blurb methods
    # ------------------------------------------------------------------
    def forecast_blurb(
        self,
        *,
        entity: str,
        measure: str,
        history: Sequence[float],
        forecast: Sequence[float],
        horizon_months: int,
        model_name: str,
        diagnostics_summary: str = "",
    ) -> str:
        """
        Narrative blurb for a single-series forecast.

        ``entity`` is a user-facing label ("Iowa", "the Midwest"),
        ``measure`` is the ontology-level measure key (e.g. "LFPR"),
        ``horizon_months`` is the forecast length, ``model_name`` is
        the winning bakeoff entry ("ets"), and ``diagnostics_summary``
        is a short string to glue into the prompt (e.g. the ADF/DM
        line).
        """
        try:
            measure_obj = ONTOLOGY.measure(measure)
            readable_measure = measure_obj.name
        except KeyError:
            readable_measure = measure.replace("_", " ").lower()

        prompt = (
            f"Write a concise 3-4 sentence analytical narrative about the "
            f"following forecast.\n\n"
            f"- Subject: {entity} {readable_measure}\n"
            f"- Forecast horizon: {horizon_months} months\n"
            f"- Historic series: {self._summarize_series(history)}\n"
            f"- Forecast series: {self._summarize_series(forecast)}\n"
            f"- Model chosen by out-of-sample RMSE: {model_name}\n"
            + (f"- Diagnostics: {diagnostics_summary}\n" if diagnostics_summary else "")
            + "\nKeep it specific to these numbers; do not invent data."
        )
        return self.agent.invoke(prompt)

    def comparison_blurb(
        self,
        *,
        measure: str,
        horizon_months: int,
        by_entity: dict[str, float],
        winners: dict[str, str] | None = None,
    ) -> str:
        """
        Cross-entity narrative ("who's expected to lead / lag").

        ``by_entity`` maps a label ("IA", "Midwest Mean") to the end-of-
        horizon forecast value. ``winners`` optionally maps label →
        per-entity winning model name so the blurb can mention the
        model mix.
        """
        try:
            readable_measure = ONTOLOGY.measure(measure).name
        except KeyError:
            readable_measure = measure.replace("_", " ").lower()
        ranked = sorted(by_entity.items(), key=lambda p: p[1], reverse=True)

        body_lines = [f"- {lbl}: {val:,.1f}" for lbl, val in ranked]
        extra = ""
        if winners:
            mix = ", ".join(f"{k} ← {v}" for k, v in winners.items())
            extra = f"\nPer-entity model mix: {mix}"

        prompt = (
            f"Summarise the following forecast results for {readable_measure} "
            f"over a {horizon_months}-month horizon in three sentences, "
            f"aimed at regional planners. Call out the top-3 and bottom-3 "
            f"entities but do not invent numbers.\n\n"
            + "\n".join(body_lines)
            + extra
        )
        return self.agent.invoke(prompt)

    def context_answer(self, question: str, context: str = "") -> str:
        """General-purpose Q+A grounded in the provided RAG context."""
        if context:
            prompt = (
                f"Answer the user's question using ONLY the context below. "
                f"If the context doesn't answer it, say so.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            )
        else:
            prompt = question
        return self.agent.invoke(prompt)

    # ------------------------------------------------------------------
    # View-grounded explanations
    # ------------------------------------------------------------------
    def explain_view(self, view_state: dict, *, mode: str = "panel") -> str:
        """
        Ground a narrative in the actual rendered panel.

        The dashboard tabs construct a small ``view_state`` dict per
        figure / table (typed by ``_kind``: ``forecast_panel``,
        ``requirements_panel``, ``trend_panel``, ``recap``). We feed
        that through :class:`SentenceRAGBuilder.render_view_sentences`
        so the LLM never sees raw key/value numbers — only ontology-
        aware natural-language sentences tagged with state, region,
        and measure. That's the same surface we use for the embedding
        corpus (``utils.agents.sentence_rag``), keeping retrieval and
        generation grounded on the same prose.

        ``mode='panel'`` produces a tight 2-3 sentence explanation
        tied to one figure; ``mode='recap'`` produces a 4-6 sentence
        end-of-tab synthesis. The "do not invent values" guardrail is
        the main hedge against hallucination — the model can only
        weave the sentences we put in front of it.
        """
        if mode not in {"panel", "recap"}:
            raise ValueError(f"explain_view mode must be 'panel' or 'recap', got {mode!r}")
        # Lazy import to keep the BlurbAgent module cheap to load — the
        # sentence-RAG module pulls in the full ontology + measures map.
        from utils.agents.sentence_rag import default_rag_builder  # noqa: PLC0415

        sentences = default_rag_builder().render_view_sentences(view_state)
        if not sentences:
            return (
                "I don't have enough panel context to explain this view. "
                "Try re-running the forecast or refreshing the page."
            )
        title = view_state.get("title") or view_state.get("metric") or "this panel"
        if mode == "recap":
            instructions = (
                "Write a 4-6 sentence end-of-tab synthesis aimed at a "
                "state workforce planner. Lead with the headline finding, "
                "then weave the supporting facts into one paragraph — "
                "do not bullet-list them."
            )
        else:
            instructions = (
                "Write a 2-3 sentence plain-English explanation of this "
                "panel. Stay grounded in the facts below — every claim "
                "must trace back to one of the sentences."
            )
        body = "\n".join(f"- {s}" for s in sentences)
        prompt = (
            f"{instructions}\n"
            f"Do NOT invent values, ranks, or trends not stated below.\n\n"
            f"Panel: {title}\nFacts:\n{body}"
        )
        return self.agent.invoke(prompt)


# --------------------------------------------------------------------------
# Module-level convenience
# --------------------------------------------------------------------------
_default_blurb: BlurbAgent | None = None


def default_blurb_agent() -> BlurbAgent:
    """Lazy singleton — the most common call site wants the default chat model."""
    global _default_blurb
    if _default_blurb is None:
        _default_blurb = BlurbAgent()
    return _default_blurb
