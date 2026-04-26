"""
Backward-compatible facade over the blurb agent.

``generate_insight`` is the function the Dash tabs have been calling
since before the LangChain harness existed. It still works the same
way — caller passes a prompt, gets a natural-language answer — but
under the hood it now routes through :class:`BlurbAgent` so every
path shares the same connection, system prompt, and timeout handling.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from utils.agents import BlurbAgent, default_blurb_agent
from utils.constants import DEFAULT_TIMEOUT  # noqa: F401 — kept for import compat

logger = logging.getLogger(__name__)

#: Generic user-facing message returned when the blurb agent fails.
#: Re-exported so the chat drawer renders the same wording (wrapping
#: with Markdown italics on its side). Avoids leaking exception text
#: to the rendered surface (OWASP API A05:2023).
AI_FAILURE_MESSAGE = (
    "Sorry, I couldn't generate a response right now. "
    "The error has been logged — please try again."
)


def _resolve_context(active_tab: str | None, prompt: str) -> str:
    """Pull RAG context for the prompt, falling back silently on failure."""
    try:
        from utils.embeddings import get_relevant_context, retrieve_context  # noqa: PLC0415

        if active_tab:
            return (get_relevant_context(active_tab, top_k=3) or "").strip()
        return (retrieve_context(prompt, top_k=3) or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.info(f"[llm] RAG retrieval skipped: {exc}")
        return ""


@lru_cache(maxsize=256)
def _cached_invoke(key: tuple[str, str, str]) -> str:
    """
    LRU cache keyed on (prompt, context, agent_model).

    Tuple-keyed so identical prompts that retrieved different context
    (e.g. the same question after the panel refreshed) are cached
    separately. Note that failure responses also get cached — a
    transient Ollama outage will stick to the LRU entry until evicted
    or the process restarts; an explicit reload bypasses by varying
    the prompt or context.
    """
    prompt, context, _model = key
    blurb: BlurbAgent = default_blurb_agent()
    try:
        return blurb.context_answer(prompt, context=context) or "[AI] empty response"
    except Exception as exc:  # noqa: BLE001 — full trace stays server-side
        logger.warning(
            "[llm] generate_insight failed: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return f"[AI] {AI_FAILURE_MESSAGE}"


def generate_insight(prompt: str, timeout: int | None = None, active_tab: str | None = None) -> str:
    """
    Public entry point used by the Dash tabs. Same signature as before;
    returns the LLM response as a plain string.
    """
    context = _resolve_context(active_tab, prompt)
    # Include the agent model in the cache key so a model swap via env
    # triggers fresh generations without restarting the app.
    blurb = default_blurb_agent()
    return _cached_invoke((prompt, context, blurb.agent.model))
