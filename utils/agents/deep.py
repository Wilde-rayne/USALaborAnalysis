"""
LangChain DeepAgents harness — phi3-backed "real" agents.

DeepAgents wraps the basic LLM-plus-tools loop with planning, a
virtual filesystem scratchpad, subagent spawning, and
interrupt-based human-in-the-loop review. Anywhere the user asks
for an "agent" (as opposed to a one-shot chat call), it should be
a deep agent — by default using phi3 via Ollama.

The ``chat_agent()`` and ``worker_agent()`` factories elsewhere in
this package are *not* deep agents; they stay as plain
``ChatOllama`` wrappers because their calls are one-shot
prompt → response and DeepAgents' planning overhead is pure cost
in that case.

Public API::

    from utils.agents.deep import (
        create_labor_deep_agent,
        data_refresh_agent,
    )
"""
from __future__ import annotations

import logging
import os
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Provider-prefixed model string expected by DeepAgents' ``model=`` arg.
# DeepAgents uses LangChain's init_chat_model under the hood, which
# understands the ``provider:name`` convention.
DEFAULT_AGENT_MODEL_SPEC = "ollama:phi3"


def _resolve_model_spec(model: str | None) -> str:
    """Pick the model string; prefer explicit, then OLLAMA_AGENT_MODEL env."""
    if model:
        return model
    env_val = os.getenv("OLLAMA_AGENT_MODEL", "").strip()
    if not env_val:
        return DEFAULT_AGENT_MODEL_SPEC
    return env_val if ":" in env_val else f"ollama:{env_val}"


def create_labor_deep_agent(
    *,
    tools: Sequence[Any],
    system_prompt: str,
    model: str | None = None,
) -> Any:
    """
    Build a ``deepagents.create_deep_agent`` bound to the labor-market
    toolbelt and the project's default phi3 model.

    Returns the raw DeepAgents agent object — callers invoke it with
    ``agent.invoke({"messages": [{"role": "user", "content": "..."}]})``
    per the DeepAgents docs.

    Raises :class:`ImportError` with a pointed message if deepagents
    isn't installed, so a CI path without the package still loads
    the rest of the package cleanly.
    """
    try:
        from deepagents import create_deep_agent  # noqa: PLC0415
    except ImportError as exc:  # noqa: F841 — re-raised with context
        raise ImportError(
            "deepagents is not installed. Add it to requirements.txt or "
            "``pip install deepagents`` to use this harness."
        ) from exc

    resolved = _resolve_model_spec(model)
    logger.info(
        f"[deep-agent] model={resolved} tools=[{','.join(t.name for t in tools)}]"
    )
    return create_deep_agent(
        model=resolved,
        tools=list(tools),
        system_prompt=system_prompt,
    )


# --------------------------------------------------------------------------
# Built-in deep agents
# --------------------------------------------------------------------------
_DATA_REFRESH_PROMPT = """\
You are a data-refresh agent for a US labor-market dashboard.

You have tools that:
- fetch BLS CES (Current Employment Statistics) for given states and year ranges
- fetch BLS LAUS (Local Area Unemployment Statistics) for given states and year ranges
- fetch US Census population estimates for given states and year ranges
- ensure the merged panel (data/all_data.json) is fresh
- describe a raw BLS series id in plain English

Rules:
- Always normalize and validate state codes — reject unknown codes with
  a clear message instead of silently skipping them.
- If the user just wants "everything up to date", prefer
  ``ensure_merged_data`` over running each fetcher individually.
- When fetching a large state list, consider using the planning
  ``write_todos`` tool to sketch the steps before calling fetchers.
- After every tool call, briefly say what you did and why.
- End with a one-line summary of what's now fresh.
"""


def data_refresh_agent(model: str | None = None) -> Any:
    """
    Ready-to-invoke deep agent bound to the Phase C5 fetch toolbelt.

    Lazily imported at call time so ``from utils.agents.deep import
    data_refresh_agent`` doesn't drag in LangGraph for unrelated
    callers.
    """
    from utils.agents.tools import ALL_TOOLS  # noqa: PLC0415

    return create_labor_deep_agent(
        tools=ALL_TOOLS,
        system_prompt=_DATA_REFRESH_PROMPT,
        model=model,
    )
