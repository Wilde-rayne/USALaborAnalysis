"""
Built-in :class:`LaborAgent` factories tied to the Ollama deployment.

``chat_agent()`` — user-facing narrative blurbs, uses the larger chat
model (``OLLAMA_MODEL``). ``worker_agent()`` — internal helpers,
uses the smaller fast model (``OLLAMA_AGENT_MODEL``). Both read
connection details from env so the same code runs in dev and CI.
"""
from __future__ import annotations

import os

from utils.agents.base import LaborAgent, LaborAgentConfig
from utils.constants import DEFAULT_TIMEOUT

CHAT_MODEL_ENV = "OLLAMA_MODEL"
AGENT_MODEL_ENV = "OLLAMA_AGENT_MODEL"
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")


_CHAT_SYSTEM_PROMPT = (
    "You are a data analyst helping interpret US labor statistics for a "
    "regional policy audience. Speak plainly and cite numbers from the "
    "context when you have them. If the context doesn't support a claim, "
    "say so instead of guessing. Keep answers under six sentences unless "
    "asked for more detail."
)

_WORKER_SYSTEM_PROMPT = (
    "You are a background helper that turns structured labor-market data "
    "into crisp natural-language sentences. Respond with the requested "
    "text only — no preamble, no JSON, no quotation marks."
)


def chat_agent(
    model: str | None = None,
    *,
    system_prompt: str = _CHAT_SYSTEM_PROMPT,
    temperature: float = 0.2,
    timeout: int = DEFAULT_TIMEOUT,
) -> LaborAgent:
    """User-facing chat agent (defaults to llama3.2:3b via OLLAMA_MODEL)."""
    config = LaborAgentConfig(
        model=model or os.getenv(CHAT_MODEL_ENV, "llama3.2:3b"),
        system_prompt=system_prompt,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        timeout=timeout,
    )
    return LaborAgent(config)


def worker_agent(
    model: str | None = None,
    *,
    system_prompt: str = _WORKER_SYSTEM_PROMPT,
    temperature: float = 0.1,
    timeout: int = DEFAULT_TIMEOUT,
) -> LaborAgent:
    """Background worker agent (defaults to phi3 via OLLAMA_AGENT_MODEL)."""
    config = LaborAgentConfig(
        model=model or os.getenv(AGENT_MODEL_ENV, "phi3"),
        system_prompt=system_prompt,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        timeout=timeout,
    )
    return LaborAgent(config)
