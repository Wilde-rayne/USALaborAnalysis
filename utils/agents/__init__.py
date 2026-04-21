"""
LangChain-based agent harness.

Two flavors ship by default:

- ``chat_agent()`` — user-facing, backed by the OLLAMA_MODEL env
  (llama3.2:3b). Used for tab insight blurbs and the chat sidebar.
- ``worker_agent()`` — internal, backed by OLLAMA_AGENT_MODEL env
  (phi3). Used for sentence-RAG preprocessing, label generation, and
  any task where latency > quality.

Both return :class:`LaborAgent` instances that expose ``invoke(prompt)``
and ``invoke_with_tools(prompt)`` — the latter wires in LangChain's
tool-calling loop once the fetch agents (Phase C5) come online.
"""
from utils.agents.base import LaborAgent, LaborAgentConfig
from utils.agents.ollama import (
    AGENT_MODEL_ENV,
    CHAT_MODEL_ENV,
    OLLAMA_BASE_URL,
    chat_agent,
    worker_agent,
)

__all__ = [
    "AGENT_MODEL_ENV",
    "CHAT_MODEL_ENV",
    "LaborAgent",
    "LaborAgentConfig",
    "OLLAMA_BASE_URL",
    "chat_agent",
    "worker_agent",
]
