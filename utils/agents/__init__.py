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
from utils.agents.blurb import BlurbAgent, default_blurb_agent
from utils.agents.deep import (
    DEFAULT_AGENT_MODEL_SPEC,
    create_labor_deep_agent,
    data_refresh_agent,
)
from utils.agents.ollama import (
    AGENT_MODEL_ENV,
    CHAT_MODEL_ENV,
    OLLAMA_BASE_URL,
    chat_agent,
    worker_agent,
)
from utils.agents.sentence_rag import SentenceRAGBuilder

__all__ = [
    "AGENT_MODEL_ENV",
    "BlurbAgent",
    "CHAT_MODEL_ENV",
    "DEFAULT_AGENT_MODEL_SPEC",
    "LaborAgent",
    "LaborAgentConfig",
    "OLLAMA_BASE_URL",
    "SentenceRAGBuilder",
    "chat_agent",
    "create_labor_deep_agent",
    "data_refresh_agent",
    "default_blurb_agent",
    "worker_agent",
]
