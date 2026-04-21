"""
Base agent + configuration types.

Keep this file pure-Python with a lazy ``langchain`` import inside the
LaborAgent class so the module is cheap to load in tests that only
exercise configuration plumbing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LaborAgentConfig:
    """Inputs to :class:`LaborAgent` — trivially serializable."""

    model: str
    system_prompt: str
    base_url: str = "http://ollama:11434"
    temperature: float = 0.2
    timeout: int = 180
    tool_names: tuple[str, ...] = ()


class LaborAgent:
    """
    Thin LangChain-ChatOllama wrapper shared by the chat / worker /
    fetch agents.

    The constructor is side-effect-free; the underlying
    ``ChatOllama`` client is created on first ``.invoke`` call so
    unit tests can instantiate the agent without statsmodels-style
    heavy-import taxes.
    """

    def __init__(
        self,
        config: LaborAgentConfig,
        tools: Sequence[Any] | None = None,
    ) -> None:
        self.config = config
        self.tools = tuple(tools) if tools else ()
        self._llm: Any = None
        self._chain: Any = None

    # --- lazy init -----------------------------------------------------
    def _get_llm(self):
        if self._llm is None:
            from langchain_ollama import ChatOllama  # noqa: PLC0415

            self._llm = ChatOllama(
                model=self.config.model,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                # LangChain's ChatOllama uses ``num_predict`` for max tokens,
                # which is a per-request knob we leave at default. Request
                # timeout is surfaced via the underlying httpx client.
                client_kwargs={"timeout": float(self.config.timeout)},
            )
        return self._llm

    def _get_chain(self):
        if self._chain is None:
            from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.config.system_prompt),
                    ("human", "{input}"),
                ]
            )
            self._chain = prompt | self._get_llm()
        return self._chain

    # --- public API ----------------------------------------------------
    def invoke(self, user_input: str, **extra_vars: Any) -> str:
        """
        Run a single-turn chat call. ``extra_vars`` is forwarded to the
        prompt template so subclasses can inject context without
        rewriting the system prompt.

        Returns the assistant's message content as a plain string.
        """
        chain = self._get_chain()
        try:
            msg = chain.invoke({"input": user_input, **extra_vars})
        except Exception as exc:  # noqa: BLE001 — surfaced to callers
            logger.warning(f"[agent:{self.config.model}] invoke failed: {exc}")
            return f"[agent:{self.config.model} error] {exc}"
        # AIMessage.content can be a string or a list of blocks (multimodal).
        content = getattr(msg, "content", msg)
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    def invoke_with_tools(self, user_input: str, **_: Any) -> str:
        """
        Tool-calling loop. Stubbed until Phase C5 lands the fetch tools
        — returns the plain-chat answer with a log note for now so
        callers can adopt the method signature today without branching.
        """
        if not self.tools:
            return self.invoke(user_input)
        # Phase C5: build a LangGraph / AgentExecutor here and bind
        # ``self.tools``. For now fall back to plain chat.
        logger.info(
            f"[agent:{self.config.model}] invoke_with_tools stub — "
            f"{len(self.tools)} tools registered but tool loop not yet wired."
        )
        return self.invoke(user_input)

    # --- introspection -------------------------------------------------
    @property
    def model(self) -> str:
        return self.config.model

    def describe(self) -> str:
        return (
            f"LaborAgent(model={self.config.model}, "
            f"base_url={self.config.base_url}, "
            f"tools={[t.__class__.__name__ for t in self.tools]})"
        )
