"""
Tests for the agent harness configuration layer.

Any test that actually invokes the LLM belongs in an integration
suite — those need a running Ollama container with the referenced
models pulled. These tests cover the plumbing: config dataclass,
factory defaults, lazy LLM init.
"""
from __future__ import annotations

import os

import pytest

from utils.agents.base import LaborAgent, LaborAgentConfig
from utils.agents.ollama import chat_agent, worker_agent


class TestLaborAgentConfig:
    def test_is_frozen_and_hashable(self) -> None:
        c1 = LaborAgentConfig(model="phi3", system_prompt="hi")
        c2 = LaborAgentConfig(model="phi3", system_prompt="hi")
        assert c1 == c2
        with pytest.raises(Exception):
            c1.model = "other"  # frozen dataclass blocks rebind

    def test_defaults(self) -> None:
        c = LaborAgentConfig(model="phi3", system_prompt="hi")
        assert c.temperature == 0.2
        assert c.base_url == "http://ollama:11434"
        assert c.timeout == 180


class TestLaborAgent:
    def test_construction_does_not_contact_ollama(self) -> None:
        """LLM client init is lazy — building the agent shouldn't error."""
        agent = LaborAgent(
            LaborAgentConfig(
                model="phi3",
                system_prompt="hi",
                base_url="http://127.0.0.1:59999",  # guaranteed unreachable
            )
        )
        assert agent.model == "phi3"
        # The underlying ChatOllama isn't built yet.
        assert agent._llm is None

    def test_describe_includes_model_name(self) -> None:
        agent = LaborAgent(LaborAgentConfig(model="phi3", system_prompt="hi"))
        desc = agent.describe()
        assert "phi3" in desc

    def test_invoke_with_tools_falls_back_to_plain_when_tools_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No tools ⇒ invoke_with_tools just calls invoke."""
        agent = LaborAgent(LaborAgentConfig(model="phi3", system_prompt="hi"))
        called: list[str] = []

        def fake_invoke(self, user_input, **_):
            called.append(user_input)
            return "ok"

        monkeypatch.setattr(LaborAgent, "invoke", fake_invoke)
        result = agent.invoke_with_tools("hello")
        assert result == "ok"
        assert called == ["hello"]


class TestFactories:
    def test_chat_agent_honours_ollama_model_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_MODEL", "custom:chat")
        monkeypatch.setenv("OLLAMA_URL", "http://ollama:11434")
        agent = chat_agent()
        assert agent.model == "custom:chat"

    def test_worker_agent_honours_agent_model_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_AGENT_MODEL", "worker:latest")
        agent = worker_agent()
        assert agent.model == "worker:latest"

    def test_explicit_model_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_MODEL", "env-model")
        agent = chat_agent(model="explicit-model")
        assert agent.model == "explicit-model"

    def test_chat_and_worker_use_distinct_prompts(self) -> None:
        c = chat_agent()
        w = worker_agent()
        assert c.config.system_prompt != w.config.system_prompt
