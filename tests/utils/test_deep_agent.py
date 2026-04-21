"""
Tests for the DeepAgents harness facade.

The DeepAgents package itself isn't installed in every test path, so
the module-level importorskip keeps the lightweight CI green. Tests
that don't touch the external library (resolution logic, import
error messaging) live OUTSIDE the skip so they always run.
"""
from __future__ import annotations

import pytest

from utils.agents.deep import DEFAULT_AGENT_MODEL_SPEC, _resolve_model_spec


class TestResolveModelSpec:
    def test_explicit_model_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_AGENT_MODEL", "env-model")
        assert _resolve_model_spec("ollama:custom") == "ollama:custom"

    def test_falls_back_to_env_and_adds_provider_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_AGENT_MODEL", "phi3")
        # Env value missing provider prefix gets "ollama:" prepended.
        assert _resolve_model_spec(None) == "ollama:phi3"

    def test_env_with_provider_passed_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_AGENT_MODEL", "ollama:mistral")
        assert _resolve_model_spec(None) == "ollama:mistral"

    def test_empty_env_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_AGENT_MODEL", "")
        assert _resolve_model_spec(None) == DEFAULT_AGENT_MODEL_SPEC

    def test_default_is_phi3(self) -> None:
        assert DEFAULT_AGENT_MODEL_SPEC == "ollama:phi3"


class TestImportErrorMessage:
    def test_create_agent_raises_actionable_error_without_deepagents(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the deepagents package is missing, the error points at pip."""
        import sys

        # Pretend deepagents isn't installed — clear any cached submodule.
        monkeypatch.setitem(sys.modules, "deepagents", None)

        from utils.agents.deep import create_labor_deep_agent

        with pytest.raises(ImportError, match="deepagents is not installed"):
            # Passing empty tools; the import error must fire before any
            # argument-validation inside deepagents.create_deep_agent.
            create_labor_deep_agent(tools=[], system_prompt="hi")


# --------------------------------------------------------------------------
# Integration — only runs when the real library is installed.
# --------------------------------------------------------------------------
_HAS_DEEPAGENTS = True
try:
    import deepagents  # type: ignore[import-not-found]  # noqa: F401
    import langchain_core  # type: ignore[import-not-found]  # noqa: F401
except ImportError:
    _HAS_DEEPAGENTS = False


@pytest.mark.skipif(
    not _HAS_DEEPAGENTS, reason="deepagents / langchain_core not installed"
)
class TestDataRefreshAgentIntegration:
    def test_build_data_refresh_agent_returns_invokable(self) -> None:
        """
        Smoke test: the factory returns an object with an ``invoke``
        method (the DeepAgents contract). We don't actually call it —
        that would hit Ollama.
        """
        from utils.agents import data_refresh_agent

        agent = data_refresh_agent()
        assert hasattr(agent, "invoke")
