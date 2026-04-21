"""
Tests for utils.agents.tools.

Every fetcher is monkey-patched so the tools can be exercised without
hitting BLS or Census. These tests verify tool metadata (names,
descriptions, arg schemas) and the validation logic inside each
tool's wrapper (state-code normalization, year-range checks).
"""
from __future__ import annotations

from typing import Any

import pytest

# The tools module uses the ``@tool`` decorator from ``langchain_core``.
# Skip this whole file if the lightweight test environment doesn't have
# the LangChain stack installed.
pytest.importorskip("langchain_core")

from utils.agents import tools as tools_mod  # noqa: E402


# --------------------------------------------------------------------------
# Tool registry metadata
# --------------------------------------------------------------------------
class TestToolRegistry:
    def test_exposes_five_tools_by_default(self) -> None:
        assert len(tools_mod.ALL_TOOLS) == 5
        names = {t.name for t in tools_mod.ALL_TOOLS}
        assert names == {
            "fetch_bls_ces",
            "fetch_bls_laus",
            "fetch_census_population",
            "ensure_merged_data",
            "describe_series",
        }

    def test_every_tool_has_description(self) -> None:
        for t in tools_mod.ALL_TOOLS:
            assert t.description, f"tool {t.name} missing description"
            # Descriptions should mention what the tool does in plain English.
            assert len(t.description) > 30


# --------------------------------------------------------------------------
# Argument validation (pure Python, no LLM/IO)
# --------------------------------------------------------------------------
class TestStateNormalization:
    def test_lowercase_codes_are_upcased(self) -> None:
        assert tools_mod._normalize_states(["ia", "il"]) == ["IA", "IL"]

    def test_unknown_state_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown state code"):
            tools_mod._normalize_states(["ZZ"])

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            tools_mod._normalize_states([])


class TestYearRangeValidation:
    def test_accepts_valid_range(self) -> None:
        tools_mod._validate_year_range(1996, 2024)

    def test_rejects_non_int(self) -> None:
        with pytest.raises(ValueError, match="integers"):
            tools_mod._validate_year_range("1996", 2024)  # type: ignore[arg-type]

    def test_rejects_too_early(self) -> None:
        with pytest.raises(ValueError, match="too early"):
            tools_mod._validate_year_range(1900, 1910)

    def test_rejects_inverted_range(self) -> None:
        with pytest.raises(ValueError, match=">= start_year"):
            tools_mod._validate_year_range(2020, 2019)


# --------------------------------------------------------------------------
# Tool behaviour (with monkey-patched fetchers)
# --------------------------------------------------------------------------
class TestFetchTools:
    def test_fetch_bls_ces_invokes_fetcher(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[Any, ...]] = []

        def fake_fetch(states, start, end):
            calls.append((list(states), start, end))

        monkeypatch.setattr("utils.fetch_ces_data.fetch_ces_data", fake_fetch)
        result = tools_mod.fetch_bls_ces.invoke(
            {"states": ["ia", "il"], "start_year": 2020, "end_year": 2021}
        )
        assert calls == [(["IA", "IL"], 2020, 2021)]
        assert "Fetched BLS CES" in result
        assert "IA" in result and "IL" in result

    def test_fetch_bls_laus_invokes_fetcher(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[Any, ...]] = []

        def fake_fetch(states, start, end):
            calls.append((list(states), start, end))

        monkeypatch.setattr("utils.fetch_laus_data.fetch_laus_data", fake_fetch)
        result = tools_mod.fetch_bls_laus.invoke(
            {"states": ["IA"], "start_year": 2020, "end_year": 2020}
        )
        assert calls == [(["IA"], 2020, 2020)]
        assert "Fetched BLS LAUS" in result

    def test_fetch_census_population_invokes_fetcher(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[Any, ...]] = []

        def fake_fetch(states, start, end):
            calls.append((list(states), start, end))

        monkeypatch.setattr("utils.fetch_population_data.fetch_population", fake_fetch)
        result = tools_mod.fetch_census_population.invoke(
            {"states": ["IA"], "start_year": 2020, "end_year": 2020}
        )
        assert calls == [(["IA"], 2020, 2020)]
        assert "Fetched Census population" in result

    def test_ensure_merged_data_routes_through_ensure_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[bool] = []

        def fake_ensure(states=None, start_year=None, end_year=None, *, force=False):
            called.append(force)
            return "/tmp/all_data.json"

        monkeypatch.setattr("utils.data_pipeline.ensure_data", fake_ensure)
        result = tools_mod.ensure_merged_data.invoke({"force": True})
        assert called == [True]
        assert "/tmp/all_data.json" in result


class TestDescribeSeriesTool:
    def test_laus_series_described(self) -> None:
        out = tools_mod.describe_series.invoke({"series_id": "LASST190000000000006"})
        assert "Iowa" in out
        assert "labor force" in out.lower()

    def test_unknown_series_returns_error_string(self) -> None:
        """LangChain tools catch and surface exceptions back to the agent."""
        with pytest.raises(ValueError):
            tools_mod.describe_series.invoke({"series_id": "BOGUS"})


class TestInvalidInputsSurfaceFromTool:
    def test_fetch_rejects_unknown_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[Any] = []

        def fake_fetch(*a, **k):
            called.append(a)

        monkeypatch.setattr("utils.fetch_ces_data.fetch_ces_data", fake_fetch)
        with pytest.raises(ValueError, match="unknown state code"):
            tools_mod.fetch_bls_ces.invoke(
                {"states": ["ZZ"], "start_year": 2020, "end_year": 2020}
            )
        # Fetcher must not have been called.
        assert called == []
