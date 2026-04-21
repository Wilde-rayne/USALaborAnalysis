"""
Unit tests for the BlurbAgent.

All tests run with a fake LaborAgent so they don't need Ollama.
Verifies prompt construction — the value the class adds is packaging
structured data into well-formed natural-language prompts.
"""
from __future__ import annotations

from typing import Any

import pytest

from utils.agents.blurb import BlurbAgent


class FakeAgent:
    """Records the prompt and returns a canned reply."""

    model: str = "fake-model"

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.calls: list[str] = []

    def invoke(self, prompt: str, **_: Any) -> str:
        self.calls.append(prompt)
        return self.reply


class TestForecastBlurb:
    def test_prompt_includes_entity_measure_and_model(self) -> None:
        fake = FakeAgent()
        ba = BlurbAgent(agent=fake)  # type: ignore[arg-type]
        ba.forecast_blurb(
            entity="Iowa",
            measure="LFPR",
            history=[50.0, 51.0, 52.0],
            forecast=[52.5, 52.8, 53.0],
            horizon_months=24,
            model_name="ets",
        )
        p = fake.calls[0]
        assert "Iowa" in p
        assert "labor force participation rate" in p.lower()
        assert "ets" in p
        assert "24 months" in p

    def test_prompt_summarises_numeric_range(self) -> None:
        fake = FakeAgent()
        ba = BlurbAgent(agent=fake)  # type: ignore[arg-type]
        ba.forecast_blurb(
            entity="Ohio",
            measure="Labor_Force",
            history=[100.0, 200.0, 50.0],
            forecast=[60.0, 70.0],
            horizon_months=12,
            model_name="naive",
        )
        p = fake.calls[0]
        # first / last / range / n should all be in the prompt
        assert "first=100" in p
        assert "last=50" in p
        assert "range=[50" in p and "200" in p
        assert "n=3" in p

    def test_handles_unknown_measure_gracefully(self) -> None:
        fake = FakeAgent()
        ba = BlurbAgent(agent=fake)  # type: ignore[arg-type]
        ba.forecast_blurb(
            entity="PR",
            measure="UnknownMeasure_XYZ",
            history=[1, 2],
            forecast=[3],
            horizon_months=6,
            model_name="naive",
        )
        # Fell through to the snake_case-lower fallback.
        p = fake.calls[0]
        assert "unknownmeasure xyz" in p.lower()

    def test_includes_diagnostics_when_provided(self) -> None:
        fake = FakeAgent()
        ba = BlurbAgent(agent=fake)  # type: ignore[arg-type]
        ba.forecast_blurb(
            entity="Iowa",
            measure="LFPR",
            history=[50.0],
            forecast=[51.0],
            horizon_months=12,
            model_name="ets",
            diagnostics_summary="ADF p=0.02, DM p=0.001",
        )
        assert "ADF p=0.02" in fake.calls[0]

    def test_empty_history_and_forecast(self) -> None:
        fake = FakeAgent()
        ba = BlurbAgent(agent=fake)  # type: ignore[arg-type]
        ba.forecast_blurb(
            entity="Iowa",
            measure="LFPR",
            history=[],
            forecast=[],
            horizon_months=12,
            model_name="naive",
        )
        assert "no data" in fake.calls[0]


class TestComparisonBlurb:
    def test_prompt_sorted_and_includes_values(self) -> None:
        fake = FakeAgent()
        ba = BlurbAgent(agent=fake)  # type: ignore[arg-type]
        ba.comparison_blurb(
            measure="Sector_Employment",
            horizon_months=24,
            by_entity={"IA": 220.0, "IL": 580.0, "WI": 470.0},
        )
        p = fake.calls[0]
        # Lines sorted desc by value.
        p_ia = p.index("IA")
        p_il = p.index("IL")
        p_wi = p.index("WI")
        # IL > WI > IA in the value ordering => IL must come first.
        assert p_il < p_wi < p_ia
        # Values formatted with thousand separator.
        assert "580.0" in p and "470.0" in p and "220.0" in p

    def test_winners_included_when_provided(self) -> None:
        fake = FakeAgent()
        ba = BlurbAgent(agent=fake)  # type: ignore[arg-type]
        ba.comparison_blurb(
            measure="Sector_Employment",
            horizon_months=24,
            by_entity={"IA": 220.0, "IL": 580.0},
            winners={"IA": "ets", "IL": "naive"},
        )
        p = fake.calls[0]
        assert "IA ← ets" in p
        assert "IL ← naive" in p


class TestContextAnswer:
    def test_with_context_wraps_prompt(self) -> None:
        fake = FakeAgent()
        ba = BlurbAgent(agent=fake)  # type: ignore[arg-type]
        ba.context_answer("How is Iowa doing?", context="Iowa LFPR averaged 52%.")
        p = fake.calls[0]
        assert "Iowa LFPR averaged 52%." in p
        assert "Question: How is Iowa doing?" in p

    def test_without_context_forwards_question(self) -> None:
        fake = FakeAgent()
        ba = BlurbAgent(agent=fake)  # type: ignore[arg-type]
        ba.context_answer("Hello?")
        assert fake.calls == ["Hello?"]


class TestDefaultBlurbAgent:
    def test_returns_a_blurb_like_singleton(self) -> None:
        """
        Must not error on first call; lazy init should be safe.

        Duck-typed rather than ``isinstance`` because the smoke-test
        helper in tests/test_smoke.py pops ``utils.*`` from
        ``sys.modules``, which would make the class objects in this
        test not identical to the class the singleton was instantiated
        against even though they live at the same path.
        """
        from utils.agents.blurb import default_blurb_agent

        ba = default_blurb_agent()
        assert ba is not None
        assert hasattr(ba, "agent")
        assert callable(getattr(ba, "forecast_blurb", None))
        assert callable(getattr(ba, "comparison_blurb", None))
        assert callable(getattr(ba, "context_answer", None))
        # Same singleton on second call.
        assert default_blurb_agent() is ba
