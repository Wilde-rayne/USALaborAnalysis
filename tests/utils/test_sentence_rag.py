"""
Unit tests for utils.agents.sentence_rag.

Deterministic layers (facts, rankings, trends) run in every CI path.
``agent_polish`` is tested against a mocked agent to avoid a real
Ollama round-trip.
"""
from __future__ import annotations

from typing import Any

import pytest

from utils.agents.sentence_rag import (
    SentenceRAGBuilder,
    _is_lower_better,
    _ordinal,
)
from utils.ontology import ONTOLOGY


@pytest.fixture
def sample_records() -> list[dict]:
    """Small panel: IA + IL + MN, 2019 + 2020, LAUS + LFPR + Population."""
    return [
        # 2019
        {"state": "IA", "year": 2019, "period": "M01",
         "Labor_Force": 1600000, "Population": 3150000, "LFPR": 50.8,
         "Unemployment_Rate": 3.1},
        {"state": "IL", "year": 2019, "period": "M01",
         "Labor_Force": 6500000, "Population": 12700000, "LFPR": 51.2,
         "Unemployment_Rate": 4.3},
        {"state": "MN", "year": 2019, "period": "M01",
         "Labor_Force": 3100000, "Population": 5700000, "LFPR": 54.4,
         "Unemployment_Rate": 3.2},
        # 2020 — pandemic dip
        {"state": "IA", "year": 2020, "period": "M01",
         "Labor_Force": 1500000, "Population": 3160000, "LFPR": 47.5,
         "Unemployment_Rate": 6.1},
        {"state": "IL", "year": 2020, "period": "M01",
         "Labor_Force": 6100000, "Population": 12680000, "LFPR": 48.1,
         "Unemployment_Rate": 9.8},
        {"state": "MN", "year": 2020, "period": "M01",
         "Labor_Force": 2950000, "Population": 5710000, "LFPR": 51.7,
         "Unemployment_Rate": 6.8},
    ]


class TestFactSentences:
    def test_one_sentence_per_state_year_measure(self, sample_records) -> None:
        builder = SentenceRAGBuilder()
        sents = builder.fact_sentences(sample_records)
        # 3 states × 2 years × 4 present measures (LF, Pop, LFPR, U-rate)
        assert len(sents) == 24
        assert any("Iowa" in s and "2020" in s for s in sents)
        assert any("Illinois" in s and "2019" in s for s in sents)

    def test_sentence_includes_unit(self, sample_records) -> None:
        sents = SentenceRAGBuilder().fact_sentences(sample_records)
        lfpr = next(s for s in sents if "participation rate" in s)
        assert "%" in lfpr
        labor_force = next(s for s in sents if "Iowa labor force" in s)
        assert "persons" in labor_force

    def test_drops_records_missing_state_or_year(self) -> None:
        bad = [
            {"state": None, "year": 2020, "Labor_Force": 1},
            {"state": "IA", "year": None, "Labor_Force": 1},
            {"state": "IA", "year": 2020, "Labor_Force": None},
        ]
        sents = SentenceRAGBuilder().fact_sentences(bad)
        assert sents == []

    def test_unknown_state_code_is_ignored(self) -> None:
        sents = SentenceRAGBuilder().fact_sentences(
            [{"state": "ZZ", "year": 2020, "Labor_Force": 1}]
        )
        assert sents == []


class TestRankingSentences:
    def test_rank_across_states_for_a_year(self, sample_records) -> None:
        sents = SentenceRAGBuilder().ranking_sentences(sample_records)
        # At least one sentence per (year, metric) × N states.
        # Year 2019, LFPR: MN (54.4) > IL (51.2) > IA (50.8). Highest = 1st.
        lfpr_2019 = [s for s in sents if "2019" in s and "participation rate" in s]
        assert any("Minnesota" in s and "1st" in s for s in lfpr_2019)
        assert any("Iowa" in s and "3rd" in s for s in lfpr_2019)

    def test_unemployment_rate_ranks_lower_is_better(self, sample_records) -> None:
        sents = SentenceRAGBuilder().ranking_sentences(sample_records)
        # 2019 unemployment: IA 3.1 < MN 3.2 < IL 4.3 → IA is 1st, IL is 3rd
        u_2019 = [s for s in sents if "2019" in s and "unemployment rate" in s]
        assert any("Iowa" in s and "1st" in s for s in u_2019)
        assert any("Illinois" in s and "3rd" in s for s in u_2019)


class TestTrendSentences:
    def test_captures_direction_and_delta(self, sample_records) -> None:
        sents = SentenceRAGBuilder().trend_sentences(sample_records)
        # Iowa labor force 2019→2020: 1.6M → 1.5M (fell, -6.25%)
        ia_lf = next(s for s in sents if s.startswith("Iowa labor force"))
        assert "fell" in ia_lf
        assert "2019" in ia_lf and "2020" in ia_lf

    def test_positive_direction_for_rising_unemployment(self, sample_records) -> None:
        sents = SentenceRAGBuilder().trend_sentences(sample_records)
        u_rows = [s for s in sents if "unemployment rate" in s]
        # All three states saw u-rate rise in the pandemic year.
        assert sum(1 for s in u_rows if "rose" in s) >= 3


class TestBuildCorpus:
    def test_combines_all_three_layers(self, sample_records) -> None:
        corpus = SentenceRAGBuilder().build_corpus(sample_records)
        # Non-empty and all three sentence styles present.
        assert len(corpus) > 0
        assert any("averaged" in s for s in corpus)            # fact
        assert any("ranked" in s for s in corpus)              # ranking
        assert any("rose" in s or "fell" in s for s in corpus) # trend

    def test_polish_off_by_default_keeps_input(self, sample_records) -> None:
        builder = SentenceRAGBuilder()  # no agent
        corpus_plain = builder.build_corpus(sample_records)
        corpus_polished = builder.build_corpus(sample_records, polish=True)
        # No agent ⇒ polish is a no-op.
        assert corpus_plain == corpus_polished


class TestAgentPolish:
    def test_no_agent_returns_inputs_unchanged(self) -> None:
        builder = SentenceRAGBuilder()
        inp = ["Alpha.", "Beta."]
        assert builder.agent_polish(inp) == inp

    def test_agent_polish_invokes_once_per_sentence(self) -> None:
        calls: list[str] = []

        class FakeAgent:
            def invoke(self, prompt: str, **_: Any) -> str:
                calls.append(prompt)
                return "polished: " + prompt.split('"')[-2]  # extract Input

        builder = SentenceRAGBuilder(agent=FakeAgent())  # type: ignore[arg-type]
        out = builder.agent_polish(["Alpha.", "Beta."])
        assert len(out) == 2
        assert out[0].startswith("polished:")
        assert out[1].startswith("polished:")
        assert len(calls) == 2

    def test_agent_polish_falls_back_on_empty_response(self) -> None:
        class EmptyAgent:
            def invoke(self, prompt: str, **_: Any) -> str:
                return "   "  # blank

        builder = SentenceRAGBuilder(agent=EmptyAgent())  # type: ignore[arg-type]
        out = builder.agent_polish(["Original."])
        # Blank response → keep the original text.
        assert out == ["Original."]


class TestHelpers:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"),
         (11, "11th"), (12, "12th"), (13, "13th"),
         (21, "21st"), (22, "22nd"), (23, "23rd"), (101, "101st")],
    )
    def test_ordinal(self, n: int, expected: str) -> None:
        assert _ordinal(n) == expected

    def test_lower_better_flags_unemployment(self) -> None:
        assert _is_lower_better(ONTOLOGY.measure("Unemployment")) is True
        assert _is_lower_better(ONTOLOGY.measure("Unemployment_Rate")) is True
        assert _is_lower_better(ONTOLOGY.measure("Labor_Force")) is False
        assert _is_lower_better(ONTOLOGY.measure("LFPR")) is False
