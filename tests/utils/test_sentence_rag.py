"""
Unit tests for utils.agents.sentence_rag.

Deterministic layers (facts, rankings, trends) run in every CI path.
"""
from __future__ import annotations

import pytest

from utils.agents.sentence_rag import (
    CONTEXT_SNIPPETS,
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

    def test_context_snippets_appended_by_default(self, sample_records) -> None:
        corpus = SentenceRAGBuilder().build_corpus(sample_records)
        for snippet in CONTEXT_SNIPPETS:
            assert snippet in corpus

    def test_context_snippets_can_be_opted_out(self, sample_records) -> None:
        corpus = SentenceRAGBuilder().build_corpus(sample_records, extra_snippets=())
        for snippet in CONTEXT_SNIPPETS:
            assert snippet not in corpus

    def test_custom_extra_snippets_replace_defaults(self, sample_records) -> None:
        custom = ("A custom grounding sentence.",)
        corpus = SentenceRAGBuilder().build_corpus(
            sample_records, extra_snippets=custom
        )
        assert "A custom grounding sentence." in corpus
        assert CONTEXT_SNIPPETS[0] not in corpus


class TestContextSnippets:
    def test_non_empty_and_well_formed(self) -> None:
        assert len(CONTEXT_SNIPPETS) >= 6
        for snippet in CONTEXT_SNIPPETS:
            assert snippet.strip() == snippet
            assert snippet.endswith(".")
            # Each snippet stays short enough to embed as one passage.
            assert len(snippet) < 400

    def test_each_snippet_names_a_source(self) -> None:
        """Every snippet must carry attribution the narrative LLM can cite."""
        sources = (
            "Estrella", "FRED", "Treasury", "H.15", "Federal Reserve",
            "FOMC", "Beige Book", "CME", "Nasdaq", "federalreserve.gov",
            "cmegroup.com", "data.nasdaq.com",
        )
        unattributed = [
            s for s in CONTEXT_SNIPPETS
            if not any(token in s for token in sources)
        ]
        # The rates→labor mechanism sentence is general macro consensus,
        # deliberately hedged ("typically") rather than source-pinned.
        assert len(unattributed) <= 1


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
