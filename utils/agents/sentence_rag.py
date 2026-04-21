"""
Sentence-RAG builder.

Turns the tabular monthly panel into natural-language sentences before
embedding. Embedding models retrieve prose far better than column
dumps, so instead of feeding ``Labor_Force: 1500000; Population:
3150000`` into the vectoriser we feed::

    In 2020, Iowa labor force averaged 1,500,000 persons.
    Iowa's 2020 labor force participation rate of 47.6% ranked fourth
    among Midwest states, below Minnesota (58.2%) and North Dakota (63.4%).

The builder is split into three layers so callers can compose what
they need:

1. :meth:`fact_sentences` — deterministic "per (state, year, metric)"
   summaries. Cheap, runs at data-refresh time.
2. :meth:`ranking_sentences` — ontology-enriched comparative framing
   that gives the embedder context (peer groups, rank positions).
3. :meth:`agent_polish` — optional pass through the worker LLM for
   stylistic polish. Expensive; off by default.

``build_corpus`` wires the first two up; the agent pass stays opt-in
via the ``polish=True`` flag so the common case is zero-LLM-call.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable, Sequence

from utils.agents.base import LaborAgent
from utils.ontology import ONTOLOGY, Measure, Ontology, State

logger = logging.getLogger(__name__)


#: Metrics that map cleanly to a single-sentence summary per (state, year).
#: Everything else on the panel (per-sector wide columns, etc.) would bloat
#: the corpus without adding signal — see embeddings._RAG_METRICS history.
_RAG_MEASURES: tuple[str, ...] = (
    "Labor_Force",
    "Employment",
    "Unemployment",
    "Unemployment_Rate",
    "LFPR",
    "Population",
)


class SentenceRAGBuilder:
    """Ontology-aware sentence generator for the embedding pipeline."""

    def __init__(
        self,
        ontology: Ontology | None = None,
        agent: LaborAgent | None = None,
    ) -> None:
        self.ontology = ontology or ONTOLOGY
        self.agent = agent
        # Cache: {metric_key: Measure}
        self._measures = {m.key: m for m in self.ontology.measures.values()}

    # ------------------------------------------------------------------
    # Layer 1 — deterministic facts
    # ------------------------------------------------------------------
    def fact_sentences(self, records: Iterable[dict]) -> list[str]:
        """One sentence per (state, year, tracked metric), yearly mean."""
        agg: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for rec in records:
            state_code = rec.get("state")
            year = rec.get("year")
            if not state_code or year is None:
                continue
            for metric_key in _RAG_MEASURES:
                val = rec.get(metric_key)
                if val is None:
                    continue
                try:
                    agg[(state_code, int(year))][metric_key].append(float(val))
                except (TypeError, ValueError):
                    continue

        out: list[str] = []
        for (state_code, year), metrics in sorted(agg.items()):
            state = self.ontology.states.get(state_code)
            if state is None:
                continue
            for metric_key in _RAG_MEASURES:
                values = metrics.get(metric_key)
                if not values:
                    continue
                measure = self._measures.get(metric_key)
                if measure is None:
                    continue
                avg = sum(values) / len(values)
                out.append(
                    self._format_fact(state=state, year=year, measure=measure, value=avg)
                )
        return out

    def _format_fact(
        self, *, state: State, year: int, measure: Measure, value: float
    ) -> str:
        """Compose a single fact sentence. Values formatted by unit."""
        if measure.unit == "percent":
            number = f"{value:.1f}%"
        else:
            number = f"{value:,.0f} {measure.unit}"
        return f"In {year}, {state.name} {measure.name} averaged {number}."

    # ------------------------------------------------------------------
    # Layer 2 — ontology-enriched ranking + trend context
    # ------------------------------------------------------------------
    def ranking_sentences(self, records: Iterable[dict]) -> list[str]:
        """
        Per (year, metric), rank every state and emit a comparative sentence
        for each. These feed the embedder with peer-group context, which is
        what the chatbot most often needs.
        """
        # {(year, metric): {state_code: mean_value}}
        table: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
        for rec in records:
            state_code = rec.get("state")
            year = rec.get("year")
            if not state_code or year is None:
                continue
            try:
                year_i = int(year)
            except (TypeError, ValueError):
                continue
            for metric_key in _RAG_MEASURES:
                val = rec.get(metric_key)
                if val is None:
                    continue
                try:
                    val_f = float(val)
                except (TypeError, ValueError):
                    continue
                existing = table[(year_i, metric_key)].get(state_code, [])
                # accumulate monthly values; we'll average at render time
                if isinstance(existing, list):
                    existing.append(val_f)
                    table[(year_i, metric_key)][state_code] = existing
                else:
                    table[(year_i, metric_key)][state_code] = [existing, val_f]

        out: list[str] = []
        for (year, metric_key), state_vals in sorted(table.items()):
            measure = self._measures.get(metric_key)
            if measure is None:
                continue
            # Collapse month lists into year means.
            means = {
                code: sum(vals) / len(vals) if isinstance(vals, list) and vals else float(vals)
                for code, vals in state_vals.items()
                if (isinstance(vals, list) and vals) or isinstance(vals, (int, float))
            }
            if len(means) < 2:
                continue
            # Higher-is-better for most measures; lower is better for
            # unemployment-style measures so we flip sort accordingly.
            reverse = not _is_lower_better(measure)
            ranked = sorted(means.items(), key=lambda p: p[1], reverse=reverse)
            total = len(ranked)
            for rank_idx, (code, value) in enumerate(ranked, start=1):
                state = self.ontology.states.get(code)
                if state is None:
                    continue
                out.append(
                    self._format_rank(
                        state=state,
                        year=year,
                        measure=measure,
                        value=value,
                        rank=rank_idx,
                        total=total,
                    )
                )
        return out

    def _format_rank(
        self,
        *,
        state: State,
        year: int,
        measure: Measure,
        value: float,
        rank: int,
        total: int,
    ) -> str:
        value_str = (
            f"{value:.1f}%" if measure.unit == "percent" else f"{value:,.0f} {measure.unit}"
        )
        return (
            f"In {year}, {state.name}'s {measure.name} of {value_str} "
            f"ranked {_ordinal(rank)} of {total} jurisdictions in the dataset."
        )

    def trend_sentences(self, records: Iterable[dict]) -> list[str]:
        """
        For each (state, metric), describe the overall direction across the
        observed span. One sentence per (state, metric, earliest vs latest).
        """
        # {(state, metric): {year: [values]}}
        years_by_state_metric: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for rec in records:
            state_code = rec.get("state")
            year = rec.get("year")
            if not state_code or year is None:
                continue
            try:
                year_i = int(year)
            except (TypeError, ValueError):
                continue
            for metric_key in _RAG_MEASURES:
                val = rec.get(metric_key)
                if val is None:
                    continue
                try:
                    years_by_state_metric[(state_code, metric_key)][year_i].append(float(val))
                except (TypeError, ValueError):
                    continue

        out: list[str] = []
        for (state_code, metric_key), by_year in sorted(years_by_state_metric.items()):
            measure = self._measures.get(metric_key)
            state = self.ontology.states.get(state_code)
            if measure is None or state is None or len(by_year) < 2:
                continue
            annual_means = {y: sum(vs) / len(vs) for y, vs in by_year.items() if vs}
            if len(annual_means) < 2:
                continue
            y0 = min(annual_means)
            y1 = max(annual_means)
            v0, v1 = annual_means[y0], annual_means[y1]
            change_pct = (v1 - v0) / v0 * 100 if v0 else 0.0
            direction = "rose" if v1 > v0 else "fell" if v1 < v0 else "held steady"
            if measure.unit == "percent":
                v0s, v1s = f"{v0:.1f}%", f"{v1:.1f}%"
            else:
                v0s, v1s = f"{v0:,.0f}", f"{v1:,.0f}"
            out.append(
                f"{state.name} {measure.name} {direction} from {v0s} in {y0} "
                f"to {v1s} in {y1} ({change_pct:+.1f}% total change)."
            )
        return out

    # ------------------------------------------------------------------
    # Layer 3 — optional LLM polish
    # ------------------------------------------------------------------
    def agent_polish(self, sentences: Sequence[str]) -> list[str]:
        """
        Send each sentence through the worker agent for stylistic polish.

        Expensive — one LLM call per sentence. Off by default; call this
        only when you're willing to pay ~1-3 s per sentence.
        """
        if self.agent is None:
            logger.info("[rag] agent_polish skipped — no agent attached")
            return list(sentences)

        polished: list[str] = []
        for s in sentences:
            prompt = (
                "Rewrite the following factual statement as one natural, "
                "fluent English sentence. Keep every number exactly as given. "
                "Do not add claims that aren't in the input. "
                f'Input: "{s}"'
            )
            out = self.agent.invoke(prompt)
            polished.append(out.strip() or s)
        return polished

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------
    def build_corpus(
        self, records: Iterable[dict], *, polish: bool = False
    ) -> list[str]:
        """Deterministic facts + ontology ranks + trends (+ optional polish)."""
        # Materialize the records once — each layer wants to iterate.
        records_list = list(records)
        sentences = (
            self.fact_sentences(records_list)
            + self.ranking_sentences(records_list)
            + self.trend_sentences(records_list)
        )
        return self.agent_polish(sentences) if polish else sentences


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _ordinal(n: int) -> str:
    """1 → '1st', 2 → '2nd', ...; handles the 11/12/13 exception."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _is_lower_better(measure: Measure) -> bool:
    """Unemployment is bad; everything else on _RAG_MEASURES is higher-better."""
    return measure.key in {"Unemployment", "Unemployment_Rate"}
