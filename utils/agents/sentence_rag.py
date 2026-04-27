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

    # ------------------------------------------------------------------
    # View-state rendering
    # ------------------------------------------------------------------
    def render_view_sentences(self, view_state: dict) -> list[str]:
        """
        Render an on-screen panel's ``view_state`` into ontology-aware
        natural-language sentences.

        This is the grounding layer for ``BlurbAgent.explain_view`` —
        the AI never sees the raw key/value dict; it sees prose tagged
        with state, region, measure, and source so its narrative stays
        on-rails. Sentences from this function are also a natural input
        to the embedding store: tag them with ``view_state['_kind']``
        and ``focus_state`` to retrieve later from the chat drawer.

        Dispatches on ``view_state['_kind']``; an unknown kind falls
        back to a flat ``key: value`` rendering so a forgotten kind
        still produces *something* (the AI just gets a dumber prompt).
        """
        kind = view_state.get("_kind", "generic")
        renderer = {
            "forecast_panel": self._render_forecast_panel,
            "requirements_panel": self._render_requirements_panel,
            "trend_panel": self._render_trend_panel,
            "recap": self._render_recap,
        }.get(kind, self._render_generic)
        return [s for s in renderer(view_state) if s]

    # --- per-kind renderers ---
    def _render_forecast_panel(self, vs: dict) -> list[str]:
        focus = self._state_label(vs.get("focus_state"))
        measure = self._measure_label(vs.get("metric_key"))
        horizon = vs.get("horizon_years")
        last_year = vs.get("last_actual_year")
        last_val = vs.get("last_actual_value")
        forecast = vs.get("forecast_point")
        ci = vs.get("forecast_ci") or [None, None]
        ci_lo, ci_hi = (ci[0], ci[1]) if isinstance(ci, (list, tuple)) and len(ci) >= 2 else (None, None)
        model = vs.get("winning_model")
        rmse = vs.get("rmse")
        peers = vs.get("peer_states") or []

        sentences: list[str] = []
        if focus and measure and horizon:
            model_clause = (
                f" using the {model.upper()} model selected from a Naive / "
                f"Seasonal-Naive / Holt-Winters / ARIMA bake-off"
                f" (out-of-sample RMSE {rmse:.2f})"
                if model and rmse is not None
                else ""
            )
            sentences.append(
                f"The {focus} {measure} forecast over the next {horizon} year(s)"
                f"{model_clause}."
            )
        if focus and measure and last_year is not None and last_val is not None:
            sentences.append(
                f"The last published {focus} {measure} was "
                f"{self._fmt_pct_or_num(last_val, vs)} in {last_year}."
            )
        if forecast is not None and horizon and measure:
            ci_clause = (
                f", with a 95% confidence interval of "
                f"{self._fmt_pct_or_num(ci_lo, vs)}–{self._fmt_pct_or_num(ci_hi, vs)}"
                if ci_lo is not None and ci_hi is not None
                else ""
            )
            sentences.append(
                f"The +{horizon}-year point forecast is "
                f"{self._fmt_pct_or_num(forecast, vs)}{ci_clause}."
            )
        peer_labels = [self._state_label(c) for c in peers if c]
        if peer_labels:
            region_hint = self._shared_region_hint(
                [vs.get("focus_state"), *peers]
            )
            sentences.append(
                f"Peer states under comparison: {', '.join(peer_labels)}"
                + (f" ({region_hint})." if region_hint else ".")
            )
        return sentences

    def _render_requirements_panel(self, vs: dict) -> list[str]:
        measure = self._measure_label(vs.get("metric_key"))
        threshold = vs.get("threshold")
        direction = vs.get("direction", "min")
        op = "≥" if direction == "min" else "≤"
        passing = vs.get("states_passing") or []
        failing = vs.get("states_failing") or []
        all_points = vs.get("state_points") or []

        sentences: list[str] = []
        # When the user leaves the threshold blank we still want the
        # AI panel to read the forecast values across the comparison —
        # the table on screen shows them, the AI should describe them.
        if threshold is None:
            if measure:
                sentences.append(
                    f"No requirement threshold was set, so no pass/fail "
                    f"verdict was computed for {measure}."
                )
            if all_points:
                parts = self._fmt_state_value_pairs(all_points, vs)
                if parts:
                    sentences.append(
                        f"Projected {measure or 'forecast'} across the "
                        f"{len(parts)} state(s) under comparison: "
                        f"{', '.join(parts)}."
                    )
            return sentences

        # Threshold-set path: describe the rule + the bucketed verdicts.
        if measure:
            sentences.append(
                f"The threshold for {measure} is set at {op} "
                f"{self._fmt_pct_or_num(threshold, vs)}."
            )
        if passing:
            parts = self._fmt_state_value_pairs(passing, vs)
            if parts:
                sentences.append(
                    f"{len(parts)} state(s) projected to meet the requirement: "
                    f"{', '.join(parts)}."
                )
        if failing:
            parts = self._fmt_state_value_pairs(failing, vs)
            if parts:
                sentences.append(
                    f"{len(parts)} state(s) projected to fall short: "
                    f"{', '.join(parts)}."
                )
        return sentences

    def _fmt_state_value_pairs(self, items: list, vs: dict) -> list[str]:
        """Helper — turn ``[{code, value}, ...]`` into ``["Iowa (65.2%)", ...]``."""
        return [
            f"{self._state_label(item.get('code'))} ({self._fmt_pct_or_num(item.get('value'), vs)})"
            for item in items
            if item.get("code") and item.get("value") is not None
        ]

    def _render_trend_panel(self, vs: dict) -> list[str]:
        measure = self._measure_label(vs.get("metric_key"))
        rows = vs.get("states") or []
        sentences: list[str] = []
        if measure and rows:
            sentences.append(
                f"Historic {measure} trend summary across {len(rows)} state(s)."
            )
        for row in rows:
            label = self._state_label(row.get("code"))
            current = row.get("current")
            change = row.get("five_yr_change")
            if not label or current is None:
                continue
            change_clause = ""
            if change is not None:
                direction = "up" if change > 0 else "down" if change < 0 else "flat"
                change_clause = (
                    f", {direction} {abs(change):.1f} pp over five years"
                    if direction != "flat"
                    else ", essentially flat over five years"
                )
            sentences.append(
                f"{label} {measure} is currently "
                f"{self._fmt_pct_or_num(current, vs)}{change_clause}."
            )
        return sentences

    def _render_recap(self, vs: dict) -> list[str]:
        # The recap re-uses the per-panel renderers so the synthesis
        # gets every sentence the user just saw on the page. The recap
        # adds a one-line headline at the top so the LLM has a clear
        # "lead" anchor to expand on.
        sentences: list[str] = []
        title = vs.get("title")
        if title:
            sentences.append(f"Recap: {title}.")
        for panel in vs.get("panels", []):
            sentences.extend(self.render_view_sentences(panel))
        return sentences

    def _render_generic(self, vs: dict) -> list[str]:
        return [f"{k}: {v}" for k, v in vs.items() if not k.startswith("_") and k != "title"]

    # --- formatting helpers ---
    def _state_label(self, code: str | None) -> str:
        if not code:
            return ""
        st = self.ontology.states.get(str(code).upper())
        return st.name if st else str(code)

    def _measure_label(self, key: str | None) -> str:
        if not key:
            return ""
        m = self._measures.get(str(key))
        return m.name if m else str(key).replace("_", " ").lower()

    def _fmt_pct_or_num(self, value, vs: dict) -> str:
        if value is None:
            return "n/a"
        measure = self._measures.get(vs.get("metric_key", ""))
        try:
            v = float(value)
        except (TypeError, ValueError):
            return str(value)
        if measure and measure.unit == "percent":
            return f"{v:.1f}%"
        return f"{v:,.2f}"

    def _shared_region_hint(self, codes: list[str | None]) -> str:
        """If every state in the list shares one Census region, name it."""
        regions = {
            self.ontology.states[c.upper()].region
            for c in codes
            if c and c.upper() in self.ontology.states
        }
        return f"{regions.pop()} census region" if len(regions) == 1 else ""


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


# --------------------------------------------------------------------------
# Module-level convenience
# --------------------------------------------------------------------------
_default_builder: SentenceRAGBuilder | None = None


def default_rag_builder() -> SentenceRAGBuilder:
    """
    Lazy singleton — the per-call view-state renderers are stateless
    apart from the ontology pointer, so re-instantiating per blurb
    just rebuilds the measure dict for nothing.
    """
    global _default_builder
    if _default_builder is None:
        _default_builder = SentenceRAGBuilder()
    return _default_builder
