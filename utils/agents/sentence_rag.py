"""
Sentence-RAG builder.

Turns the tabular monthly panel into natural-language sentences before
embedding. Embedding models retrieve prose far better than column
dumps, so instead of feeding ``Labor_Force: 1500000; Population:
3150000`` into the vectoriser we feed::

    In 2020, Iowa labor force averaged 1,500,000 persons.
    Iowa's 2020 labor force participation rate of 47.6% ranked fourth
    among Midwest states, below Minnesota (58.2%) and North Dakota (63.4%).

The builder is split into two layers so callers can compose what
they need:

1. :meth:`fact_sentences` — deterministic "per (state, year, metric)"
   summaries. Cheap, runs at data-refresh time.
2. :meth:`ranking_sentences` — ontology-enriched comparative framing
   that gives the embedder context (peer groups, rank positions).

``build_corpus`` wires both up — plus the static
:data:`CONTEXT_SNIPPETS` rate/policy grounding layer — and the corpus
is always zero-LLM-call.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable

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


#: Static macro-context snippets embedded alongside the panel-derived
#: sentences (Track H). Each names its source so the narrative LLM can
#: attribute the claim; the matching registry entries live in
#: ``utils.citations`` (``estrella_mishkin_1996``,
#: ``treasury_yield_curve``, ``fed_h15``, ``fed_fomc_statements``,
#: ``fed_beige_book``, ``nasdaq_data_link``, ``cme_fedwatch``). The
#: snippets flow through :meth:`SentenceRAGBuilder.build_corpus` like
#: any other corpus sentence, so the e5 ``passage:`` prefix is applied
#: automatically at embed time (see ``utils.embeddings``).
CONTEXT_SNIPPETS: tuple[str, ...] = (
    "The Treasury yield-curve spread (10-year minus 3-month "
    "constant-maturity yield) is a classic leading indicator of U.S. "
    "recessions, and an inverted spread has preceded most postwar U.S. "
    "recessions (Estrella & Mishkin, 1996; data: FRED series GS10 and "
    "GS3M).",
    "Treasury constant-maturity yields in this dashboard are national "
    "monthly averages from the Federal Reserve Board's H.15 Selected "
    "Interest Rates release, retrieved via FRED; they are broadcast to "
    "every state as shared macro context rather than measured per state.",
    "Constant-maturity Treasury yields are interpolated by the U.S. "
    "Department of the Treasury from its daily par yield curve, "
    "published as the Daily Treasury Par Yield Curve Rates.",
    "In the merged panel, TREASURY_SPREAD_10Y3M is the FRED GS10 yield "
    "minus the GS3M yield for months where both are available; a "
    "negative value means the yield curve is inverted.",
    "The Federal Open Market Committee (FOMC) sets the target range for "
    "the federal funds rate; its meeting statements and minutes are "
    "published on the Federal Reserve Board website (federalreserve.gov).",
    "The Beige Book, published eight times per year by the Federal "
    "Reserve, summarizes qualitative economic conditions — including "
    "labor-market commentary — across the twelve Federal Reserve "
    "Districts.",
    "CME Group's FedWatch Tool publishes market-implied probabilities "
    "of Federal Reserve rate decisions derived from federal funds "
    "futures prices (cmegroup.com).",
    "Nasdaq Data Link (data.nasdaq.com, formerly Quandl) aggregates "
    "financial and economic datasets, including equity-index and "
    "macroeconomic series, under a mix of free and licensed terms.",
    "Higher policy interest rates raise borrowing costs and typically "
    "cool hiring in rate-sensitive sectors such as construction and "
    "manufacturing, with a lag of several quarters.",
)


class SentenceRAGBuilder:
    """Ontology-aware sentence generator for the embedding pipeline."""

    def __init__(
        self,
        ontology: Ontology | None = None,
    ) -> None:
        self.ontology = ontology or ONTOLOGY
        # Cache: {metric_key: Measure}
        self._measures = {m.key: m for m in self.ontology.measures.values()}

    # ------------------------------------------------------------------
    # Layer 1 — deterministic facts
    # ------------------------------------------------------------------
    def fact_sentences(self, records: Iterable[dict]) -> list[str]:
        """Emit one sentence per (state, year, tracked metric) using the yearly mean.

        Parameters
        ----------
        records : Iterable[dict]
            Long-form rows with at minimum ``state``, ``year``, and one of
            :data:`_RAG_MEASURES`.

        Returns
        -------
        list[str]
            One fact sentence per (state, year, metric) cell with data.
        """
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
        """Emit per-state comparative ranking sentences per (year, metric).

        These feed the embedder with peer-group context, which is what
        the chatbot most often needs.

        Parameters
        ----------
        records : Iterable[dict]
            Long-form rows; see :meth:`fact_sentences`.

        Returns
        -------
        list[str]
            One sentence per (year, metric, state) ranking position.
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
        """Emit one direction sentence per (state, metric) over the observed span.

        Parameters
        ----------
        records : Iterable[dict]
            Long-form rows; see :meth:`fact_sentences`.

        Returns
        -------
        list[str]
            One sentence per (state, metric) summarising earliest → latest.
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
    # Top-level
    # ------------------------------------------------------------------
    def build_corpus(
        self,
        records: Iterable[dict],
        extra_snippets: Iterable[str] | None = None,
    ) -> list[str]:
        """Return facts, rankings, trends, plus static context snippets.

        Parameters
        ----------
        records : Iterable[dict]
            Long-form panel rows; see :meth:`fact_sentences`.
        extra_snippets : Iterable[str], optional
            Static context sentences appended verbatim to the corpus.
            Defaults to :data:`CONTEXT_SNIPPETS` (rate/policy grounding
            for the narrative agent); pass ``()`` to opt out.

        Returns
        -------
        list[str]
            Corpus sentences ready for the embedding pipeline.
        """
        # Materialize the records once — each layer wants to iterate.
        records_list = list(records)
        snippets = (
            CONTEXT_SNIPPETS if extra_snippets is None else tuple(extra_snippets)
        )
        return (
            self.fact_sentences(records_list)
            + self.ranking_sentences(records_list)
            + self.trend_sentences(records_list)
            + list(snippets)
        )

    # ------------------------------------------------------------------
    # View-state rendering
    # ------------------------------------------------------------------
    def render_view_sentences(self, view_state: dict) -> list[str]:
        """Render an on-screen panel's ``view_state`` into ontology-aware sentences.

        This is the grounding layer for ``BlurbAgent.explain_view`` — the
        AI never sees the raw key/value dict; it sees prose tagged with
        state, region, measure, and source so its narrative stays
        on-rails. Sentences from this function are also a natural input
        to the embedding store: tag them with ``view_state['_kind']``
        and ``focus_state`` to retrieve later from the chat drawer.

        Dispatches on ``view_state['_kind']``; an unknown kind falls back
        to a flat ``key: value`` rendering so a forgotten kind still
        produces *something* (the AI just gets a dumber prompt).

        Parameters
        ----------
        view_state : dict
            Typed payload emitted by a tab callback.

        Returns
        -------
        list[str]
            Non-empty sentences ready for the embedder / specialist prompt.
        """
        kind = view_state.get("_kind", "generic")
        renderer = {
            "forecast_panel": self._render_forecast_panel,
            "requirements_panel": self._render_requirements_panel,
            "trend_panel": self._render_trend_panel,
            "recap": self._render_recap,
            # EDA tab
            "eda_timeseries": self._render_eda_timeseries,
            "eda_distribution": self._render_eda_distribution,
            "eda_volatility": self._render_eda_volatility,
            # Super (supersector) tab
            "supersector_forecast": self._render_supersector_forecast,
            "supersector_recommendation": self._render_supersector_recommendation,
            "supersector_models": self._render_supersector_models,
        }.get(kind, self._render_generic)
        return [s for s in renderer(view_state) if s]

    # --- per-kind renderers ---
    def _render_forecast_panel(self, vs: dict) -> list[str]:
        focus = self._state_label(vs.get("focus_state"))
        measure = self._measure_label(vs.get("metric_key"))
        horizon = vs.get("horizon_years")
        last_year = vs.get("last_actual_year")
        last_val = vs.get("last_actual_value")
        forecast_start = vs.get("forecast_start_year")
        forecast_end = vs.get("forecast_end_year")
        data_lag = vs.get("data_lag_months") or 0
        forecast = vs.get("forecast_point")
        ci = vs.get("forecast_ci") or [None, None]
        ci_lo, ci_hi = (
            (ci[0], ci[1]) if isinstance(ci, (list, tuple)) and len(ci) >= 2 else (None, None)
        )
        model = vs.get("winning_model")
        rmse = vs.get("rmse")
        peers = vs.get("peer_states") or []

        sentences: list[str] = []
        if focus and measure and horizon and forecast_start:
            model_clause = (
                f" using the {model.upper()} model selected from a Naive / "
                f"Seasonal-Naive / Holt-Winters / ARIMA bake-off"
                f" (out-of-sample RMSE {rmse:.2f})"
                if model and rmse is not None
                else ""
            )
            window = (
                f"{forecast_start}–{forecast_end}"
                if forecast_end and forecast_end != forecast_start
                else str(forecast_start)
            )
            sentences.append(
                f"The {focus} {measure} forecast covers {window} "
                f"({horizon}-year window){model_clause}."
            )
        if focus and measure and last_year is not None and last_val is not None:
            lag_clause = (
                f"; the {data_lag}-month gap between that and the "
                f"forecast start represents months that have elapsed but "
                f"BLS hasn't published yet"
                if data_lag and data_lag > 0
                else ""
            )
            sentences.append(
                f"The last published {focus} {measure} was "
                f"{self._fmt_pct_or_num(last_val, vs)} in {last_year}{lag_clause}."
            )
        if forecast is not None and forecast_end and measure:
            ci_clause = (
                f", with a 95% confidence interval of "
                f"{self._fmt_pct_or_num(ci_lo, vs)}–{self._fmt_pct_or_num(ci_hi, vs)}"
                if ci_lo is not None and ci_hi is not None
                else ""
            )
            sentences.append(
                f"The end-of-window point forecast (year {forecast_end}) is "
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
        # Statistical context — gives the AI specific numbers to
        # interpret against rather than just restating the forecast.
        hist_mean = vs.get("historical_mean")
        hist_vol = vs.get("historical_volatility")
        peer_median = vs.get("peer_median_forecast")
        if hist_mean is not None and hist_vol is not None:
            sentences.append(
                f"Historical context: the {focus} {measure} averaged "
                f"{self._fmt_pct_or_num(hist_mean, vs)} across the "
                f"observed window with σ "
                f"{self._fmt_pct_or_num(hist_vol, vs)}; deviations "
                f"larger than ~2σ are noteworthy."
            )
        if peer_median is not None and forecast is not None:
            delta = forecast - peer_median
            direction_word = (
                "above" if delta > 0
                else "below" if delta < 0
                else "in line with"
            )
            sentences.append(
                f"Peer-state median forecast at the same horizon is "
                f"{self._fmt_pct_or_num(peer_median, vs)}; {focus}'s "
                f"point estimate is {direction_word} the peer median "
                f"by {self._fmt_pct_or_num(abs(delta), vs)}."
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

    # --- EDA tab renderers ---
    def _render_eda_timeseries(self, vs: dict) -> list[str]:
        measures = vs.get("measures") or []
        states = vs.get("states") or []
        rows = vs.get("series") or []
        sentences: list[str] = []
        if measures and states:
            measure_labels = [self._measure_label(m) for m in measures if m]
            sentences.append(
                f"The time-series chart overlays "
                f"{', '.join(measure_labels) or 'the selected measures'} "
                f"for {len(states)} state(s) ({', '.join(self._state_label(s) for s in states[:8])}"
                + (f", plus {len(states)-8} more" if len(states) > 8 else "")
                + ")."
            )
        for row in rows[:6]:
            label = self._state_label(row.get("state"))
            measure = self._measure_label(row.get("measure"))
            first, last = row.get("first"), row.get("last")
            if not (label and measure and first is not None and last is not None):
                continue
            change = (last - first)
            direction = "up" if change > 0 else "down" if change < 0 else "essentially flat"
            sentences.append(
                f"{label} {measure} moved from "
                f"{self._fmt_value_with_unit(first, row.get('unit'))} to "
                f"{self._fmt_value_with_unit(last, row.get('unit'))} "
                f"({direction})."
            )
        return sentences

    def _render_eda_distribution(self, vs: dict) -> list[str]:
        measures = vs.get("measures") or []
        rows = vs.get("series") or []
        sentences: list[str] = []
        if measures:
            measure_labels = [self._measure_label(m) for m in measures if m]
            sentences.append(
                f"The distribution view summarises the spread of "
                f"{', '.join(measure_labels) or 'the selected measures'} "
                f"across the active scope."
            )
        for row in rows[:6]:
            label = self._state_label(row.get("state"))
            measure = self._measure_label(row.get("measure"))
            mean = row.get("mean")
            std = row.get("std")
            if not (label and measure and mean is not None):
                continue
            unit = row.get("unit")
            std_clause = (
                f" (σ ≈ {self._fmt_value_with_unit(std, unit)})"
                if std is not None
                else ""
            )
            sentences.append(
                f"{label} {measure}: mean "
                f"{self._fmt_value_with_unit(mean, unit)}{std_clause}."
            )
        return sentences

    def _render_eda_volatility(self, vs: dict) -> list[str]:
        rows = vs.get("series") or []
        window = vs.get("rolling_window_months")
        sentences: list[str] = []
        if window:
            sentences.append(
                f"The trend view shows {window}-month rolling means and "
                f"year-over-year percent change for each selected series."
            )
        for row in rows[:6]:
            label = self._state_label(row.get("state"))
            measure = self._measure_label(row.get("measure"))
            yoy_pct = row.get("yoy_pct")
            if not (label and measure and yoy_pct is not None):
                continue
            direction = "above" if yoy_pct > 0 else "below"
            sentences.append(
                f"{label} {measure} is currently {abs(yoy_pct):.1f}% "
                f"{direction} its level twelve months earlier."
            )
        return sentences

    # --- Super (supersector) tab renderers ---
    def _render_supersector_forecast(self, vs: dict) -> list[str]:
        sector = vs.get("sector_label") or vs.get("sector") or "the selected sector"
        horizon = vs.get("horizon_years")
        region = vs.get("region_label") or vs.get("region")
        rows = vs.get("states") or []
        sentences: list[str] = []
        scope_clause = (
            f" across the {region} census region"
            if region and region != "All states"
            else " across the active scope"
        )
        if horizon:
            sentences.append(
                f"The {sector} +{horizon}-year forecast{scope_clause} "
                f"covers {len(rows)} state(s), each forecast by the model "
                f"its bake-off selected."
            )
        leaders = sorted(
            [r for r in rows if r.get("forecast") is not None],
            key=lambda r: r["forecast"],
            reverse=True,
        )[:3]
        for r in leaders:
            label = self._state_label(r.get("code"))
            ci_low, ci_high = self._unpack_pair([r.get("lower_ci"), r.get("upper_ci")])
            ci_clause = (
                f" (95% CI {self._fmt_value(ci_low)}–{self._fmt_value(ci_high)})"
                if ci_low is not None and ci_high is not None
                else ""
            )
            model = r.get("model")
            model_clause = f", selected model {model.upper()}" if model else ""
            sentences.append(
                f"{label}: forecast {self._fmt_value(r.get('forecast'))}"
                f"{ci_clause}{model_clause}."
            )
        return sentences

    def _render_supersector_recommendation(self, vs: dict) -> list[str]:
        sector = vs.get("sector_label") or vs.get("sector") or "the selected sector"
        horizon = vs.get("horizon_years")
        median = vs.get("median")
        top = vs.get("top") or []
        bottom = vs.get("bottom") or []
        sentences: list[str] = []
        if median is not None:
            sentences.append(
                f"The peer median for {sector} at the +{horizon}-year horizon "
                f"is {self._fmt_value(median)}."
            )
        if top:
            top_parts = [
                f"{self._state_label(r.get('code'))} ({self._fmt_value(r.get('value'))})"
                for r in top[:3]
                if r.get("code") and r.get("value") is not None
            ]
            if top_parts:
                sentences.append(
                    f"Top recommendations for siting / expansion: {', '.join(top_parts)}."
                )
        if bottom:
            bottom_parts = [
                f"{self._state_label(r.get('code'))} ({self._fmt_value(r.get('value'))})"
                for r in bottom[:3]
                if r.get("code") and r.get("value") is not None
            ]
            if bottom_parts:
                sentences.append(
                    f"Trailing peers (recovery / divestment risk): {', '.join(bottom_parts)}."
                )
        return sentences

    def _render_supersector_models(self, vs: dict) -> list[str]:
        winners = vs.get("winners") or []
        sentences: list[str] = []
        if not winners:
            return sentences
        # Tally model picks across the in-scope states.
        tally: dict[str, int] = {}
        for w in winners:
            m = (w.get("model") or "").lower()
            if m:
                tally[m] = tally.get(m, 0) + 1
        if tally:
            mix = ", ".join(
                f"{count} × {name.upper()}"
                for name, count in sorted(tally.items(), key=lambda kv: kv[1], reverse=True)
            )
            sentences.append(
                f"Per-state bake-off model mix across {len(winners)} state(s): {mix}."
            )
        # Highlight outliers — states where RMSE is notably higher.
        rmses = [w.get("rmse") for w in winners if w.get("rmse") is not None]
        if rmses:
            avg_rmse = sum(rmses) / len(rmses)
            sentences.append(
                f"Average out-of-sample RMSE across the selected models is "
                f"{avg_rmse:.2f}."
            )
        return sentences

    # --- formatting helpers (extras for new renderers) ---
    def _unpack_pair(self, pair) -> tuple:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            return pair[0], pair[1]
        return None, None

    def _fmt_value(self, value) -> str:
        """Generic numeric format for sector-level values (no unit / no percent)."""
        if value is None:
            return "n/a"
        try:
            return f"{float(value):,.1f}"
        except (TypeError, ValueError):
            return str(value)

    def _fmt_value_with_unit(self, value, unit: str | None) -> str:
        if value is None:
            return "n/a"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return str(value)
        if unit == "percent":
            return f"{v:.1f}%"
        if unit == "persons":
            return f"{v:,.0f}"
        return f"{v:,.2f}"

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
    """Return the lazy module-level :class:`SentenceRAGBuilder` singleton.

    The per-call view-state renderers are stateless apart from the
    ontology pointer, so re-instantiating per blurb just rebuilds the
    measure dict for nothing.
    """
    global _default_builder
    if _default_builder is None:
        _default_builder = SentenceRAGBuilder()
    return _default_builder
