"""
Reusable Methodology + References panel for the forecasting tabs.

Surfaces, in one collapsible block per tab:

1. What the forecast actually did (scoring rule + backtest protocol),
2. Which candidate models competed and the paper each cites,
3. Which statistical tests we ran on the residuals + series,
4. Which BLS / Census / BEA / Treasury / FRED datasets feed the
   pipeline, plus the qualitative macro & policy context sources
   (FOMC, Beige Book, Nasdaq Data Link, CME FedWatch) surfaced to the
   narrative agent.

Every claim carries an inline citation that round-trips through
``utils.citations``; the references panel surfaces the academic and
methodological literature underpinning the dashboard's forecasting,
retrieval, and software stack, with the canonical registry defined
once in ``utils/citations.py`` and reused everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from dash import html

from utils.citations import CITATIONS, citation, render_inline
from utils.forecasting.models import ARIMAForecaster


def _format_arima_grid(grid: tuple[tuple[int, int, int], ...]) -> str:
    """Render ``((1,1,1), (2,1,1), ...)`` as ``{(1,1,1), (2,1,1), ...}``."""
    return "{" + ", ".join(f"({p},{d},{q})" for (p, d, q) in grid) + "}"


@dataclass(frozen=True, slots=True)
class MethodologyNote:
    """One bullet of "what we did" + the citation keys backing it."""

    text: str
    cites: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Default bullets for the forecasting tabs.
# ---------------------------------------------------------------------------
#: Applies to both LFP + Super — they share the bakeoff harness.
FORECAST_NOTES: tuple[MethodologyNote, ...] = (
    MethodologyNote(
        "The model for each series is selected by minimum out-of-sample "
        "RMSE over a 3-fold expanding-window backtest: every candidate "
        "forecasts three held-out stretches of past data, and its errors "
        "are averaged across the three. This follows standard practice "
        "in the forecasting literature.",
        cites=("hyndman_athanasopoulos_2018", "tashman_2000"),
    ),
    MethodologyNote(
        "Candidate models include random-walk persistence (Naive), "
        "seasonal persistence (Seasonal-Naive) and classical "
        "exponential smoothing (Holt–Winters).",
        cites=("holt_1957", "winters_1960", "hyndman_athanasopoulos_2018"),
    ),
    MethodologyNote(
        # Rendered from ARIMAForecaster.DEFAULT_GRID at module load so
        # editing the grid in the model class flows through to the
        # methodology panel automatically — no second source of truth.
        f"ARIMA orders are AIC-selected from a small grid "
        f"{_format_arima_grid(ARIMAForecaster.DEFAULT_GRID)}.",
        cites=("box_jenkins_1970",),
    ),
    MethodologyNote(
        "LSTM models are available behind an opt-in flag; they are not "
        "enabled by default because on ~300-observation monthly series "
        "they rarely beat Holt–Winters and cost ~10× more compute.",
        cites=("hochreiter_schmidhuber_1997",),
    ),
    MethodologyNote(
        "95 % prediction intervals come from model-native variance "
        "(statsmodels' ``get_forecast`` for ETS + ARIMA) or from a "
        "Brownian residual-bootstrap band for the baselines.",
        cites=("hyndman_athanasopoulos_2018",),
    ),
)

DIAGNOSTIC_NOTES: tuple[MethodologyNote, ...] = (
    MethodologyNote(
        "Series stationarity is checked with the Augmented Dickey–Fuller "
        "and KPSS tests — used together because each has blind spots "
        "the other catches.",
        cites=("dickey_fuller_1979", "kpss_1992"),
    ),
    MethodologyNote(
        "Residual autocorrelation is tested with the Ljung–Box Q at "
        "lag 10; residual normality with Jarque–Bera.",
        cites=("ljung_box_1978", "jarque_bera_1980"),
    ),
    MethodologyNote(
        "The selected model is compared to a Naive baseline using the "
        "Diebold–Mariano test with the Harvey–Leybourne–Newbold "
        "small-sample correction. The comparison currently runs on "
        "in-sample fit errors from the full-history refit — not on the "
        "held-out backtest windows — so read it as a goodness-of-fit "
        "check rather than an independent out-of-sample test.",
        cites=("diebold_mariano_1995", "harvey_leybourne_newbold_1997"),
    ),
    MethodologyNote(
        "Variable independence + double-counting is analysed offline "
        "with Spearman correlations, Variance Inflation Factors, and a "
        "KSG mutual-information ranking, with results in "
        "`docs/methodology/variable_selection.md`. The selection step "
        "is not yet wired into the bakeoff (Phase 2 task).",
        cites=(
            "belsley_kuh_welsch_1980",
            "kraskov_stoegbauer_grassberger_2004",
            "james_witten_hastie_tibshirani_2013",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Macro & policy context (Track H)
# ---------------------------------------------------------------------------
#: Single bullet rendered inside the "Data sources" block: explains how
#: Treasury yields enter the panel (national broadcast + computed
#: 10y−3m spread) and which qualitative sources ground the narrative
#: agent. Kept to ONE note so the panel stays scannable.
CONTEXT_NOTES: tuple[MethodologyNote, ...] = (
    MethodologyNote(
        "National macro context: Treasury constant-maturity yields "
        "(10-year, 2-year, 3-month; FRED series GS10 / GS2 / GS3M, "
        "republished from the Federal Reserve H.15 release) enter the "
        "panel as national context variables broadcast to all states, "
        "and the 10-year-minus-3-month term spread is included as a "
        "classic recession leading indicator. FOMC statements, the "
        "Beige Book, Nasdaq Data Link, and the CME FedWatch Tool are "
        "linked below as qualitative context sources surfaced to the "
        "narrative agent.",
        cites=(
            "estrella_mishkin_1996",
            "fed_fomc_statements",
            "fed_beige_book",
            "nasdaq_data_link",
            "cme_fedwatch",
        ),
    ),
)

DATA_SOURCE_KEYS: tuple[str, ...] = (
    "bls_ces_handbook",
    "bls_laus_handbook",
    "bls_jolts_handbook",
    "bls_qcew_handbook",
    "bls_cpi_handbook",
    "census_acs_handbook",
    "bea_regional_handbook",
    "fred_api",
    "fhfa_hpi_handbook",
    # Treasury yield lineage (Track H): Treasury par yield curve →
    # H.15 constant-maturity rates → FRED GS-series monthly averages.
    "treasury_yield_curve",
    "fred_treasury_cmt",
    "fed_h15",
)

#: Qualitative macro & policy context sources (Track H). These do not
#: feed any chart — they are linked for the narrative agent's
#: rate/policy grounding (see ``utils.agents.sentence_rag``
#: ``CONTEXT_SNIPPETS``) — so they render as their own linked list
#: under the data sources rather than claiming figure provenance.
CONTEXT_SOURCE_KEYS: tuple[str, ...] = (
    "fed_fomc_statements",
    "fed_beige_book",
    "nasdaq_data_link",
    "cme_fedwatch",
)

#: Short, one-line attribution string suitable for a chart caption /
#: layout-annotation footer. Each upstream agency expects a "Source:
#: ..." line under any figure that uses their data (BLS, Census, BEA,
#: FRED, FHFA all publish a citation policy). One consolidated line
#: keeps the footer visually compact while satisfying every policy.
DATA_SOURCE_FOOTER: str = (
    "Source: U.S. Bureau of Labor Statistics (CES, LAUS, JOLTS, QCEW, "
    "CPI); U.S. Census Bureau (ACS, PEP); U.S. Bureau of Economic "
    "Analysis; U.S. Department of the Treasury; Federal Reserve Bank "
    "of St. Louis (FRED); Federal Housing Finance Agency."
)


def chart_source_annotation(
    *,
    x: float = 0.0,
    y: float = -0.18,
    text: str = DATA_SOURCE_FOOTER,
) -> dict:
    """Build a Plotly annotation dict that renders the source citation footer.

    Add to a figure via ``fig.add_annotation(**chart_source_annotation())``
    or ``fig.update_layout(annotations=[chart_source_annotation()])``.
    Keep the wording neutral and short; one consolidated line satisfies
    BLS / Census / BEA / FRED / FHFA citation policies.

    Parameters
    ----------
    x, y : float, optional
        Paper-anchored coordinates for the caption.
    text : str, optional
        Override the default consolidated source line.

    Returns
    -------
    dict
        Plotly ``layout.annotations`` entry suitable for ``add_annotation``.
    """
    return {
        "text": text,
        "xref": "paper",
        "yref": "paper",
        "x": x,
        "y": y,
        "xanchor": "left",
        "yanchor": "top",
        "showarrow": False,
        "font": {"size": 10, "color": "rgba(80, 80, 80, 0.85)"},
        "align": "left",
    }


#: Software / model attribution keys — surfaces the embedding model,
#: the sentence-transformers library, the numeric/statistical stack,
#: the agent-architecture antecedents, and the Llama 3.2 community
#: license. Rendered as its own "Software & model attribution" block
#: alongside the data sources so an auditor sees the same provenance
#: layer for code that they see for data.
SOFTWARE_ATTRIBUTION_KEYS: tuple[str, ...] = (
    "wang_e5_2022",
    "reimers_gurevych_2019",
    "seabold_perktold_2010",
    "harris_numpy_2020",
    "mckinney_pandas_2010",
    "pedregosa_sklearn_2011",
    "abadi_tensorflow_2016",
    "tukey_1977",
    "akaike_1974",
    "efron_1979",
    "shinn_reflexion_2023",
    "minsky_1986",
    "llama3_license_2024",
)


# ---------------------------------------------------------------------------
# Definitional caveats — surfaces as its own block on the LFP tab
# ---------------------------------------------------------------------------
#: Plain-text body of the LFPR-denominator audit, mirrored in
#: ``docs/methodology/lfpr_denominator.md``. Rendered as a yellow
#: callout above the chart on the LFP tab so a researcher hits the
#: caveat before they screenshot the number.
LFPR_DENOMINATOR_NOTE: str = (
    "**LFPR denominator caveat.** BLS defines LFPR as "
    "`100 × civilian_labor_force / civilian_noninstitutional_"
    "population_16+`. The Census Population Estimates Program ships "
    "*total* resident population, which includes children under 16, "
    "active-duty military, and the institutionalized — all three are "
    "excluded from the BLS denominator. We apply a **0.78** "
    "working-age civilian-noninstitutional correction (US average per "
    "BLS Handbook of Methods, ch. 1) so the displayed LFPR lands "
    "approximately ±2 pp on average of the published BLS state LFPR, "
    "with up to ~3 pp residual error at the extremes (e.g., UT, ID, "
    "ME, FL — whose civilian-noninstitutional share of total "
    "population sits noticeably above or below the US average). The "
    "uncorrected ratio is preserved as `{state}_LFPR_RAW` in the "
    "panel. See [`docs/methodology/lfpr_denominator.md`]"
    "(docs/methodology/lfpr_denominator.md) for the full audit + the "
    "per-state ACS-B23025-based fix that is planned to become the "
    "default denominator in Phase 2."
)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _inline_cites_suffix(cites: Iterable[str]) -> str:
    keys = list(cites)
    if not keys:
        return ""
    return " " + "; ".join(render_inline(k) for k in keys)


def _render_notes(notes: Iterable[MethodologyNote]) -> html.Ul:
    return html.Ul(
        [
            html.Li(note.text + _inline_cites_suffix(note.cites))
            for note in notes
        ],
        className="pi-method-list",
    )


def _render_references(keys: Iterable[str]) -> html.Ol:
    """Render the full bibliography as a numbered, author-year-sorted list."""
    items = []
    for key in sorted(set(keys), key=lambda k: (CITATIONS[k].year, CITATIONS[k].authors)):
        c = citation(key)
        line: list = [f"{c.authors} ({c.year}). ", html.I(c.title), f". {c.venue}."]
        if c.url:
            line.append(" ")
            line.append(html.A(c.url, href=c.url, target="_blank", rel="noopener"))
        items.append(html.Li(line))
    return html.Ol(items, className="pi-method-refs")


def _render_source_links(keys: Iterable[str]) -> html.Ul:
    """Render citation keys as a linked ``Title — venue (inline cite)`` list.

    Shared by the data-source, macro-context, and software-attribution
    blocks so the three lists stay visually identical.
    """
    return html.Ul(
        [
            html.Li(
                [
                    citation(k).title,
                    " — ",
                    html.A(
                        citation(k).venue,
                        href=citation(k).url,
                        target="_blank",
                        rel="noopener",
                    )
                    if citation(k).url
                    else citation(k).venue,
                    f" {render_inline(k)}",
                ]
            )
            for k in keys
        ],
        className="pi-method-list",
    )


def methodology_panel(
    *,
    forecast_notes: Iterable[MethodologyNote] = FORECAST_NOTES,
    diagnostic_notes: Iterable[MethodologyNote] = DIAGNOSTIC_NOTES,
    context_notes: Iterable[MethodologyNote] = CONTEXT_NOTES,
    data_source_keys: Iterable[str] = DATA_SOURCE_KEYS,
    context_source_keys: Iterable[str] = CONTEXT_SOURCE_KEYS,
    software_keys: Iterable[str] = SOFTWARE_ATTRIBUTION_KEYS,
    open_by_default: bool = False,
    summary_text: str = "Methodology & references",
) -> html.Details:
    """Render the collapsible methodology + references card.

    Drop into any tab that exposes forecast output; defaults cover both
    the LFP and the Super tabs — callers only need to override
    ``forecast_notes`` if their methodology diverges.

    Parameters
    ----------
    forecast_notes, diagnostic_notes : Iterable[MethodologyNote], optional
        Bullet sets shown under "Forecasting & scoring" and "Residual &
        series diagnostics", respectively.
    context_notes : Iterable[MethodologyNote], optional
        Macro & policy context bullets rendered inside the "Data
        sources" block (default :data:`CONTEXT_NOTES`).
    data_source_keys, software_keys : Iterable[str], optional
        Citation keys surfaced under "Data sources" and "Software & model
        attribution".
    context_source_keys : Iterable[str], optional
        Citation keys for the qualitative macro/policy sources linked
        under the data sources (default :data:`CONTEXT_SOURCE_KEYS`).
    open_by_default : bool, optional
        When True, the ``<details>`` element renders open.
    summary_text : str, optional
        Text rendered inside the ``<summary>`` toggle.

    Returns
    -------
    dash.html.Details
        The composed panel element.
    """
    forecast_notes = tuple(forecast_notes)
    diagnostic_notes = tuple(diagnostic_notes)
    context_notes = tuple(context_notes)
    data_source_keys = tuple(data_source_keys)
    context_source_keys = tuple(context_source_keys)
    software_keys = tuple(software_keys)

    all_keys: set[str] = set()
    for n in forecast_notes + diagnostic_notes + context_notes:
        all_keys.update(n.cites)
    all_keys.update(data_source_keys)
    all_keys.update(context_source_keys)
    all_keys.update(software_keys)

    return html.Details(
        [
            html.Summary(summary_text, className="pi-method-summary"),
            html.Div(
                [
                    html.H6("Forecasting & scoring"),
                    _render_notes(forecast_notes),
                    html.H6("Residual & series diagnostics"),
                    _render_notes(diagnostic_notes),
                    html.H6("Data sources"),
                    html.P(
                        "Every figure on this dashboard is traceable to "
                        "one of the following published data programs.",
                        className="text-muted small",
                    ),
                    _render_source_links(data_source_keys),
                    _render_notes(context_notes),
                    html.P(
                        "Qualitative macro & policy context linked for "
                        "the narrative layer (not chart inputs):",
                        className="text-muted small",
                    ),
                    _render_source_links(context_source_keys),
                    html.H6("Software & model attribution"),
                    html.P(
                        [
                            "Built with Llama (Meta's Llama 3.2 Community "
                            "License — see ",
                            html.A(
                                "license",
                                href="https://www.llama.com/llama3_2/license/",
                                target="_blank",
                                rel="noopener",
                            ),
                            "). The numeric, statistical, embedding, and "
                            "agent-architecture references below underlie "
                            "the implementation; software is cited "
                            "alongside the seminal papers so reviewers can "
                            "audit code provenance the same way they audit "
                            "data provenance.",
                        ],
                        className="text-muted small",
                    ),
                    _render_source_links(software_keys),
                    html.H6("References"),
                    _render_references(all_keys),
                ],
                className="pi-method-body",
            ),
        ],
        className="pi-method-panel",
        open=open_by_default,
    )
