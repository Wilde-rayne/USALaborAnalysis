"""
Reusable Methodology + References panel for the forecasting tabs.

Surfaces, in one collapsible block per tab:

1. What the forecast actually did (scoring rule + backtest protocol),
2. Which candidate models competed and the paper each cites,
3. Which statistical tests we ran on the residuals + series,
4. Which BLS / Census / BEA / FRED datasets feed the pipeline.

Every claim carries an inline citation that round-trips through
``utils.citations``, so the list of 20+ references is defined once
and used everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from dash import html

from utils.citations import CITATIONS, citation, render_inline


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
        "The winning model for each series is chosen by minimum "
        "out-of-sample RMSE over a 3-fold expanding-window backtest, "
        "following standard practice in the forecasting literature.",
        cites=("hyndman_athanasopoulos_2018", "tashman_2000"),
    ),
    MethodologyNote(
        "Candidate models include random-walk persistence (Naive), "
        "seasonal persistence (Seasonal-Naive) and classical "
        "exponential smoothing (Holt–Winters).",
        cites=("holt_1957", "winters_1960", "hyndman_athanasopoulos_2018"),
    ),
    MethodologyNote(
        "ARIMA orders are AIC-selected from a small grid "
        "{(1,1,1), (2,1,1), (1,1,2), (2,1,2)}.",
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
        "The winning model is compared to a Naive baseline using the "
        "Diebold–Mariano test with the Harvey–Leybourne–Newbold "
        "small-sample correction, so the implied significance stays "
        "honest on short held-out windows.",
        cites=("diebold_mariano_1995", "harvey_leybourne_newbold_1997"),
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
    "within ~2 pp of the published BLS state LFPR. The uncorrected "
    "ratio is preserved as `{state}_LFPR_RAW` in the panel. See "
    "[`docs/methodology/lfpr_denominator.md`]"
    "(https://github.com/) for the full audit + the per-state "
    "ACS-B23025-based fix that's queued for the next data refresh."
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
    """
    Full bibliography rendering — numbered list with author-year sort.
    Each entry gets a DOI / handbook link when we have one.
    """
    items = []
    for key in sorted(set(keys), key=lambda k: (CITATIONS[k].year, CITATIONS[k].authors)):
        c = citation(key)
        line: list = [f"{c.authors} ({c.year}). ", html.I(c.title), f". {c.venue}."]
        if c.url:
            line.append(" ")
            line.append(html.A(c.url, href=c.url, target="_blank", rel="noopener"))
        items.append(html.Li(line))
    return html.Ol(items, className="pi-method-refs")


def methodology_panel(
    *,
    forecast_notes: Iterable[MethodologyNote] = FORECAST_NOTES,
    diagnostic_notes: Iterable[MethodologyNote] = DIAGNOSTIC_NOTES,
    data_source_keys: Iterable[str] = DATA_SOURCE_KEYS,
    open_by_default: bool = False,
    summary_text: str = "Methodology & references",
) -> html.Details:
    """
    Render a collapsible methodology card. Drop into any tab that
    exposes forecast output; defaults cover both the LFP and the
    Super tabs — callers only need to override ``forecast_notes`` if
    their methodology diverges.
    """
    forecast_notes = tuple(forecast_notes)
    diagnostic_notes = tuple(diagnostic_notes)
    data_source_keys = tuple(data_source_keys)

    all_keys: set[str] = set()
    for n in forecast_notes + diagnostic_notes:
        all_keys.update(n.cites)
    all_keys.update(data_source_keys)

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
                    html.Ul(
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
                            for k in data_source_keys
                        ],
                        className="pi-method-list",
                    ),
                    html.H6("References"),
                    _render_references(all_keys),
                ],
                className="pi-method-body",
            ),
        ],
        className="pi-method-panel",
        open=open_by_default,
    )
