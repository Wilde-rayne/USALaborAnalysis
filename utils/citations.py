"""
Citation registry for the forecasting, diagnostics, and data layers.

Kept as plain dataclasses so every piece of methodology the dashboard
surfaces can point back at the seminal literature — both for
research-engineer credibility and to make the implementation's
choices auditable.

Public API:
    CITATIONS           — ``dict[str, Citation]`` keyed by short id.
    citation(key)       — quick lookup; raises KeyError on unknown.
    render_inline(key)  — short "(Authors, year)" string for in-text
                          references.
    render_full(key)    — full bibliography-style string.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Citation:
    """A single bibliographic entry."""

    key: str               # short identifier ("box_jenkins_1970")
    authors: str           # "Box, G. E. P., and Jenkins, G. M."
    year: int
    title: str             # "Time Series Analysis: Forecasting and Control"
    venue: str             # "Holden-Day" / "Journal of ..."
    url: str | None = None


#: Ordered registry. New citations land here; callers look up by key.
_CITATIONS_LIST: tuple[Citation, ...] = (
    # ----- Forecasting models --------------------------------------------
    Citation(
        key="hyndman_athanasopoulos_2018",
        authors="Hyndman, R. J., and Athanasopoulos, G.",
        year=2018,
        title="Forecasting: Principles and Practice",
        venue="OTexts (2nd ed.)",
        url="https://otexts.com/fpp2/",
    ),
    Citation(
        key="holt_1957",
        authors="Holt, C. C.",
        year=1957,
        title="Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages",
        venue="ONR Research Memorandum 52, Carnegie Institute of Technology",
    ),
    Citation(
        key="winters_1960",
        authors="Winters, P. R.",
        year=1960,
        title="Forecasting Sales by Exponentially Weighted Moving Averages",
        venue="Management Science, 6(3), 324–342",
        url="https://doi.org/10.1287/mnsc.6.3.324",
    ),
    Citation(
        key="box_jenkins_1970",
        authors="Box, G. E. P., and Jenkins, G. M.",
        year=1970,
        title="Time Series Analysis: Forecasting and Control",
        venue="Holden-Day, San Francisco",
    ),
    Citation(
        key="hochreiter_schmidhuber_1997",
        authors="Hochreiter, S., and Schmidhuber, J.",
        year=1997,
        title="Long Short-Term Memory",
        venue="Neural Computation, 9(8), 1735–1780",
        url="https://doi.org/10.1162/neco.1997.9.8.1735",
    ),
    # ----- Stationarity + residual tests ---------------------------------
    Citation(
        key="dickey_fuller_1979",
        authors="Dickey, D. A., and Fuller, W. A.",
        year=1979,
        title="Distribution of the Estimators for Autoregressive Time Series with a Unit Root",
        venue="Journal of the American Statistical Association, 74(366), 427–431",
        url="https://doi.org/10.2307/2286348",
    ),
    Citation(
        key="kpss_1992",
        authors="Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., and Shin, Y.",
        year=1992,
        title=(
            "Testing the null hypothesis of stationarity against the "
            "alternative of a unit root"
        ),
        venue="Journal of Econometrics, 54(1–3), 159–178",
        url="https://doi.org/10.1016/0304-4076(92)90104-Y",
    ),
    Citation(
        key="ljung_box_1978",
        authors="Ljung, G. M., and Box, G. E. P.",
        year=1978,
        title="On a Measure of Lack of Fit in Time Series Models",
        venue="Biometrika, 65(2), 297–303",
        url="https://doi.org/10.1093/biomet/65.2.297",
    ),
    Citation(
        key="jarque_bera_1980",
        authors="Jarque, C. M., and Bera, A. K.",
        year=1980,
        title="Efficient tests for normality, homoscedasticity and serial independence of regression residuals",
        venue="Economics Letters, 6(3), 255–259",
        url="https://doi.org/10.1016/0165-1765(80)90024-5",
    ),
    Citation(
        key="diebold_mariano_1995",
        authors="Diebold, F. X., and Mariano, R. S.",
        year=1995,
        title="Comparing Predictive Accuracy",
        venue="Journal of Business & Economic Statistics, 13(3), 253–263",
        url="https://doi.org/10.1080/07350015.1995.10524599",
    ),
    Citation(
        key="harvey_leybourne_newbold_1997",
        authors="Harvey, D., Leybourne, S., and Newbold, P.",
        year=1997,
        title="Testing the equality of prediction mean squared errors",
        venue="International Journal of Forecasting, 13(2), 281–291",
        url="https://doi.org/10.1016/S0169-2070(96)00719-4",
    ),
    # ----- Backtesting + scoring -----------------------------------------
    Citation(
        key="tashman_2000",
        authors="Tashman, L. J.",
        year=2000,
        title="Out-of-sample tests of forecasting accuracy: an analysis and review",
        venue="International Journal of Forecasting, 16(4), 437–450",
        url="https://doi.org/10.1016/S0169-2070(00)00065-0",
    ),
    # ----- Data sources --------------------------------------------------
    Citation(
        key="bls_ces_handbook",
        authors="US Bureau of Labor Statistics",
        year=2024,
        title="Current Employment Statistics — State and Area Employment",
        venue="BLS Handbook of Methods",
        url="https://www.bls.gov/opub/hom/sae/home.htm",
    ),
    Citation(
        key="bls_laus_handbook",
        authors="US Bureau of Labor Statistics",
        year=2024,
        title="Local Area Unemployment Statistics — Technical Documentation",
        venue="BLS Handbook of Methods",
        url="https://www.bls.gov/lau/laumthd.htm",
    ),
    Citation(
        key="bls_jolts_handbook",
        authors="US Bureau of Labor Statistics",
        year=2024,
        title="Job Openings and Labor Turnover Survey (JOLTS) — Technical Note",
        venue="BLS Handbook of Methods",
        url="https://www.bls.gov/jlt/jlt_statedata.htm",
    ),
    Citation(
        key="bls_qcew_handbook",
        authors="US Bureau of Labor Statistics",
        year=2024,
        title="Quarterly Census of Employment and Wages — Overview",
        venue="BLS Handbook of Methods",
        url="https://www.bls.gov/cew/overview.htm",
    ),
    Citation(
        key="bls_cpi_handbook",
        authors="US Bureau of Labor Statistics",
        year=2024,
        title="Consumer Price Index — Methodology",
        venue="BLS Handbook of Methods",
        url="https://www.bls.gov/opub/hom/cpi/home.htm",
    ),
    Citation(
        key="census_acs_handbook",
        authors="US Census Bureau",
        year=2024,
        title="American Community Survey Design and Methodology",
        venue="US Census Bureau",
        url="https://www.census.gov/programs-surveys/acs/methodology.html",
    ),
    Citation(
        key="bea_regional_handbook",
        authors="US Bureau of Economic Analysis",
        year=2024,
        title="Regional Economic Accounts — Methodology",
        venue="BEA",
        url="https://www.bea.gov/resources/methodologies/regional-economic-accounts",
    ),
    Citation(
        key="fred_api",
        authors="Federal Reserve Bank of St. Louis",
        year=2024,
        title="FRED® — Federal Reserve Economic Data API",
        venue="Federal Reserve Bank of St. Louis",
        url="https://fred.stlouisfed.org/docs/api/fred/",
    ),
    Citation(
        key="fhfa_hpi_handbook",
        authors="Federal Housing Finance Agency",
        year=2024,
        title="House Price Index Technical Description",
        venue="FHFA",
        url="https://www.fhfa.gov/data/hpi/technical-documentation",
    ),
)


#: Lookup-by-key dict.
CITATIONS: dict[str, Citation] = {c.key: c for c in _CITATIONS_LIST}


def citation(key: str) -> Citation:
    """Raise-on-miss helper — better than a silent None for rendering."""
    try:
        return CITATIONS[key]
    except KeyError:
        raise KeyError(f"unknown citation key: {key!r}") from None


def render_inline(key: str) -> str:
    """Compact in-text form: ``(Box & Jenkins, 1970)``."""
    c = citation(key)
    # Shorten authors: drop initials for the parenthetical form. "Box, G. E. P.,
    # and Jenkins, G. M." → "Box & Jenkins".
    authors = _shorten_authors(c.authors)
    return f"({authors}, {c.year})"


def render_full(key: str) -> str:
    """Full bibliography line: authors (year). Title. Venue."""
    c = citation(key)
    parts = [f"{c.authors} ({c.year}). {c.title}."]
    if c.venue:
        parts.append(c.venue + ".")
    if c.url:
        parts.append(c.url)
    return " ".join(parts)


def _shorten_authors(authors: str) -> str:
    """
    Compact "Surname, Initials." style author strings to a short form:

      "Holt, C. C."                                      → "Holt"
      "Box, G. E. P., and Jenkins, G. M."                → "Box & Jenkins"
      "Harvey, D., Leybourne, S., and Newbold, P."       → "Harvey et al."
      "Kwiatkowski, D., Phillips, P. C. B., …, Shin, Y." → "Kwiatkowski et al."
      "US Bureau of Labor Statistics"                    → "US Bureau of Labor Statistics"

    The approach is a single regex that pulls surnames by looking for
    ``Surname, I.`` patterns — works for all the academic-style
    citations we carry and degrades to the raw string for institutional
    authors (who have no trailing initial to match).
    """
    import re  # noqa: PLC0415 — only needed here

    surnames = re.findall(
        r"([A-Z][a-zA-Z\u00C0-\u017F'\-]+),\s+[A-Z]\.",
        authors,
    )
    if not surnames:
        # Institutional author or unexpected format — fall back to the
        # text before the first comma (author list), or the whole string.
        return authors.split(",", 1)[0].strip() or authors
    if len(surnames) == 1:
        return surnames[0]
    if len(surnames) == 2:
        return f"{surnames[0]} & {surnames[1]}"
    return f"{surnames[0]} et al."
