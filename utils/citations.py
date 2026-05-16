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
        title="Local Area Unemployment Statistics",
        venue="BLS Handbook of Methods",
        url="https://www.bls.gov/opub/hom/lau/home.htm",
    ),
    Citation(
        key="bls_jolts_handbook",
        authors="US Bureau of Labor Statistics",
        year=2024,
        title="Job Openings and Labor Turnover Survey (JOLTS)",
        venue="BLS Handbook of Methods",
        url="https://www.bls.gov/opub/hom/jlt/home.htm",
    ),
    Citation(
        key="bls_qcew_handbook",
        authors="US Bureau of Labor Statistics",
        year=2024,
        title="Quarterly Census of Employment and Wages",
        venue="BLS Handbook of Methods",
        url="https://www.bls.gov/opub/hom/cew/home.htm",
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
    # ----- Software / model attribution ----------------------------------
    Citation(
        key="reimers_gurevych_2019",
        authors="Reimers, N., and Gurevych, I.",
        year=2019,
        title="Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        venue="Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
        url="https://arxiv.org/abs/1908.10084",
    ),
    Citation(
        key="wang_e5_2022",
        authors="Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., and Wei, F.",
        year=2022,
        title="Text Embeddings by Weakly-Supervised Contrastive Pre-training",
        venue="arXiv:2212.03533",
        url="https://arxiv.org/abs/2212.03533",
    ),
    Citation(
        key="seabold_perktold_2010",
        authors="Seabold, S., and Perktold, J.",
        year=2010,
        title="Statsmodels: Econometric and Statistical Modeling with Python",
        venue="Proceedings of the 9th Python in Science Conference",
        url="https://conference.scipy.org/proceedings/scipy2010/seabold.html",
    ),
    Citation(
        key="pedregosa_sklearn_2011",
        authors=(
            "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., "
            "Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., "
            "Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., "
            "Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E."
        ),
        year=2011,
        title="Scikit-learn: Machine Learning in Python",
        venue="Journal of Machine Learning Research, 12, 2825–2830",
        url="https://www.jmlr.org/papers/v12/pedregosa11a.html",
    ),
    Citation(
        key="harris_numpy_2020",
        authors=(
            "Harris, C. R., Millman, K. J., van der Walt, S. J., "
            "Gommers, R., Virtanen, P., Cournapeau, D., et al."
        ),
        year=2020,
        title="Array programming with NumPy",
        venue="Nature, 585(7825), 357–362",
        url="https://doi.org/10.1038/s41586-020-2649-2",
    ),
    Citation(
        key="mckinney_pandas_2010",
        authors="McKinney, W.",
        year=2010,
        title="Data Structures for Statistical Computing in Python",
        venue="Proceedings of the 9th Python in Science Conference, 56–61",
        url="https://conference.scipy.org/proceedings/scipy2010/mckinney.html",
    ),
    Citation(
        key="abadi_tensorflow_2016",
        authors="Abadi, M., Barham, P., Chen, J., et al.",
        year=2016,
        title="TensorFlow: A System for Large-Scale Machine Learning",
        venue="12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)",
        url="https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi",
    ),
    Citation(
        key="tukey_1977",
        authors="Tukey, J. W.",
        year=1977,
        title="Exploratory Data Analysis",
        venue="Addison-Wesley",
    ),
    Citation(
        key="akaike_1974",
        authors="Akaike, H.",
        year=1974,
        title="A new look at the statistical model identification",
        venue="IEEE Transactions on Automatic Control, 19(6), 716–723",
        url="https://doi.org/10.1109/TAC.1974.1100705",
    ),
    Citation(
        key="efron_1979",
        authors="Efron, B.",
        year=1979,
        title="Bootstrap Methods: Another Look at the Jackknife",
        venue="The Annals of Statistics, 7(1), 1–26",
        url="https://doi.org/10.1214/aos/1176344552",
    ),
    # ----- Agent architecture antecedents --------------------------------
    Citation(
        key="shinn_reflexion_2023",
        authors="Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., and Yao, S.",
        year=2023,
        title="Reflexion: Language Agents with Verbal Reinforcement Learning",
        venue="Advances in Neural Information Processing Systems 36 (NeurIPS 2023)",
        url="https://arxiv.org/abs/2303.11366",
    ),
    Citation(
        key="minsky_1986",
        authors="Minsky, M.",
        year=1986,
        title="The Society of Mind",
        venue="Simon & Schuster",
    ),
    # ----- Model licenses (non-academic but worth surfacing) -------------
    Citation(
        key="llama3_license_2024",
        authors="Meta Platforms, Inc.",
        year=2024,
        title="Llama 3.2 Community License Agreement",
        venue="Meta",
        url="https://www.llama.com/llama3_2/license/",
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
