"""
LangChain tool wrappers around the data fetchers.

Each ``@tool`` here is a self-contained capability an agent can call:
refresh BLS CES, refresh BLS LAUS, refresh Census population, or
ensure the merged panel is fresh. The tools delegate to the
underlying ``utils.fetch_*`` functions but add:

- typed Pydantic-style argument schemas so the LLM sees clean
  parameter descriptions;
- state-code normalization (case-insensitive, ontology-checked);
- structured return messages that summarize what was fetched.

The tools are intentionally side-effectful — they write to
``data/raw/``. Tests monkeypatch the underlying fetchers so nothing
hits the internet.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.tools import tool

from utils.ontology import ONTOLOGY

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _normalize_states(codes: list[str]) -> list[str]:
    """Upper-case + validate every state code against the ontology."""
    out: list[str] = []
    for code in codes:
        c = str(code).strip().upper()
        if c not in ONTOLOGY.states:
            raise ValueError(f"unknown state code: {code!r}")
        out.append(c)
    if not out:
        raise ValueError("states list must be non-empty")
    return out


def _validate_year_range(start_year: int, end_year: int) -> None:
    if not isinstance(start_year, int) or not isinstance(end_year, int):
        raise ValueError("start_year and end_year must be integers")
    if start_year < 1948:
        raise ValueError("BLS state-level series start around 1948; start_year too early")
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
@tool
def fetch_bls_ces(states: list[str], start_year: int, end_year: int) -> str:
    """
    Download BLS Current Employment Statistics (CES) for the given
    states and year range. Writes per-series text files into
    data/raw/ces/. Returns a summary of what was fetched.
    """
    from utils.fetch_ces_data import fetch_ces_data  # noqa: PLC0415

    codes = _normalize_states(states)
    _validate_year_range(start_year, end_year)
    fetch_ces_data(codes, start_year, end_year)
    return (
        f"Fetched BLS CES for {len(codes)} state(s): "
        f"{', '.join(codes)} over {start_year}-{end_year}."
    )


@tool
def fetch_bls_laus(states: list[str], start_year: int, end_year: int) -> str:
    """
    Download BLS Local Area Unemployment Statistics (LAUS) for the
    given states and year range. Writes per-series text files into
    data/raw/laus/. Returns a summary of what was fetched.
    """
    from utils.fetch_laus_data import fetch_laus_data  # noqa: PLC0415

    codes = _normalize_states(states)
    _validate_year_range(start_year, end_year)
    fetch_laus_data(codes, start_year, end_year)
    return (
        f"Fetched BLS LAUS for {len(codes)} state(s): "
        f"{', '.join(codes)} over {start_year}-{end_year}."
    )


@tool
def fetch_census_population(states: list[str], start_year: int, end_year: int) -> str:
    """
    Download US Census Bureau population estimates (ACS/PEP) for the
    given states and year range. Writes per-state text files into
    data/raw/laus/. Returns a summary.
    """
    from utils.fetch_population_data import fetch_population  # noqa: PLC0415

    codes = _normalize_states(states)
    _validate_year_range(start_year, end_year)
    fetch_population(codes, start_year, end_year)
    return (
        f"Fetched Census population for {len(codes)} state(s): "
        f"{', '.join(codes)} over {start_year}-{end_year}."
    )


@tool
def ensure_merged_data(force: bool = False) -> str:
    """
    Make sure data/all_data.json is present and fresh. If the cache is
    older than CACHE_MAX_AGE_SECONDS (default 7 days) or ``force=True``
    is passed, re-runs the full refresh pipeline (CES + LAUS + Census
    + merge). Returns the output path.
    """
    from utils.data_pipeline import ensure_data  # noqa: PLC0415

    path = ensure_data(force=force)
    return f"Merged panel at {path} (force={force})."


@tool
def describe_series(series_id: str) -> str:
    """
    Resolve a BLS series id to a one-line plain-English description
    using the labor-data ontology. Useful when a user pastes a raw
    series id and wants to know what it represents before fetching.
    """
    return ONTOLOGY.describe(series_id)


@tool
def fetch_bea_personal_income(states: list[str], start_year: int, end_year: int) -> str:
    """
    Download BEA SAINC1 per-capita personal income (annual, 1929+) for
    the given states and years. Writes data/raw/bea/BEA_SAINC1_L3_<ST>.txt
    files alongside BLS output. Requires the ``BEA_API_KEY`` env var;
    register a free key at https://apps.bea.gov/API/signup/.
    """
    from utils.fetch_bea_data import fetch_bea_sainc1  # noqa: PLC0415

    codes = _normalize_states(states)
    _validate_year_range(start_year, end_year)
    written = fetch_bea_sainc1(codes, start_year, end_year)
    return (
        f"Fetched BEA SAINC1 (per-capita personal income) for "
        f"{len(codes)} state(s): {', '.join(codes)} "
        f"over {start_year}-{end_year}; {written} rows written."
    )


@tool
def fetch_bls_jolts(
    series_ids: list[str] | None, start_year: int, end_year: int
) -> str:
    """
    Download BLS JOLTS (Job Openings and Labor Turnover Survey) rows.
    Pass ``None`` or an empty list to fetch the national preset —
    total nonfarm openings / hires / quits / layoffs / separations,
    seasonally adjusted, monthly from 2000. Pass an explicit list of
    series ids (including BLS's experimental state-JOLTS codes) for a
    targeted pull. Uses the existing BLS_API_KEY.
    """
    from utils.fetch_jolts_data import (  # noqa: PLC0415
        NATIONAL_JOLTS_SERIES,
        fetch_jolts,
    )

    if not series_ids:
        series_ids = list(NATIONAL_JOLTS_SERIES)
    _validate_year_range(start_year, end_year)
    written = fetch_jolts(series_ids, start_year, end_year)
    return (
        f"Fetched BLS JOLTS for {len(series_ids)} series over "
        f"{start_year}-{end_year}; {written} monthly rows written."
    )


@tool
def fetch_bls_regional_cpi(
    regions: list[str] | None, start_year: int, end_year: int
) -> str:
    """
    Download BLS CPI-U (Consumer Price Index, all items) for the four
    Census regions (Northeast / Midwest / South / West) + US city
    average. Pass ``None`` or an empty list to fetch all five. Uses
    the existing BLS_API_KEY; no additional signup required.
    """
    from utils.fetch_cpi_data import CPI_REGIONS, fetch_cpi_regional  # noqa: PLC0415

    if not regions:
        regions = list(CPI_REGIONS.keys())
    bad = [r for r in regions if r not in CPI_REGIONS]
    if bad:
        raise ValueError(
            f"unknown CPI region codes: {bad}; allowed: "
            f"{sorted(CPI_REGIONS)}"
        )
    _validate_year_range(start_year, end_year)
    written = fetch_cpi_regional(regions, start_year, end_year)
    labels = ", ".join(CPI_REGIONS[r][0] for r in regions)
    return (
        f"Fetched BLS CPI-U for regions [{labels}] "
        f"{start_year}-{end_year}; {written} rows written."
    )


@tool
def fetch_fred_state_indicator(
    states: list[str], indicator: str, start_year: int, end_year: int
) -> str:
    """
    Download a FRED state-level indicator family. ``indicator`` must be
    one of: "UR" (unemployment rate, monthly), "PI" (personal income,
    quarterly), "NGSP" (nominal gross state product, annual), "STHPI"
    (FHFA all-transactions state house price index, quarterly —
    1975Q1-present), or "MHI" (state median household income, annual —
    FRED series ``MEHOINUS{ST}A646N``, 1984-present, useful as a
    real-household-earnings companion to BEA per-capita and BLS
    earnings-per-hour measures). Writes
    data/raw/fred/FRED_<series_id>.txt files. Requires ``FRED_API_KEY``
    env; register at https://fred.stlouisfed.org/docs/api/api_key.html.
    """
    from utils.fetch_fred_data import (  # noqa: PLC0415
        FRED_INDICATORS,
        fetch_fred_state_series,
    )

    ind = str(indicator).upper()
    if ind not in FRED_INDICATORS:
        raise ValueError(
            f"unknown FRED indicator: {indicator!r}; supported: "
            f"{sorted(FRED_INDICATORS)}"
        )
    codes = _normalize_states(states)
    _validate_year_range(start_year, end_year)
    written = fetch_fred_state_series(codes, ind, start_year, end_year)
    desc, _ = FRED_INDICATORS[ind]
    return (
        f"Fetched FRED {ind} ({desc}) for {len(codes)} state(s): "
        f"{', '.join(codes)} over {start_year}-{end_year}; "
        f"{written} rows written."
    )


#: The canonical toolbelt an orchestration agent gets by default.
ALL_TOOLS: tuple["BaseTool", ...] = (
    fetch_bls_ces,
    fetch_bls_laus,
    fetch_census_population,
    fetch_bea_personal_income,
    fetch_fred_state_indicator,
    fetch_bls_regional_cpi,
    fetch_bls_jolts,
    ensure_merged_data,
    describe_series,
)


def build_data_refresh_agent() -> Any:
    """
    Return a ready-to-invoke DeepAgents-based orchestrator bound to
    ``ALL_TOOLS`` and phi3 via ``utils.agents.deep``. Thin shim so
    existing callers that used this function don't have to re-import.
    """
    from utils.agents.deep import data_refresh_agent  # noqa: PLC0415

    return data_refresh_agent()
