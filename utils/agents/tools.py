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


#: The canonical toolbelt an orchestration agent gets by default.
ALL_TOOLS: tuple["BaseTool", ...] = (
    fetch_bls_ces,
    fetch_bls_laus,
    fetch_census_population,
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
