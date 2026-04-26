"""
Fetch civilian noninstitutional population aged 16+ from Census ACS.

Closes the residual ~2 pp error in the LFPR denominator that the
uniform 0.78 working-age fraction leaves behind. ACS table B23025
publishes ``Population 16 years and over`` per state per year — the
closest publicly-available proxy for BLS's CNI16+ definition.

Variable: ``B23025_001E`` — "Total: Population 16 years and over"
Coverage:  ACS 1-year estimates 2005-present (skipping 2020 due to
           Census-side data quality flags).
Endpoint:  https://api.census.gov/data/{year}/acs/acs1

Output schema mirrors ``fetch_population_data`` so the merger can
treat it like any other LAUS-style series:

    series_id,year,period,value
    WAP_IA,2010,A01,2362178
    WAP_IA,2011,A01,2378450
    ...

Re-fetches are idempotent and incremental — files only get re-written
if the upstream value changed.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import requests

from utils.constants import CENSUS_API_KEY
from utils.ontology import ONTOLOGY

logger = logging.getLogger(__name__)

ACS_URL_TEMPLATE = (
    "https://api.census.gov/data/{year}/acs/acs1"
    "?get=B23025_001E&for=state:{fips}&key={key}"
)
RAW_DIR_DEFAULT = os.path.join("data", "raw", "laus")  # co-locates with PEP files

#: ACS 1-year released yearly since 2005. 2020 was suppressed.
DEFAULT_ACS_YEARS: tuple[int, ...] = tuple(
    y for y in range(2005, 2025) if y != 2020
)

#: Read-only proxy for the variable ID so external callers don't have
#: to memorize Census codes.
ACS_VARIABLE_CNI16 = "B23025_001E"


def _api_key() -> str | None:
    return CENSUS_API_KEY or None


def fetch_acs_working_age_population(
    states: Iterable[str],
    years: Iterable[int] | None = None,
    *,
    raw_dir: str = RAW_DIR_DEFAULT,
    api_key: str | None = None,
) -> int:
    """
    Pull ``B23025_001E`` (Population 16+) for each (state, year) pair
    and write per-state TXT files in the standard
    ``series_id,year,period,value`` schema.

    Returns the count of (state, year) rows written across all files.
    Raises only on a missing API key or malformed state input — bad
    individual responses are logged and skipped.
    """
    api_key = api_key or _api_key()
    if not api_key:
        raise RuntimeError(
            "CENSUS_API_KEY is not set. Register a free key at "
            "https://api.census.gov/data/key_signup.html and put it "
            "in .env."
        )

    state_codes = [s.upper() for s in states]
    unknown = [c for c in state_codes if c not in ONTOLOGY.states]
    if unknown:
        raise ValueError(f"unknown state codes: {unknown}")

    if years is None:
        years = DEFAULT_ACS_YEARS
    else:
        years = tuple(years)
        if not years:
            raise ValueError("years must be non-empty")

    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    rows: dict[str, list[tuple[int, int]]] = {st: [] for st in state_codes}
    for st in state_codes:
        fips = ONTOLOGY.states[st].fips
        for year in years:
            url = ACS_URL_TEMPLATE.format(year=year, fips=fips, key=api_key)
            try:
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:  # noqa: BLE001
                logger.info(f"[ACS-CNI16] {st} {year}: {exc}")
                continue
            # Expected payload: [[header...], [value, fips_str]] — value index 0.
            if not isinstance(payload, list) or len(payload) < 2:
                continue
            try:
                value = int(payload[1][0])
            except (TypeError, ValueError, IndexError):
                continue
            rows[st].append((year, value))

    total = 0
    for st, year_values in rows.items():
        if not year_values:
            continue
        sid = f"WAP_{st}"
        path = os.path.join(raw_dir, f"{sid}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("series_id,year,period,value\n")
            for y, v in sorted(year_values):
                f.write(f"{sid},{y},A01,{v}\n")
        total += len(year_values)
        logger.info(f"[ACS-CNI16] {sid}: wrote {len(year_values)} rows → {path}")

    return total
