"""Fetch state populations from the Census ACS/PEP APIs."""
from __future__ import annotations

import logging
import os
from typing import List

import requests

from .constants import CENSUS_API_KEY
from .ontology import ONTOLOGY

logger = logging.getLogger(__name__)

RAW_DIR = "data/raw/laus"
ACS_URL = "https://api.census.gov/data/{year}/acs/acs1"
PEP_URL = "https://api.census.gov/data/{year}/pep/population"


def _fetch_acs(year: int, fips: str) -> int | None:
    url = (
        f"{ACS_URL.format(year=year)}"
        f"?get=B01003_001E&for=state:{fips}"
        f"&key={CENSUS_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return int(data[1][0]) if len(data) > 1 else None
    except Exception as e:
        logger.warning(f"[POP] ACS lookup failed for fips={fips} year={year}: {e}")
        return None


def _fetch_pep(year: int, fips: str) -> int | None:
    url = (
        f"{PEP_URL.format(year=year)}"
        f"?get=POP&for=state:{fips}"
        f"&key={CENSUS_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return int(data[1][0]) if len(data) > 1 else None
    except Exception as e:
        logger.warning(f"[POP] PEP lookup failed for fips={fips} year={year}: {e}")
        return None


def _fetch_population(year: int, fips: str) -> int | None:
    """Try ACS for year ≥ 2005 then PEP; fall back to PEP if ACS fails."""
    if year >= 2005:
        pop = _fetch_acs(year, fips)
        if pop is None:
            pop = _fetch_pep(year, fips)
    else:
        pop = _fetch_pep(year, fips)
    return pop


def fetch_population(states: List[str], start: int, end: int) -> None:
    """Download state populations from Census ACS/PEP into ``POP_{ST}.txt`` files.

    Each file has ``series_id,year,period,value`` rows with ``period=M01``
    (January). Years with neither ACS nor PEP available are filled by
    linear interpolation between the closest known years, or held at the
    edge value when outside the coverage window.

    Parameters
    ----------
    states : list[str]
        USPS state codes.
    start, end : int
        Inclusive year range.

    Raises
    ------
    RuntimeError
        When ``CENSUS_API_KEY`` is unset.
    """
    if not CENSUS_API_KEY:
        raise RuntimeError(
            "CENSUS_API_KEY is not set. The Census ACS/PEP API requires a key. "
            "Copy .env.example to .env and set CENSUS_API_KEY (free signup at "
            "https://api.census.gov/data/key_signup.html)."
        )
    os.makedirs(RAW_DIR, exist_ok=True)
    for st in states:
        state = ONTOLOGY.states.get(st.upper())
        if state is None:
            logger.warning(f"[POP] Unknown state {st}")
            continue
        fips = state.fips
        pop_data: dict[int, int] = {}
        missing_years: list[int] = []
        for yr in range(start, end + 1):
            pop = _fetch_population(yr, fips)
            if pop is not None:
                pop_data[yr] = pop
            else:
                missing_years.append(yr)
                logger.warning(f"[POP] Missing census population for {st} {yr}")

        if missing_years:
            sorted_years = sorted(pop_data.keys())
            for yr in missing_years:
                if not sorted_years:
                    logger.warning(f"[POP] No available data to interpolate for {st} {yr}")
                    continue
                if yr < sorted_years[0]:
                    pop_est = pop_data[sorted_years[0]]
                elif yr > sorted_years[-1]:
                    pop_est = pop_data[sorted_years[-1]]
                else:
                    lower = max(y for y in sorted_years if y < yr)
                    upper = min(y for y in sorted_years if y > yr)
                    pop_lower = pop_data[lower]
                    pop_upper = pop_data[upper]
                    pop_est = int(
                        pop_lower
                        + (pop_upper - pop_lower) * (yr - lower) / (upper - lower)
                    )
                pop_data[yr] = pop_est
                logger.info(f"[POP] Interpolated population for {st} {yr}: {pop_est}")

        path = os.path.join(RAW_DIR, f"POP_{st}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("series_id,year,period,value\n")
            for yr in sorted(pop_data):
                f.write(f"POP_{st},{yr},M01,{pop_data[yr]}\n")
        logger.info(f"[POP] Saved {path}")
