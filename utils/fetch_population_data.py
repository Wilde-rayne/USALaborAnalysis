from __future__ import annotations
import os, requests
from typing import List
import logging
from .constants import CENSUS_API_KEY

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

RAW_DIR = "data/raw/laus"
ACS_URL = "https://api.census.gov/data/{year}/acs/acs1"
PEP_URL = "https://api.census.gov/data/{year}/pep/population"

# FIPS codes for states
STATE_FIPS = {
    "AL": "01","AK": "02","AZ": "04","AR": "05","CA": "06","CO": "08","CT": "09","DE": "10",
    "FL": "12","GA": "13","HI": "15","ID": "16","IL": "17","IN": "18","IA": "19","KS": "20",
    "KY": "21","LA": "22","ME": "23","MD": "24","MA": "25","MI": "26","MN": "27","MS": "28",
    "MO": "29","MT": "30","NE": "31","NV": "32","NH": "33","NJ": "34","NM": "35","NY": "36",
    "NC": "37","ND": "38","OH": "39","OK": "40","OR": "41","PA": "42","RI": "44","SC": "45",
    "SD": "46","TN": "47","TX": "48","UT": "49","VT": "50","VA": "51","WA": "53","WV": "54",
    "WI": "55","WY": "56"
}

def _fetch_acs(year: int, fips: str) -> int | None:
    url = (
        f"{ACS_URL.format(year=year)}"
        f"?get=B01003_001E&for=state:{fips}"
        f"&key={CENSUS_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10); resp.raise_for_status()
        data = resp.json()
        return int(data[1][0]) if len(data) > 1 else None
    except:
        return None

def _fetch_pep(year: int, fips: str) -> int | None:
    url = (
        f"{PEP_URL.format(year=year)}"
        f"?get=POP&for=state:{fips}"
        f"&key={CENSUS_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10); resp.raise_for_status()
        data = resp.json()
        return int(data[1][0]) if len(data) > 1 else None
    except:
        return None

def _fetch_population(year: int, fips: str) -> int | None:
    """
    Try ACS for year>=2005, otherwise PEP. If ACS fails for recent year, fall back to PEP.
    """
    if year >= 2005:
        pop = _fetch_acs(year, fips)
        if pop is None:
            pop = _fetch_pep(year, fips)
    else:
        pop = _fetch_pep(year, fips)
    return pop

def fetch_population(states: List[str], start: int, end: int) -> None:
    """
    Download state populations via Census (ACS/PEP) and write to data/raw/laus/POP_{ST}.txt.
    Each file has series_id,year,period,value (with period M01 for January).
    """
    if not CENSUS_API_KEY:
        raise RuntimeError(
            "CENSUS_API_KEY is not set. The Census ACS/PEP API requires a key. "
            "Copy .env.example to .env and set CENSUS_API_KEY (free signup at "
            "https://api.census.gov/data/key_signup.html)."
        )
    os.makedirs(RAW_DIR, exist_ok=True)
    for st in states:
        if st not in STATE_FIPS:
            logger.warning(f"[POP] Unknown state {st}")
            continue
        fips = STATE_FIPS[st]
        pop_data = {}
        missing_years = []
        for yr in range(start, end+1):
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
                    pop_est = int(pop_lower + (pop_upper - pop_lower) * (yr - lower) / (upper - lower))
                pop_data[yr] = pop_est
                logger.info(f"[POP] Interpolated population for {st} {yr}: {pop_est}")

        path = os.path.join(RAW_DIR, f"POP_{st}.txt")
        with open(path, "w") as f:
            f.write("series_id,year,period,value\n")
            for yr in sorted(pop_data):
                val = pop_data[yr]
                f.write(f"POP_{st},{yr},M01,{val}\n")
        logger.info(f"[POP] Saved {path}")
