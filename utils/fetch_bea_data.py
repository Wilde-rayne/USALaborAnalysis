"""
Fetch state-level annual data from the US Bureau of Economic Analysis.

BEA's Regional API exposes the SAINC1 (state annual personal income) and
SAGDP2N (state GDP by industry) tables, among many others. This module
focuses on SAINC1 — per-capita personal income is a strong companion
signal to BLS labor-force data because it separates "number of earners"
from "earnings per earner", and it extends back to 1929 so trend
analysis is not storage-bound.

API docs: https://apps.bea.gov/api/_pdf/bea_web_service_api_user_guide.pdf
Free key:  https://apps.bea.gov/API/signup/
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable

import requests

from utils.ontology import ONTOLOGY

logger = logging.getLogger(__name__)

BEA_API_URL = "https://apps.bea.gov/api/data/"

# BEA's "Regional" dataset has a single "ALL STATES" synthetic GeoFips
# that returns all states in one call; we iterate year-by-year because
# the API paginates by year param rather than by page number.
DEFAULT_TABLE = "SAINC1"  # State annual personal income
DEFAULT_LINECODE = "3"    # Line 3 = Per capita personal income (dollars)

RAW_DIR_DEFAULT = os.path.join("data", "raw", "bea")


def _get_api_key() -> str | None:
    return os.getenv("BEA_API_KEY") or None


def _parse_year(year: int | str | None) -> int | None:
    """BEA returns year as int but be defensive about strings / None."""
    if year is None:
        return None
    try:
        return int(year)
    except (TypeError, ValueError):
        return None


def fetch_bea_sainc1(
    states: Iterable[str],
    start_year: int,
    end_year: int,
    *,
    linecode: str = DEFAULT_LINECODE,
    raw_dir: str = RAW_DIR_DEFAULT,
    api_key: str | None = None,
) -> int:
    """
    Download state personal-income rows into data/raw/bea/BEA_<ST>.txt.

    Writes a single line per (state, year) in ``series_id,year,period,value``
    so the merge layer can consume it alongside BLS LAUS files. ``period``
    is always ``"A01"`` (annual) because BEA SAINC1 is an annual table.

    Returns the number of rows actually written across all states.
    """
    api_key = api_key or _get_api_key()
    if not api_key:
        raise RuntimeError(
            "BEA_API_KEY is not set. Register a free key at "
            "https://apps.bea.gov/API/signup/ and add it to .env."
        )

    codes = [s.upper() for s in states]
    unknown = [c for c in codes if c not in ONTOLOGY.states]
    if unknown:
        raise ValueError(f"unknown state codes: {unknown}")
    if start_year < 1929 or end_year < start_year:
        raise ValueError(
            f"invalid year range: start={start_year} end={end_year} "
            "(BEA SAINC1 starts 1929)"
        )

    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    # One API call per year keeps each response small; for 96+ years
    # of history that's fine given BEA's generous rate limits.
    years = list(range(start_year, end_year + 1))
    # BEA accepts a comma-separated list of GeoFips — use FIPS codes.
    fips_by_state = {c: ONTOLOGY.states[c].fips for c in codes}
    geo_fips_param = ",".join(fips_by_state.values())

    written = 0
    rows_by_state: dict[str, list[tuple[int, float]]] = {c: [] for c in codes}
    fips_to_state = {v: k for k, v in fips_by_state.items()}

    for yr in years:
        params = {
            "UserID": api_key,
            "method": "GetData",
            "datasetname": "Regional",
            "TableName": DEFAULT_TABLE,
            "LineCode": linecode,
            "GeoFips": geo_fips_param,
            "Year": str(yr),
            "ResultFormat": "JSON",
        }
        try:
            resp = requests.get(BEA_API_URL, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[BEA] {yr}: request failed — {exc}")
            continue

        results = (
            payload.get("BEAAPI", {})
            .get("Results", {})
            .get("Data", [])
        )
        if not results:
            logger.info(f"[BEA] {yr}: no rows returned")
            continue

        for row in results:
            fips = str(row.get("GeoFips") or "")[:2]
            state = fips_to_state.get(fips)
            if state is None:
                continue
            try:
                value = float(str(row.get("DataValue", "")).replace(",", ""))
            except ValueError:
                continue
            rows_by_state[state].append((yr, value))
            written += 1

    series_id_tpl = f"BEA_{DEFAULT_TABLE}_L{linecode}"
    for state, rows in rows_by_state.items():
        if not rows:
            continue
        path = os.path.join(raw_dir, f"{series_id_tpl}_{state}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("series_id,year,period,value\n")
            for yr, val in sorted(rows):
                f.write(f"{series_id_tpl}_{state},{yr},A01,{val}\n")
        logger.info(f"[BEA] {state}: wrote {len(rows)} rows → {path}")

    return written
