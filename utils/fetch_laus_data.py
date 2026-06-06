#!/usr/bin/env python3
"""Fetch LAUS data via the BLS API and save one TXT per series.

Series ids come from ``data/laus_state_codes.json``; each is pulled in
:data:`YEAR_SLICE`-year chunks and saved under :data:`RAW_DIR`.
"""
import json
import logging
import os
import sys

import requests

from .constants import API_KEY

# --- Configuration ---
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
CODES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "data", "laus_state_codes.json")
RAW_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw", "laus")
BATCH_SIZE = 10  # number of series per request
YEAR_SLICE = 5   # years per API call
MONTHS_IN_YEAR = 12
START_YEAR = 1996
END_YEAR = 2024

logger = logging.getLogger(__name__)


def load_codes(path=CODES_PATH):
    """Load the LAUS codes JSON and flatten it to ``(measure, state, series_id)`` rows.

    Parameters
    ----------
    path : str, optional
        Path to the codes JSON (default :data:`CODES_PATH`).

    Returns
    -------
    list[tuple[str, str, str]]
        One row per ``(measure, state, series_id)`` triple in the JSON.
    """
    if not os.path.exists(path):
        logger.error(f"LAUS codes file not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        (measure, state, sid)
        for measure, state_map in data.items()
        for state, sid in state_map.items()
    ]


def fetch_laus_data(states, start_year, end_year):
    """Fetch LAUS series for ``states`` between ``start_year`` and ``end_year``.

    Saves each series to ``{RAW_DIR}/{series_id}.txt`` with a header and
    monthly data rows.

    Parameters
    ----------
    states : list[str]
        USPS state codes to fetch.
    start_year, end_year : int
        Inclusive year range.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    # Clean out old .txt files
    for fn in os.listdir(RAW_DIR):
        if fn.endswith(".txt"):
            os.remove(os.path.join(RAW_DIR, fn))

    codes = load_codes()
    series_ids = [sid for _, state, sid in codes if state in states]
    total = len(series_ids)
    logger.info(f"Fetching {total} LAUS series for states: {states}")

    # Batch series and year slices
    batches = [series_ids[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    for b_idx, batch in enumerate(batches, start=1):
        logger.info(f"Series batch {b_idx}/{len(batches)}: {len(batch)} IDs")
        for year_start in range(start_year, end_year + 1, YEAR_SLICE):
            year_end = min(year_start + YEAR_SLICE - 1, end_year)
            logger.info(f"  Years slice: {year_start}-{year_end}")

            payload = {
                "seriesid": batch,
                "startyear": str(year_start),
                "endyear":   str(year_end),
                "registrationkey": API_KEY,
            }
            try:
                resp = requests.post(
                    BLS_API_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                resp.raise_for_status()
            except Exception as e:
                logger.error(f"HTTP error for {year_start}-{year_end}: {e}")
                continue

            result = resp.json()
            if result.get("status") != "REQUEST_SUCCEEDED":
                msg = result.get("message", [])
                logger.error(f"API failed for {year_start}-{year_end}: {msg}")
                continue

            for series in result["Results"]["series"]:
                sid = series.get("seriesID")
                data_pts = [
                    d for d in series.get("data", [])
                    if d.get("period", "").startswith("M")
                ]
                if not data_pts:
                    logger.warning(f"No monthly data for {sid} in {year_start}-{year_end}")
                    continue

                path = os.path.join(RAW_DIR, f"{sid}.txt")
                write_header = not os.path.exists(path)
                with open(path, "a", encoding="utf-8") as f:
                    if write_header:
                        f.write("series_id,year,period,value\n")
                    # write oldest first
                    for d in reversed(data_pts):
                        f.write(f"{sid},{d['year']},{d['period']},{d['value']}\n")
                logger.info(f"Saved {len(data_pts)} pts for {sid} ({year_start}-{year_end})")

                expected = MONTHS_IN_YEAR * (year_end - year_start + 1)
                if len(data_pts) != expected:
                    logger.warning(
                        f"{sid} has {len(data_pts)}/{expected} pts in "
                        f"{year_start}-{year_end}"
                    )


if __name__ == "__main__":
    # Automatically fetch for all states present in the JSON codes file
    codes = load_codes()
    states = sorted({state for _, state, _ in codes})
    fetch_laus_data(states, START_YEAR, END_YEAR)
