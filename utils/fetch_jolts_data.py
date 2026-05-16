"""
Fetch BLS JOLTS — Job Openings and Labor Turnover Survey.

JOLTS is the official source for US job-openings, hires, quits, and
layoffs counts. National JOLTS has been published monthly since 2000;
the BLS experimental "State JOLTS" program released state-level
openings / hires / separations in 2023. Series id layout is the same
as the other BLS v2 endpoints the rest of the pipeline already uses
(CES, LAUS, CPI), so this fetcher mirrors their POST-to-seriesid
pattern.

Because the state-level series ids are still settling (BLS itself
labels them "experimental"), this fetcher takes an explicit list of
series ids instead of building them from an ontology — let the
caller pick the slice they want and leave the resolution layer
(agent / tool) to validate it against a known set.

National presets ship in ``NATIONAL_JOLTS_SERIES`` so a minimal
invocation "fetch JOLTS for the US since 2020" doesn't require the
caller to memorize id formats.

API docs:
    https://www.bls.gov/jlt/
    https://www.bls.gov/jlt/jlt_statedata.htm   (state data + series codes)
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Iterable

import requests

from utils.constants import API_KEY  # backward-compat alias for BLS_API_KEY

logger = logging.getLogger(__name__)

BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

#: National JOLTS totals — Total Nonfarm private + public, seasonally
#: adjusted level (thousands). Monthly from 2000-12 onward.
NATIONAL_JOLTS_SERIES: tuple[str, ...] = (
    "JTS000000000000000JOL",  # Total job openings, seasonally adjusted, level
    "JTS000000000000000HIL",  # Total hires
    "JTS000000000000000QUL",  # Total quits
    "JTS000000000000000LDL",  # Total layoffs/discharges
    "JTS000000000000000TSL",  # Total separations
)

RAW_DIR_DEFAULT = os.path.join("data", "raw", "jolts")
BLS_MAX_YEARS_PER_REQUEST = 20
BLS_MAX_SERIES_PER_REQUEST = 50


def fetch_jolts(
    series_ids: Iterable[str] | None = None,
    start_year: int = 2000,
    end_year: int = 2024,
    *,
    raw_dir: str = RAW_DIR_DEFAULT,
    api_key: str | None = None,
) -> int:
    """
    Download BLS JOLTS series and write one TXT per series id.

    ``series_ids``: any collection of valid BLS JOLTS series ids. When
    ``None`` the national preset (openings / hires / quits / layoffs /
    separations, seasonally adjusted, total nonfarm) is fetched.
    """
    if series_ids is None:
        series_ids = list(NATIONAL_JOLTS_SERIES)
    series_ids = list(series_ids)
    if not series_ids:
        raise ValueError("series_ids must be non-empty")
    if start_year < 2000 or end_year < start_year:
        raise ValueError(
            f"invalid year range: {start_year}-{end_year} "
            "(JOLTS starts 2000-12)"
        )

    # Guard against callers passing pathologically large series sets in
    # one call — BLS v2 caps at 50 series per request.
    if len(series_ids) > BLS_MAX_SERIES_PER_REQUEST:
        raise ValueError(
            f"BLS v2 allows at most {BLS_MAX_SERIES_PER_REQUEST} "
            f"series per request; got {len(series_ids)}"
        )

    key = api_key if api_key is not None else API_KEY
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    # Split the year range into ≤20-year batches (BLS-with-key limit).
    span = end_year - start_year + 1
    n_batches = max(1, math.ceil(span / BLS_MAX_YEARS_PER_REQUEST))
    batches: list[tuple[int, int]] = []
    for i in range(n_batches):
        lo = start_year + i * BLS_MAX_YEARS_PER_REQUEST
        hi = min(end_year, lo + BLS_MAX_YEARS_PER_REQUEST - 1)
        batches.append((lo, hi))

    rows_by_sid: dict[str, list[tuple[int, str, float]]] = {
        sid: [] for sid in series_ids
    }

    for lo, hi in batches:
        payload: dict = {
            "seriesid": series_ids,
            "startyear": str(lo),
            "endyear": str(hi),
        }
        if key:
            payload["registrationkey"] = key
        try:
            resp = requests.post(
                BLS_API_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[JOLTS] {lo}-{hi}: request failed — {exc}")
            continue

        if body.get("status") != "REQUEST_SUCCEEDED":
            logger.warning(
                f"[JOLTS] {lo}-{hi}: API status {body.get('status')}: "
                f"{body.get('message')}"
            )
            continue

        for series in body.get("Results", {}).get("series", []):
            sid = series.get("seriesID")
            if sid not in rows_by_sid:
                continue
            for rec in series.get("data", []):
                period = rec.get("period") or ""
                # Keep monthly rows; drop annual (M13) and any other period.
                if not period.startswith("M") or period == "M13":
                    continue
                try:
                    year = int(rec.get("year"))
                    value = float(rec.get("value"))
                except (TypeError, ValueError):
                    continue
                rows_by_sid[sid].append((year, period, value))

    total = 0
    for sid, rows in rows_by_sid.items():
        if not rows:
            continue
        path = os.path.join(raw_dir, f"{sid}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("series_id,year,period,value\n")
            for y, p, v in sorted(rows, key=lambda r: (r[0], r[1])):
                f.write(f"{sid},{y},{p},{v}\n")
        total += len(rows)
        logger.info(f"[JOLTS] {sid}: wrote {len(rows)} rows → {path}")

    return total
