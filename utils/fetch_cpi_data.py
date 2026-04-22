"""
Fetch BLS Consumer Price Index (CPI) regional indices.

BLS publishes CPI at several geographies; state-level isn't one of
them, but regional CPI (4 Census regions + the U.S. city average)
is, and it's enough to deflate dollar-denominated series (wages,
income, GDP) to real terms — which is the missing piece when the
dashboard compares 1996-vintage payroll counts against 2024-vintage
ones.

Series id format: ``CUUR{area:4}{item:3}0``
- Area codes: 0000 (US city avg), 0100 (Northeast), 0200 (Midwest),
  0300 (South), 0400 (West).
- Item: SA (All items) → suffixed with 0 → "SA0".

Uses the existing BLS API key (same endpoint / rate limits as CES
and LAUS); no new signup required.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Iterable

import requests

from utils.constants import API_KEY  # backward-compat alias for BLS_API_KEY

logger = logging.getLogger(__name__)

BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

#: Regional area code → (ontology-region-label, human description).
#: US-city-average is included so the output can deflate national
#: averages on the same footing.
CPI_REGIONS: dict[str, tuple[str, str]] = {
    "0000": ("US",        "U.S. city average"),
    "0100": ("Northeast", "Northeast urban"),
    "0200": ("Midwest",   "Midwest urban"),
    "0300": ("South",     "South urban"),
    "0400": ("West",      "West urban"),
}

ITEM_ALL_ITEMS = "SA0"   # CPI-U, all items
RAW_DIR_DEFAULT = os.path.join("data", "raw", "cpi")
# BLS v2 limits: 50 series per payload, 20 years per payload with a key.
BLS_MAX_YEARS_PER_REQUEST = 20


def _series_id(area_code: str, item_code: str = ITEM_ALL_ITEMS) -> str:
    return f"CUUR{area_code}{item_code}"


def fetch_cpi_regional(
    regions: Iterable[str] | None = None,
    start_year: int = 2000,
    end_year: int = 2024,
    *,
    raw_dir: str = RAW_DIR_DEFAULT,
    api_key: str | None = None,
    item_code: str = ITEM_ALL_ITEMS,
) -> int:
    """
    Download BLS regional CPI series and write per-region TXTs.

    ``regions``: any subset of ``CPI_REGIONS`` keys ("0000", "0100",
    "0200", "0300", "0400"). ``None`` fetches all five.

    Returns the total number of ``(region, year, month)`` rows written.
    Individual year-batch failures are logged and skipped; the rest
    still write their files.
    """
    if regions is None:
        regions = list(CPI_REGIONS.keys())
    regions = list(regions)
    unknown = [r for r in regions if r not in CPI_REGIONS]
    if unknown:
        raise ValueError(f"unknown CPI region codes: {unknown}")
    if start_year < 1913 or end_year < start_year:
        raise ValueError(
            f"invalid year range: {start_year}-{end_year} "
            "(BLS CPI-U starts 1913)"
        )

    key = api_key if api_key is not None else API_KEY
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    # Batch years into ≤20-year slices (BLS v2 limit with a key).
    span = end_year - start_year + 1
    n_batches = max(1, math.ceil(span / BLS_MAX_YEARS_PER_REQUEST))
    year_batches: list[tuple[int, int]] = []
    for i in range(n_batches):
        lo = start_year + i * BLS_MAX_YEARS_PER_REQUEST
        hi = min(end_year, lo + BLS_MAX_YEARS_PER_REQUEST - 1)
        year_batches.append((lo, hi))

    series_ids = [_series_id(r, item_code) for r in regions]
    rows_by_sid: dict[str, list[tuple[int, str, float]]] = {
        sid: [] for sid in series_ids
    }

    for batch_lo, batch_hi in year_batches:
        payload: dict = {
            "seriesid": series_ids,
            "startyear": str(batch_lo),
            "endyear": str(batch_hi),
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
            logger.warning(
                f"[CPI] {batch_lo}-{batch_hi}: request failed — {exc}"
            )
            continue

        if body.get("status") != "REQUEST_SUCCEEDED":
            logger.warning(
                f"[CPI] {batch_lo}-{batch_hi}: API status "
                f"{body.get('status')}: {body.get('message')}"
            )
            continue

        for series in body.get("Results", {}).get("series", []):
            sid = series.get("seriesID")
            if sid not in rows_by_sid:
                continue
            for rec in series.get("data", []):
                period = rec.get("period") or ""
                # Keep monthly periods (M01..M12); drop annual averages
                # (M13), semiannuals (S01/S02), and anything else.
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
            for y, p, v in sorted(rows, key=lambda x: (x[0], x[1])):
                f.write(f"{sid},{y},{p},{v}\n")
        total += len(rows)
        logger.info(f"[CPI] {sid}: wrote {len(rows)} rows → {path}")

    return total
