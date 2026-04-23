"""
Fetch BLS Quarterly Census of Employment and Wages (QCEW).

QCEW is the BLS's comprehensive quarterly payroll file: every
state-level UI-covered job is in it. Publishes quarterly employment
counts, total wages, and average weekly wages by state, county,
ownership, and NAICS industry.

Rather than use the BLS v2 timeseries JSON API (which requires knowing
the exact 14-character series id format in advance), this fetcher hits
the QCEW ``data_slices`` CSV endpoint — one file per (state, year,
quarter). The response has dozens of columns; we extract the
state-total row (``agglvl_code='50'``, all ownerships, all industries)
and emit the canonical ``series_id,year,period,value`` shape the
merger already consumes.

API docs:
    https://data.bls.gov/cew/doc/access/csv_data_slices.htm

Endpoint pattern:
    https://data.bls.gov/cew/data/api/{year}/{qtr}/area/{FIPS}000.csv
"""
from __future__ import annotations

import csv
import io
import logging
import os
from pathlib import Path
from typing import Iterable

import requests

from utils.ontology import ONTOLOGY

logger = logging.getLogger(__name__)

QCEW_URL_TEMPLATE = "https://data.bls.gov/cew/data/api/{year}/{qtr}/area/{area}.csv"

#: QCEW aggregation level for statewide totals (all industries, all
#: ownerships). BLS publishes many roll-up levels; 50 is the state
#: summary used by the public data-views UI.
STATE_TOTAL_AGGLVL = "50"

#: Columns we persist. QCEW emits many more but these are the ones
#: wages-and-headcount analysis actually cares about.
METRICS: tuple[tuple[str, str, str], ...] = (
    # (csv_column,           readable,            canonical_suffix)
    ("month1_emplvl",        "employment level",  "EMP"),
    ("total_qtrly_wages",    "total quarterly wages", "TQW"),
    ("avg_wkly_wage",        "average weekly wage",   "AWW"),
    ("qtrly_estabs_count",   "establishments",    "EST"),
)

RAW_DIR_DEFAULT = os.path.join("data", "raw", "qcew")


def fetch_qcew_state_totals(
    states: Iterable[str],
    start_year: int,
    end_year: int,
    *,
    quarters: Iterable[int] | None = None,
    raw_dir: str = RAW_DIR_DEFAULT,
) -> int:
    """
    Download QCEW state-total rows and write per-(state, metric) TXTs.

    ``quarters``: optional restriction, e.g. ``[1, 2]``. Defaults to
    all four. Returns the total number of (state, year, quarter,
    metric) rows written across every output file.
    """
    quarters = tuple(quarters) if quarters else (1, 2, 3, 4)
    bad_q = [q for q in quarters if q not in (1, 2, 3, 4)]
    if bad_q:
        raise ValueError(f"invalid quarters: {bad_q}")

    codes = [c.upper() for c in states]
    unknown = [c for c in codes if c not in ONTOLOGY.states]
    if unknown:
        raise ValueError(f"unknown state codes: {unknown}")
    if start_year < 1990 or end_year < start_year:
        raise ValueError(
            f"invalid year range: {start_year}-{end_year} "
            "(QCEW coverage starts 1990)"
        )

    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    # {(state, metric_suffix): [(year, period, value), ...]}
    rows: dict[tuple[str, str], list[tuple[int, str, float]]] = {}
    for code in codes:
        for _, _, suffix in METRICS:
            rows[(code, suffix)] = []

    for code in codes:
        fips = ONTOLOGY.states[code].fips
        area = f"{fips}000"
        for year in range(start_year, end_year + 1):
            for qtr in quarters:
                url = QCEW_URL_TEMPLATE.format(year=year, qtr=qtr, area=area)
                try:
                    resp = requests.get(url, timeout=60)
                    resp.raise_for_status()
                    text = resp.text
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[QCEW] {code} {year}Q{qtr}: request failed — {exc}")
                    continue

                reader = csv.DictReader(io.StringIO(text))
                hit = False
                for rec in reader:
                    if rec.get("agglvl_code") != STATE_TOTAL_AGGLVL:
                        continue
                    # agglvl 50 + own_code 0 is the "all ownerships" roll-up.
                    if rec.get("own_code") not in ("", "0"):
                        continue
                    if rec.get("industry_code") not in ("", "10"):
                        continue
                    # Found the state total row — extract metrics.
                    period = f"Q{int(rec.get('qtr', qtr)):02d}"
                    for col, _readable, suffix in METRICS:
                        raw = rec.get(col, "").strip()
                        try:
                            value = float(raw)
                        except (TypeError, ValueError):
                            continue
                        rows[(code, suffix)].append((int(year), period, value))
                    hit = True
                    break
                if not hit:
                    logger.info(
                        f"[QCEW] {code} {year}Q{qtr}: no state-total row in CSV"
                    )

    total = 0
    for (code, suffix), entries in rows.items():
        if not entries:
            continue
        sid = f"QCEW_{code}_{suffix}"
        path = os.path.join(raw_dir, f"{sid}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("series_id,year,period,value\n")
            for y, p, v in sorted(entries, key=lambda r: (r[0], r[1])):
                f.write(f"{sid},{y},{p},{v}\n")
        total += len(entries)
        logger.info(f"[QCEW] {sid}: wrote {len(entries)} rows → {path}")
    return total
