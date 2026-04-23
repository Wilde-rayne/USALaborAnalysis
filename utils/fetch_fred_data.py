"""
Fetch state-level indicators from FRED (Federal Reserve Economic Data).

FRED's naming convention ``{2-letter state code}{indicator}`` covers a
huge swath of state-level macro series — the most obvious starter is
``UR`` (unemployment rate, monthly) which aligns with the Super tab's
"where to hire / avoid" recommendation flow. The fetcher is
generic so adding new indicator families is a one-line addition to
``FRED_INDICATORS``.

Free API key: https://fred.stlouisfed.org/docs/api/api_key.html
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import requests

from utils.ontology import ONTOLOGY

logger = logging.getLogger(__name__)

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

#: Indicator suffix → (description, period-code-BLS-style).
#: The period code is stored in the output so the merger can line up
#: FRED rows with BLS monthly series that use M01..M12.
#:
#: All four indicators follow FRED's ``{STATE}{INDICATOR}`` naming:
#: e.g. ``IAUR`` (Iowa unemployment rate), ``IAPI`` (Iowa personal
#: income), ``IANGSP`` (Iowa nominal gross state product),
#: ``IASTHPI`` (Iowa FHFA all-transactions house price index).
FRED_INDICATORS: dict[str, tuple[str, str]] = {
    "UR":    ("unemployment rate",                   "monthly"),
    "PI":    ("personal income",                     "quarterly"),
    "NGSP":  ("nominal gross state product",         "annual"),
    "STHPI": ("FHFA state house price index",        "quarterly"),
}

RAW_DIR_DEFAULT = os.path.join("data", "raw", "fred")


def _api_key() -> str | None:
    return os.getenv("FRED_API_KEY") or None


def _fred_series_id(state_code: str, indicator: str) -> str:
    """FRED state-level convention: '<ST><IND>', e.g. 'IAUR'."""
    if state_code not in ONTOLOGY.states:
        raise ValueError(f"unknown state code: {state_code!r}")
    if indicator not in FRED_INDICATORS:
        raise ValueError(
            f"unknown FRED indicator: {indicator!r}; supported: "
            f"{sorted(FRED_INDICATORS)}"
        )
    return f"{state_code}{indicator}"


def _bls_period_from_date(date_str: str, cadence: str) -> str:
    """
    FRED returns YYYY-MM-DD observation dates. Map to BLS-shaped
    period codes so the merger sees familiar strings.
    """
    yyyy, mm, _ = date_str.split("-", 2)
    if cadence == "monthly":
        return f"M{int(mm):02d}"
    if cadence == "quarterly":
        # FRED quarterlies are always stamped on the first day of the
        # quarter (01-01, 04-01, 07-01, 10-01).
        q = (int(mm) - 1) // 3 + 1
        return f"Q{q:02d}"
    return "A01"


def fetch_fred_state_series(
    states: Iterable[str],
    indicator: str,
    start_year: int,
    end_year: int,
    *,
    raw_dir: str = RAW_DIR_DEFAULT,
    api_key: str | None = None,
) -> int:
    """
    Download a FRED state-level indicator family into per-state TXTs.

    Writes ``data/raw/fred/FRED_<ST><IND>.txt`` with the usual
    ``series_id,year,period,value`` columns. Returns the number of
    observation rows written.
    """
    api_key = api_key or _api_key()
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Register a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and put "
            "it in .env."
        )
    if indicator not in FRED_INDICATORS:
        raise ValueError(
            f"unknown FRED indicator: {indicator!r}; supported: "
            f"{sorted(FRED_INDICATORS)}"
        )
    if start_year < 1948 or end_year < start_year:
        raise ValueError(
            f"invalid year range: {start_year}-{end_year} "
            "(FRED state series start ~1976; keep the floor at 1948)"
        )

    codes = [c.upper() for c in states]
    unknown = [c for c in codes if c not in ONTOLOGY.states]
    if unknown:
        raise ValueError(f"unknown state codes: {unknown}")

    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    _, cadence = FRED_INDICATORS[indicator]

    total_rows = 0
    for code in codes:
        sid = _fred_series_id(code, indicator)
        params = {
            "series_id": sid,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": f"{start_year}-01-01",
            "observation_end": f"{end_year}-12-31",
        }
        try:
            resp = requests.get(FRED_API_URL, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[FRED] {sid}: request failed — {exc}")
            continue

        observations = payload.get("observations", [])
        if not observations:
            logger.info(f"[FRED] {sid}: no observations returned")
            continue

        path = os.path.join(raw_dir, f"FRED_{sid}.txt")
        written = 0
        with open(path, "w", encoding="utf-8") as f:
            f.write("series_id,year,period,value\n")
            for obs in observations:
                date = obs.get("date") or ""
                raw_val = (obs.get("value") or "").strip()
                if not date or raw_val in {"", "."}:
                    continue
                try:
                    value = float(raw_val)
                except ValueError:
                    continue
                year = int(date.split("-", 1)[0])
                period = _bls_period_from_date(date, cadence)
                f.write(f"FRED_{sid},{year},{period},{value}\n")
                written += 1
        if written:
            logger.info(f"[FRED] {sid}: wrote {written} rows → {path}")
        total_rows += written

    return total_rows
