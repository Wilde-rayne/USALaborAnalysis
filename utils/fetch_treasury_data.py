"""
Fetch national Treasury constant-maturity yields from FRED.

The Board of Governors' H.15 "Selected Interest Rates" release is the
canonical free source for Treasury constant-maturity (CMT) yields;
FRED republishes it with stable series ids. The monthly-average
variants (``GS10`` / ``GS2`` / ``GS3M``) fit the dashboard's monthly
panel natively, unlike their daily ``DGS*`` siblings.

The classic recession leading indicator — the 10-year-minus-3-month
term spread (Estrella & Mishkin, 1996) — is deliberately NOT fetched:
FRED's ready-made spread series (``T10Y3M``) is daily-only, so the
merger derives the monthly spread as ``GS10 − GS3M`` at read time
(see :func:`utils.merge_all_data.read_treasury`).

Raw output: one ``data/raw/treasury/{series}.json`` per series, each a
JSON list of ``{"series_id", "year", "period", "value"}`` records —
the same row shape the FRED fetcher writes to its TXT files, so the
two readers stay symmetric.

Free API key: https://fred.stlouisfed.org/docs/api/api_key.html
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable

import requests

from utils.fetch_fred_data import FRED_API_URL, _bls_period_from_date

logger = logging.getLogger(__name__)

#: Series id → (description, cadence). All entries are national FRED
#: monthly series (averages of business days, percent, not seasonally
#: adjusted) sourced from the Federal Reserve H.15 release; adding a
#: maturity is a one-line addition here. Coverage starts: GS10
#: 1953-04, GS2 1976-06, GS3M 1981-09 — so the computed 10y−3m spread
#: only exists from late 1981 onward.
#:
#: Cadence strings drive the BLS-style period mapping shared with the
#: FRED fetcher (``_bls_period_from_date``).
TREASURY_SERIES: dict[str, tuple[str, str]] = {
    "GS10": ("10-year Treasury constant-maturity yield", "monthly"),
    "GS2":  ("2-year Treasury constant-maturity yield",  "monthly"),
    "GS3M": ("3-month Treasury constant-maturity yield", "monthly"),
}

RAW_DIR_DEFAULT = os.path.join("data", "raw", "treasury")

#: GS10 is the longest-running series in the registry (1953-04);
#: reject obviously wrong ranges below it, mirroring the FRED
#: fetcher's 1948 floor.
_MIN_YEAR = 1953


def _api_key() -> str | None:
    return os.getenv("FRED_API_KEY") or None


def fetch_treasury_series(
    start_year: int,
    end_year: int,
    series_ids: Iterable[str] | None = None,
    *,
    raw_dir: str = RAW_DIR_DEFAULT,
    api_key: str | None = None,
) -> int:
    """Download national Treasury CMT yield series into per-series JSON files.

    Writes ``data/raw/treasury/{series}.json`` containing a JSON list
    of ``{"series_id", "year", "period", "value"}`` records (period is
    BLS-style ``M01``–``M12``).

    Parameters
    ----------
    start_year, end_year : int
        Inclusive year range (GS10 starts 1953-04).
    series_ids : Iterable[str], optional
        Keys of :data:`TREASURY_SERIES` to fetch. ``None`` fetches the
        full registry.
    raw_dir : str, optional
        Output directory (default :data:`RAW_DIR_DEFAULT`).
    api_key : str, optional
        FRED API key; falls back to the ``FRED_API_KEY`` env var.

    Returns
    -------
    int
        Total number of observation rows written across all series.

    Raises
    ------
    RuntimeError
        When no FRED API key is available.
    ValueError
        On an unknown series id or an invalid year range.
    """
    api_key = api_key or _api_key()
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Register a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and put "
            "it in .env."
        )
    if series_ids is None:
        series_ids = list(TREASURY_SERIES)
    series_ids = list(series_ids)
    unknown = [s for s in series_ids if s not in TREASURY_SERIES]
    if unknown:
        raise ValueError(
            f"unknown Treasury series: {unknown}; supported: "
            f"{sorted(TREASURY_SERIES)}"
        )
    if start_year < _MIN_YEAR or end_year < start_year:
        raise ValueError(
            f"invalid year range: {start_year}-{end_year} "
            f"(Treasury CMT series start {_MIN_YEAR}; GS2 1976, GS3M 1981)"
        )

    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for sid in series_ids:
        _desc, cadence = TREASURY_SERIES[sid]
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
            logger.warning(f"[TREASURY] {sid}: request failed — {exc}")
            continue

        observations = payload.get("observations", [])
        if not observations:
            logger.info(f"[TREASURY] {sid}: no observations returned")
            continue

        records: list[dict] = []
        for obs in observations:
            date = obs.get("date") or ""
            raw_val = (obs.get("value") or "").strip()
            # FRED marks missing observations with a literal ".".
            if not date or raw_val in {"", "."}:
                continue
            try:
                value = float(raw_val)
            except ValueError:
                continue
            year = int(date.split("-", 1)[0])
            period = _bls_period_from_date(date, cadence)
            records.append(
                {"series_id": sid, "year": year, "period": period, "value": value}
            )

        if not records:
            logger.info(f"[TREASURY] {sid}: no valid rows after filtering")
            continue
        path = os.path.join(raw_dir, f"{sid}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        logger.info(f"[TREASURY] {sid}: wrote {len(records)} rows → {path}")
        total_rows += len(records)

    return total_rows
