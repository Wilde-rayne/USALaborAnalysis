"""
Central pipeline orchestration: fetch CES, fetch LAUS, merge into JSON.

Two entry points:
- ``refresh_all`` — always fetches from source APIs (explicit refresh).
- ``ensure_data`` — lazy wrapper: returns cached data if fresh, else
  calls refresh_all. Used by the reactive startup path so a container
  with a recent ``all_data.json`` in its mounted volume skips the
  15-20 min BLS/Census round-trip.
"""
import logging
import os
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

import pandas as pd

from .fetch_ces_data import fetch_ces_data
from .fetch_laus_data import fetch_laus_data
from .fetch_population_data import fetch_population
from .merge_all_data import merge_all_data, save_data

logger = logging.getLogger(__name__)

OUTPUT_JSON = "data/all_data.json"
OUTPUT_CSV  = "data/all_data.csv"

# How long a cached merge is considered fresh before we re-fetch.
# Override with CACHE_MAX_AGE_SECONDS env var at container start.
DEFAULT_CACHE_MAX_AGE_SECONDS = 7 * 24 * 3600  # 7 days


def cache_is_fresh(path: str | None = None, max_age_seconds: int | None = None) -> bool:
    """
    True if ``path`` exists and was modified within the freshness window.

    ``path`` defaults to the module-level ``OUTPUT_JSON`` resolved at call
    time (not at function-definition time) so tests can monkeypatch
    ``data_pipeline.OUTPUT_JSON`` without re-importing.
    """
    if path is None:
        path = OUTPUT_JSON
    if not os.path.exists(path):
        return False
    if max_age_seconds is None:
        max_age_seconds = int(os.getenv("CACHE_MAX_AGE_SECONDS", DEFAULT_CACHE_MAX_AGE_SECONDS))
    age = time.time() - os.path.getmtime(path)
    return age < max_age_seconds


def ensure_data(
    states: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    *,
    force: bool = False,
) -> str:
    """
    Lazy data loader. Returns the path to the merged data file, fetching
    from BLS/Census only when the cache is missing or stale. This is the
    entry point reactive callers (Dash callbacks, app startup) should use.
    """
    from .constants import ALL_STATES, END_YEAR, START_YEAR

    states = states or ALL_STATES
    start_year = start_year or START_YEAR
    end_year = end_year or END_YEAR

    if not force and cache_is_fresh(OUTPUT_JSON):
        age_h = (time.time() - os.path.getmtime(OUTPUT_JSON)) / 3600
        logger.info(f"[PIPE] Cache hit: {OUTPUT_JSON} (age {age_h:.1f}h)")
        return OUTPUT_JSON

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"[PIPE] Cache miss or forced refresh — fetching {start_year}-{end_year} "
        f"for {len(states)} states"
    )
    refresh_all(states, start_year, end_year)

    # Invalidate the LRU cache in llm_utils so any question whose answer
    # was computed against the old corpus doesn't stick around. Wrapped
    # in a lazy try/except so this module stays free of an import cycle
    # and works even if llm_utils hasn't been loaded yet.
    try:
        from utils import llm_utils as _llm  # noqa: PLC0415

        _llm._cached_invoke.cache_clear()
    except Exception:  # noqa: BLE001
        pass

    return OUTPUT_JSON

def validate_lfpr_data(
    df: pd.DataFrame,
    states: list[str],
    start_year: int,
    end_year: int,
) -> dict[str, dict[int, list[int]]]:
    """
    Report which ``{state}_Labor_Force_Participation_Rate`` wide-column
    observations are missing across the requested span.

    Returns a ``{state: {year: [missing_month_ints]}}`` dict and emits a
    ``WARNING`` log line listing the gaps. Callers can branch on a
    truthy return value to signal to the UI that a refresh is worth
    running. Uses the integer ``month`` column the merger now produces,
    not the old "period == 'January'" string shape.
    """
    issues: dict[str, dict[int, list[int]]] = {}
    year_range = range(start_year, end_year + 1)
    month_range = range(1, 13)
    panel = df.drop_duplicates(subset="date") if "date" in df.columns else df
    for state in states:
        col = f"{state}_Labor_Force_Participation_Rate"
        if col not in panel.columns:
            issues[state] = {yr: list(month_range) for yr in year_range}
            continue

        mask = panel[col].notna()
        present: set[tuple[int, int]] = {
            (int(y), int(m))
            for y, m in zip(panel.loc[mask, "year"], panel.loc[mask, "month"])
        }
        expected = {(yr, m) for yr in year_range for m in month_range}
        missing = sorted(expected - present)
        if missing:
            gaps: dict[int, list[int]] = {}
            for yr, m in missing:
                gaps.setdefault(yr, []).append(m)
            issues[state] = gaps

    if issues:
        lines = [
            f"{st}: missing {sum(len(ms) for ms in gaps.values())} months "
            f"across {len(gaps)} year(s)"
            for st, gaps in issues.items()
        ]
        logger.warning("[LFPR-validate] gaps — " + "; ".join(lines))
    return issues


def refresh_all(states: list[str], start_year: int, end_year: int):
    """
    Runs the full data pipeline:
      1. Fetch CES data for given states and years
      2. Fetch LAUS data for given states and years
      3. Fetch Population data for given states and years
      4. Merge all into a single dataset
      5. Save to CSV and JSON
      6. Validate completeness of LFPR data
    """
    try:
        logger.info(f"[PIPE] Fetching data for states: {states} ({start_year}-{end_year})")
        fetch_ces_data(states, start_year, end_year)
        fetch_laus_data(states, start_year, end_year)
        fetch_population(states, start_year, end_year)
        logger.info("[PIPE] Merging data")
        merged_df = merge_all_data(states, start_year, end_year)
        merged_df = merged_df.sort_values(["year","period"])
        save_data(merged_df, OUTPUT_CSV, OUTPUT_JSON)
        logger.info(f"[PIPE] Data saved to {OUTPUT_CSV} and {OUTPUT_JSON}")
        #validate_lfpr_data(merged_df, states, start_year, end_year)
    except Exception as e:
        logger.error(f"[PIPE] Error in refresh_all: {e}")

if __name__ == "__main__":
    import argparse

    from .constants import ALL_STATES, END_YEAR, START_YEAR

    parser = argparse.ArgumentParser(description="USA Labor Analysis data pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fetch even if a fresh cache exists.",
    )
    args = parser.parse_args()

    print("[PIPE] Starting data pipeline")
    t0 = datetime.now()
    ensure_data(ALL_STATES, START_YEAR, END_YEAR, force=args.force)
    t1 = datetime.now()
    print(f"[PIPE] Completed in {t1 - t0}")
    print(f"[PIPE] Results at data/all_data.csv and {OUTPUT_JSON}")
