"""
Central pipeline orchestration: fetch CES, fetch LAUS, merge into JSON.
"""
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
from .fetch_ces_data import fetch_ces_data
from .fetch_laus_data import fetch_laus_data
from .fetch_population_data import fetch_population
from .merge_all_data import merge_all_data, save_data
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_JSON = "data/all_data.json"
OUTPUT_CSV  = "data/all_data.csv"

def validate_lfpr_data(df: pd.DataFrame, states: list[str], start_year: int, end_year: int):
    """
    Validates that each state has complete monthly LFPR data.
    Logs a warning if any (state, year, month) entry is missing.
    """
    months = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]
    issues: dict[str, dict[int, list[str]]] = {}
    for state in states:
        col = f"{state}_Labor_Force_Participation_Rate"
        if col not in df.columns:
            issues[state] = {year: months[:] for year in range(start_year, end_year + 1)}
            continue

        present: set[tuple[int, str]] = set()
        for _, row in df.iterrows():
            year = int(row["year"])
            period = row["period"]
            value = row.get(col)
            if period == "Annual":
                continue
            if pd.notna(value):
                present.add((year, period))

        expected = {(yr, month) for yr in range(start_year, end_year + 1) for month in months}
        missing = sorted(expected - present)
        if missing:
            missing_by_year: dict[int, list[str]] = {}
            for yr, month in missing:
                missing_by_year.setdefault(yr, []).append(month)
            issues[state] = missing_by_year

    if issues:
        msg_lines = []
        for state, missing_info in issues.items():
            for yr, months_missing in missing_info.items():
                months_str = ", ".join(months_missing)
                msg_lines.append(f"{state}: missing {months_str} in {yr}")
        msg = (
            "Incomplete Labor_Force_Participation_Rate data for the following states:\n"
            + "\n".join(msg_lines)
        )
        logger.warning(msg)

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
        from .constants import ALL_STATES
        states = ALL_STATES
    except ImportError:
        pass

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
    from .constants import ALL_STATES, START_YEAR, END_YEAR
    print("[PIPE] Starting data pipeline")
    t0 = datetime.now()
    fetch_ces_data(ALL_STATES, START_YEAR, END_YEAR)
    fetch_laus_data(ALL_STATES, START_YEAR, END_YEAR)
    fetch_population(ALL_STATES, START_YEAR, END_YEAR)
    panel = merge_all_data(ALL_STATES, START_YEAR, END_YEAR)
    panel = panel.sort_values(["state", "year", "month"])
    save_data(panel, "data/all_data.csv", OUTPUT_JSON)
    t1 = datetime.now()
    print(f"[PIPE] Completed in {t1 - t0}")
    print(f"[PIPE] Results saved to data/all_data.csv and {OUTPUT_JSON}")
