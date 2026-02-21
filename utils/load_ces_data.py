import os
import json
import pandas as pd

from .constants import ALL_STATES, START_YEAR, END_YEAR

CES_CODES_JSON = "data/ces_state_sms_codes.json"
RAW_DIR = "data/raw/ces"

def load_ces_data(states: list[str], start: int, end: int) -> pd.DataFrame:
    """
    Load CES supersector data from raw text files in data/raw/ces.
    Returns DataFrame (long format) with columns: state, year, period, sector, value.
    """
    # Load mapping of (sector -> {state: series_id})
    if not os.path.exists(CES_CODES_JSON):
        raise FileNotFoundError(f"Missing {CES_CODES_JSON}")
    codes = json.load(open(CES_CODES_JSON, encoding="utf-8"))
    # Invert mapping: series_id -> (state, sector)
    series_map = {}
    for sector, stmap in codes.items():
        for st, sid in stmap.items():
            if st in states and sid.startswith("SMS"):
                series_map[sid] = (st, sector)
    df_list = []
    for fn in os.listdir(RAW_DIR):
        if not fn.endswith(".txt"):
            continue
        sid = fn.replace(".txt", "")
        mapping = series_map.get(sid)
        if mapping is None:
            continue
        state, sector = mapping
        path = os.path.join(RAW_DIR, fn)
        try:
            df = pd.read_csv(path)
        except Exception:
            print(f"[CES] Warning: Failed to read {fn}")
            continue
        if df.empty:
            continue
        df = df.rename(columns={"value": "value"})
        df = df[["year", "period", "value"]]
        df["state"] = state
        df["sector"] = sector
        df["year"] = df["year"].astype(int)
        # Keep only monthly data
        df = df[df["period"].str.startswith("M")]
        df = df[(df["year"] >= start) & (df["year"] <= end)]
        df_list.append(df)
    if not df_list:
        return pd.DataFrame(columns=["state", "year", "period", "sector", "value"])
    df_all = pd.concat(df_list, ignore_index=True)
    return df_all

if __name__ == "__main__":
    # CLI test
    df = load_ces_data(ALL_STATES, START_YEAR, END_YEAR)
    print(f"[CES] Loaded data: {len(df)} rows")
