import os
import pandas as pd

from .constants import ALL_STATES, START_YEAR, END_YEAR

RAW_DIR = "data/raw/laus"

def load_population_data(states: list[str], start: int, end: int) -> pd.DataFrame:
    """
    Load state population data from Census (files POP_{ST}.txt in data/raw/laus).
    Returns DataFrame with columns: state, year, period, Population (M01 only).
    """
    df_list = []
    for fn in os.listdir(RAW_DIR):
        if not fn.startswith("POP_") or not fn.endswith(".txt"):
            continue
        state = fn.split("_")[1].split(".")[0]
        if state not in states:
            continue
        path = os.path.join(RAW_DIR, fn)
        try:
            df = pd.read_csv(path)
        except Exception:
            print(f"[POP] Warning: Failed to read {fn}")
            continue
        if df.empty:
            continue
        df = df.rename(columns={"value": "Population"})
        df["state"] = state
        df["year"] = df["year"].astype(int)
        df = df[["state", "year", "period", "Population"]]
        df_list.append(df)
    if not df_list:
        return pd.DataFrame(columns=["state", "year", "period", "Population"])
    df_all = pd.concat(df_list, ignore_index=True)
    # Filter year range
    df_all = df_all[(df_all["year"] >= start) & (df_all["year"] <= end)]
    return df_all

if __name__ == "__main__":
    # CLI test
    df = load_population_data(ALL_STATES, START_YEAR, END_YEAR)
    print(f"[POP] Loaded data: {len(df)} rows")
