import os
import pandas as pd

from .constants import ALL_STATES, START_YEAR, END_YEAR

RAW_DIR = "data/raw/laus"

def load_laus_data(states: list[str], start: int, end: int) -> pd.DataFrame:
    """
    Load LAUS data (Labor Force, Employment, Unemployment, Unemployment Rate)
    from raw text files in data/raw/laus.
    Returns DataFrame with columns: state, year, period, Labor_Force, Employment,
    Unemployment, Unemployment_Rate.
    """
    # Map FIPS codes to state abbreviations
    fips_map = {
        "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
        "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS",
        "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
        "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY",
        "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC",
        "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
        "55": "WI", "56": "WY"
    }
    # Map suffix to measure name
    measure_map = {
        "006": "Labor_Force",
        "005": "Employment",
        "004": "Unemployment",
        "003": "Unemployment_Rate"
    }
    df_all = None
    # Iterate through LAUS series files
    for fn in os.listdir(RAW_DIR):
        if not fn.startswith("LASST") or not fn.endswith(".txt"):
            continue
        path = os.path.join(RAW_DIR, fn)
        try:
            df = pd.read_csv(path)
        except Exception:
            print(f"[LAUS] Warning: Failed to read {fn}")
            continue
        if df.empty:
            continue
        sid = df.at[0, "series_id"]
        # Determine state from FIPS code in series_id (positions 5-6)
        fips = sid[5:7]
        state = fips_map.get(fips)
        if state is None or state not in states:
            continue
        suffix = sid[-3:]
        measure = measure_map.get(suffix)
        # Skip population series or unknown suffix
        if measure is None:
            continue
        # Rename 'value' column to the measure name
        df = df.rename(columns={"value": measure})
        # Keep only needed columns
        df = df[["year", "period", measure]]
        # Filter to monthly periods (drop "A01" etc.)
        df = df[df["period"].str.startswith("M")]
        df["year"] = df["year"].astype(int)
        df["state"] = state
        # Filter year range
        df = df[(df["year"] >= start) & (df["year"] <= end)]
        # Merge into combined DataFrame
        if df_all is None:
            df_all = df.copy()
        else:
            df_all = pd.merge(df_all, df, on=["state", "year", "period"], how="outer")
    if df_all is None:
        # Return empty DataFrame if no data
        columns = ["state", "year", "period"] + list(measure_map.values())
        return pd.DataFrame(columns=columns)
    # Ensure correct data types
    df_all["year"] = df_all["year"].astype(int)
    df_all.sort_values(["state", "year", "period"], inplace=True)
    return df_all

if __name__ == "__main__":
    # CLI test
    df = load_laus_data(ALL_STATES, START_YEAR, END_YEAR)
    print(f"[LAUS] Loaded data: {len(df)} rows")
