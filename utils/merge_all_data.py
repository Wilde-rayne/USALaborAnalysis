import os
import pandas as pd
import logging
import json

# configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# directories and constants
RAW_DIR_LAUS = "data/raw/laus"
RAW_DIR_CES  = "data/raw/ces"
CES_JSON     = "data/ces_state_sms_codes.json"


def read_laus_series(states: list, start: int, end: int) -> pd.DataFrame:
    """
    Read LAUS series files and return DataFrame with 
    (state, year, month, Labor_Force, Employment, Unemployment, Population).
    """
    fips_map = {
        "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10",
        "FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20",
        "KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28",
        "MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36",
        "NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45",
        "SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54",
        "WI":"55","WY":"56"
    }
    measure_map = {"006": "Labor_Force", "005": "Employment", "004": "Unemployment", "009": "Population"}

    df_all = pd.DataFrame()
    for fn in os.listdir(RAW_DIR_LAUS):
        if fn.startswith("LASST") and fn.endswith(".txt"):
            df = pd.read_csv(os.path.join(RAW_DIR_LAUS, fn))
            if df.empty:
                continue
            sid = str(df.at[0, "series_id"])
            fips = sid[5:7]
            state = next((k for k,v in fips_map.items() if v==fips), None)
            if state not in states:
                continue
            suffix = sid[-3:]
            measure = measure_map.get(suffix)
            if not measure:
                continue
            df = df.rename(columns={"value": measure})
            df["state"] = state
            df = df[["state","year","period", measure]]
            df_all = pd.concat([df_all, df], ignore_index=True)

    if df_all.empty:
        return pd.DataFrame()

    df_all = df_all[df_all["period"].str.startswith("M")]
    df_all["year"] = df_all["year"].astype(int)
    df_all["month"] = df_all["period"].str[1:].astype(int)
    df_all = df_all[(df_all.year>=start) & (df_all.year<=end) & (df_all.state.isin(states))]

    # Each measure file contributes a row with its own column populated and
    # the other measure columns as NaN. Collapse those per (state, year,
    # period, month) so downstream merges don't fan out into duplicates.
    id_cols = ["state", "year", "period", "month"]
    measure_cols = [c for c in df_all.columns if c not in id_cols]
    if measure_cols:
        df_all = (
            df_all
            .groupby(id_cols, as_index=False, dropna=False)
            [measure_cols]
            .first()
        )
    return df_all


def read_population(states: list, start: int, end: int) -> pd.DataFrame:
    """
    Read POP_{ST}.txt files (Census fallback) and return DataFrame with 
    (state, year, month=1, Population).
    """
    df_list = []
    for fn in os.listdir(RAW_DIR_LAUS):
        if fn.startswith("POP_") and fn.endswith(".txt"):
            st = fn.split("_")[1].split(".")[0]
            if st not in states:
                continue
            df = pd.read_csv(os.path.join(RAW_DIR_LAUS, fn))
            if df.empty:
                continue
            df = df.rename(columns={"value": "Population"})
            df["state"] = st
            df["year"] = df["year"].astype(int)
            df["month"] = 1  # January
            df_list.append(df[["state","year","month","Population"]])
    pop_df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    if pop_df.empty:
        return pop_df
    return pop_df[(pop_df.year>=start) & (pop_df.year<=end)]


def merge_all_data(states: list, start: int, end: int) -> pd.DataFrame:
    """
    Merge LAUS, population, and CES data into a single wide DataFrame.
    Proceeds even if some sources are missing (partial data).
    """
    # fetch LAUS and population
    df_laus = read_laus_series(states, start, end)
    df_pop  = read_population(states, start, end)

    # standard grid
    years = list(range(start, end+1))
    months = list(range(1,13))
    grid = pd.DataFrame([(st, y, m) for st in states for y in years for m in months],
                        columns=["state","year","month"])

    panel = grid.copy()

    # merge LAUS if available
    if not df_laus.empty:
        panel = pd.merge(panel, df_laus, on=["state","year","month"], how="left")
    else:
        logger.warning("[MERGE] No LAUS data available - skipping LAUS merge.")

    # merge population
    if not df_pop.empty:
        panel = pd.merge(panel, df_pop, on=["state","year","month"], how="left", suffixes=("_laus","_pop"))
    else:
        logger.warning("[MERGE] No population data available - skipping population merge.")
        # Only backfill a Population column if LAUS didn't already supply one.
        if "Population" not in panel.columns:
            panel["Population"] = pd.NA

    # consolidate population columns
    if "Population_laus" in panel.columns and "Population_pop" in panel.columns:
        panel["Population"] = panel["Population_laus"].fillna(panel["Population_pop"])
        panel.drop(columns=["Population_laus","Population_pop"], inplace=True)
    elif "Population_laus" in panel.columns:
        panel.rename(columns={"Population_laus": "Population"}, inplace=True)

    # propagate population
    if "Population" in panel.columns:
        panel["Population"] = panel.groupby(["state","year"])["Population"].transform(lambda x: x.ffill().bfill())

    # compute LFPR if possible
    if "Labor_Force" in panel.columns and "Population" in panel.columns:
        panel["LFPR"] = 100 * pd.to_numeric(panel["Labor_Force"], errors="coerce") / pd.to_numeric(panel["Population"], errors="coerce")
    else:
        logger.warning("[MERGE] Cannot compute LFPR - missing Labor_Force or Population.")

    # incorporate CES sector data
    ces_rows = []
    if os.path.exists(CES_JSON):
        with open(CES_JSON, encoding="utf-8") as f:
            ces_codes = json.load(f)
        for sector, mapping in ces_codes.items():
            for st in states:
                sid = mapping.get(st)
                if not sid:
                    logger.warning(f"[MERGE] No CES series for {st} - {sector}")
                    continue
                path = os.path.join(RAW_DIR_CES, f"{sid}.txt")
                if not os.path.exists(path):
                    logger.warning(f"[MERGE] Missing CES file: {sid}.txt")
                    continue
                df = pd.read_csv(path)
                df = df[df.period.str.startswith("M")].copy()
                df["year"] = df["year"].astype(int)
                df["month"] = df["period"].str[1:].astype(int)
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df[(df.year>=start) & (df.year<=end)]
                for _, row in df.iterrows():
                    ces_rows.append({
                        "year": row["year"],
                        "month": row["month"],
                        "state": st,
                        f"{st}_{sector}": row["value"]
                    })
        if ces_rows:
            df_ces = pd.DataFrame(ces_rows)
            ces_wide = df_ces.pivot_table(
                index=["year","month"],
                values=[c for c in df_ces.columns if c not in ["year","month","state"]],
                aggfunc="first"
            )
            ces_wide.reset_index(inplace=True)
            panel = pd.merge(panel, ces_wide, on=["year","month"], how="left")
    else:
        logger.error(f"[MERGE] CES JSON not found: {CES_JSON}")

    return panel


def save_data(df: pd.DataFrame, csv_path: str, json_path: str) -> None:
    """Save merged panel to CSV and JSON."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=4)
