import os
import pandas as pd
import logging
import json

from .ontology import ONTOLOGY

logger = logging.getLogger(__name__)

# directories and constants
RAW_DIR_LAUS = "data/raw/laus"
RAW_DIR_CES  = "data/raw/ces"
CES_JSON     = "data/ces_state_sms_codes.json"

#: US-wide civilian noninstitutional population aged 16+ as a share of
#: total resident population, per BLS Handbook of Methods (ch. 1).
#: Used as the LFPR denominator correction factor — see the
#: ``compute LFPR`` block below for the methodology audit reference.
LFPR_WORKING_AGE_FRACTION: float = 0.78


def read_laus_series(states: list, start: int, end: int) -> pd.DataFrame:
    """
    Read LAUS series files and return DataFrame with 
    (state, year, month, Labor_Force, Employment, Unemployment, Population).
    """
    measure_map = {"006": "Labor_Force", "005": "Employment", "004": "Unemployment", "009": "Population"}

    df_all = pd.DataFrame()
    for fn in os.listdir(RAW_DIR_LAUS):
        if fn.startswith("LASST") and fn.endswith(".txt"):
            df = pd.read_csv(os.path.join(RAW_DIR_LAUS, fn))
            if df.empty:
                continue
            sid = str(df.at[0, "series_id"])
            fips = sid[5:7]
            state_obj = ONTOLOGY.states_by_fips.get(fips)
            state = state_obj.code if state_obj is not None else None
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


def read_working_age_population(states: list, start: int, end: int) -> pd.DataFrame:
    """
    Read ``WAP_{ST}.txt`` files written by
    ``utils.fetch_working_age_population`` and return a DataFrame of
    ACS B23025_001E ("Population 16 years and over") observations.

    Returns an empty DataFrame if no WAP files exist — the merger
    treats that as "fall back to the uniform 0.78 fraction", so this
    layer is fully opt-in. Years between two known ACS releases are
    linearly interpolated per state; years outside the ACS coverage
    window are forward / backward-filled from the nearest available
    year so every (state, year) the merger asks about gets a value.
    """
    import numpy as np  # noqa: PLC0415 — local import keeps top-level lean

    rows: list[dict] = []
    if not os.path.isdir(RAW_DIR_LAUS):
        return pd.DataFrame()

    for fn in os.listdir(RAW_DIR_LAUS):
        if not (fn.startswith("WAP_") and fn.endswith(".txt")):
            continue
        st = fn.split("_", 1)[1].split(".", 1)[0]
        if st not in states:
            continue
        df = pd.read_csv(os.path.join(RAW_DIR_LAUS, fn))
        if df.empty:
            continue
        df["state"] = st
        df["year"] = df["year"].astype(int)
        df["working_age_population"] = pd.to_numeric(df["value"], errors="coerce")
        rows.extend(df[["state", "year", "working_age_population"]].to_dict("records"))

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    # Densify per-state across the full requested year range so
    # downstream callers can do a clean (state, year) merge.
    out_frames: list[pd.DataFrame] = []
    full_years = list(range(start, end + 1))
    for st, sub in raw.groupby("state"):
        s = sub.set_index("year")["working_age_population"].sort_index()
        s = s.reindex(full_years)
        # Linear interpolation between known years; forward + backward
        # fill at the edges so 1996-2004 and post-2024 still get a
        # value (uses the closest known ACS release as the proxy).
        s = s.interpolate(method="linear", limit_direction="both")
        out_frames.append(
            pd.DataFrame(
                {
                    "state": st,
                    "year": full_years,
                    "working_age_population": s.values,
                }
            )
        )
    return pd.concat(out_frames, ignore_index=True)


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

    # ---------------------------------------------------------------- *
    # Labor Force Participation Rate
    # ---------------------------------------------------------------- *
    # BLS defines LFPR as 100 * civilian_labor_force /
    # civilian_noninstitutional_population_16_and_over (CNI16+).
    # Census PEP / ACS 1-year ship total resident population, which
    # includes children under 16, active-duty military, and the
    # institutionalized (prison, long-term care). Across US states
    # CNI16+ is consistently ~78 % of the total resident
    # population — see BLS Handbook of Methods, ch. 1, table 1.
    #
    # Strategy:
    #   1. If ACS B23025_001E ("Population 16+") is present (via the
    #      ``utils.fetch_working_age_population`` fetcher writing
    #      ``WAP_{ST}.txt`` files), use the per-state per-year
    #      working-age population directly. Residual error vs. BLS
    #      published state LFPR drops to ~0.5 pp.
    #   2. Otherwise fall back to multiplying the total population by
    #      the uniform US-wide ``LFPR_WORKING_AGE_FRACTION`` (0.78).
    #      Residual error: ~2 pp.
    #
    # ``LFPR_RAW`` is preserved as ``Labor_Force`` / total population
    # so anyone needing the exact raw ratio can still get it.
    if "Labor_Force" in panel.columns and "Population" in panel.columns:
        lf  = pd.to_numeric(panel["Labor_Force"], errors="coerce")
        pop = pd.to_numeric(panel["Population"], errors="coerce")
        panel["LFPR_RAW"] = 100.0 * lf / pop

        wap_df = read_working_age_population(states, start, end)
        if not wap_df.empty:
            # Per-state, per-year denominator. Merge into the panel and
            # use it for any (state, year) where ACS gave us a number.
            panel = pd.merge(
                panel,
                wap_df[["state", "year", "working_age_population"]],
                on=["state", "year"],
                how="left",
            )
            wap = pd.to_numeric(panel["working_age_population"], errors="coerce")
            denom = wap.where(wap > 0, pop * LFPR_WORKING_AGE_FRACTION)
            panel["LFPR"] = 100.0 * lf / denom
            n_acs = int((wap > 0).sum())
            n_total = int(panel.shape[0])
            logger.info(
                f"[MERGE] LFPR: {n_acs}/{n_total} rows used per-state ACS B23025; "
                f"others fell back to {LFPR_WORKING_AGE_FRACTION:.2f} multiplier."
            )
        else:
            panel["LFPR"] = 100.0 * lf / (pop * LFPR_WORKING_AGE_FRACTION)
            logger.info(
                "[MERGE] LFPR: no ACS WAP files present — using uniform "
                f"{LFPR_WORKING_AGE_FRACTION:.2f} working-age fraction."
            )
    else:
        logger.warning("[MERGE] Cannot compute LFPR - missing Labor_Force or Population.")

    # Widen per-state LAUS measures into {state}_{measure} columns so the
    # forecast/EDA tabs can pick them up with the same `startswith("{st}_")`
    # pattern they use for CES sector columns. This keeps the long-format
    # originals too so any downstream consumer can pick its preferred shape.
    wide_measures = [
        m for m in (
            "Labor_Force", "Employment", "Unemployment",
            "Population", "LFPR", "LFPR_RAW",
        )
        if m in panel.columns
    ]
    for measure in wide_measures:
        wide = panel.pivot_table(
            index=["year", "month"],
            columns="state",
            values=measure,
            aggfunc="first",
        )
        if measure == "LFPR":
            wide.columns = [f"{st}_Labor_Force_Participation_Rate" for st in wide.columns]
        else:
            wide.columns = [f"{st}_{measure}" for st in wide.columns]
        wide = wide.reset_index()
        panel = pd.merge(panel, wide, on=["year", "month"], how="left")

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
