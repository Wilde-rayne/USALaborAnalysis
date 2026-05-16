import os
import pandas as pd
import logging
import json

from .ontology import ONTOLOGY

logger = logging.getLogger(__name__)

# directories and constants
RAW_DIR_LAUS = "data/raw/laus"
RAW_DIR_CES  = "data/raw/ces"
RAW_DIR_QCEW = "data/raw/qcew"
RAW_DIR_JOLTS = "data/raw/jolts"
RAW_DIR_CPI  = "data/raw/cpi"
RAW_DIR_FRED = "data/raw/fred"
RAW_DIR_BEA  = "data/raw/bea"
CES_JSON     = "data/ces_state_sms_codes.json"

# Map ONTOLOGY region label → BLS CPI region code. Used to broadcast
# regional CPI rows back to the per-state panel so each state inherits
# the index of its Census region.
_REGION_TO_CPI_AREA: dict[str, str] = {
    "Northeast": "0100",
    "Midwest":   "0200",
    "South":     "0300",
    "West":      "0400",
    # Territories + DC get the US city average as a best-effort proxy.
    "Caribbean": "0000",
    "Pacific":   "0000",
}

#: US-wide civilian noninstitutional population aged 16+ as a share of
#: total resident population, per BLS Handbook of Methods (ch. 1).
#: Used as the LFPR denominator correction factor — see the
#: ``compute LFPR`` block below for the methodology audit reference.
LFPR_WORKING_AGE_FRACTION: float = 0.78


def read_laus_series(states: list, start: int, end: int) -> pd.DataFrame:
    """
    Read LAUS series files and return DataFrame with
    (state, year, month, Labor_Force, Employment, Unemployment).

    Population is NOT a LAUS state-level measure (the BLS LAUS suffix
    009 is not published per-state); it comes from Census PEP/ACS via
    :mod:`utils.fetch_population_data` and is joined separately by
    :func:`merge_all_data` below.
    """
    # population comes from Census PEP/ACS via fetch_population_data.py
    measure_map = {"006": "Labor_Force", "005": "Employment", "004": "Unemployment"}

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


# ---------------------------------------------------------------------------
# Phase E/F readers — pull QCEW / JOLTS / CPI / FRED / BEA into the panel.
# Each helper returns a long-format DataFrame with at least
# (year, month) plus one or more metric columns; per-state files also
# carry a ``state`` column. The merger broadcasts national-only and
# annual-only series across the requested grid so downstream consumers
# can still join on (state, year, month) without missing rows.
# ---------------------------------------------------------------------------
def _broadcast_quarter_to_months(period: str) -> list[int]:
    """``Q01`` → [1,2,3]; ``Q02`` → [4,5,6]; etc. Returns ``[]`` on bad input."""
    if not period or not period.startswith("Q"):
        return []
    try:
        q = int(period[1:])
    except ValueError:
        return []
    if q < 1 or q > 4:
        return []
    base = (q - 1) * 3 + 1
    return [base, base + 1, base + 2]


def read_qcew(states: list, start: int, end: int) -> pd.DataFrame:
    """
    Read QCEW state-total files into a long DataFrame.

    File layout (per :mod:`utils.fetch_qcew_data`):
        ``data/raw/qcew/QCEW_{ST}_{SUFFIX}.txt``
        with ``SUFFIX`` in {EMP, TQW, AWW, EST} and period ``QNN``.

    Returns columns ``(state, year, month, metric, value)`` where the
    quarterly value is broadcast forward into each of the three months
    of the quarter. The merger then pivots to ``{ST}_QCEW_{metric}``
    wide columns; ``metric`` is the human-readable name (e.g.
    ``AverageWeeklyWage``) rather than the 3-letter file suffix so
    downstream column names self-describe.
    """
    if not os.path.isdir(RAW_DIR_QCEW):
        return pd.DataFrame()

    metric_names: dict[str, str] = {
        "EMP": "QCEW_Employment",
        "TQW": "QCEW_TotalQuarterlyWages",
        "AWW": "QCEW_AverageWeeklyWage",
        "EST": "QCEW_Establishments",
    }
    rows: list[dict] = []
    for fn in os.listdir(RAW_DIR_QCEW):
        if not fn.startswith("QCEW_") or not fn.endswith(".txt"):
            continue
        parts = fn[:-4].split("_")
        if len(parts) != 3:
            continue
        _, st, suffix = parts
        if st not in states or suffix not in metric_names:
            continue
        df = pd.read_csv(os.path.join(RAW_DIR_QCEW, fn))
        if df.empty:
            continue
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["year", "value"])
        df = df[(df["year"] >= start) & (df["year"] <= end)]
        for _, row in df.iterrows():
            months = _broadcast_quarter_to_months(str(row["period"]))
            if not months:
                continue
            for m in months:
                rows.append(
                    {
                        "state": st,
                        "year": int(row["year"]),
                        "month": m,
                        "metric": metric_names[suffix],
                        "value": float(row["value"]),
                    }
                )
    return pd.DataFrame(rows)


def read_jolts(states: list, start: int, end: int) -> pd.DataFrame:
    """
    Read JOLTS files into a long DataFrame keyed by (year, month).

    Caveat: ``utils.fetch_jolts_data`` defaults to NATIONAL JOLTS
    series. State-level series ids exist (experimental program) but
    aren't fetched by default. National rows are broadcast across all
    ``states`` so each per-state row of the panel inherits the
    national series — this is the standard pattern for national
    macro covariates joined onto a per-state grid.

    JOLTS series ids encode the metric in the last 3 characters:
    ``JOL`` (openings), ``HIL`` (hires), ``QUL`` (quits),
    ``LDL`` (layoffs/discharges), ``TSL`` (total separations).
    """
    if not os.path.isdir(RAW_DIR_JOLTS):
        return pd.DataFrame()

    suffix_to_metric: dict[str, str] = {
        "JOL": "JOLTS_JobOpenings",
        "HIL": "JOLTS_Hires",
        "QUL": "JOLTS_Quits",
        "LDL": "JOLTS_Layoffs",
        "TSL": "JOLTS_Separations",
    }
    long_rows: list[dict] = []
    for fn in os.listdir(RAW_DIR_JOLTS):
        if not fn.endswith(".txt"):
            continue
        sid = fn[:-4]
        if not sid.startswith("JTS"):
            continue
        suffix = sid[-3:]
        metric = suffix_to_metric.get(suffix)
        if metric is None:
            continue
        df = pd.read_csv(os.path.join(RAW_DIR_JOLTS, fn))
        if df.empty:
            continue
        df = df[df["period"].astype(str).str.startswith("M")].copy()
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["year", "value"])
        df["month"] = df["period"].str[1:].astype(int)
        df = df[(df["year"] >= start) & (df["year"] <= end)]
        for _, row in df.iterrows():
            long_rows.append(
                {
                    "year": int(row["year"]),
                    "month": int(row["month"]),
                    "metric": metric,
                    "value": float(row["value"]),
                }
            )

    if not long_rows:
        return pd.DataFrame()
    national = pd.DataFrame(long_rows).drop_duplicates(
        subset=["year", "month", "metric"]
    )
    # Broadcast to every state. The series is national; the join is
    # informational (every IA row sees the same national openings count
    # for that month).
    out_rows: list[dict] = []
    for st in states:
        sub = national.copy()
        sub["state"] = st
        out_rows.append(sub)
    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


def read_cpi(states: list, start: int, end: int) -> pd.DataFrame:
    """
    Read regional CPI files and broadcast each region's index back to
    the states that belong to it (per ``ONTOLOGY.states_by_region``).

    Returns long format ``(state, year, month, metric, value)`` with
    one metric: ``CPI_AllItems``. Useful as a deflator for any
    dollar-denominated wage series joined later (BEA, QCEW total wages).
    """
    if not os.path.isdir(RAW_DIR_CPI):
        return pd.DataFrame()

    # area_code -> long-format frame
    by_area: dict[str, pd.DataFrame] = {}
    for fn in os.listdir(RAW_DIR_CPI):
        if not fn.endswith(".txt") or not fn.startswith("CUUR"):
            continue
        sid = fn[:-4]
        # series id layout: CUUR{area:4}{item:3}
        area = sid[4:8]
        df = pd.read_csv(os.path.join(RAW_DIR_CPI, fn))
        if df.empty:
            continue
        df = df[df["period"].astype(str).str.startswith("M")].copy()
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["year", "value"])
        df["month"] = df["period"].str[1:].astype(int)
        df = df[(df["year"] >= start) & (df["year"] <= end)]
        by_area[area] = df[["year", "month", "value"]].reset_index(drop=True)

    if not by_area:
        return pd.DataFrame()

    rows: list[dict] = []
    us_fallback = by_area.get("0000")
    for st in states:
        state_obj = ONTOLOGY.states.get(st)
        if state_obj is None:
            continue
        area = _REGION_TO_CPI_AREA.get(state_obj.region, "0000")
        df = by_area.get(area)
        if df is None or df.empty:
            df = us_fallback
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            rows.append(
                {
                    "state": st,
                    "year": int(row["year"]),
                    "month": int(row["month"]),
                    "metric": "CPI_AllItems",
                    "value": float(row["value"]),
                }
            )
    return pd.DataFrame(rows)


def read_fred(states: list, start: int, end: int) -> pd.DataFrame:
    """
    Read FRED per-state series into long format.

    ``utils.fetch_fred_data`` writes ``FRED_<sid>.txt`` files with
    period codes ``M01..M12`` (monthly), ``Q01..Q04`` (quarterly), or
    ``A01`` (annual). Quarterly values are broadcast into the three
    months of the quarter; annual values are broadcast across all 12
    months of the year — a coarse but honest join (the caller can
    flag the column name, e.g. ``FRED_NGSP``, as annual-derived).

    We extract the state code from the series id using the FRED
    indicator registry (``utils.fetch_fred_data.FRED_INDICATORS``):
    most use ``{ST}{IND}`` (prefix) but MHI uses
    ``MEHOINUS{ST}A646N`` (infix), so simple slicing isn't safe.
    """
    if not os.path.isdir(RAW_DIR_FRED):
        return pd.DataFrame()

    # Lazy import: fetch_fred_data imports `requests`, which would be
    # an unnecessary hard dependency for the merger on systems where
    # FRED data hasn't been fetched.
    try:
        from .fetch_fred_data import FRED_INDICATORS  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        FRED_INDICATORS = {}

    # Pre-compute (indicator_key, state) → series_id so we can do an
    # O(1) reverse-lookup per file.
    expected: dict[str, tuple[str, str, str]] = {}
    for ind_key, (_desc, cadence, template) in FRED_INDICATORS.items():
        for st in states:
            sid = template.format(st=st)
            expected[sid] = (st, ind_key, cadence)

    rows: list[dict] = []
    for fn in os.listdir(RAW_DIR_FRED):
        if not fn.startswith("FRED_") or not fn.endswith(".txt"):
            continue
        sid = fn[5:-4]  # strip 'FRED_' prefix and '.txt' suffix
        match = expected.get(sid)
        if match is None:
            continue
        st, ind_key, cadence = match
        df = pd.read_csv(os.path.join(RAW_DIR_FRED, fn))
        if df.empty:
            continue
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["year", "value"])
        df = df[(df["year"] >= start) & (df["year"] <= end)]
        metric = f"FRED_{ind_key}"
        for _, row in df.iterrows():
            period = str(row["period"])
            year = int(row["year"])
            value = float(row["value"])
            if period.startswith("M"):
                try:
                    m = int(period[1:])
                except ValueError:
                    continue
                rows.append({"state": st, "year": year, "month": m,
                             "metric": metric, "value": value})
            elif period.startswith("Q"):
                for m in _broadcast_quarter_to_months(period):
                    rows.append({"state": st, "year": year, "month": m,
                                 "metric": metric, "value": value})
            elif period.startswith("A"):
                # Broadcast annual value across all 12 months.
                for m in range(1, 13):
                    rows.append({"state": st, "year": year, "month": m,
                                 "metric": metric, "value": value})
    return pd.DataFrame(rows)


def read_bea(states: list, start: int, end: int) -> pd.DataFrame:
    """
    Read BEA per-state annual personal income files.

    File pattern: ``BEA_SAINC1_L<linecode>_<ST>.txt`` with one annual
    row per year. Each value is broadcast across all 12 months of the
    matching year — annual joins are inherently coarse, but a constant
    per-state per-year is the right semantic for this dataset.
    """
    if not os.path.isdir(RAW_DIR_BEA):
        return pd.DataFrame()

    rows: list[dict] = []
    for fn in os.listdir(RAW_DIR_BEA):
        if not fn.startswith("BEA_") or not fn.endswith(".txt"):
            continue
        # File: BEA_<TABLE>_L<line>_<ST>.txt → split on '_'.
        base = fn[:-4]
        parts = base.split("_")
        if len(parts) < 4:
            continue
        st = parts[-1]
        if st not in states:
            continue
        # Build a stable metric name from the table + linecode so a
        # caller running both L3 (per-capita) and L1 (total) doesn't
        # collide in the wide pivot.
        table = parts[1]
        linecode = parts[2]
        metric = f"BEA_{table}_{linecode}"
        df = pd.read_csv(os.path.join(RAW_DIR_BEA, fn))
        if df.empty:
            continue
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["year", "value"])
        df = df[(df["year"] >= start) & (df["year"] <= end)]
        for _, row in df.iterrows():
            year = int(row["year"])
            value = float(row["value"])
            for m in range(1, 13):
                rows.append({"state": st, "year": year, "month": m,
                             "metric": metric, "value": value})
    return pd.DataFrame(rows)


def _join_long_phase_ef(panel: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a (state, year, month, metric, value) frame into
    ``{ST}_{METRIC}`` wide columns and merge into ``panel``.

    Used uniformly by the Phase E/F readers (QCEW, JOLTS, CPI, FRED,
    BEA) so column-naming and the join keys stay symmetric with the
    existing CES wide-pivot.
    """
    if long_df is None or long_df.empty:
        return panel
    # Build {ST}_{METRIC} column key.
    long_df = long_df.copy()
    long_df["col"] = long_df["state"] + "_" + long_df["metric"]
    # If two source files happen to publish overlapping rows (e.g.
    # both CPI us-avg and CPI midwest broadcast to OH), keep the first.
    long_df = long_df.drop_duplicates(subset=["year", "month", "col"], keep="first")
    wide = long_df.pivot_table(
        index=["year", "month"],
        columns="col",
        values="value",
        aggfunc="first",
    )
    wide = wide.reset_index()
    return pd.merge(panel, wide, on=["year", "month"], how="left")


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

    # ---------------------------------------------------------------- *
    # Phase E/F: QCEW, JOLTS, CPI, FRED, BEA
    # ---------------------------------------------------------------- *
    # Each reader returns (state, year, month, metric, value) long form.
    # ``_join_long_phase_ef`` pivots into ``{ST}_{METRIC}`` columns and
    # left-joins by (year, month). Sources that publish at coarser
    # cadences than monthly (QCEW = quarterly, FRED-annual, BEA = annual)
    # are pre-broadcast by their reader; sources without a per-state
    # signal (CPI regional, JOLTS national) are pre-broadcast to every
    # state so the column shape is always ``{ST}_{METRIC}``.
    for source_name, reader in (
        ("QCEW",  read_qcew),
        ("JOLTS", read_jolts),
        ("CPI",   read_cpi),
        ("FRED",  read_fred),
        ("BEA",   read_bea),
    ):
        try:
            long_df = reader(states, start, end)
        except Exception as exc:  # noqa: BLE001 — Phase E/F is best-effort
            logger.warning(f"[MERGE] {source_name} read failed: {exc}")
            continue
        if long_df is None or long_df.empty:
            logger.info(f"[MERGE] {source_name}: no rows on disk — skipping")
            continue
        before = panel.shape[1]
        panel = _join_long_phase_ef(panel, long_df)
        added = panel.shape[1] - before
        logger.info(f"[MERGE] {source_name}: added {added} columns")

    return panel


def save_data(df: pd.DataFrame, csv_path, json_path) -> None:
    """
    Save merged panel to CSV and JSON.

    Accepts ``str`` or :class:`pathlib.Path` for both paths; we coerce
    to ``str`` before handing off to pandas / ``os.path`` to keep the
    behavior identical on Python 3.6+.
    """
    csv_path = str(csv_path)
    json_path = str(json_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=4)
