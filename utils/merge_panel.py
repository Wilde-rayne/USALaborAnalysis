import pandas as pd

from .constants import ALL_STATES, START_YEAR, END_YEAR, SUPERSECTORS, OUTPUT_JSON
from .load_laus_data import load_laus_data
from .load_population_data import load_population_data
from .load_ces_data import load_ces_data

def merge_panel(states: list[str], start: int, end: int) -> pd.DataFrame:
    """
    Merge LAUS, population, and CES data into a single panel DataFrame.
    Computes Labor Force Participation Rate and includes all supersectors.
    """
    # Create full grid of state, year, period (M01-M12)
    periods = [f"M{m:02d}" for m in range(1, 13)]
    years = list(range(start, end + 1))
    grid = pd.DataFrame([(st, yr, pr) for st in states for yr in years for pr in periods],
                        columns=["state", "year", "period"])

    # Load datasets
    df_laus = load_laus_data(states, start, end)
    df_pop = load_population_data(states, start, end)
    df_ces = load_ces_data(states, start, end)  # long format

    # Merge LAUS and population onto the grid
    panel = pd.merge(grid, df_laus, on=["state", "year", "period"], how="left")
    panel = pd.merge(panel, df_pop, on=["state", "year", "period"], how="left")

    # Fill population for all months within each year
    panel["Population"] = panel.groupby(["state", "year"])["Population"]\
                             .transform(lambda x: x.ffill().bfill())

    # Compute Labor Force Participation Rate
    panel["Labor_Force_Participation_Rate"] = 100 * panel["Labor_Force"] / panel["Population"]

    # Merge CES supersector data: add columns like "ST_Sector"
    panel = panel.set_index(["state", "year", "period"])
    for state in states:
        df_state = df_ces[df_ces.state == state]
        if df_state.empty:
            print(f"[MERGE] No CES data for state {state} – skipping supersectors")
            continue
        pivot_state = df_state.pivot(index=["year", "period"], columns="sector", values="value")
        for sector in pivot_state.columns:
            if sector not in SUPERSECTORS:
                continue
            col = f"{state}_{sector}"
            series = pivot_state[sector]
            # Align series to multi-index
            tuples = [(state, year, period) for (year, period) in series.index]
            mi = pd.MultiIndex.from_tuples(tuples, names=["state", "year", "period"])
            series.index = mi
            panel[col] = series
    panel = panel.reset_index()

    # Sort for readability
    panel = panel.sort_values(["state", "year", "period"])
    return panel

def save_data(df: pd.DataFrame, csv_path: str, json_path: str) -> None:
    """Save merged panel to CSV and JSON."""
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=4)

if __name__ == "__main__":
    print(f"[MERGE] Merging data for states {ALL_STATES}")
    panel = merge_panel(ALL_STATES, START_YEAR, END_YEAR)
    save_data(panel, "data/all_data.csv", OUTPUT_JSON)
    print(f"[MERGE] Data saved to data/all_data.csv and {OUTPUT_JSON}")
