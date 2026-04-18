"""
Golden-file tests for ``utils.merge_all_data``.

We build a minimal BLS-shaped fixture on tmpdir, point the module's
directory constants at it, and verify the end-to-end panel output —
column set, row grid, and LFPR arithmetic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from utils import merge_all_data as merge_mod


@pytest.fixture
def fixture_data_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Stand up a miniature data/raw/ layout on tmp_path:

      raw/laus/LASST19…006.txt   Iowa Labor Force, 2020 M01–M02
      raw/laus/LASST19…009.txt   Iowa Population,   2020 M01–M02
      raw/ces/SMS19…001.txt      Iowa Manufacturing, 2020 M01–M02

    Plus a `ces_state_sms_codes.json` mapping, so ``merge_all_data`` can
    locate the CES series for Iowa.
    """
    laus_dir = tmp_path / "raw" / "laus"
    ces_dir = tmp_path / "raw" / "ces"
    laus_dir.mkdir(parents=True)
    ces_dir.mkdir(parents=True)

    # IA Labor Force (measure 006). FIPS '19' lives at chars [5:7] of the ID.
    (laus_dir / "LASST190000000000006.txt").write_text(
        "series_id,year,period,value\n"
        "LASST190000000000006,2020,M01,1500000\n"
        "LASST190000000000006,2020,M02,1510000\n",
        encoding="utf-8",
    )
    # IA Population (measure 009).
    (laus_dir / "LASST190000000000009.txt").write_text(
        "series_id,year,period,value\n"
        "LASST190000000000009,2020,M01,3150000\n"
        "LASST190000000000009,2020,M02,3150000\n",
        encoding="utf-8",
    )
    # IA Manufacturing employment (CES).
    (ces_dir / "SMS19000001000000001.txt").write_text(
        "series_id,year,period,value\n"
        "SMS19000001000000001,2020,M01,220000\n"
        "SMS19000001000000001,2020,M02,221000\n",
        encoding="utf-8",
    )

    ces_json_path = tmp_path / "ces_state_sms_codes.json"
    ces_json_path.write_text(
        json.dumps({"Manufacturing": {"IA": "SMS19000001000000001"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(merge_mod, "RAW_DIR_LAUS", str(laus_dir))
    monkeypatch.setattr(merge_mod, "RAW_DIR_CES", str(ces_dir))
    monkeypatch.setattr(merge_mod, "CES_JSON", str(ces_json_path))
    return tmp_path


class TestReadLausSeries:
    def test_parses_labor_force_and_population(self, fixture_data_dirs: Path) -> None:
        df = merge_mod.read_laus_series(["IA"], 2020, 2020)
        assert not df.empty
        # Both measures should come through under their canonical column names.
        assert "Labor_Force" in df.columns
        assert "Population" in df.columns
        assert (df["state"] == "IA").all()
        # Parsed month should be integer 1 for M01.
        jan = df[df["period"] == "M01"]
        assert (jan["month"] == 1).all()

    def test_filters_out_of_range_years(self, fixture_data_dirs: Path) -> None:
        df = merge_mod.read_laus_series(["IA"], 2030, 2031)
        assert df.empty

    def test_filters_out_states_not_requested(self, fixture_data_dirs: Path) -> None:
        df = merge_mod.read_laus_series(["CA"], 2020, 2020)
        assert df.empty


class TestMergeAllData:
    def test_produces_12_month_grid_per_state(self, fixture_data_dirs: Path) -> None:
        panel = merge_mod.merge_all_data(["IA"], 2020, 2020)
        # 1 state × 1 year × 12 months = 12 rows.
        assert len(panel) == 12
        assert set(panel["state"].unique()) == {"IA"}
        assert set(panel["month"].unique()) == set(range(1, 13))

    def test_lfpr_arithmetic_matches_definition(self, fixture_data_dirs: Path) -> None:
        """LFPR = 100 * labor_force / population. Verify against the fixture."""
        panel = merge_mod.merge_all_data(["IA"], 2020, 2020)
        jan = panel[(panel["year"] == 2020) & (panel["month"] == 1)].iloc[0]
        # 1,500,000 / 3,150,000 * 100 ≈ 47.619%.
        expected = 100.0 * 1_500_000 / 3_150_000
        assert pd.notna(jan["LFPR"])
        assert abs(jan["LFPR"] - expected) < 1e-6

    def test_ces_manufacturing_column_is_created(self, fixture_data_dirs: Path) -> None:
        panel = merge_mod.merge_all_data(["IA"], 2020, 2020)
        assert "IA_Manufacturing" in panel.columns
        jan = panel[(panel["year"] == 2020) & (panel["month"] == 1)].iloc[0]
        assert jan["IA_Manufacturing"] == 220000

    def test_months_without_source_data_are_nan(self, fixture_data_dirs: Path) -> None:
        """The fixture only has Jan+Feb; Mar–Dec LFPR should be NaN."""
        panel = merge_mod.merge_all_data(["IA"], 2020, 2020)
        mar = panel[(panel["year"] == 2020) & (panel["month"] == 3)].iloc[0]
        assert pd.isna(mar["Labor_Force"])
        # Population is forward/back-filled within state-year, so it stays populated.
        # LFPR = 100 * NaN / 3,150,000 → NaN.
        assert pd.isna(mar["LFPR"])
