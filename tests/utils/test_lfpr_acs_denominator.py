"""
Integration tests for the ACS B23025 LFPR-denominator path in
``utils.merge_all_data``.

When ``WAP_{state}.txt`` files exist, the merger uses them; otherwise
it falls back to the uniform 0.78 multiplier. Both code paths get an
end-to-end test against a tiny on-disk fixture.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _write_laus_fixture(tmp_path: Path) -> None:
    """Build the minimum LAUS + CES + codes-JSON layout the merger expects."""
    laus = tmp_path / "raw" / "laus"
    ces = tmp_path / "raw" / "ces"
    laus.mkdir(parents=True)
    ces.mkdir(parents=True)
    (laus / "LASST190000000000006.txt").write_text(
        "series_id,year,period,value\n"
        "LASST190000000000006,2020,M01,1700000\n"
        "LASST190000000000006,2020,M02,1710000\n"
    )
    # ACS PEP-style population rows (total resident population).
    (laus / "POP_IA.txt").write_text(
        "series_id,year,period,value\n"
        "POP_IA,2020,M01,3155000\n"
    )
    # CES totally empty placeholder so merge doesn't fail.
    import json
    (tmp_path / "ces_state_sms_codes.json").write_text(json.dumps({}))


class TestACSPathPresent:
    def test_uses_per_state_wap_when_file_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from utils import merge_all_data as merge_mod

        _write_laus_fixture(tmp_path)
        # Add the WAP file — Iowa CNI16+ for 2020 ≈ 2,404,000.
        wap = tmp_path / "raw" / "laus" / "WAP_IA.txt"
        wap.write_text(
            "series_id,year,period,value\n"
            "WAP_IA,2020,A01,2404000\n"
        )
        monkeypatch.setattr(merge_mod, "RAW_DIR_LAUS", str(tmp_path / "raw" / "laus"))
        monkeypatch.setattr(merge_mod, "RAW_DIR_CES",  str(tmp_path / "raw" / "ces"))
        monkeypatch.setattr(merge_mod, "CES_JSON",     str(tmp_path / "ces_state_sms_codes.json"))

        panel = merge_mod.merge_all_data(["IA"], 2020, 2020)
        jan = panel[(panel["year"] == 2020) & (panel["month"] == 1)].iloc[0]

        # LFPR_RAW = 100 * 1,700,000 / 3,155,000 ≈ 53.88%
        raw_expected = 100.0 * 1_700_000 / 3_155_000
        assert jan["LFPR_RAW"] == pytest.approx(raw_expected, abs=1e-4)

        # ACS-corrected LFPR uses the actual CNI16+ figure.
        acs_expected = 100.0 * 1_700_000 / 2_404_000
        assert jan["LFPR"] == pytest.approx(acs_expected, abs=1e-4)
        # ~70.7%, well above the uniform-0.78 prediction (~69%).
        assert 65.0 <= jan["LFPR"] <= 75.0

    def test_acs_value_lifts_lfpr_above_uniform_0_78_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """For Iowa in 2020 the ACS denominator (2.40M) is *smaller* than
        ``Pop * 0.78`` (2.46M), so the ACS-path LFPR comes out higher
        than the uniform-fraction fallback. Lock that in."""
        from utils import merge_all_data as merge_mod

        _write_laus_fixture(tmp_path)
        (tmp_path / "raw" / "laus" / "WAP_IA.txt").write_text(
            "series_id,year,period,value\n"
            "WAP_IA,2020,A01,2404000\n"
        )
        monkeypatch.setattr(merge_mod, "RAW_DIR_LAUS", str(tmp_path / "raw" / "laus"))
        monkeypatch.setattr(merge_mod, "RAW_DIR_CES",  str(tmp_path / "raw" / "ces"))
        monkeypatch.setattr(merge_mod, "CES_JSON",     str(tmp_path / "ces_state_sms_codes.json"))

        panel = merge_mod.merge_all_data(["IA"], 2020, 2020)
        jan = panel[(panel["year"] == 2020) & (panel["month"] == 1)].iloc[0]

        uniform_expected = 100.0 * 1_700_000 / (3_155_000 * merge_mod.LFPR_WORKING_AGE_FRACTION)
        acs_expected = 100.0 * 1_700_000 / 2_404_000
        assert acs_expected > uniform_expected
        assert jan["LFPR"] == pytest.approx(acs_expected, abs=1e-4)


class TestACSPathFallback:
    def test_no_wap_file_uses_uniform_0_78(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from utils import merge_all_data as merge_mod

        _write_laus_fixture(tmp_path)
        # NO WAP file written — fallback path should engage.
        monkeypatch.setattr(merge_mod, "RAW_DIR_LAUS", str(tmp_path / "raw" / "laus"))
        monkeypatch.setattr(merge_mod, "RAW_DIR_CES",  str(tmp_path / "raw" / "ces"))
        monkeypatch.setattr(merge_mod, "CES_JSON",     str(tmp_path / "ces_state_sms_codes.json"))

        panel = merge_mod.merge_all_data(["IA"], 2020, 2020)
        jan = panel[(panel["year"] == 2020) & (panel["month"] == 1)].iloc[0]
        expected = 100.0 * 1_700_000 / (3_155_000 * merge_mod.LFPR_WORKING_AGE_FRACTION)
        assert jan["LFPR"] == pytest.approx(expected, abs=1e-4)


class TestReadWorkingAgePopulation:
    def test_interpolates_missing_years(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Years between two ACS releases get linear interpolation."""
        from utils import merge_all_data as merge_mod

        laus = tmp_path / "raw" / "laus"
        laus.mkdir(parents=True)
        # ACS only has 2018 and 2022 — 2019/2020/2021 should be filled.
        (laus / "WAP_IA.txt").write_text(
            "series_id,year,period,value\n"
            "WAP_IA,2018,A01,2_400_000\n"
            "WAP_IA,2022,A01,2_440_000\n".replace("_", "")
        )
        monkeypatch.setattr(merge_mod, "RAW_DIR_LAUS", str(laus))

        df = merge_mod.read_working_age_population(["IA"], 2018, 2022)
        years = df.set_index("year")["working_age_population"]
        # 2018 = 2.40M, 2022 = 2.44M → 2020 should be ~2.42M.
        assert years[2018] == pytest.approx(2_400_000, abs=1)
        assert years[2020] == pytest.approx(2_420_000, abs=1)
        assert years[2022] == pytest.approx(2_440_000, abs=1)

    def test_extrapolates_via_ffill_outside_range(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from utils import merge_all_data as merge_mod

        laus = tmp_path / "raw" / "laus"
        laus.mkdir(parents=True)
        (laus / "WAP_IA.txt").write_text(
            "series_id,year,period,value\n"
            "WAP_IA,2010,A01,2362000\n"
        )
        monkeypatch.setattr(merge_mod, "RAW_DIR_LAUS", str(laus))

        df = merge_mod.read_working_age_population(["IA"], 1996, 2024)
        years = df.set_index("year")["working_age_population"]
        # Pre-2010 and post-2010 both fill from the single known year.
        assert years[1996] == pytest.approx(2_362_000, abs=1)
        assert years[2024] == pytest.approx(2_362_000, abs=1)

    def test_no_files_returns_empty_dataframe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from utils import merge_all_data as merge_mod

        laus = tmp_path / "raw" / "laus"
        laus.mkdir(parents=True)
        monkeypatch.setattr(merge_mod, "RAW_DIR_LAUS", str(laus))
        df = merge_mod.read_working_age_population(["IA"], 2020, 2024)
        assert df.empty
