"""
Tests for ``utils.fetch_fred_data``. requests.get is monkey-patched so
nothing hits the live FRED API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from utils import fetch_fred_data


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


def _obs(date: str, value: str) -> dict:
    return {"date": date, "value": value}


class TestValidation:
    def test_missing_key_raises_with_signup_link(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="fred.stlouisfed.org"):
            fetch_fred_data.fetch_fred_state_series(["IA"], "UR", 2020, 2021)

    def test_unknown_indicator_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FRED_API_KEY", "fake")
        with pytest.raises(ValueError, match="unknown FRED indicator"):
            fetch_fred_data.fetch_fred_state_series(["IA"], "XYZ", 2020, 2021)

    def test_unknown_state_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FRED_API_KEY", "fake")
        with pytest.raises(ValueError, match="unknown state codes"):
            fetch_fred_data.fetch_fred_state_series(["ZZ"], "UR", 2020, 2021)

    def test_year_range_validated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FRED_API_KEY", "fake")
        with pytest.raises(ValueError, match="invalid year range"):
            fetch_fred_data.fetch_fred_state_series(["IA"], "UR", 1900, 1910)
        with pytest.raises(ValueError, match="invalid year range"):
            fetch_fred_data.fetch_fred_state_series(["IA"], "UR", 2020, 2019)


class TestSeriesIdConstruction:
    def test_id_is_state_plus_indicator(self) -> None:
        assert fetch_fred_data._fred_series_id("IA", "UR") == "IAUR"
        assert fetch_fred_data._fred_series_id("MI", "PI") == "MIPI"
        # STHPI: FHFA state house price index through FRED.
        assert fetch_fred_data._fred_series_id("CA", "STHPI") == "CASTHPI"


class TestIndicatorRegistry:
    def test_registry_includes_housing(self) -> None:
        assert "STHPI" in fetch_fred_data.FRED_INDICATORS
        desc, cadence = fetch_fred_data.FRED_INDICATORS["STHPI"]
        assert "house price" in desc.lower()
        assert cadence == "quarterly"

    def test_every_indicator_has_cadence_the_period_mapper_accepts(self) -> None:
        for ind, (_, cadence) in fetch_fred_data.FRED_INDICATORS.items():
            # Must not raise and must return a non-empty period code.
            got = fetch_fred_data._bls_period_from_date("2020-04-01", cadence)
            assert got and got[:1] in {"M", "Q", "A"}, ind


class TestPeriodMapping:
    def test_monthly(self) -> None:
        assert fetch_fred_data._bls_period_from_date("2020-03-01", "monthly") == "M03"

    def test_quarterly(self) -> None:
        assert fetch_fred_data._bls_period_from_date("2020-04-01", "quarterly") == "Q02"
        assert fetch_fred_data._bls_period_from_date("2020-10-01", "quarterly") == "Q04"

    def test_annual_fallback(self) -> None:
        assert fetch_fred_data._bls_period_from_date("2020-01-01", "annual") == "A01"


class TestSuccessfulFetch:
    def test_writes_file_per_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FRED_API_KEY", "fake")
        # Two months for IA (IAUR) and two months for IL (ILUR).
        by_series = {
            "IAUR": {"observations": [
                _obs("2020-01-01", "3.1"),
                _obs("2020-02-01", "3.3"),
            ]},
            "ILUR": {"observations": [
                _obs("2020-01-01", "4.2"),
                _obs("2020-02-01", "4.5"),
            ]},
        }

        def fake_get(url, params=None, timeout=None):
            return FakeResponse(by_series[params["series_id"]])

        monkeypatch.setattr("utils.fetch_fred_data.requests.get", fake_get)

        written = fetch_fred_data.fetch_fred_state_series(
            ["IA", "IL"], "UR", 2020, 2020, raw_dir=str(tmp_path)
        )
        assert written == 4

        ia = (tmp_path / "FRED_IAUR.txt").read_text(encoding="utf-8").splitlines()
        assert ia[0] == "series_id,year,period,value"
        assert ia[1] == "FRED_IAUR,2020,M01,3.1"
        assert ia[2] == "FRED_IAUR,2020,M02,3.3"
        il = (tmp_path / "FRED_ILUR.txt").read_text(encoding="utf-8").splitlines()
        assert il[1] == "FRED_ILUR,2020,M01,4.2"

    def test_drops_missing_value_markers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FRED represents missing observations as a literal '.'."""
        monkeypatch.setenv("FRED_API_KEY", "fake")

        def fake_get(url, params=None, timeout=None):
            return FakeResponse({
                "observations": [
                    _obs("2020-01-01", "3.1"),
                    _obs("2020-02-01", "."),   # missing
                    _obs("2020-03-01", ""),    # missing
                    _obs("2020-04-01", "not-a-number"),  # unparseable
                    _obs("2020-05-01", "3.5"),
                ]
            })

        monkeypatch.setattr("utils.fetch_fred_data.requests.get", fake_get)

        written = fetch_fred_data.fetch_fred_state_series(
            ["IA"], "UR", 2020, 2020, raw_dir=str(tmp_path)
        )
        # Only two valid rows.
        assert written == 2

    def test_api_key_sent_in_params(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FRED_API_KEY", "abc-123")
        captured: list[dict] = []

        def fake_get(url, params=None, timeout=None):
            captured.append(dict(params))
            return FakeResponse({"observations": []})

        monkeypatch.setattr("utils.fetch_fred_data.requests.get", fake_get)
        fetch_fred_data.fetch_fred_state_series(
            ["IA"], "UR", 2020, 2020, raw_dir=str(tmp_path)
        )
        assert captured[0]["api_key"] == "abc-123"
        assert captured[0]["series_id"] == "IAUR"
        assert captured[0]["file_type"] == "json"

    def test_http_error_on_one_state_does_not_abort_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FRED_API_KEY", "fake")

        def fake_get(url, params=None, timeout=None):
            if params["series_id"] == "ILUR":
                raise RuntimeError("boom")
            return FakeResponse({"observations": [_obs("2020-01-01", "3.0")]})

        monkeypatch.setattr("utils.fetch_fred_data.requests.get", fake_get)
        written = fetch_fred_data.fetch_fred_state_series(
            ["IA", "IL"], "UR", 2020, 2020, raw_dir=str(tmp_path)
        )
        # Only IA (1 row) succeeded.
        assert written == 1
        assert (tmp_path / "FRED_IAUR.txt").exists()
        assert not (tmp_path / "FRED_ILUR.txt").exists()
