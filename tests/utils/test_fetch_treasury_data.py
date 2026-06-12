"""
Tests for ``utils.fetch_treasury_data``. requests.get is monkey-patched
so nothing hits the live FRED API.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from utils import fetch_treasury_data


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
            fetch_treasury_data.fetch_treasury_series(2020, 2021)

    def test_unknown_series_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FRED_API_KEY", "fake")
        with pytest.raises(ValueError, match="unknown Treasury series"):
            fetch_treasury_data.fetch_treasury_series(2020, 2021, ["DGS10"])

    def test_year_range_validated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FRED_API_KEY", "fake")
        with pytest.raises(ValueError, match="invalid year range"):
            fetch_treasury_data.fetch_treasury_series(1900, 1910)
        with pytest.raises(ValueError, match="invalid year range"):
            fetch_treasury_data.fetch_treasury_series(2020, 2019)


class TestSeriesRegistry:
    def test_registry_includes_all_three_maturities(self) -> None:
        assert {"GS10", "GS2", "GS3M"} == set(fetch_treasury_data.TREASURY_SERIES)

    def test_descriptions_name_their_maturity(self) -> None:
        desc_10y, _ = fetch_treasury_data.TREASURY_SERIES["GS10"]
        desc_2y, _ = fetch_treasury_data.TREASURY_SERIES["GS2"]
        desc_3m, _ = fetch_treasury_data.TREASURY_SERIES["GS3M"]
        assert "10-year" in desc_10y.lower()
        assert "2-year" in desc_2y.lower()
        assert "3-month" in desc_3m.lower()

    def test_every_entry_is_monthly_with_a_mappable_cadence(self) -> None:
        """All registry series are FRED monthly averages; the shared
        BLS-style period mapper must accept the cadence string."""
        for sid, (desc, cadence) in fetch_treasury_data.TREASURY_SERIES.items():
            assert desc.strip(), sid
            assert cadence == "monthly", sid
            got = fetch_treasury_data._bls_period_from_date("2020-04-01", cadence)
            assert got == "M04", sid


class TestSuccessfulFetch:
    def test_writes_json_file_per_series(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FRED_API_KEY", "fake")
        by_series = {
            "GS10": {"observations": [
                _obs("2020-01-01", "1.76"),
                _obs("2020-02-01", "1.50"),
            ]},
            "GS3M": {"observations": [
                _obs("2020-01-01", "1.52"),
            ]},
        }

        def fake_get(url, params=None, timeout=None):
            return FakeResponse(by_series[params["series_id"]])

        monkeypatch.setattr("utils.fetch_treasury_data.requests.get", fake_get)

        written = fetch_treasury_data.fetch_treasury_series(
            2020, 2020, ["GS10", "GS3M"], raw_dir=str(tmp_path)
        )
        assert written == 3

        gs10 = json.loads((tmp_path / "GS10.json").read_text(encoding="utf-8"))
        assert gs10 == [
            {"series_id": "GS10", "year": 2020, "period": "M01", "value": 1.76},
            {"series_id": "GS10", "year": 2020, "period": "M02", "value": 1.50},
        ]
        gs3m = json.loads((tmp_path / "GS3M.json").read_text(encoding="utf-8"))
        assert gs3m[0]["period"] == "M01"
        assert gs3m[0]["value"] == pytest.approx(1.52)

    def test_default_series_ids_cover_the_whole_registry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FRED_API_KEY", "fake")
        requested: list[str] = []

        def fake_get(url, params=None, timeout=None):
            requested.append(params["series_id"])
            return FakeResponse({"observations": []})

        monkeypatch.setattr("utils.fetch_treasury_data.requests.get", fake_get)
        fetch_treasury_data.fetch_treasury_series(2020, 2020, raw_dir=str(tmp_path))
        assert set(requested) == set(fetch_treasury_data.TREASURY_SERIES)

    def test_drops_missing_value_markers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FRED represents missing observations as a literal '.'."""
        monkeypatch.setenv("FRED_API_KEY", "fake")

        def fake_get(url, params=None, timeout=None):
            return FakeResponse({
                "observations": [
                    _obs("2020-01-01", "1.76"),
                    _obs("2020-02-01", "."),   # missing
                    _obs("2020-03-01", ""),    # missing
                    _obs("2020-04-01", "not-a-number"),  # unparseable
                    _obs("2020-05-01", "0.69"),
                ]
            })

        monkeypatch.setattr("utils.fetch_treasury_data.requests.get", fake_get)

        written = fetch_treasury_data.fetch_treasury_series(
            2020, 2020, ["GS10"], raw_dir=str(tmp_path)
        )
        # Only two valid rows.
        assert written == 2
        records = json.loads((tmp_path / "GS10.json").read_text(encoding="utf-8"))
        assert [r["period"] for r in records] == ["M01", "M05"]

    def test_api_key_sent_in_params(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FRED_API_KEY", "abc-123")
        captured: list[dict] = []

        def fake_get(url, params=None, timeout=None):
            captured.append(dict(params))
            return FakeResponse({"observations": []})

        monkeypatch.setattr("utils.fetch_treasury_data.requests.get", fake_get)
        fetch_treasury_data.fetch_treasury_series(
            2020, 2020, ["GS10"], raw_dir=str(tmp_path)
        )
        assert captured[0]["api_key"] == "abc-123"
        assert captured[0]["series_id"] == "GS10"
        assert captured[0]["file_type"] == "json"

    def test_http_error_on_one_series_does_not_abort_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FRED_API_KEY", "fake")

        def fake_get(url, params=None, timeout=None):
            if params["series_id"] == "GS3M":
                raise RuntimeError("boom")
            return FakeResponse({"observations": [_obs("2020-01-01", "1.76")]})

        monkeypatch.setattr("utils.fetch_treasury_data.requests.get", fake_get)
        written = fetch_treasury_data.fetch_treasury_series(
            2020, 2020, ["GS10", "GS3M"], raw_dir=str(tmp_path)
        )
        # Only GS10 (1 row) succeeded.
        assert written == 1
        assert (tmp_path / "GS10.json").exists()
        assert not (tmp_path / "GS3M.json").exists()
