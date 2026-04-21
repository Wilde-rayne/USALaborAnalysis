"""
Tests for ``utils.fetch_bea_data``.

Requests.get is monkey-patched so nothing hits BEA's live API. These
tests cover input validation (unknown states, bad year ranges, missing
key), response parsing, and output file format.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from utils import fetch_bea_data


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


def _bea_payload(rows: list[tuple[str, float]]) -> dict:
    """Build a realistic SAINC1 response body from (fips, value) pairs."""
    return {
        "BEAAPI": {
            "Results": {
                "Data": [
                    {"GeoFips": fips, "DataValue": f"{val:,.0f}"}
                    for fips, val in rows
                ]
            }
        }
    }


class TestValidation:
    def test_missing_key_raises_with_signup_link(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("BEA_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="bea.gov/API/signup"):
            fetch_bea_data.fetch_bea_sainc1(["IA"], 2020, 2020)

    def test_unknown_state_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BEA_API_KEY", "fake")
        with pytest.raises(ValueError, match="unknown state codes"):
            fetch_bea_data.fetch_bea_sainc1(["ZZ"], 2020, 2020)

    def test_year_range_rejects_before_1929(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BEA_API_KEY", "fake")
        with pytest.raises(ValueError, match="invalid year range"):
            fetch_bea_data.fetch_bea_sainc1(["IA"], 1900, 2000)

    def test_year_range_rejects_inverted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BEA_API_KEY", "fake")
        with pytest.raises(ValueError, match="invalid year range"):
            fetch_bea_data.fetch_bea_sainc1(["IA"], 2020, 2019)


class TestSuccessfulFetch:
    def test_writes_one_file_per_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BEA_API_KEY", "fake")
        payloads = {
            2020: _bea_payload([("19", 55000), ("17", 62000)]),
            2021: _bea_payload([("19", 58000), ("17", 65000)]),
        }
        calls: list[dict] = []

        def fake_get(url, params=None, timeout=None):
            calls.append(dict(params))
            year = int(params["Year"])
            return FakeResponse(payloads[year])

        monkeypatch.setattr("utils.fetch_bea_data.requests.get", fake_get)

        written = fetch_bea_data.fetch_bea_sainc1(
            ["IA", "IL"], 2020, 2021, raw_dir=str(tmp_path)
        )
        # 2 states × 2 years = 4 rows.
        assert written == 4

        # One file per state, properly named.
        ia_file = tmp_path / "BEA_SAINC1_L3_IA.txt"
        il_file = tmp_path / "BEA_SAINC1_L3_IL.txt"
        assert ia_file.exists() and il_file.exists()

        ia_lines = ia_file.read_text(encoding="utf-8").splitlines()
        # Header + 2 data rows.
        assert ia_lines[0] == "series_id,year,period,value"
        assert ia_lines[1] == "BEA_SAINC1_L3_IA,2020,A01,55000.0"
        assert ia_lines[2] == "BEA_SAINC1_L3_IA,2021,A01,58000.0"

    def test_one_request_per_year(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BEA_API_KEY", "fake")
        call_years: list[str] = []

        def fake_get(url, params=None, timeout=None):
            call_years.append(params["Year"])
            return FakeResponse(_bea_payload([("19", 1.0)]))

        monkeypatch.setattr("utils.fetch_bea_data.requests.get", fake_get)

        fetch_bea_data.fetch_bea_sainc1(
            ["IA"], 2020, 2022, raw_dir=str(tmp_path)
        )
        assert call_years == ["2020", "2021", "2022"]

    def test_api_key_is_sent_in_request(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BEA_API_KEY", "my-secret-key")
        captured: list[dict] = []

        def fake_get(url, params=None, timeout=None):
            captured.append(dict(params))
            return FakeResponse(_bea_payload([("19", 100.0)]))

        monkeypatch.setattr("utils.fetch_bea_data.requests.get", fake_get)

        fetch_bea_data.fetch_bea_sainc1(
            ["IA"], 2020, 2020, raw_dir=str(tmp_path)
        )
        assert captured[0]["UserID"] == "my-secret-key"
        assert captured[0]["TableName"] == "SAINC1"
        assert captured[0]["LineCode"] == "3"

    def test_skips_unknown_fips_in_response(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """BEA sometimes returns extra aggregated rows — ignore them."""
        monkeypatch.setenv("BEA_API_KEY", "fake")
        payload = _bea_payload(
            [("19", 100.0), ("00", 999.0), ("99", 999.0)]  # national + unknown
        )

        def fake_get(url, params=None, timeout=None):
            return FakeResponse(payload)

        monkeypatch.setattr("utils.fetch_bea_data.requests.get", fake_get)

        written = fetch_bea_data.fetch_bea_sainc1(
            ["IA"], 2020, 2020, raw_dir=str(tmp_path)
        )
        assert written == 1
        assert (tmp_path / "BEA_SAINC1_L3_IA.txt").exists()
        # No file for the synthetic "99" / "00" rows.
        for p in tmp_path.iterdir():
            assert p.name.startswith("BEA_SAINC1_L3_IA"), p


class TestGracefulFailures:
    def test_http_error_on_a_year_does_not_kill_whole_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Network errors on one year should skip that year, not abort."""
        monkeypatch.setenv("BEA_API_KEY", "fake")
        calls = {"n": 0}

        def fake_get(url, params=None, timeout=None):
            calls["n"] += 1
            if int(params["Year"]) == 2021:
                raise RuntimeError("network down")
            return FakeResponse(_bea_payload([("19", 50.0)]))

        monkeypatch.setattr("utils.fetch_bea_data.requests.get", fake_get)

        written = fetch_bea_data.fetch_bea_sainc1(
            ["IA"], 2020, 2022, raw_dir=str(tmp_path)
        )
        # 2 successful years (2020 + 2022), one skipped year (2021).
        assert written == 2

    def test_empty_results_still_writes_no_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BEA_API_KEY", "fake")

        def fake_get(url, params=None, timeout=None):
            return FakeResponse({"BEAAPI": {"Results": {"Data": []}}})

        monkeypatch.setattr("utils.fetch_bea_data.requests.get", fake_get)

        written = fetch_bea_data.fetch_bea_sainc1(
            ["IA"], 2020, 2020, raw_dir=str(tmp_path)
        )
        assert written == 0
        # Directory exists (tmp_path), but no BEA file in it.
        assert not any(p.name.startswith("BEA_") for p in tmp_path.iterdir())
