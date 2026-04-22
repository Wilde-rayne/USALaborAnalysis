"""
Tests for ``utils.fetch_cpi_data``. requests.post is monkey-patched so
nothing reaches BLS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from utils import fetch_cpi_data


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


def _bls_body(series_rows: dict[str, list[tuple[int, str, float]]]) -> dict:
    return {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": sid,
                    "data": [
                        {"year": str(y), "period": p, "value": str(v)}
                        for y, p, v in rows
                    ],
                }
                for sid, rows in series_rows.items()
            ]
        },
    }


class TestSeriesIdFormat:
    def test_all_items_series(self) -> None:
        assert fetch_cpi_data._series_id("0200") == "CUUR0200SA0"

    def test_custom_item(self) -> None:
        assert fetch_cpi_data._series_id("0000", "SAF0") == "CUUR0000SAF0"


class TestValidation:
    def test_unknown_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BLS_API_KEY", "fake")
        with pytest.raises(ValueError, match="unknown CPI region"):
            fetch_cpi_data.fetch_cpi_regional(["XXXX"], 2020, 2021)

    def test_invalid_year_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BLS_API_KEY", "fake")
        with pytest.raises(ValueError, match="invalid year range"):
            fetch_cpi_data.fetch_cpi_regional(["0200"], 1900, 1910)
        with pytest.raises(ValueError, match="invalid year range"):
            fetch_cpi_data.fetch_cpi_regional(["0200"], 2024, 2020)


class TestSuccessfulFetch:
    def test_writes_one_file_per_region(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BLS_API_KEY", "fake")

        def fake_post(url, json=None, headers=None, timeout=None):
            sids = json["seriesid"]
            body = {
                sid: [
                    (2020, "M01", 100.0 + i),
                    (2020, "M02", 101.5 + i),
                ]
                for i, sid in enumerate(sids)
            }
            return FakeResponse(_bls_body(body))

        monkeypatch.setattr("utils.fetch_cpi_data.requests.post", fake_post)

        written = fetch_cpi_data.fetch_cpi_regional(
            ["0200", "0300"], 2020, 2020, raw_dir=str(tmp_path)
        )
        assert written == 4
        mw = (tmp_path / "CUUR0200SA0.txt").read_text(encoding="utf-8").splitlines()
        assert mw[0] == "series_id,year,period,value"
        assert mw[1].startswith("CUUR0200SA0,2020,M01,")

    def test_defaults_to_all_regions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BLS_API_KEY", "fake")
        captured_series: list[list[str]] = []

        def fake_post(url, json=None, headers=None, timeout=None):
            captured_series.append(list(json["seriesid"]))
            return FakeResponse(_bls_body({sid: [(2020, "M01", 100.0)] for sid in json["seriesid"]}))

        monkeypatch.setattr("utils.fetch_cpi_data.requests.post", fake_post)
        fetch_cpi_data.fetch_cpi_regional(None, 2020, 2020, raw_dir=str(tmp_path))
        # Default fetches all 5 region codes.
        assert len(captured_series) == 1
        assert len(captured_series[0]) == 5
        assert "CUUR0000SA0" in captured_series[0]
        assert "CUUR0400SA0" in captured_series[0]

    def test_batches_years_under_20_year_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BLS_API_KEY", "fake")
        batches: list[tuple[str, str]] = []

        def fake_post(url, json=None, headers=None, timeout=None):
            batches.append((json["startyear"], json["endyear"]))
            return FakeResponse(_bls_body({sid: [] for sid in json["seriesid"]}))

        monkeypatch.setattr("utils.fetch_cpi_data.requests.post", fake_post)
        # 35-year span should split into 2 batches (≤20 each).
        fetch_cpi_data.fetch_cpi_regional(
            ["0200"], 1990, 2024, raw_dir=str(tmp_path)
        )
        assert len(batches) == 2
        assert batches[0] == ("1990", "2009")
        assert batches[1] == ("2010", "2024")

    def test_api_key_included_when_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BLS_API_KEY", "secret-key")
        seen: list[dict] = []

        def fake_post(url, json=None, headers=None, timeout=None):
            seen.append(dict(json))
            return FakeResponse(_bls_body({"CUUR0200SA0": []}))

        monkeypatch.setattr("utils.fetch_cpi_data.requests.post", fake_post)
        # Inject API key explicitly — constants.API_KEY is read at import
        # time so monkeypatching the env alone isn't enough.
        fetch_cpi_data.fetch_cpi_regional(
            ["0200"], 2020, 2020, raw_dir=str(tmp_path), api_key="secret-key"
        )
        assert seen[0].get("registrationkey") == "secret-key"

    def test_drops_non_monthly_periods(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BLS sometimes returns M13 (annual avg) or S01/S02 (semi) — skip."""
        monkeypatch.setenv("BLS_API_KEY", "fake")

        def fake_post(url, json=None, headers=None, timeout=None):
            return FakeResponse(_bls_body({
                "CUUR0200SA0": [
                    (2020, "M01", 100.0),
                    (2020, "M13", 100.9),  # annual avg
                    (2020, "S01", 100.5),  # semi-annual
                    (2020, "M02", 101.5),
                ]
            }))

        monkeypatch.setattr("utils.fetch_cpi_data.requests.post", fake_post)
        written = fetch_cpi_data.fetch_cpi_regional(
            ["0200"], 2020, 2020, raw_dir=str(tmp_path)
        )
        # Only the two M-prefixed monthly rows survive.
        assert written == 2


class TestGracefulFailures:
    def test_http_error_on_a_batch_does_not_abort_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BLS_API_KEY", "fake")
        calls = {"n": 0}

        def fake_post(url, json=None, headers=None, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("upstream timeout")
            return FakeResponse(_bls_body({sid: [(2020, "M01", 100.0)] for sid in json["seriesid"]}))

        monkeypatch.setattr("utils.fetch_cpi_data.requests.post", fake_post)
        # 21-year span → 2 batches; first fails, second succeeds.
        written = fetch_cpi_data.fetch_cpi_regional(
            ["0200"], 2000, 2020, raw_dir=str(tmp_path)
        )
        assert written == 1

    def test_request_failed_status_is_logged_and_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BLS_API_KEY", "fake")

        def fake_post(url, json=None, headers=None, timeout=None):
            return FakeResponse({
                "status": "REQUEST_NOT_PROCESSED",
                "message": ["Over rate limit"],
            })

        monkeypatch.setattr("utils.fetch_cpi_data.requests.post", fake_post)
        written = fetch_cpi_data.fetch_cpi_regional(
            ["0200"], 2020, 2020, raw_dir=str(tmp_path)
        )
        assert written == 0
