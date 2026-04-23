"""Tests for ``utils.fetch_jolts_data``. requests.post is monkey-patched."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from utils import fetch_jolts_data


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


def _body(series_rows: dict[str, list[tuple[int, str, float]]]) -> dict:
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


class TestPresets:
    def test_national_preset_has_five_series(self) -> None:
        assert len(fetch_jolts_data.NATIONAL_JOLTS_SERIES) == 5
        # All must start with the JTS prefix.
        assert all(
            s.startswith("JTS") for s in fetch_jolts_data.NATIONAL_JOLTS_SERIES
        )

    def test_national_preset_covers_canonical_metrics(self) -> None:
        suffixes = {s[-3:] for s in fetch_jolts_data.NATIONAL_JOLTS_SERIES}
        # JOL = openings level, HIL = hires level, QUL = quits, LDL =
        # layoffs/discharges, TSL = total separations.
        assert suffixes == {"JOL", "HIL", "QUL", "LDL", "TSL"}


class TestValidation:
    def test_empty_series_list_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            fetch_jolts_data.fetch_jolts([], 2020, 2020)

    def test_pre_2000_year_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid year range"):
            fetch_jolts_data.fetch_jolts(None, 1999, 2020)

    def test_inverted_year_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid year range"):
            fetch_jolts_data.fetch_jolts(None, 2020, 2019)

    def test_more_than_fifty_series_rejected(self) -> None:
        too_many = [f"JTS000000000000000JOL{i}" for i in range(60)]
        with pytest.raises(ValueError, match="at most 50 series"):
            fetch_jolts_data.fetch_jolts(too_many, 2020, 2020)


class TestSuccessfulFetch:
    def test_defaults_to_national_preset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen_sids: list[list[str]] = []

        def fake_post(url, json=None, headers=None, timeout=None):
            seen_sids.append(list(json["seriesid"]))
            return FakeResponse(
                _body({sid: [(2020, "M01", 10.0)] for sid in json["seriesid"]})
            )

        monkeypatch.setattr("utils.fetch_jolts_data.requests.post", fake_post)

        written = fetch_jolts_data.fetch_jolts(
            None, 2020, 2020, raw_dir=str(tmp_path)
        )
        # Preset of 5 series × 1 monthly row each.
        assert written == 5
        assert set(seen_sids[0]) == set(fetch_jolts_data.NATIONAL_JOLTS_SERIES)

    def test_writes_one_file_per_series(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_post(url, json=None, headers=None, timeout=None):
            return FakeResponse(
                _body(
                    {
                        "JTS000000000000000JOL": [
                            (2020, "M01", 7200.0),
                            (2020, "M02", 7300.0),
                        ]
                    }
                )
            )

        monkeypatch.setattr("utils.fetch_jolts_data.requests.post", fake_post)
        written = fetch_jolts_data.fetch_jolts(
            ["JTS000000000000000JOL"], 2020, 2020, raw_dir=str(tmp_path)
        )
        assert written == 2
        out = (tmp_path / "JTS000000000000000JOL.txt").read_text(encoding="utf-8").splitlines()
        assert out[0] == "series_id,year,period,value"
        assert out[1] == "JTS000000000000000JOL,2020,M01,7200.0"

    def test_batches_years_under_20_year_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        batch_ranges: list[tuple[str, str]] = []

        def fake_post(url, json=None, headers=None, timeout=None):
            batch_ranges.append((json["startyear"], json["endyear"]))
            return FakeResponse(_body({sid: [] for sid in json["seriesid"]}))

        monkeypatch.setattr("utils.fetch_jolts_data.requests.post", fake_post)
        # 22 years → 2 batches.
        fetch_jolts_data.fetch_jolts(
            ["JTS000000000000000JOL"], 2003, 2024, raw_dir=str(tmp_path)
        )
        assert len(batch_ranges) == 2
        assert batch_ranges[0] == ("2003", "2022")
        assert batch_ranges[1] == ("2023", "2024")

    def test_drops_m13_and_non_monthly_periods(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_post(url, json=None, headers=None, timeout=None):
            return FakeResponse(
                _body(
                    {
                        "JTS000000000000000JOL": [
                            (2020, "M01", 100.0),
                            (2020, "M13", 110.0),   # annual average
                            (2020, "S01", 105.0),   # semi-annual
                            (2020, "M02", 101.0),
                        ]
                    }
                )
            )

        monkeypatch.setattr("utils.fetch_jolts_data.requests.post", fake_post)
        written = fetch_jolts_data.fetch_jolts(
            ["JTS000000000000000JOL"], 2020, 2020, raw_dir=str(tmp_path)
        )
        assert written == 2

    def test_registration_key_included_when_provided(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[dict] = []

        def fake_post(url, json=None, headers=None, timeout=None):
            seen.append(dict(json))
            return FakeResponse(_body({sid: [] for sid in json["seriesid"]}))

        monkeypatch.setattr("utils.fetch_jolts_data.requests.post", fake_post)
        fetch_jolts_data.fetch_jolts(
            ["JTS000000000000000JOL"], 2020, 2020,
            raw_dir=str(tmp_path), api_key="my-bls-key",
        )
        assert seen[0].get("registrationkey") == "my-bls-key"


class TestGracefulFailures:
    def test_http_error_on_one_batch_does_not_abort(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def fake_post(url, json=None, headers=None, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("upstream timeout")
            return FakeResponse(
                _body({sid: [(2024, "M01", 99.0)] for sid in json["seriesid"]})
            )

        monkeypatch.setattr("utils.fetch_jolts_data.requests.post", fake_post)
        written = fetch_jolts_data.fetch_jolts(
            ["JTS000000000000000JOL"], 2003, 2024, raw_dir=str(tmp_path)
        )
        # First batch failed → second batch only wrote 1 row.
        assert written == 1

    def test_request_not_processed_logged_and_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_post(url, json=None, headers=None, timeout=None):
            return FakeResponse(
                {"status": "REQUEST_NOT_PROCESSED", "message": ["Over quota"]}
            )

        monkeypatch.setattr("utils.fetch_jolts_data.requests.post", fake_post)
        written = fetch_jolts_data.fetch_jolts(
            ["JTS000000000000000JOL"], 2020, 2020, raw_dir=str(tmp_path)
        )
        assert written == 0
