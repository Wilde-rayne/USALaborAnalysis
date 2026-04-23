"""Tests for ``utils.fetch_qcew_data``. requests.get is monkey-patched."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from utils import fetch_qcew_data


class FakeResponse:
    def __init__(self, text: str, status: int = 200) -> None:
        self.text = text
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _qcew_csv(rows: list[dict]) -> str:
    """Render a QCEW-shaped CSV with a realistic column set."""
    headers = [
        "area_fips", "own_code", "industry_code", "agglvl_code", "qtr",
        "year", "month1_emplvl", "total_qtrly_wages", "avg_wkly_wage",
        "qtrly_estabs_count",
    ]
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(r.get(h, "")) for h in headers))
    return "\n".join(lines)


def _state_total_row(**over: Any) -> dict:
    base = {
        "area_fips": "19000",
        "own_code": "0",
        "industry_code": "10",
        "agglvl_code": "50",
        "qtr": "1",
        "year": "2020",
        "month1_emplvl": "1500000",
        "total_qtrly_wages": "12000000000",
        "avg_wkly_wage": "980",
        "qtrly_estabs_count": "75000",
    }
    base.update(over)
    return base


class TestValidation:
    def test_unknown_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValueError, match="unknown state codes"):
            fetch_qcew_data.fetch_qcew_state_totals(["ZZ"], 2020, 2020)

    def test_pre_1990_year(self) -> None:
        with pytest.raises(ValueError, match="invalid year range"):
            fetch_qcew_data.fetch_qcew_state_totals(["IA"], 1970, 2020)

    def test_inverted_year(self) -> None:
        with pytest.raises(ValueError, match="invalid year range"):
            fetch_qcew_data.fetch_qcew_state_totals(["IA"], 2020, 2019)

    def test_invalid_quarter(self) -> None:
        with pytest.raises(ValueError, match="invalid quarters"):
            fetch_qcew_data.fetch_qcew_state_totals(
                ["IA"], 2020, 2020, quarters=[5]
            )


class TestSuccessfulFetch:
    def test_writes_one_file_per_state_per_metric(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_get(url, timeout=None):
            # Return the state-total row plus some other roll-ups that
            # the filter should skip.
            rows = [
                {"area_fips": "19000", "agglvl_code": "10",
                 "own_code": "0", "industry_code": "10"},  # national, skip
                {"area_fips": "19000", "agglvl_code": "50",
                 "own_code": "5", "industry_code": "10",
                 "qtr": "1", "year": "2020",
                 "month1_emplvl": "999"},  # private only, skip
                _state_total_row(),
            ]
            return FakeResponse(_qcew_csv(rows))

        monkeypatch.setattr("utils.fetch_qcew_data.requests.get", fake_get)
        written = fetch_qcew_data.fetch_qcew_state_totals(
            ["IA"], 2020, 2020, quarters=[1], raw_dir=str(tmp_path)
        )
        # One quarter × 4 metrics = 4 rows.
        assert written == 4
        # One file per metric.
        metric_files = sorted(p.name for p in tmp_path.iterdir())
        assert metric_files == [
            "QCEW_IA_AWW.txt",
            "QCEW_IA_EMP.txt",
            "QCEW_IA_EST.txt",
            "QCEW_IA_TQW.txt",
        ]
        # File content is header + one row.
        emp = (tmp_path / "QCEW_IA_EMP.txt").read_text(encoding="utf-8").splitlines()
        assert emp[0] == "series_id,year,period,value"
        assert emp[1] == "QCEW_IA_EMP,2020,Q01,1500000.0"

    def test_multiple_quarters_roll_up_per_metric(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call_urls: list[str] = []

        def fake_get(url, timeout=None):
            call_urls.append(url)
            # URL: .../api/{year}/{qtr}/area/{area}.csv
            #      [...][-5][-4][-3]   [-2] [-1]
            parts = url.split("/")
            qtr = parts[-3]
            return FakeResponse(_qcew_csv([_state_total_row(qtr=qtr)]))

        monkeypatch.setattr("utils.fetch_qcew_data.requests.get", fake_get)
        written = fetch_qcew_data.fetch_qcew_state_totals(
            ["IA"], 2020, 2020, raw_dir=str(tmp_path)
        )
        # Default is all 4 quarters × 4 metrics = 16 rows.
        assert written == 16
        assert len(call_urls) == 4  # one GET per (state, quarter)

        emp = (tmp_path / "QCEW_IA_EMP.txt").read_text(encoding="utf-8").splitlines()
        # Header + 4 quarterly rows.
        assert len(emp) == 5
        periods = [line.split(",")[2] for line in emp[1:]]
        assert periods == ["Q01", "Q02", "Q03", "Q04"]

    def test_url_targets_state_level_area_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Area code must be 2-digit FIPS + '000' for statewide."""
        urls: list[str] = []

        def fake_get(url, timeout=None):
            urls.append(url)
            return FakeResponse(_qcew_csv([_state_total_row()]))

        monkeypatch.setattr("utils.fetch_qcew_data.requests.get", fake_get)
        fetch_qcew_data.fetch_qcew_state_totals(
            ["CA"], 2020, 2020, quarters=[1], raw_dir=str(tmp_path)
        )
        # California FIPS = 06; statewide is 06000.
        assert "/area/06000.csv" in urls[0]

    def test_non_numeric_value_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Garbage in a column shouldn't poison the other metrics."""
        def fake_get(url, timeout=None):
            return FakeResponse(_qcew_csv([
                _state_total_row(avg_wkly_wage="N/A"),  # bad
            ]))

        monkeypatch.setattr("utils.fetch_qcew_data.requests.get", fake_get)
        written = fetch_qcew_data.fetch_qcew_state_totals(
            ["IA"], 2020, 2020, quarters=[1], raw_dir=str(tmp_path)
        )
        # 3 of 4 metrics parsed; AWW skipped.
        assert written == 3
        assert not (tmp_path / "QCEW_IA_AWW.txt").exists()
        assert (tmp_path / "QCEW_IA_EMP.txt").exists()

    def test_missing_state_row_logs_and_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """CSV with no agglvl=50 row → nothing written for that quarter."""
        def fake_get(url, timeout=None):
            # Only national-level rows.
            return FakeResponse(_qcew_csv([
                {"area_fips": "19000", "agglvl_code": "10",
                 "own_code": "0", "industry_code": "10", "year": "2020"}
            ]))

        monkeypatch.setattr("utils.fetch_qcew_data.requests.get", fake_get)
        written = fetch_qcew_data.fetch_qcew_state_totals(
            ["IA"], 2020, 2020, quarters=[1], raw_dir=str(tmp_path)
        )
        assert written == 0


class TestGracefulFailures:
    def test_http_error_on_one_quarter_does_not_abort_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def fake_get(url, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("upstream 503")
            return FakeResponse(_qcew_csv([_state_total_row(qtr=str(calls["n"]))]))

        monkeypatch.setattr("utils.fetch_qcew_data.requests.get", fake_get)
        written = fetch_qcew_data.fetch_qcew_state_totals(
            ["IA"], 2020, 2020, raw_dir=str(tmp_path)
        )
        # 1 failed quarter + 3 successful × 4 metrics = 12 rows.
        assert written == 12
