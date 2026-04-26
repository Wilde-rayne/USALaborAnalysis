"""
Tests for ``utils.fetch_working_age_population``.

Mocks ``requests.get`` so nothing hits the live Census ACS API. These
lock in the validation gates, the per-(state, year) loop, and the
output file format the merger consumes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def _import_module():
    """Lazy import — keeps test collection cheap when the module isn't installed."""
    import importlib

    return importlib.import_module("utils.fetch_working_age_population")


class FakeResponse:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._payload


def _acs_payload(value: int, fips: str) -> list:
    """Match Census ACS shape: ``[[header...], [value, fips_str]]``."""
    return [["B23025_001E", "state"], [str(value), fips]]


class TestValidation:
    def test_missing_key_raises_with_signup_link(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CENSUS_API_KEY", "")
        # The module reads the key via utils.constants at import time;
        # patch that rather than the env so the test isn't sensitive to
        # import ordering.
        mod = _import_module()
        monkeypatch.setattr(mod, "CENSUS_API_KEY", None)
        with pytest.raises(RuntimeError, match="api.census.gov"):
            mod.fetch_acs_working_age_population(["IA"], [2020])

    def test_unknown_state_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _import_module()
        monkeypatch.setattr(mod, "CENSUS_API_KEY", "fake-key")
        with pytest.raises(ValueError, match="unknown state codes"):
            mod.fetch_acs_working_age_population(["ZZ"], [2020])

    def test_empty_year_list_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _import_module()
        monkeypatch.setattr(mod, "CENSUS_API_KEY", "fake-key")
        with pytest.raises(ValueError, match="years must be non-empty"):
            mod.fetch_acs_working_age_population(["IA"], years=[])


class TestSuccessfulFetch:
    def test_writes_one_file_per_state_with_correct_format(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _import_module()
        monkeypatch.setattr(mod, "CENSUS_API_KEY", "fake-key")

        # Fake responses keyed by (year, fips).
        responses = {
            (2020, "19"): _acs_payload(2_404_000, "19"),  # Iowa
            (2021, "19"): _acs_payload(2_412_500, "19"),
            (2020, "17"): _acs_payload(10_133_700, "17"),  # Illinois
            (2021, "17"): _acs_payload(10_120_500, "17"),
        }

        def fake_get(url, timeout=None):
            year = int(url.split("/data/")[1].split("/", 1)[0])
            fips = url.split("for=state:", 1)[1].split("&", 1)[0]
            return FakeResponse(responses[(year, fips)])

        monkeypatch.setattr(mod.requests, "get", fake_get)

        written = mod.fetch_acs_working_age_population(
            ["IA", "IL"], years=[2020, 2021], raw_dir=str(tmp_path)
        )
        # 2 states × 2 years = 4 rows.
        assert written == 4

        ia = (tmp_path / "WAP_IA.txt").read_text(encoding="utf-8").splitlines()
        assert ia[0] == "series_id,year,period,value"
        assert ia[1] == "WAP_IA,2020,A01,2404000"
        assert ia[2] == "WAP_IA,2021,A01,2412500"

        il = (tmp_path / "WAP_IL.txt").read_text(encoding="utf-8").splitlines()
        assert il[1] == "WAP_IL,2020,A01,10133700"

    def test_request_failure_skips_rather_than_aborts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _import_module()
        monkeypatch.setattr(mod, "CENSUS_API_KEY", "fake-key")

        # 2020 fails for IA; 2021 succeeds.
        def fake_get(url, timeout=None):
            year = int(url.split("/data/")[1].split("/", 1)[0])
            if year == 2020:
                raise RuntimeError("upstream 503")
            return FakeResponse(_acs_payload(2_412_500, "19"))

        monkeypatch.setattr(mod.requests, "get", fake_get)

        written = mod.fetch_acs_working_age_population(
            ["IA"], years=[2020, 2021], raw_dir=str(tmp_path)
        )
        # 1 successful year only.
        assert written == 1
        # File still written (single row).
        assert (tmp_path / "WAP_IA.txt").exists()

    def test_default_year_range_excludes_2020(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ACS 1-year suppressed 2020; the default tuple drops it."""
        mod = _import_module()
        assert 2020 not in mod.DEFAULT_ACS_YEARS
        assert min(mod.DEFAULT_ACS_YEARS) == 2005
        assert max(mod.DEFAULT_ACS_YEARS) >= 2024


class TestVariableConstant:
    def test_acs_variable_is_b23025_001e(self) -> None:
        mod = _import_module()
        assert mod.ACS_VARIABLE_CNI16 == "B23025_001E"
