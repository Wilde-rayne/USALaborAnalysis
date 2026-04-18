"""
Unit tests for the pure logic inside ``utils.fetch_ces_data``.

Network calls (fetch_ces_data itself) are exercised in the integration
suite. Here we cover ``_series_ids`` — the part of the fetcher that can
fail silently and leave the BLS batch empty.
"""
from __future__ import annotations

from typing import Any

import pytest

from utils import fetch_ces_data


@pytest.fixture
def fake_ces_codes(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, str]]:
    """Install a small in-memory CES code table."""
    fake: dict[str, dict[str, str]] = {
        "Manufacturing": {
            "IA": "SMS19000001000000001",
            "IL": "SMS17000001000000001",
            "WI": "POP_55_BAD_PREFIX",  # intentionally non-SMS
        },
        "Construction": {
            "IA": "SMS19000002000000001",
            "IL": "SMS17000002000000001",
        },
    }
    monkeypatch.setattr(fetch_ces_data, "_load_json", lambda: fake)
    return fake


class TestSeriesIds:
    def test_collects_sms_codes_for_requested_state(self, fake_ces_codes: Any) -> None:
        ids = fetch_ces_data._series_ids(["IA"])
        assert ids == [
            "SMS19000001000000001",  # Manufacturing IA
            "SMS19000002000000001",  # Construction IA
        ]

    def test_ignores_non_sms_prefix_codes(self, fake_ces_codes: Any) -> None:
        # WI has a POP-prefixed code in Manufacturing and nothing in Construction.
        assert fetch_ces_data._series_ids(["WI"]) == []

    def test_skips_states_with_no_entry(self, fake_ces_codes: Any) -> None:
        # CA is absent from every supersector.
        assert fetch_ces_data._series_ids(["CA"]) == []

    def test_empty_states_returns_empty(self, fake_ces_codes: Any) -> None:
        assert fetch_ces_data._series_ids([]) == []

    def test_deduplicates_identical_codes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Same SID appears under two supersectors — should not double-count.
        duplicated = {
            "Sector_A": {"IA": "SMS19000001000000001"},
            "Sector_B": {"IA": "SMS19000001000000001"},
        }
        monkeypatch.setattr(fetch_ces_data, "_load_json", lambda: duplicated)
        ids = fetch_ces_data._series_ids(["IA"])
        assert ids == ["SMS19000001000000001"]

    def test_result_is_sorted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        unsorted_input = {
            "Sector_X": {"IA": "SMS19000003000000001"},
            "Sector_Y": {"IA": "SMS19000001000000001"},
            "Sector_Z": {"IA": "SMS19000002000000001"},
        }
        monkeypatch.setattr(fetch_ces_data, "_load_json", lambda: unsorted_input)
        ids = fetch_ces_data._series_ids(["IA"])
        assert ids == sorted(ids) == [
            "SMS19000001000000001",
            "SMS19000002000000001",
            "SMS19000003000000001",
        ]

    def test_multiple_states_aggregate(self, fake_ces_codes: Any) -> None:
        ids = fetch_ces_data._series_ids(["IA", "IL"])
        # Four total SMS codes across IA + IL; WI's non-SMS code stays excluded.
        assert ids == [
            "SMS17000001000000001",
            "SMS17000002000000001",
            "SMS19000001000000001",
            "SMS19000002000000001",
        ]
