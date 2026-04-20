"""
Tests for the reactive cache layer in ``utils.data_pipeline``.

These lock in the behavior of the cache-aware wrappers added to enable
the reactive startup path — so future refactors don't silently
reintroduce the old always-fetch behavior.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from utils import data_pipeline as dp


@pytest.fixture
def fake_output_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point OUTPUT_JSON at a tmp path and return the path."""
    path = tmp_path / "all_data.json"
    monkeypatch.setattr(dp, "OUTPUT_JSON", str(path))
    return path


class TestCacheIsFresh:
    def test_false_when_missing(self, fake_output_json: Path) -> None:
        assert not fake_output_json.exists()
        assert dp.cache_is_fresh(str(fake_output_json)) is False

    def test_true_for_recent_file(self, fake_output_json: Path) -> None:
        fake_output_json.write_text("[]", encoding="utf-8")
        assert dp.cache_is_fresh(str(fake_output_json), max_age_seconds=60) is True

    def test_false_for_stale_file(self, fake_output_json: Path) -> None:
        fake_output_json.write_text("[]", encoding="utf-8")
        # Backdate mtime 10 days.
        old = time.time() - 10 * 24 * 3600
        os.utime(fake_output_json, (old, old))
        assert dp.cache_is_fresh(str(fake_output_json), max_age_seconds=7 * 24 * 3600) is False

    def test_honors_max_age_env_override(
        self, fake_output_json: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 1-second max_age env override makes any existing file go stale."""
        fake_output_json.write_text("[]", encoding="utf-8")
        # Backdate 5 seconds so a 1-second window is exceeded.
        past = time.time() - 5
        os.utime(fake_output_json, (past, past))
        monkeypatch.setenv("CACHE_MAX_AGE_SECONDS", "1")
        assert dp.cache_is_fresh(str(fake_output_json)) is False


class TestEnsureData:
    def test_cache_hit_skips_refresh(
        self,
        fake_output_json: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If OUTPUT_JSON is fresh, refresh_all should not be called."""
        fake_output_json.write_text(json.dumps([{"x": 1}]), encoding="utf-8")

        called: dict[str, Any] = {"refresh_all": 0}

        def _refuse_refresh(*args: Any, **kwargs: Any) -> None:
            called["refresh_all"] += 1
            raise AssertionError("refresh_all should not run on cache hit")

        monkeypatch.setattr(dp, "refresh_all", _refuse_refresh)

        path = dp.ensure_data(["IA"], 2020, 2020)
        assert path == str(fake_output_json)
        assert called["refresh_all"] == 0

    def test_cache_miss_triggers_refresh(
        self,
        fake_output_json: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No file on disk → refresh_all is called exactly once."""
        assert not fake_output_json.exists()

        called: dict[str, Any] = {"refresh_all_args": None}

        def _fake_refresh(states, start, end):  # noqa: ANN001
            called["refresh_all_args"] = (list(states), start, end)
            fake_output_json.write_text("[]", encoding="utf-8")

        monkeypatch.setattr(dp, "refresh_all", _fake_refresh)

        dp.ensure_data(["IA", "IL"], 2019, 2020)
        assert called["refresh_all_args"] == (["IA", "IL"], 2019, 2020)

    def test_force_bypasses_cache(
        self,
        fake_output_json: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """force=True refreshes even when a fresh cache exists."""
        fake_output_json.write_text("[]", encoding="utf-8")

        called: dict[str, Any] = {"count": 0}

        def _fake_refresh(states, start, end):  # noqa: ANN001
            called["count"] += 1

        monkeypatch.setattr(dp, "refresh_all", _fake_refresh)

        dp.ensure_data(["IA"], 2020, 2020, force=True)
        assert called["count"] == 1

    def test_defaults_to_constants_values(
        self,
        fake_output_json: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Calling with no args should pull ALL_STATES/START_YEAR/END_YEAR."""
        # Force cache miss so refresh_all is invoked.
        captured: dict[str, Any] = {}

        def _fake_refresh(states, start, end):  # noqa: ANN001
            captured["states"] = list(states)
            captured["start"] = start
            captured["end"] = end
            fake_output_json.write_text("[]", encoding="utf-8")

        monkeypatch.setattr(dp, "refresh_all", _fake_refresh)

        dp.ensure_data()

        from utils.constants import ALL_STATES, END_YEAR, START_YEAR

        assert captured["states"] == list(ALL_STATES)
        assert captured["start"] == START_YEAR
        assert captured["end"] == END_YEAR
