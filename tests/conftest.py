"""
Repo-wide pytest fixtures and configuration.

Keep this file intentionally small. Domain-specific fixtures (sample BLS
responses, golden data-pipeline output, etc.) live next to the tests that
use them under tests/<area>/conftest.py.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture
def clean_api_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Strip BLS/Census credentials from the environment for a single test
    and stub out python-dotenv so a stray ``.env`` anywhere up the tree
    can't silently repopulate them.
    """
    import dotenv

    for var in ("BLS_API_KEY", "CENSUS_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # python-dotenv walks up from cwd looking for a .env file; in a dev
    # checkout one exists and may contain legacy keys. Neutralize it for
    # the duration of the test.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
