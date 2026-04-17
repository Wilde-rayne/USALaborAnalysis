"""
Smoke tests: prove the project imports and basic invariants hold.

These exist to catch accidental breakage (syntax errors, import cycles,
renamed constants) before the heavier numerical/integration suites run.
"""
from __future__ import annotations

import pytest  # noqa: F401 — imported for TYPE_CHECKING-style annotations


def test_constants_module_imports() -> None:
    from utils import constants  # noqa: PLC0415 — intentional lazy import

    assert constants.START_YEAR < constants.END_YEAR
    assert constants.START_YEAR == 1996
    assert constants.END_YEAR == 2024


def test_state_list_shape() -> None:
    from utils import constants

    # 12 Midwest states per MILESTONE.md; all 2-letter USPS codes.
    assert len(constants.ALL_STATES) == 12
    assert all(isinstance(s, str) and len(s) == 2 and s.isupper() for s in constants.ALL_STATES)
    # No duplicates.
    assert len(set(constants.ALL_STATES)) == len(constants.ALL_STATES)


def test_supersector_list_matches_bls_taxonomy() -> None:
    from utils import constants

    # BLS CES publishes 9 supersectors at the state level (NAICS roll-up).
    assert len(constants.SUPERSECTORS) == 9
    # Canonical names include Manufacturing and Government.
    assert "Manufacturing" in constants.SUPERSECTORS
    assert "Government" in constants.SUPERSECTORS


def test_month_map_covers_all_bls_periods() -> None:
    from utils import constants

    # BLS period codes: M01..M12 (monthly) + A01 (annual average).
    assert set(constants.MONTH_MAP.keys()) == {
        "M01", "M02", "M03", "M04", "M05", "M06",
        "M07", "M08", "M09", "M10", "M11", "M12",
        "A01",
    }
    assert constants.MONTH_MAP["M01"] == "January"
    assert constants.MONTH_MAP["M12"] == "December"
    assert constants.MONTH_MAP["A01"] == "Annual"


def _cold_reimport_constants():
    """
    Drop any cached import of utils.constants (and its parent) so the next
    ``from utils import constants`` runs module-level code against the
    current os.environ. Necessary because ``from utils import constants``
    otherwise resolves against the already-loaded ``utils`` package attribute.
    """
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "utils" or mod_name.startswith("utils."):
            sys.modules.pop(mod_name, None)


def test_api_key_defaults_are_none_when_env_unset(clean_api_env) -> None:
    """Fresh import with no env vars must yield None for both keys."""
    _cold_reimport_constants()
    from utils import constants

    assert constants.BLS_API_KEY is None
    assert constants.CENSUS_API_KEY is None
    # Backward-compat alias mirrors BLS_API_KEY.
    assert constants.API_KEY is constants.BLS_API_KEY


def test_api_key_picked_up_from_env(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Setting env vars before import should populate the constants."""
    monkeypatch.setenv("BLS_API_KEY", "fake-bls-xxxxxxxx")
    monkeypatch.setenv("CENSUS_API_KEY", "fake-census-yyyyyyyy")

    _cold_reimport_constants()
    from utils import constants

    assert constants.BLS_API_KEY == "fake-bls-xxxxxxxx"
    assert constants.CENSUS_API_KEY == "fake-census-yyyyyyyy"
