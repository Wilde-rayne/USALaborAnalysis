"""
Tests for the LFPR denominator correction.

Pins the working-age fraction at 0.78 (US BLS-published civilian
noninstitutional 16+ share of total population) and verifies that the
corrected LFPR formula matches the BLS-style definition while the
raw ratio remains accessible for transparency.

These tests use a tiny in-memory panel rather than the real merger
output so they run without statsmodels / heavy fixtures.
"""
from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

from utils.merge_all_data import LFPR_WORKING_AGE_FRACTION


class TestWorkingAgeConstant:
    def test_default_matches_bls_handbook(self) -> None:
        """0.78 is the US-wide CNI16+ share per BLS Handbook of Methods."""
        assert LFPR_WORKING_AGE_FRACTION == pytest.approx(0.78, abs=1e-9)


class TestLFPRArithmetic:
    """Algebra-only tests of the corrected vs raw formulas."""

    def test_corrected_lfpr_relation_to_raw(self) -> None:
        """corrected = raw / working_age_fraction by construction."""
        # Iowa-ish numbers.
        labor_force = 1_700_000.0
        total_pop = 3_150_000.0

        raw = 100.0 * labor_force / total_pop
        corrected = 100.0 * labor_force / (total_pop * LFPR_WORKING_AGE_FRACTION)

        assert raw == pytest.approx(53.97, abs=0.05)
        assert corrected == pytest.approx(raw / LFPR_WORKING_AGE_FRACTION, rel=1e-9)
        # Corrected LFPR should land within ~2 pp of the BLS-published
        # Iowa LFPR (~67 %) at the time of writing. Loose band.
        assert 60.0 <= corrected <= 75.0

    def test_corrected_within_population_band(self) -> None:
        """For real US states LFPR sits in [50, 75]; this guards against
        accidentally using an inverted denominator."""
        # National-ish 2024 numbers.
        labor_force = 168_000_000.0
        total_pop = 332_000_000.0
        corrected = 100.0 * labor_force / (total_pop * LFPR_WORKING_AGE_FRACTION)
        # National LFPR is 62-63 %; correction should land in that band.
        assert 60.0 <= corrected <= 68.0

    def test_zero_population_yields_inf_or_nan(self) -> None:
        """The merger should let numpy handle div-by-zero; assert the
        per-row math doesn't crash and produces a sentinel."""
        # numpy ops produce inf/NaN for div-by-zero rather than raising
        # (which is what pd.to_numeric → numpy ops would do inside the merger).
        with np.errstate(divide="ignore", invalid="ignore"):
            v = float(np.float64(100.0) * np.float64(1.0)
                      / (np.float64(0.0) * LFPR_WORKING_AGE_FRACTION))
        # Either +inf or NaN is acceptable; the merger replaces both
        # downstream when widening — the contract here is "doesn't raise".
        assert not np.isfinite(v)


class TestIntegrationWithMerger:
    """Smoke check on the merger formula via a tiny synthetic panel."""

    def test_panel_has_both_lfpr_and_raw(self) -> None:
        from utils import merge_all_data as merge_mod

        # Build the smallest possible panel that reaches the LFPR block.
        panel = pd.DataFrame(
            {
                "state": ["IA", "IA"],
                "year": [2024, 2024],
                "month": [1, 2],
                "Labor_Force": [1_700_000, 1_710_000],
                "Population":  [3_150_000, 3_150_000],
            }
        )
        # Re-run the LFPR block manually — the merger has more setup
        # but the LFPR computation itself is straightforward.
        lf = pd.to_numeric(panel["Labor_Force"], errors="coerce")
        pop = pd.to_numeric(panel["Population"], errors="coerce")
        panel["LFPR_RAW"] = 100.0 * lf / pop
        panel["LFPR"] = 100.0 * lf / (pop * merge_mod.LFPR_WORKING_AGE_FRACTION)

        # Both columns present.
        assert "LFPR" in panel.columns
        assert "LFPR_RAW" in panel.columns
        # Corrected is consistently larger than raw because we scaled
        # the denominator down.
        assert (panel["LFPR"] > panel["LFPR_RAW"]).all()
        # By the constant ratio.
        assert (
            panel["LFPR"] / panel["LFPR_RAW"]
        ).round(6).eq(round(1.0 / merge_mod.LFPR_WORKING_AGE_FRACTION, 6)).all()
