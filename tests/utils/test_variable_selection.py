"""
Tests for :mod:`utils.forecasting.variable_selection`.

Pure-Python, deterministic synthetic fixtures: no fetching, no real
panel needed. We exercise:

* ``compute_correlation_matrix`` — known structure (independent,
  correlated, perfectly collinear) returns the expected entries; NaN
  handling is pairwise; method validation.
* ``compute_variance_inflation_factors`` — perfectly collinear pair
  → ∞; independent block → ≈1; rank-deficient input → ∞ across the
  board; single-column edge case returns 1.0.
* ``mutual_information_ranking`` — perfect copy ranks above noisy copy
  ranks above an independent series; constant columns → 0; basic
  invariance to monotone transforms.
* ``identify_potential_double_counting`` — CES sum-of-sectors fixture
  fires; LFPR-vs-components fixture fires; unrelated columns do not.
* ``panel_analysis_report`` — synthetic 2-state 2-year panel returns a
  well-formed summary dict with a non-empty MI top list.

Property tests use ``hypothesis`` for two invariants that should hold
on any well-formed input:

* The correlation matrix is symmetric and has unit diagonal (where
  the column is non-constant).
* VIF is always ≥ 1 for a well-conditioned design.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from utils.forecasting.variable_selection import (
    _mi_histogram,
    _mi_knn_continuous,
    compute_correlation_matrix,
    compute_variance_inflation_factors,
    identify_potential_double_counting,
    mutual_information_ranking,
    panel_analysis_report,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def known_structure_frame() -> pd.DataFrame:
    """Three columns: independent, monotone-correlated, perfectly collinear.

    * ``x``       — i.i.d. N(0, 1)
    * ``x_twice`` — ``2 * x`` (perfectly collinear)
    * ``x_noisy`` — ``x + N(0, 0.5)`` (highly but not perfectly correlated)
    * ``z``       — independent N(0, 1)
    """
    rng = np.random.default_rng(seed=42)
    n = 200
    x = rng.standard_normal(n)
    return pd.DataFrame(
        {
            "x":       x,
            "x_twice": 2.0 * x,
            "x_noisy": x + rng.normal(scale=0.5, size=n),
            "z":       rng.standard_normal(n),
        }
    )


@pytest.fixture
def small_panel() -> pd.DataFrame:
    """2-state × 2-year × 12-month panel with a handful of synthetic columns."""
    rng = np.random.default_rng(seed=7)
    rows = []
    for state_code in ("IA", "IL"):
        for y in (2020, 2021):
            for m in range(1, 13):
                rows.append({"state": state_code, "year": y, "month": m})
    df = pd.DataFrame(rows)
    n = len(df)
    # Per-state synthetic columns.
    df["IA_LFPR"] = rng.normal(size=n) + 65.0
    df["IA_Employment"] = rng.normal(size=n) + 1_500_000
    df["IA_QCEW_Employment"] = df["IA_Employment"] + rng.normal(scale=100, size=n)
    df["IA_Population"] = rng.normal(size=n) + 3_200_000
    df["IL_LFPR"] = rng.normal(size=n) + 64.0
    df["IL_Employment"] = rng.normal(size=n) + 6_200_000
    return df


# --------------------------------------------------------------------------
# compute_correlation_matrix
# --------------------------------------------------------------------------
class TestCorrelationMatrix:
    def test_perfect_collinearity_gives_unit_correlation(
        self, known_structure_frame: pd.DataFrame
    ) -> None:
        corr = compute_correlation_matrix(known_structure_frame, method="pearson")
        assert corr.loc["x", "x_twice"] == pytest.approx(1.0, abs=1e-9)

    def test_independent_columns_correlate_near_zero(
        self, known_structure_frame: pd.DataFrame
    ) -> None:
        corr = compute_correlation_matrix(known_structure_frame, method="pearson")
        assert abs(corr.loc["x", "z"]) < 0.15

    def test_diagonal_is_one(self, known_structure_frame: pd.DataFrame) -> None:
        corr = compute_correlation_matrix(known_structure_frame)
        np.testing.assert_allclose(np.diag(corr.values), 1.0)

    def test_invalid_method_raises(
        self, known_structure_frame: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError):
            compute_correlation_matrix(known_structure_frame, method="kendall")

    def test_non_numeric_columns_dropped(self) -> None:
        df = pd.DataFrame(
            {"num": [1.0, 2.0, 3.0], "txt": ["a", "b", "c"]}
        )
        corr = compute_correlation_matrix(df)
        assert list(corr.columns) == ["num"]

    def test_pairwise_nan_handling(self) -> None:
        """NaN in some rows of one column should not poison other pairs."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [2.0, 4.0, 6.0, 8.0, 10.0],
                "c": [1.0, np.nan, np.nan, 4.0, 5.0],
            }
        )
        corr = compute_correlation_matrix(df, method="pearson")
        # a vs b: perfect monotone, all 5 rows used.
        assert corr.loc["a", "b"] == pytest.approx(1.0)
        # a vs c: 3 rows used, still monotone.
        assert corr.loc["a", "c"] == pytest.approx(1.0)

    @given(
        n=st.integers(min_value=20, max_value=80),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=10)
    def test_symmetric_diagonal_one(self, n: int, seed: int) -> None:
        """Property: corr matrix symmetric, diagonal = 1, |r| ≤ 1."""
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(rng.standard_normal((n, 4)),
                          columns=list("abcd"))
        corr = compute_correlation_matrix(df, method="pearson")
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-9)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-9)
        assert (corr.abs() <= 1.0 + 1e-9).all().all()


# --------------------------------------------------------------------------
# compute_variance_inflation_factors
# --------------------------------------------------------------------------
class TestVIF:
    def test_perfectly_collinear_pair_returns_infinity(
        self, known_structure_frame: pd.DataFrame
    ) -> None:
        # x and x_twice: VIF should be ∞ for both — column j fully
        # explained by column k. Use just the collinear pair.
        df = known_structure_frame[["x", "x_twice"]]
        vifs = compute_variance_inflation_factors(df)
        assert vifs["x"] == np.inf
        assert vifs["x_twice"] == np.inf

    def test_independent_columns_vif_near_one(self) -> None:
        rng = np.random.default_rng(seed=1)
        df = pd.DataFrame(
            rng.standard_normal((300, 4)), columns=list("abcd")
        )
        vifs = compute_variance_inflation_factors(df)
        assert (vifs < 2).all()

    def test_single_column_returns_one(self) -> None:
        df = pd.DataFrame({"only": [1.0, 2.0, 3.0]})
        vifs = compute_variance_inflation_factors(df)
        assert vifs["only"] == pytest.approx(1.0)

    def test_fewer_rows_than_columns_returns_inf(self) -> None:
        # 3 obs, 5 columns: cannot fit OLS.
        rng = np.random.default_rng(seed=2)
        df = pd.DataFrame(
            rng.standard_normal((3, 5)), columns=list("abcde")
        )
        vifs = compute_variance_inflation_factors(df)
        assert (vifs == np.inf).all()

    def test_non_numeric_column_dropped_with_warning(self) -> None:
        rng = np.random.default_rng(seed=3)
        df = pd.DataFrame(
            {
                "a": rng.standard_normal(50),
                "b": rng.standard_normal(50),
                "txt": ["x"] * 50,
            }
        )
        with pytest.warns(UserWarning, match="non-numeric"):
            vifs = compute_variance_inflation_factors(df, columns=["a", "b", "txt"])
        assert "txt" not in vifs.index

    def test_vif_at_least_one(self) -> None:
        """VIF ≥ 1 always — R² ∈ [0, 1] so 1/(1-R²) ≥ 1."""
        rng = np.random.default_rng(seed=4)
        df = pd.DataFrame(
            rng.standard_normal((100, 3)), columns=list("xyz")
        )
        vifs = compute_variance_inflation_factors(df)
        assert (vifs >= 1.0 - 1e-9).all()


# --------------------------------------------------------------------------
# mutual_information_ranking
# --------------------------------------------------------------------------
class TestMIRanking:
    def test_perfect_copy_ranks_highest(self) -> None:
        rng = np.random.default_rng(seed=5)
        n = 500
        y = rng.standard_normal(n)
        df = pd.DataFrame(
            {
                "target":  y,
                "perfect": y,                            # full information
                "noisy":   y + rng.normal(scale=0.5, size=n),
                "indep":   rng.standard_normal(n),
            }
        )
        ranking = mutual_information_ranking(df, target_column="target")
        # perfect > noisy > indep is the expected order.
        assert ranking.index[0] == "perfect"
        assert ranking["perfect"] > ranking["noisy"] > ranking["indep"]

    def test_constant_column_scores_zero(self) -> None:
        df = pd.DataFrame(
            {"target": [1.0, 2.0, 3.0, 4.0, 5.0], "const": [7.0] * 5}
        )
        ranking = mutual_information_ranking(df, target_column="target")
        assert ranking["const"] == 0.0

    def test_target_excluded_from_ranking(self) -> None:
        rng = np.random.default_rng(seed=6)
        df = pd.DataFrame(
            {
                "target": rng.standard_normal(100),
                "other":  rng.standard_normal(100),
            }
        )
        ranking = mutual_information_ranking(df, target_column="target")
        assert "target" not in ranking.index
        assert "other" in ranking.index

    def test_missing_target_raises(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(KeyError):
            mutual_information_ranking(df, target_column="missing")

    def test_non_numeric_target_raises(self) -> None:
        df = pd.DataFrame({"target": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})
        with pytest.raises(TypeError):
            mutual_information_ranking(df, target_column="target")

    def test_monotone_transform_invariance_directionally(self) -> None:
        """A monotone transform shouldn't tank MI ranking position."""
        rng = np.random.default_rng(seed=9)
        n = 400
        y = rng.standard_normal(n)
        df = pd.DataFrame(
            {
                "target": y,
                "linear": 2 * y + 1,
                "expd":   np.exp(y),  # monotone
            }
        )
        ranking = mutual_information_ranking(df, target_column="target")
        # Both should score well above an independent series. Use a
        # within-frame sanity check rather than an absolute threshold.
        assert ranking["linear"] > 0.5
        assert ranking["expd"] > 0.5


class TestMIHistogramFallback:
    def test_histogram_estimator_returns_nonnegative(self) -> None:
        rng = np.random.default_rng(seed=10)
        x = rng.standard_normal(100)
        y = x.copy()
        mi = _mi_histogram(x, y)
        assert mi > 0.0

    def test_knn_estimator_returns_zero_on_singleton(self) -> None:
        x = np.array([1.0])
        y = np.array([2.0])
        assert _mi_knn_continuous(x, y) == 0.0


# --------------------------------------------------------------------------
# identify_potential_double_counting
# --------------------------------------------------------------------------
class TestDoubleCounting:
    def test_ces_sum_of_sectors_fires(self) -> None:
        """A panel whose sum of supersectors matches Total_Nonfarm must flag."""
        rng = np.random.default_rng(seed=11)
        n = 100
        # Build sub-sectors and a matching total.
        sectors = {
            "IA_Construction":                rng.uniform(10, 50, n),
            "IA_Education_Health_Services":   rng.uniform(80, 120, n),
            "IA_Government":                  rng.uniform(40, 60, n),
            "IA_Manufacturing":               rng.uniform(70, 100, n),
        }
        df = pd.DataFrame(sectors)
        df["IA_Total_Nonfarm"] = df.sum(axis=1)
        flags = identify_potential_double_counting(df)
        reasons = [reason for _, _, reason in flags]
        assert any("supersector sum" in r for r in reasons), reasons

    def test_ces_vs_qcew_employment_fires_when_correlated(self) -> None:
        rng = np.random.default_rng(seed=12)
        n = 100
        base = rng.uniform(1_000_000, 2_000_000, n)
        df = pd.DataFrame(
            {
                "IA_Total_Nonfarm":   base,
                "IA_QCEW_Employment": base + rng.normal(scale=500, size=n),
            }
        )
        flags = identify_potential_double_counting(df)
        # Should flag because correlation will be ≈ 1.
        joined = [(a, b) for a, b, _ in flags]
        assert ("IA_Total_Nonfarm", "IA_QCEW_Employment") in joined

    def test_lfpr_self_consistency_fires(self) -> None:
        rng = np.random.default_rng(seed=13)
        n = 60
        lf = rng.uniform(800_000, 1_200_000, n)
        pop = rng.uniform(2_500_000, 3_500_000, n)
        # LFPR = 100 * lf/pop * (1/0.78); but the heuristic checks the
        # raw ratio, so just provide an LFPR proportional to lf/pop.
        ratio = 100.0 * lf / pop
        df = pd.DataFrame(
            {
                "IA_Labor_Force_Participation_Rate": ratio,
                "IA_Labor_Force": lf,
                "IA_Population": pop,
            }
        )
        flags = identify_potential_double_counting(df)
        assert any("LFPR is its own components" in r for _, _, r in flags)

    def test_no_flags_on_unrelated_columns(self) -> None:
        rng = np.random.default_rng(seed=14)
        df = pd.DataFrame(
            {
                "year":  np.arange(2000, 2020),
                "month": np.arange(1, 21),
                "weird_a": rng.standard_normal(20),
                "weird_b": rng.standard_normal(20),
            }
        )
        flags = identify_potential_double_counting(df)
        assert flags == []

    def test_empty_frame_returns_empty(self) -> None:
        flags = identify_potential_double_counting(pd.DataFrame())
        assert flags == []


# --------------------------------------------------------------------------
# panel_analysis_report
# --------------------------------------------------------------------------
class TestPanelAnalysisReport:
    def test_synthetic_panel_returns_well_formed_dict(
        self, small_panel: pd.DataFrame
    ) -> None:
        report = panel_analysis_report(small_panel, target_column="IA_LFPR")
        assert isinstance(report, dict)
        assert set(report.keys()) >= {
            "shape",
            "n_states",
            "year_range",
            "correlation_summary",
            "vif_summary",
            "mi_top_20",
            "double_counting_flags",
        }
        assert report["shape"] == small_panel.shape
        assert report["n_states"] == 2
        assert report["year_range"] == (2020, 2021)
        # MI list should be non-empty (we have ≥4 numeric features beyond target).
        assert len(report["mi_top_20"]) >= 1
        # Each MI tuple is (str, float).
        for col, score in report["mi_top_20"]:
            assert isinstance(col, str)
            assert isinstance(score, float)
            assert score >= 0.0

    def test_missing_target_gracefully_skips_mi(
        self, small_panel: pd.DataFrame
    ) -> None:
        report = panel_analysis_report(
            small_panel, target_column="not_in_panel"
        )
        assert report["mi_top_20"] == []
        # VIF skipped because no MI to anchor on.
        assert report["vif_summary"]["computed"] is False

    def test_correlation_summary_counts(self) -> None:
        rng = np.random.default_rng(seed=15)
        n = 100
        x = rng.standard_normal(n)
        # Three highly-correlated columns + one independent — expect
        # multiple |r| > 0.9 pairs.
        df = pd.DataFrame(
            {
                "year":  [2020] * n,
                "month": list(range(1, n + 1)),
                "state": ["IA"] * n,
                "a": x,
                "b": x + rng.normal(scale=0.01, size=n),
                "c": x * 2,
                "d": rng.standard_normal(n),
                # Target column for the MI step.
                "IA_LFPR": x,
            }
        )
        report = panel_analysis_report(df, target_column="IA_LFPR")
        assert report["correlation_summary"]["n_pairs_above_0.9"] >= 3
