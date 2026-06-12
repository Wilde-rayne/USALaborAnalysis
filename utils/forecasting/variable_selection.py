"""
Variable independence + double-counting analysis for the wide panel.

This module is **analysis-only**: nothing here modifies how a forecaster
picks features. It exists so a researcher (or a Phase 2 wiring task) can
ask, "which of the ~915 columns produced by the Phase E/F merger are
redundant, collinear, or already implied by another column?" and get a
defensible numeric answer.

Public surface
--------------
compute_correlation_matrix
    Pearson or Spearman correlation over numeric columns, with pairwise
    NaN handling. Spearman is the default for time-series macro panels
    because it is rank-based, less sensitive to outliers, and captures
    monotone non-linear relationships.

compute_variance_inflation_factors
    Classical VIF (Belsley, Kuh & Welsch, 1980): for column ``j``,
    ``VIF_j = 1 / (1 - R²_j)`` where ``R²_j`` is the coefficient of
    determination of an OLS of column ``j`` on every other column. The
    standard heuristic thresholds are ``> 5`` (moderate concern) and
    ``> 10`` (severe).

mutual_information_ranking
    KNN estimator of mutual information (Kraskov, Stögbauer & Grassberger,
    2004) between each feature and a chosen target column. MI is the
    preferred ranking statistic for the bias-discipline reasons explained
    in :mod:`docs.methodology.variable_selection` — it captures
    non-linear monotone or non-monotone relationships and is invariant to
    monotone transforms.

identify_potential_double_counting
    Heuristic-driven flagger that checks the panel's known suspect pairs
    (CES vs QCEW employment, LAUS-derived rates vs raw components,
    sector-hierarchy sums, etc.) and reports any with correlation above
    ``|r| > 0.95`` ("near-identical") or in the 0.70–0.95 band
    ("potentially redundant").

panel_analysis_report
    Orchestrator that runs all of the above against the merged panel and
    returns a structured summary dictionary suitable for serialization
    into the methodology markdown.

Design notes
------------
* All functions are pure: no IO at import, no logging side effects.
* NaN handling is pairwise (correlation / MI) or row-wise dropped (VIF).
* ``hypothesis`` and ``statsmodels`` are the only non-numpy/non-pandas
  dependencies; both are already pinned in ``requirements.txt``.
* The KNN MI estimator is implemented from scratch using ``scipy.spatial``
  for the KDTree; it does not require ``sklearn`` (deliberately removed
  in Track A).

References
----------
* Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). *Regression
  Diagnostics: Identifying Influential Data and Sources of
  Collinearity*. Wiley.
* Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). "Estimating
  mutual information." *Physical Review E*, 69(6), 066138.
* James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An
  Introduction to Statistical Learning*. Springer.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------
def compute_correlation_matrix(
    df: pd.DataFrame, method: str = "spearman"
) -> pd.DataFrame:
    """Compute a numeric-column correlation matrix with pairwise NaN deletion.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide-format frame. Non-numeric columns are silently dropped before
        the correlation is computed; constant or all-NaN columns are
        retained but their row/column will be NaN.
    method : {"pearson", "spearman"}, optional
        Correlation flavour. ``"spearman"`` (default) is rank-based and
        captures monotone non-linear relationships — recommended for the
        macro-economic panels this module is built for. ``"pearson"`` is
        the standard linear correlation, included so callers can compare
        the two on the same dataset.

    Returns
    -------
    pandas.DataFrame
        Symmetric ``n × n`` matrix indexed and columned by the numeric
        columns of ``df``. The diagonal is 1.0 for any column with at
        least one non-NaN value, and NaN for all-NaN columns. NaN
        handling is pairwise: ``corr(a, b)`` uses every row where both
        ``a`` and ``b`` are finite.

    Raises
    ------
    ValueError
        If ``method`` is not ``"pearson"`` or ``"spearman"``.

    Notes
    -----
    Spearman is the default because most series in the labor panel are
    growth/level indicators that are monotone but not linear in each
    other (e.g., total nonfarm employment vs. CPI). Pearson would
    underestimate that association; Spearman captures it.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
    >>> compute_correlation_matrix(df, method="pearson").loc["a", "b"]
    1.0
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(
            f"method must be 'pearson' or 'spearman', got {method!r}"
        )
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return pd.DataFrame()
    # pandas' ``corr`` already does pairwise NaN deletion.
    return numeric.corr(method=method)


# --------------------------------------------------------------------------
# Variance Inflation Factors
# --------------------------------------------------------------------------
def compute_variance_inflation_factors(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.Series:
    """Compute classical Variance Inflation Factors for numeric columns.

    For each column ``j``, the VIF is ``1 / (1 - R²_j)`` where ``R²_j`` is
    the coefficient of determination of an OLS regression of column
    ``j`` on every other column in ``columns``. Standard heuristic
    thresholds (Belsley, Kuh & Welsch, 1980) treat ``VIF > 5`` as a
    moderate concern and ``VIF > 10`` as severe.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide frame containing the columns to score.
    columns : list of str, optional
        Subset of columns to compute VIF over. Defaults to all numeric
        columns. Non-numeric columns in the list are dropped with a
        warning.

    Returns
    -------
    pandas.Series
        VIF score per column, indexed by column name. A perfectly
        collinear column (``R² → 1``) returns ``np.inf``; a column with
        no linear relationship to any other returns ``1.0``.

    Notes
    -----
    Rows containing any NaN in the analyzed columns are dropped before
    the OLS — pairwise deletion would make the regressors disagree on
    sample size and yield incomparable VIFs. The result is sensitive to
    that drop: if many rows are dropped, the reported VIF reflects only
    the complete-case subset.

    For rank-deficient or near-singular cases ``statsmodels`` raises;
    we catch this and return ``np.inf`` for the affected column with a
    logged warning so downstream code can keep iterating.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "x": rng.normal(size=100),
    ...     "y": rng.normal(size=100),
    ...     "z": rng.normal(size=100),
    ... })
    >>> vifs = compute_variance_inflation_factors(df)
    >>> (vifs < 2).all()
    True
    """
    from statsmodels.stats.outliers_influence import (
        variance_inflation_factor,
    )

    if columns is None:
        columns = list(df.select_dtypes(include=[np.number]).columns)
    else:
        bad = [c for c in columns if c not in df.columns]
        if bad:
            warnings.warn(
                f"VIF: columns not in frame, dropped: {bad}", stacklevel=2
            )
            columns = [c for c in columns if c in df.columns]
        non_numeric = [
            c for c in columns
            if not np.issubdtype(df[c].dtype, np.number)
        ]
        if non_numeric:
            warnings.warn(
                f"VIF: non-numeric columns dropped: {non_numeric}",
                stacklevel=2,
            )
            columns = [c for c in columns if c not in non_numeric]

    if len(columns) < 2:
        # VIF is only meaningful with ≥2 regressors; a single column has
        # no "others" to regress on. Return 1.0 (no inflation) by
        # convention.
        return pd.Series({c: 1.0 for c in columns}, dtype=float)

    sub = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.empty or sub.shape[0] < len(columns) + 1:
        # Cannot fit OLS with fewer observations than regressors.
        return pd.Series({c: np.inf for c in columns}, dtype=float)

    # Add a constant for the intercept; the standard VIF formula assumes
    # one. statsmodels' variance_inflation_factor expects the column
    # index in the design matrix.
    design = sub.to_numpy(dtype=float)
    design_with_const = np.column_stack([np.ones(design.shape[0]), design])

    vifs: dict[str, float] = {}
    for j, col in enumerate(columns, start=1):  # j=0 is the constant
        try:
            with warnings.catch_warnings():
                # statsmodels emits a RuntimeWarning when R² is exactly 1
                # (division by zero in 1 / (1 - 1)). We translate that
                # into np.inf below; no need to surface the raw warning.
                warnings.simplefilter("ignore")
                v = variance_inflation_factor(design_with_const, j)
        except Exception as exc:  # noqa: BLE001 — never let one column kill the loop
            logger.warning(
                f"[varsel] VIF for {col!r} failed: {exc.__class__.__name__}: {exc}"
            )
            v = np.inf
        if not np.isfinite(v):
            v = np.inf
        vifs[col] = float(v)

    return pd.Series(vifs, dtype=float)


# --------------------------------------------------------------------------
# Mutual Information
# --------------------------------------------------------------------------
def _mi_knn_continuous(
    x: np.ndarray, y: np.ndarray, k: int = 3
) -> float:
    """KSG-1 estimator of mutual information for two continuous variables.

    Implements equation (8) of Kraskov, Stögbauer & Grassberger (2004),
    the first KSG estimator, with k-NN distances measured in the
    Chebyshev (L∞) metric. Returns MI in nats.

    Parameters
    ----------
    x, y : numpy.ndarray
        1-D float arrays of equal length, NaN-free.
    k : int, optional
        Number of nearest neighbours. Higher k reduces variance at the
        cost of bias. Three is the canonical choice in Kraskov 2004 and
        a good default for n in the low thousands.

    Returns
    -------
    float
        Estimated mutual information in nats. May go slightly negative on
        small samples; clipped to 0 for stability in downstream rankings.
    """
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    n = x.size
    if n <= k + 1:
        return 0.0

    # Add tiny noise to break ties — KSG assumes a continuous joint
    # distribution. Empirically negligible at 1e-10 scale.
    rng = np.random.default_rng(0)
    x_jit = x + 1e-10 * rng.standard_normal(n)
    y_jit = y + 1e-10 * rng.standard_normal(n)

    xy = np.column_stack([x_jit, y_jit])
    # Chebyshev metric: max(|Δx|, |Δy|).
    tree_xy = cKDTree(xy)
    # Query k+1 neighbours: the point itself + k true neighbours.
    dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    # eps_i = distance to the k-th true neighbour (column k).
    eps = dists[:, k]

    # Strictly-less-than-eps counts in marginals, per KSG eq. 8.
    # cKDTree.query_ball_point uses inclusive radius, so we step the
    # radius one ULP toward 0 to enforce strict inequality (matching
    # sklearn's mutual_info_regression and the KSG paper). The "-1" at
    # the end of each count removes the self-match (point i is at
    # distance 0 from itself, so it's always inside the ball).
    tree_x = cKDTree(x_jit.reshape(-1, 1))
    tree_y = cKDTree(y_jit.reshape(-1, 1))
    eps_strict = np.nextafter(eps, 0.0)
    n_x = np.array(
        [
            len(tree_x.query_ball_point([x_jit[i]], r=eps_strict[i], p=np.inf)) - 1
            for i in range(n)
        ]
    )
    n_y = np.array(
        [
            len(tree_y.query_ball_point([y_jit[i]], r=eps_strict[i], p=np.inf)) - 1
            for i in range(n)
        ]
    )
    # Clip to 0: with strict inequality, n_x / n_y can be 0 when the
    # k-th NN is the only neighbour at that radius. digamma(0+1)=digamma(1)
    # is finite, but a negative count would crash digamma.
    n_x = np.maximum(n_x, 0)
    n_y = np.maximum(n_y, 0)
    # KSG eq. 8: I = ψ(k) + ψ(N) - <ψ(n_x+1) + ψ(n_y+1)>
    mi = (
        digamma(k)
        + digamma(n)
        - float(np.mean(digamma(n_x + 1) + digamma(n_y + 1)))
    )
    # Floor at 0: MI is non-negative; KSG estimator can dip slightly
    # below on small samples due to variance.
    return max(0.0, float(mi))


def _mi_histogram(x: np.ndarray, y: np.ndarray) -> float:
    """Sturges-rule histogram estimator of mutual information.

    Used as a fallback when the KNN estimator cannot run (e.g., scipy
    unavailable). Far less precise than KNN, but order-preserving in
    most regimes and adequate for rough rankings. Returns MI in nats.

    Parameters
    ----------
    x, y : numpy.ndarray
        1-D float arrays, equal length, NaN-free.

    Returns
    -------
    float
        Estimated mutual information in nats, floored at 0.
    """
    n = x.size
    if n < 2:
        return 0.0
    bins = max(2, int(np.ceil(np.log2(n) + 1)))  # Sturges' rule
    p_xy, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = p_xy / p_xy.sum()
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    # MI = sum p_xy * log(p_xy / (p_x p_y)), skipping zero cells.
    mask = p_xy > 0
    if not mask.any():
        return 0.0
    pxpy = p_x @ p_y
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mask, p_xy / pxpy, 1.0)
        terms = np.where(mask, p_xy * np.log(ratio), 0.0)
    return max(0.0, float(terms.sum()))


def mutual_information_ranking(
    df: pd.DataFrame,
    target_column: str,
    discrete_features: str = "auto",
    k: int = 3,
) -> pd.Series:
    """Rank columns by mutual information with a target column.

    Uses the KSG-1 KNN estimator (Kraskov, Stögbauer & Grassberger, 2004)
    by default — robust to non-linear, non-monotone associations, and
    invariant to monotone transforms of either variable. Falls back to a
    Sturges-rule histogram estimator if ``scipy.spatial`` is unavailable.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide frame.
    target_column : str
        Name of the column to rank against. Must be numeric and present
        in ``df``.
    discrete_features : {"auto"}, optional
        Reserved for future API compatibility with sklearn's
        ``mutual_info_regression``. Currently only ``"auto"`` is
        supported and all features are treated as continuous (the macro
        panel has effectively no discrete columns).
    k : int, optional
        Nearest-neighbour count for the KNN estimator. Three is the
        Kraskov-canonical default.

    Returns
    -------
    pandas.Series
        MI score (nats) per feature column, sorted descending. The
        target column itself is omitted from the result.

    Notes
    -----
    NaN handling is pairwise: for each ``(feature, target)`` pair, rows
    where either value is non-finite are dropped before the estimate.
    Constant columns return ``0.0`` (MI with anything is zero).

    The KNN estimator is more accurate than a histogram approach for
    continuous variables and avoids the binning-choice headache, but is
    O(n log n) per pair. For ~340 obs (per-state monthly panel) and ~900
    features this is comfortable on a laptop in <1 minute.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> y = rng.normal(size=n)
    >>> df = pd.DataFrame({
    ...     "target": y,
    ...     "perfect": y,
    ...     "noisy":   y + rng.normal(scale=0.5, size=n),
    ...     "indep":   rng.normal(size=n),
    ... })
    >>> ranking = mutual_information_ranking(df, target_column="target")
    >>> ranking.index[0]
    'perfect'
    >>> ranking["indep"] < ranking["noisy"]
    True
    """
    if discrete_features != "auto":
        raise NotImplementedError(
            f"discrete_features={discrete_features!r} not yet supported"
        )
    if target_column not in df.columns:
        raise KeyError(f"target_column {target_column!r} not in df.columns")

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if target_column not in numeric_cols:
        raise TypeError(
            f"target_column {target_column!r} must be numeric, "
            f"got {df[target_column].dtype}"
        )
    feature_cols = [c for c in numeric_cols if c != target_column]

    target = pd.to_numeric(df[target_column], errors="coerce").to_numpy()

    # Try the KNN path first; if scipy is missing, fall back to the
    # histogram estimator (lower precision, but no extra dependency).
    try:
        from scipy.spatial import cKDTree  # noqa: F401

        def estimator(x: np.ndarray, y: np.ndarray) -> float:
            return _mi_knn_continuous(x, y, k=k)
    except Exception:  # noqa: BLE001 — broad: any import issue → fallback
        logger.info(
            "[varsel] scipy.spatial unavailable; falling back to histogram MI"
        )
        estimator = _mi_histogram

    scores: dict[str, float] = {}
    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(target)
        if mask.sum() < 5:
            scores[col] = 0.0
            continue
        x_clean = x[mask]
        y_clean = target[mask]
        # Constant column has zero MI with anything.
        if np.allclose(x_clean.std(), 0.0) or np.allclose(y_clean.std(), 0.0):
            scores[col] = 0.0
            continue
        try:
            scores[col] = estimator(x_clean, y_clean)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[varsel] MI({col!r}) failed: {exc.__class__.__name__}: {exc}"
            )
            scores[col] = 0.0

    return pd.Series(scores, dtype=float).sort_values(ascending=False)


# --------------------------------------------------------------------------
# Double-counting heuristics
# --------------------------------------------------------------------------
# Hardcoded suspect pairs. Each entry is (pattern_a, pattern_b, reason).
# Patterns are substrings searched against column names; the heuristic
# checks every (col_a, col_b) where col_a matches pattern_a and col_b
# matches pattern_b and they share the same state prefix where present.
_CES_SECTORS_OF_TOTAL = (
    "Construction",
    "Education_Health_Services",
    "Financial_Activities",
    "Government",
    "Information",
    "Leisure_Hospitality",
    "Manufacturing",
    "Mining_and_Logging",
    "Other_Services",
    "Professional_Business_Services",
    "Trade_Transportation_Utilities",
)


def _pearson_pairwise(a: pd.Series, b: pd.Series) -> float | None:
    """Pearson correlation with pairwise NaN deletion; None if too few obs."""
    pair = pd.concat([a, b], axis=1).dropna()
    if pair.shape[0] < 5:
        return None
    a_arr = pd.to_numeric(pair.iloc[:, 0], errors="coerce").to_numpy()
    b_arr = pd.to_numeric(pair.iloc[:, 1], errors="coerce").to_numpy()
    # Joint finite mask so the two arrays remain the same length.
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    a_arr = a_arr[mask]
    b_arr = b_arr[mask]
    if a_arr.size < 5:
        return None
    if a_arr.std() == 0 or b_arr.std() == 0:
        return None
    return float(np.corrcoef(a_arr, b_arr)[0, 1])


def _state_prefixes(columns: list[str]) -> list[str]:
    """Extract candidate state prefixes (2-letter codes) from column names.

    A column like ``IA_Total_Nonfarm`` has prefix ``IA``. Non-prefixed
    columns (``year``, ``month``, etc.) are ignored.
    """
    prefixes: set[str] = set()
    for col in columns:
        if "_" in col:
            head = col.split("_", 1)[0]
            if len(head) == 2 and head.isupper() and head.isalpha():
                prefixes.add(head)
    return sorted(prefixes)


def identify_potential_double_counting(
    df: pd.DataFrame, panel_schema: dict[str, Any] | None = None
) -> list[tuple[str, str, str]]:
    """Flag potentially redundant column pairs in the merged labor panel.

    Heuristics applied (each rooted in a known concept overlap):

    1. **CES total nonfarm vs QCEW total employment**: different sources,
       same concept (covered employment in the state economy).
    2. **CES employment by supersector sum-of-parts vs total**: the
       sectors sum to Total_Nonfarm (with non-covered slack); flag if
       the sum's correlation with Total_Nonfarm is in the suspect band.
    3. **LAUS unemployment rate vs unemployed/labor_force**: should be
       mechanically identical up to rounding.
    4. **LAUS LFPR vs labor_force/working_age_population**: same.
    5. **ACS working-age population vs Census PEP total population**:
       overlapping vintages of related concepts.
    6. **FRED median income vs BEA personal income**: proxy-related but
       different concepts; flag if the correlation is suspiciously high.
    7. **CPI vs deflated FRED nominal series**: any deflated column is
       mechanically related to CPI.
    8. **CES Manufacturing-Durable + Manufacturing-NonDurable vs
       Manufacturing**: within-sector hierarchy. (Skipped if the panel
       does not carry the sub-totals — the default sectors registry
       does not, but defensive coding for the future case.)

    Parameters
    ----------
    df : pandas.DataFrame
        Merged panel.
    panel_schema : dict, optional
        Reserved for future structured-schema-driven checks (not yet
        used; passed-through verbatim for forward-compat).

    Returns
    -------
    list of (str, str, str)
        Tuples of ``(col_a, col_b, reason)``. ``reason`` includes the
        correlation magnitude when flagging on correlation strength.
        Empty list if nothing crosses the suspect threshold.
    """
    del panel_schema  # reserved
    flags: list[tuple[str, str, str]] = []
    cols = list(df.columns)
    prefixes = _state_prefixes(cols)

    def _classify(r: float) -> str | None:
        if abs(r) > 0.95:
            return f"near-identical (|r|={abs(r):.3f})"
        if 0.70 <= abs(r) <= 0.95:
            return f"potentially redundant (|r|={abs(r):.3f})"
        return None

    # 1. CES Total_Nonfarm vs QCEW_Employment — per state.
    for st in prefixes:
        a = f"{st}_Total_Nonfarm"
        b = f"{st}_QCEW_Employment"
        if a in cols and b in cols:
            r = _pearson_pairwise(df[a], df[b])
            if r is not None and (cls := _classify(r)):
                flags.append(
                    (a, b, f"CES vs QCEW total employment — {cls}")
                )

    # 2. CES supersector sum vs Total_Nonfarm — per state.
    for st in prefixes:
        total = f"{st}_Total_Nonfarm"
        if total not in cols:
            continue
        present_sectors = [
            f"{st}_{sector}"
            for sector in _CES_SECTORS_OF_TOTAL
            if f"{st}_{sector}" in cols
        ]
        if len(present_sectors) < 2:
            continue
        # Build sum-of-sectors series, NaN if any contributor is NaN.
        sector_sum = df[present_sectors].sum(axis=1, min_count=len(present_sectors))
        r = _pearson_pairwise(sector_sum, df[total])
        if r is not None and (cls := _classify(r)):
            flags.append(
                (
                    f"sum({len(present_sectors)} sectors of {st})",
                    total,
                    f"supersector sum reproduces total — {cls}",
                )
            )

    # 3. LAUS unemployment_rate consistency: if a state ships both a
    #    Unemployment_Rate and the components (Unemployment, Labor_Force),
    #    compute the implied rate and flag if they agree very tightly.
    for st in prefixes:
        rate = f"{st}_Unemployment_Rate"
        unemp = f"{st}_Unemployment"
        lf = f"{st}_Labor_Force"
        if rate in cols and unemp in cols and lf in cols:
            implied = 100.0 * pd.to_numeric(df[unemp], errors="coerce") / pd.to_numeric(
                df[lf], errors="coerce"
            )
            r = _pearson_pairwise(implied, df[rate])
            if r is not None and (cls := _classify(r)):
                flags.append(
                    (
                        rate,
                        f"{unemp} / {lf}",
                        f"LAUS unemployment_rate is its own components — {cls}",
                    )
                )

    # 4. LFPR vs Labor_Force / Population — per state. (Two flavours:
    #    LFPR vs raw-ratio LFPR_RAW, and LFPR vs computed-from-components.)
    for st in prefixes:
        lfpr = f"{st}_Labor_Force_Participation_Rate"
        lfpr_raw = f"{st}_LFPR_RAW"
        lf = f"{st}_Labor_Force"
        pop = f"{st}_Population"
        if lfpr in cols and lfpr_raw in cols:
            r = _pearson_pairwise(df[lfpr], df[lfpr_raw])
            if r is not None and (cls := _classify(r)):
                flags.append(
                    (lfpr, lfpr_raw, f"LFPR vs LFPR_RAW (same numerator) — {cls}")
                )
        if lfpr in cols and lf in cols and pop in cols:
            implied = 100.0 * pd.to_numeric(df[lf], errors="coerce") / pd.to_numeric(
                df[pop], errors="coerce"
            )
            r = _pearson_pairwise(implied, df[lfpr])
            if r is not None and (cls := _classify(r)):
                flags.append(
                    (
                        lfpr,
                        f"{lf} / {pop}",
                        f"LFPR is its own components (modulo working-age scale) — {cls}",
                    )
                )

    # 5. ACS working_age_population vs Census Population — per state.
    for st in prefixes:
        for wap_name in (
            "working_age_population",
            f"{st}_working_age_population",
        ):
            if wap_name in cols and f"{st}_Population" in cols:
                r = _pearson_pairwise(df[wap_name], df[f"{st}_Population"])
                if r is not None and (cls := _classify(r)):
                    flags.append(
                        (
                            wap_name,
                            f"{st}_Population",
                            f"ACS WAP vs Census PEP total — {cls}",
                        )
                    )
                break

    # 6. FRED median income vs BEA personal income — per state.
    for st in prefixes:
        for fred_key in ("FRED_MHI", "FRED_MedianHouseholdIncome"):
            for bea_key in ("BEA_SAINC1_3", "BEA_SAINC1_1"):
                a = f"{st}_{fred_key}"
                b = f"{st}_{bea_key}"
                if a in cols and b in cols:
                    r = _pearson_pairwise(df[a], df[b])
                    if r is not None and (cls := _classify(r)):
                        flags.append(
                            (
                                a,
                                b,
                                f"FRED household income vs BEA personal income — {cls}",
                            )
                        )

    # 7. CPI vs any deflated-looking column. Heuristic: column names
    #    containing "deflated" or "real_" alongside a CPI column for the
    #    same state.
    for st in prefixes:
        cpi = f"{st}_CPI_AllItems"
        if cpi not in cols:
            continue
        deflated_like = [
            c for c in cols
            if c.startswith(f"{st}_") and ("deflated" in c.lower() or "_real_" in c.lower())
        ]
        for d in deflated_like:
            r = _pearson_pairwise(df[d], df[cpi])
            if r is not None and (cls := _classify(r)):
                flags.append(
                    (d, cpi, f"deflated series mechanically tied to CPI — {cls}")
                )

    # 8. Manufacturing-Durable + Manufacturing-NonDurable vs Manufacturing.
    for st in prefixes:
        total_mfg = f"{st}_Manufacturing"
        dur = f"{st}_Manufacturing_Durable"
        nondur = f"{st}_Manufacturing_NonDurable"
        if total_mfg in cols and dur in cols and nondur in cols:
            implied = pd.to_numeric(df[dur], errors="coerce") + pd.to_numeric(
                df[nondur], errors="coerce"
            )
            r = _pearson_pairwise(implied, df[total_mfg])
            if r is not None and (cls := _classify(r)):
                flags.append(
                    (
                        f"{dur} + {nondur}",
                        total_mfg,
                        f"manufacturing sub-totals sum to total — {cls}",
                    )
                )

    return flags


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
def panel_analysis_report(
    df: pd.DataFrame, target_column: str = "IA_LFPR"
) -> dict[str, Any]:
    """Run the full independence + double-counting analysis on a panel.

    Composes :func:`compute_correlation_matrix`,
    :func:`compute_variance_inflation_factors`,
    :func:`mutual_information_ranking`, and
    :func:`identify_potential_double_counting` into a single structured
    summary suitable for serialisation into the methodology markdown.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged labor panel. Must have ``state``, ``year``, ``month``
        columns when those summaries are wanted; the rest of the
        analysis tolerates their absence.
    target_column : str, optional
        Column to use as the target for the MI ranking. Default is
        ``"IA_LFPR"`` (Iowa LFPR), the panel's working forecast target.
        Falls back gracefully if the column is absent.

    Returns
    -------
    dict
        Dictionary with the following keys::

            shape: (n_rows, n_cols)
            n_states: int
            year_range: (min_year, max_year)
            correlation_summary:
                max_off_diagonal: float
                n_pairs_above_0.9: int
                n_pairs_above_0.7: int
            vif_summary:
                computed: bool
                max: float
                n_above_5: int
                n_above_10: int
                top_offenders: list[(col, vif)]
            mi_top_20: list[(col, mi)]
            double_counting_flags: list[(col_a, col_b, reason)]

    Notes
    -----
    For large panels (~900 columns), VIF is computed only on the top-50
    columns by MI rank — a full 900-column OLS is O(n²) memory and
    rarely informative across that many regressors. If MI is unavailable
    (no target), VIF is skipped (``vif_summary["computed"]`` is False).

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "state": ["IA"] * 24,
    ...     "year":  [2020]*12 + [2021]*12,
    ...     "month": list(range(1,13))*2,
    ...     "IA_LFPR": rng.normal(size=24),
    ...     "IA_Employment": rng.normal(size=24),
    ... })
    >>> report = panel_analysis_report(df)
    >>> "double_counting_flags" in report
    True
    """
    summary: dict[str, Any] = {
        "shape": tuple(df.shape),
    }
    summary["n_states"] = (
        int(df["state"].nunique()) if "state" in df.columns else 0
    )
    if "year" in df.columns:
        years = pd.to_numeric(df["year"], errors="coerce").dropna()
        if not years.empty:
            summary["year_range"] = (int(years.min()), int(years.max()))
        else:
            summary["year_range"] = (0, 0)
    else:
        summary["year_range"] = (0, 0)

    # Correlation summary.
    corr = compute_correlation_matrix(df, method="spearman")
    if corr.empty or corr.shape[0] < 2:
        summary["correlation_summary"] = {
            "max_off_diagonal": None,
            "n_pairs_above_0.9": 0,
            "n_pairs_above_0.7": 0,
        }
    else:
        c_arr = corr.to_numpy(dtype=float, copy=True)
        np.fill_diagonal(c_arr, np.nan)
        abs_arr = np.abs(c_arr)
        # Upper triangle (k=1) only, to avoid counting each pair twice.
        ut_mask = np.triu(np.ones_like(c_arr, dtype=bool), k=1)
        ut = abs_arr[ut_mask]
        ut = ut[np.isfinite(ut)]
        summary["correlation_summary"] = {
            "max_off_diagonal": float(np.nanmax(abs_arr))
            if np.isfinite(abs_arr).any()
            else None,
            "n_pairs_above_0.9": int((ut > 0.9).sum()),
            "n_pairs_above_0.7": int((ut > 0.7).sum()),
        }

    # MI ranking against the chosen target (if present). Keep the full
    # ranking around: the report surfaces the top-20, but VIF below uses
    # the top-50 (its documented computational cap).
    mi_top: list[tuple[str, float]] = []
    mi_vif_cols: list[str] = []
    if target_column in df.columns:
        try:
            mi = mutual_information_ranking(df, target_column=target_column)
            mi_top = [(c, float(v)) for c, v in mi.head(20).items()]
            mi_vif_cols = list(mi.head(50).index)
        except Exception as exc:  # noqa: BLE001 — never crash the report
            logger.warning(
                f"[varsel] MI ranking failed: {exc.__class__.__name__}: {exc}"
            )
    summary["mi_top_20"] = mi_top

    # VIF on the top-50 MI columns (computational cap; full-panel VIF is
    # rarely informative and quadratically expensive).
    vif_summary: dict[str, Any] = {
        "computed": False,
        "max": None,
        "n_above_5": 0,
        "n_above_10": 0,
        "top_offenders": [],
    }
    if mi_vif_cols:
        top_cols = [c for c in mi_vif_cols if c in df.columns]
        if len(top_cols) >= 2:
            try:
                vifs = compute_variance_inflation_factors(df, columns=top_cols)
                vif_summary["computed"] = True
                vif_summary["max"] = (
                    float(vifs.max()) if np.isfinite(vifs).any() else float("inf")
                )
                vif_summary["n_above_5"] = int((vifs > 5).sum())
                vif_summary["n_above_10"] = int((vifs > 10).sum())
                vif_summary["top_offenders"] = [
                    (c, float(v))
                    for c, v in vifs.sort_values(ascending=False).head(5).items()
                ]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"[varsel] VIF failed: {exc.__class__.__name__}: {exc}"
                )
    summary["vif_summary"] = vif_summary

    # Double-counting heuristics.
    summary["double_counting_flags"] = identify_potential_double_counting(df)

    return summary
