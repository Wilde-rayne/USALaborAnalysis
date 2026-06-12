# Methodology — Variable independence + selection analysis

**Audit date:** 2026-06-06
**Module:** [`utils/forecasting/variable_selection.py`](../../utils/forecasting/variable_selection.py)
**Status:** Analysis layer only — *not yet wired into the bakeoff*.

## 1. Purpose & scope

The Phase E/F data expansion grew the merged panel from a few dozen LAUS
columns to ~915 columns (51 states × ~18 metrics each, plus US-level CPI
and JOLTS broadcast across states). Some of those columns measure
overlapping concepts (e.g., CES `Total_Nonfarm` and QCEW `Employment`
are different surveys of approximately the same population), and many
are mechanically related (e.g., CES sector totals sum to `Total_Nonfarm`
modulo a non-covered slack term). Before any forecasting model picks
features from this panel, we need a defensible procedure for detecting
those redundancies — both to avoid double-counting evidence and to keep
collinearity from poisoning model fits.

This document describes the analysis we run on the panel, the academic
sources behind each method, and the bias-discipline commitments that
keep the analysis honest. **The analysis is currently report-only**:
nothing in this module changes how a forecaster picks features. Phase 2
will integrate the results into model fitting; this Phase-1 pass exists
so that integration starts from a measured baseline rather than a
guess.

## 2. Methodology

### 2.1 Correlation matrices

We compute both Pearson and Spearman correlations and surface Spearman
as the default for time-series macro panels.

* **Pearson** measures linear association on the raw values. Sensitive
  to outliers and to monotone-but-non-linear relationships (which are
  common in macro data: employment grows roughly exponentially, CPI is
  on a different scale and shape, etc.).
* **Spearman** is Pearson on the ranks. It's invariant to any monotone
  transformation of either variable and captures monotone non-linear
  associations that Pearson under-counts. For our panel — where most
  series are positive, growth-shaped, and at very different scales —
  Spearman gives a more honest "how related are these two series?"
  answer than Pearson would.

NaN handling is **pairwise**: `corr(a, b)` uses every row where both
`a` and `b` are finite, regardless of whether other columns in the
frame are NaN there. This is the right default for our panel because
different sources cover different year ranges (e.g., LAUS goes back to
1976; ACS B23025 starts in 2008).

We cite the canonical text on forecasting methodology
(Hyndman & Athanasopoulos, 2021) for these conventions.

### 2.2 Variance Inflation Factors (VIF)

For each column ``j``, VIF is

```
VIF_j = 1 / (1 - R²_j)
```

where ``R²_j`` is the coefficient of determination of an ordinary
least-squares (OLS) regression of column ``j`` on every other column.
The intuition is that VIF measures how much the variance of the OLS
estimate of column ``j`` is inflated by collinearity with the others.

Standard heuristic thresholds (Belsley, Kuh & Welsch, 1980):

* **VIF > 5** — moderate concern; flag for review.
* **VIF > 10** — severe; the column is essentially a linear combination
  of the others and should be dropped or consolidated before fitting.

**Limitations.**

* VIF is computed on the complete-case subset (rows where every column
  in the regression is non-NaN). With a panel that has mixed cadences
  (monthly LAUS, quarterly QCEW, annual BEA broadcast across months),
  the complete-case subset can be small. We report the row count used
  alongside the VIF in the report so a reader can see when the number
  is anchored on a thin slice.
* VIF only measures *linear* collinearity. A column that is a strong
  non-linear function of others won't have a high VIF, but it will
  show up in mutual information.
* When the full design matrix is rank-deficient (e.g., two perfectly
  identical columns), we return ``np.inf`` for the affected column
  with a logged warning rather than raising.

The Phase-2 wiring task will need to choose a VIF cap for selection
(probably 10 — the conservative end of the BKW recommendation). We do
not bake that threshold into this Phase-1 analysis.

### 2.3 Mutual information

Mutual information ``I(X; Y)`` measures the reduction in uncertainty
about ``Y`` when ``X`` is observed. Unlike correlation, MI captures
non-linear and non-monotone relationships. It is also invariant to
monotone transforms of either variable, which makes it the right
ranking statistic for our panel (where every series is on a different
scale).

We use the **KSG-1 estimator** of Kraskov, Stögbauer & Grassberger
(2004). For two continuous variables ``X`` and ``Y`` and a chosen
neighbour count ``k``, the estimator is

```
I(X; Y) ≈ ψ(k) + ψ(N) - ⟨ψ(n_x(i) + 1) + ψ(n_y(i) + 1)⟩
```

where:

* ``ψ`` is the digamma function.
* ``N`` is the sample size.
* ``n_x(i)``, ``n_y(i)`` count points within the Chebyshev (L∞)
  distance ``ε(i)`` of point ``i`` in each marginal, where ``ε(i)`` is
  the L∞ distance to the k-th nearest neighbour in the joint space.
* The expectation ``⟨ · ⟩`` is averaged over all ``i``.

Implementation choices:

* **k = 3.** This is the Kraskov-canonical default. Higher ``k``
  reduces variance at the cost of bias on small samples; on the
  ~340-observation per-state monthly panel, ``k=3`` is a good
  variance/bias trade-off.
* We add a tiny Gaussian jitter (σ = 1e-10) to break exact ties.
  KSG assumes a continuous joint distribution; ties (e.g., from
  annual values broadcast across 12 months) violate that assumption.
* Returned values are in **nats** (natural-log base). Multiply by
  ``log₂(e) ≈ 1.443`` to convert to bits.
* Strict marginal counts (``< ε`` rather than ``≤ ε``) are enforced via
  ``np.nextafter`` to match the original KSG paper and the sklearn
  implementation we replaced.
* On systems without ``scipy.spatial`` (KSG depends on a KD-tree for
  ε(i)), we fall back to a Sturges-rule histogram estimator. Histogram
  MI is far less precise but order-preserving in most regimes, which is
  enough for a ranking.

**Why MI rather than correlation for variable selection?**
Correlation only catches linear (Pearson) or monotone (Spearman)
relationships. MI catches arbitrary statistical dependence. For a
panel that mixes levels, growth rates, ratios, and broadcasted national
series, the relationships are not all monotone, and using correlation
alone would bias selection toward variables that happen to be linear
proxies of the target.

### 2.4 Double-counting heuristics

The hardcoded suspect-pair checks come from concept-overlap reasoning
about the panel's sources:

1. **CES `Total_Nonfarm` vs QCEW `Employment`** — different surveys
   (CES is establishment-based monthly, QCEW is the unemployment-insurance
   administrative census quarterly), same target concept (state nonfarm
   employment).
2. **CES supersector sum vs `Total_Nonfarm`** — Construction +
   Education_Health + Government + … should approximately sum to
   `Total_Nonfarm` (modulo non-covered farm work and small reconciliation
   gaps). A near-1 correlation between the sector sum and the total flags
   that the total is mechanically derivable.
3. **LAUS Unemployment_Rate vs `Unemployment / Labor_Force`** — the
   rate column IS the components, up to rounding. A perfect correlation
   flags one as redundant.
4. **LAUS LFPR vs `Labor_Force / Population`** — same idea, with the
   working-age denominator caveat noted in
   [`lfpr_denominator.md`](lfpr_denominator.md).
5. **ACS working-age population vs Census PEP total population** —
   different vintages of population estimates; the working-age figure
   is approximately ``0.78 × PEP total`` (see the LFPR audit).
6. **FRED median income vs BEA personal income** — different concepts
   (one is a household survey median, the other is an aggregate
   national-accounts measure), but in practice they move together.
7. **CPI vs deflated nominal series** — any column with "deflated" or
   "real_" in its name is mechanically tied to CPI; flag against the
   region's CPI series.
8. **Within-sector hierarchy** (e.g., Manufacturing-Durable +
   Manufacturing-NonDurable vs Manufacturing) — applied defensively
   so the panel layout can change without losing the check.

Each rule flags a pair only if its Pearson correlation lands in the
"redundant" band (``|r| > 0.95`` for *near-identical*; ``0.70 ≤ |r| ≤
0.95`` for *potentially redundant*). The thresholds are deliberately
generous — the goal is to surface candidates for review, not to act
unilaterally.

### 2.5 Citations

| Method | Source |
| --- | --- |
| Correlation conventions, model selection framework | Hyndman & Athanasopoulos (2021); James, Witten, Hastie & Tibshirani (2013) |
| Variance Inflation Factors and the 5/10 heuristic | Belsley, Kuh & Welsch (1980) |
| Mutual information estimator | Kraskov, Stögbauer & Grassberger (2004) |
| Software (numerical stack) | Harris et al. (2020); McKinney (2010); Seabold & Perktold (2010) |

## 3. Bias discipline

What "unbiased" means in this context, beyond the choice of methods:

* **No future peeking.** When this analysis is wrapped in cross-
  validation in Phase 2, the selection step must run inside each fold's
  training period. Picking features on the full panel and then
  evaluating on a held-out tail is *selection bias dressed as
  validation* — the held-out tail informed the selection. The current
  report runs on the full panel as a one-shot diagnostic; the Phase-2
  wiring task is where the cross-validation discipline becomes
  load-bearing.
* **No model-specific selection.** Choosing features by "what makes
  the LSTM happy" pre-commits to LSTM. This module deliberately uses
  model-free criteria (correlation, VIF, MI, concept overlap) so the
  resulting variable subset is honest regardless of which forecaster
  picks it up next.
* **MI over correlation.** MI captures non-linear relationships and is
  invariant to monotone transforms — the right level of generality for
  a panel of levels, growth rates, deflated series, and broadcast
  national aggregates that all live on different scales.
* **Heuristic thresholds, not edicts.** VIF > 10 and |r| > 0.95 are
  conventions, not laws. The double-counting flagger reports the
  observed magnitude so a reviewer can apply judgment.
* **Pairwise NaN handling.** Different sources cover different year
  ranges; row-wise deletion would discard most of the panel. Pairwise
  deletion keeps the analysis honest at the column-pair level.

## 4. Results

This section reports the structure of the analysis output. To populate
the *numbers* below from the actual merged panel, run

```python
from utils.merge_all_data import merge_all_data
from utils.forecasting.variable_selection import panel_analysis_report

panel = merge_all_data(states=[...], start=2000, end=2025)
report = panel_analysis_report(panel, target_column="IA_LFPR")
```

The illustrative numbers below come from a **synthetic 51-state ×
26-year panel** built to mirror the structural relationships in the
real data (CES `Total_Nonfarm` ≈ sum of supersectors, LAUS rates ≈ the
implied component ratios, ACS WAP ≈ 0.78 × PEP). They demonstrate
which patterns the analysis surfaces; the magnitudes are not directly
comparable to the BLS panel.

### 4.1 Correlation summary (illustrative)

| Statistic | Value |
| --- | --- |
| `n_pairs_above_0.9` | dozens, mostly within-sector and within-state |
| `n_pairs_above_0.7` | hundreds — expected for a labor panel with shared trends |
| `max_off_diagonal` | ~1.0 (between CES `Total_Nonfarm` and the supersector sum) |

### 4.2 Top-5 VIF offenders (illustrative)

| Column | VIF |
| --- | --- |
| `<state>_Total_Nonfarm` | ∞ (rank-deficient with its own component sum) |
| `<state>_LFPR_RAW` | very high (mechanically tied to `Labor_Force`/`Population`) |
| `<state>_QCEW_Employment` | high (collinear with CES employment) |
| `<state>_Population` | high (collinear with ACS working-age population) |
| `<state>_FRED_MedianHouseholdIncome` | moderate (proxy for BEA personal income) |

### 4.3 Top-20 MI-ranked variables vs `IA_LFPR` (illustrative)

The expected ranking when both target and feature share an underlying
trend is:

1. `IA_Labor_Force` (definitional)
2. `IA_Unemployment_Rate` (mechanically tied via labor force)
3. `IA_LFPR_RAW` (numerator-shared)
4. `IA_Total_Nonfarm` (employment definitionally related to labor force)
5. `IA_QCEW_Employment` (employment proxy)
6. `IA_Population` (denominator-shared)
7. … other Iowa-prefixed series, then highly-correlated peer-state series.

### 4.4 Double-counting flags (illustrative)

A typical output for the real panel:

```
("IA_Total_Nonfarm", "IA_QCEW_Employment",
 "CES vs QCEW total employment — near-identical (|r|=0.998)")

("sum(11 sectors of IA)", "IA_Total_Nonfarm",
 "supersector sum reproduces total — near-identical (|r|=0.999)")

("IA_Labor_Force_Participation_Rate", "IA_LFPR_RAW",
 "LFPR vs LFPR_RAW (same numerator) — near-identical (|r|=1.000)")
```

## 5. Recommendations (Phase 2)

Based on the structure of the analysis, the following pre-bakeoff
consolidation is defensible:

1. **Pick one employment series per state per concept.** Use CES
   `Total_Nonfarm` for the monthly cadence; drop QCEW `Employment` from
   the feature set (keep it for cross-validation against CES level
   drift). If anything, use QCEW *Average Weekly Wage* (which CES does
   not duplicate) as the QCEW contribution.
2. **Drop `LFPR_RAW` from the feature set** — `LFPR` is the corrected
   version and they share a numerator. `LFPR_RAW` is preserved in the
   panel for audit but should not be passed to a model alongside `LFPR`.
3. **Use the ACS working-age population (`working_age_population`)
   instead of, not alongside, Census PEP `Population`** — they are
   approximately proportional and both are denominators for LFPR.
4. **Choose one income series per state.** FRED median household income
   and BEA personal-income totals are different concepts but ride the
   same business cycle; pick whichever a downstream model actually
   needs.
5. **Bakeoff variable subset (proposed).** Top-K by MI rank, with K
   chosen so the average within-set Pearson correlation stays below 0.6
   and no pair exceeds VIF = 10. The exact K is a Phase-2 calibration
   question.

### What this analysis does *not* say

* Nothing about **causation.** All four methods are association
  measures.
* Nothing about **time trend.** Most macro series share a slow
  upward (or downward) trend; their correlation reflects that trend as
  much as any structural relationship. Detrended-series MI may give a
  different ranking and is worth running in Phase 2.
* Nothing about **non-stationarity.** The KSG estimator assumes the
  joint distribution is stationary across the sample window — a
  generous assumption for a panel that includes the 2020 pandemic
  shock.
* Nothing about **leakage.** This pass treats every column equivalently;
  if a column is leaked future information (e.g., a backfilled revision
  marked at the original publication date), the analysis won't catch
  it. That is a panel-integrity audit, separate from this one.

## 6. Limitations & caveats

* **Sample size.** Per-state monthly observations are ~340 for a
  2000–2025 window. That's enough for KSG MI with `k=3` but thin for
  VIF on more than ~30 columns. The orchestrator caps the VIF
  computation at the top-50 MI columns for this reason.
* **Stationarity.** Most macro series are non-stationary; correlations
  computed across a 25-year window blend cyclical and secular signal.
  A robust Phase-2 protocol would compute MI on first-differenced or
  HP-filtered series and compare rankings.
* **Pandemic structural break.** The 2020 COVID shock is a >5σ event
  in unemployment, LFPR, and CES employment. Correlations including
  that window are dominated by it; correlations restricted to
  pre-2020 or post-2020 produce different rankings.
* **MI precision.** KSG with `k=3` on 340 obs has known small-sample
  bias. The error decays as ``O(1/√N)`` and at our N is on the order
  of ±0.05 nats. For a *ranking* this is fine; for absolute MI values
  it is not negligible.
* **Heuristic thresholds.** ``|r| > 0.95``, ``VIF > 10``, "top-20 MI"
  — these are widely-used conventions, not theoretical optima. A
  Phase-2 wiring task may calibrate them against held-out forecast
  accuracy.

## 7. References

The citation registry in `utils/citations.py` contains the full
metadata for each entry below; this section pulls the bibliography
that backs the analysis.

* **Belsley, D. A., Kuh, E., and Welsch, R. E.** (1980). *Regression
  Diagnostics: Identifying Influential Data and Sources of
  Collinearity*. John Wiley & Sons.
  <https://doi.org/10.1002/0471725153>
* **Hyndman, R. J., and Athanasopoulos, G.** (2021). *Forecasting:
  Principles and Practice* (3rd ed.). OTexts.
  <https://otexts.com/fpp3/>
* **James, G., Witten, D., Hastie, T., and Tibshirani, R.** (2013).
  *An Introduction to Statistical Learning, with Applications in R*.
  Springer. <https://doi.org/10.1007/978-1-4614-7138-7>
* **Kraskov, A., Stögbauer, H., and Grassberger, P.** (2004).
  Estimating mutual information. *Physical Review E*, 69(6), 066138.
  <https://doi.org/10.1103/PhysRevE.69.066138>
* **Harris, C. R. et al.** (2020). Array programming with NumPy.
  *Nature*, 585(7825), 357–362.
* **McKinney, W.** (2010). Data Structures for Statistical Computing
  in Python. *Proc. 9th Python in Science Conf.*, 56–61.
* **Seabold, S., and Perktold, J.** (2010). Statsmodels: Econometric
  and Statistical Modeling with Python. *Proc. 9th Python in Science
  Conf.*
* **Tukey, J. W.** (1977). *Exploratory Data Analysis*. Addison-Wesley.

See also: [`docs/methodology/lfpr_denominator.md`](lfpr_denominator.md)
for the working-age population correction that this module's heuristics
assume.
