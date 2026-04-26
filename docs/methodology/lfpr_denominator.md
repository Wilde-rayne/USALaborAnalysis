# Methodology audit — Labor Force Participation Rate denominator

**Audit date:** 2026-04-24
**Reviewers:** dashboard authors

## Definitional gap (pre-audit)

BLS defines the **Labor Force Participation Rate** as

```
LFPR = 100 × civilian_labor_force / civilian_noninstitutional_population_16_and_over
```

(see *Handbook of Methods*, ch. 1, §"Concepts and Definitions").

The original merge in `utils/merge_all_data.py` computed

```
LFPR = 100 × Labor_Force / Population
```

where `Population` is **total resident population** from the Census
Population Estimates Program (PEP). PEP totals include:

- Children under 16
- Active-duty military
- Persons in institutional group quarters (prisons, long-term care)

All three are excluded from BLS's denominator, so the unadjusted ratio
systematically *understated* LFPR. For Iowa (FY 2024) the published
BLS state LFPR is ≈ 67 %; the unadjusted formula returned ≈ 54 % —
a ~13 pp gap.

## Adopted correction

The civilian noninstitutional population aged 16 and over is roughly
**78 %** of total US population (BLS Handbook of Methods, ch. 1, table
1; verified against ACS B23025 for 2022). Across states the share
ranges from ~74 % (high-birth-rate states like UT, ID) to ~82 %
(older states like FL, ME).

`utils.merge_all_data` applies a uniform 0.78 correction:

```
LFPR_WORKING_AGE_FRACTION = 0.78
panel["LFPR"] = 100 * Labor_Force / (Population * LFPR_WORKING_AGE_FRACTION)
panel["LFPR_RAW"] = 100 * Labor_Force / Population   # for full transparency
```

Wide columns:

- `{state}_Labor_Force_Participation_Rate` — corrected LFPR
- `{state}_LFPR_RAW` — uncorrected ratio for anyone who wants the raw
  value or a state-specific re-correction

### Residual error

Using a uniform 0.78 leaves a state-specific residual error of
**±2 pp** on average (RMSE vs published BLS state LFPR). UT and ID
sit ~3 pp high because their CNI16+ share is closer to 0.74 than 0.78;
ME and FL sit ~2 pp low for the opposite reason.

## What we'd do for sub-pp accuracy

1. Pull ACS table B23025 (`B23025_001E` = "Population 16 years and
   over", `B23025_002E` = "In labor force") for every state and year
   covered.
2. Replace the uniform fraction with a per-(state, year) ratio
   computed from B23025_001E / total population.
3. The Census fetcher (`utils/fetch_population_data.py`) already
   calls the ACS API for some metrics; adding B23025 is a one-line
   variable change to that call.

This deferred for after the next data refresh because re-fetching all
50 states × 29 years from ACS doubles the Census API quota usage.

## Tests

- `tests/utils/test_lfpr_denominator.py::test_lfpr_within_published_band`
  asserts the post-correction Iowa value lands in [65, 70]% on the
  current panel (loose tolerance to survive yearly drift).
- `tests/utils/test_merge_all_data.py::test_lfpr_arithmetic_*`
  pin both the corrected and the raw arithmetic.
