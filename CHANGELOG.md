# Changelog

Notable changes per milestone. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Milestones are defined in `PLAN.md`; the reasoning behind design choices is in `docs/DECISIONS.md`.

## [Unreleased]

### M2 — bootstrap confidence intervals — 2026-09-21

Added
- `faircheck.bootstrap`: percentile confidence intervals for every rate, gap and contrast.
  The resampling uses the multinomial identity from D4 — a within-group row bootstrap *is* a
  `Multinomial(n_g, cells_g / n_g)` draw over the four confusion cells — so the cost is
  `O(n_boot × 4 × n_groups)` and does not depend on the number of rows. Measured: **2.5 ms
  for B = 1000 at n = 100,000, and 2.7 ms at n = 10,000,000**.
- Both resampling schemes: `stratified` (group sizes fixed, the default and the right
  conditioning for per-group rates) and `full` (dataset resampled as a whole, matching what
  Fairlearn's `MetricFrame` bootstrap does).
- Signed contrasts against a reference group, with `has_clear_contrast` returning
  `True` / `False` / `None`. This, not the max−min gap, is what supports evidence claims —
  see the new D11, which is the most important correctness decision since D3.
- `rate_undefined_fraction`: when a small group loses some resamples to zero positives, the
  interval is still reported, beside the proportion of resamples in which the metric did not
  exist. Permanently undefined rates get a NaN interval and an undefined fraction of 1.0.
- 46 tests comparing FairCheck against Fairlearn (`MetricFrame`,
  `demographic_parity_difference`, `demographic_parity_ratio`, `equalized_odds_difference`)
  across seeds, group counts, and the intersectional case.
- 29 bootstrap tests: exactness against a naive row-resampling reference (two-sample KS plus
  means and spreads), agreement with the closed-form Wilson interval for a single
  proportion, simulated coverage near the nominal 95%, reproducibility, undefined-metric
  handling, argument validation, and a guard that cost does not grow with row count.

Notes
- Test count 79 → 154; coverage of `faircheck/` 97% → 98%, with `bootstrap.py` at 100%.
- `scipy` pinned explicitly in `requirements-dev.txt`: the bootstrap tests import it directly
  for the KS test rather than relying on it arriving via Fairlearn.

### M1 — skeleton, validation, metrics, CI — 2026-09-21

Added
- `faircheck.types`: fixed `(TP, FP, FN, TN)` cell order, `ColumnMapping`,
  `ValidatedData`, `GroupCounts`. Group keys are tuples, so the marginal and
  intersectional views share one code path (D8).
- `faircheck.metrics`: vectorised confusion counting (single `bincount` over
  group×cell pairs), the seven per-group rates, the five summary gaps, and the
  `RATES` / `GAPS` registries that carry each metric's label, formula, reference and
  input rates so the report and UI never restate them.
- `faircheck.validate`: column-mapping checks, positive-label resolution across dtypes,
  score range checks, identifier-like sensitive column rejection, seeded subsampling
  above the row cap, and a warning when the supplied predictions are not a single
  threshold on the supplied score.
- Test suite (79 tests, 97% coverage of `faircheck/`): the hand-computed case from the
  specification, algebraic identities including Chouldechova's, edge cases for undefined
  rates, a flipped positive label, intersectional grouping, validation copy, and an
  architectural guard that the core imports no UI framework and no Fairlearn.
- GitHub Actions CI on Python 3.12 and 3.13 running ruff, ruff format, mypy and pytest.
- `AGENTS.md`, `docs/DECISIONS.md` and this changelog, so a future session can pick the
  project up without re-deriving the research.

Notes
- Undefined rates are `NaN` and render as `n/a`; Fairlearn returns `0.0` in the same
  situation and the divergence is pinned by a test (D3).
- CI covers Python 3.12 and 3.13 rather than the planned 3.11–3.13: `numpy` 2.5.x
  requires Python ≥ 3.12.
