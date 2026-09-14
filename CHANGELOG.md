# Changelog

Notable changes per milestone. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Milestones are defined in `PLAN.md`; the reasoning behind design choices is in `docs/DECISIONS.md`.

## [Unreleased]

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
