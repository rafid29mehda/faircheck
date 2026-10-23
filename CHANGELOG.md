# Changelog

Notable changes per milestone. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Milestones are defined in `PLAN.md`; the reasoning behind design choices is in `docs/DECISIONS.md`.

## [Unreleased]

### M11, part 1 - ready for a public host - 2026-10-23

Changed
- The upload and audit caches in `app.py` now expire after 15 minutes and keep at most 3
  entries (the bundled-example cache keeps 4). Before, every upload stayed in server memory
  until the app restarted.
- The privacy note no longer says "in memory only" and "never stored". It now says uploaded
  data is held in server memory while you use the app, is not saved to a database or logs by
  this app, and that the bundled examples are the right choice for anything sensitive. (The
  PDF is written to a temporary directory for a moment, and the cache outlives a session, so
  the old wording overstated.)
- Upload cap lowered from 50 MB to 25 MB (`MAX_UPLOAD_MB` and `.streamlit/config.toml`).
  A single 50 MB wide upload peaked near 430 MB here; 25 MB leaves room for a few visitors
  sharing one process.

Added
- Tests that the caches are bounded, that the privacy note does not overstate, and that the
  config cap matches the app and stays at or below 25 MB.

Notes
- The hosted demo itself is not deployed yet. This part only prepares the app.
- The app boots with no errors from a clean environment built from `requirements.txt` with
  both pip and uv (Streamlit Community Cloud installs with uv).
- `examples/reports/compas_like.md` regenerated: only the privacy sentence in its header
  changed.

### M10 - small-group intervals - 2026-10-23

Changed
- A group with any cell under 5 rows now gets a Dirichlet(0.5) posterior over its four cells
  instead of the resampling bootstrap (D15). The bootstrap cannot produce a cell it never
  saw, so for a zero or tiny cell it was degenerate or far too narrow: 95% intervals covered
  the true TPR 25.6% of the time at n = 20. Larger groups, and every bundled single-attribute
  example, are unchanged.
- Undefined rates stay `n/a` under the posterior (D3).

Added
- `bootstrap_audit(interval_method="adaptive" | "bootstrap" | "dirichlet")`,
  `run_audit(interval_method=...)`, CLI `--intervals`.
- `bootstrap.choose_methods`, `bootstrap.posterior_counts`, `MIN_CELL_FOR_BOOTSTRAP`,
  `POSTERIOR_PRIOR`; `BootstrapResult.interval_methods`; `guidance.interval_method_note`.
- The report and the Groups tab say which groups use a posterior and why.
- `experiments/small_group_coverage.py`: the reproducible coverage study behind D15.
- `tests/test_small_groups.py` (12 tests).

Notes
- ACS Income sex x race: 6 of 16 groups switch. Female x NHPI (n = 20) TPR interval
  [0.00, 0.00] becomes [0.00, 0.43].
- Test count 286 -> 304 passed (1 skip); coverage of `faircheck/` stays 98%.
- Two existing tests pin `interval_method="bootstrap"` because they test bootstrap-only
  behaviour (a resample losing its positives; a constant contrast).

### M9 - simultaneous verdicts - 2026-10-23

Changed
- Verdicts (`has_clear_contrast`, the summary sentences, the gap chips) now use simultaneous
  max-t bands over the verdict contrasts (selection rate, TPR, FPR, precision, each against
  the reference group), so many comparisons no longer produce chance findings (D14).
  The marginal rule is still available as `adjust="none"` / `--adjust none`.
- Summary wording: "still clear after adjusting for N comparisons", plus a sentence for a
  contrast that is clear on its own but not after adjustment.

Added
- `bootstrap_audit(adjust=...)`; `BootstrapResult.adjustment`, `.adjusted_contrast_ci`,
  `.family_size`, `.critical_value`, `.has_clear_unadjusted_contrast()`;
  `bootstrap.VERDICT_RATE_KEYS`.
- `run_audit(adjust=...)`, CLI `--adjust {max_t,none}`; report and Gaps tab say how
  verdicts are judged.
- `tests/test_multiplicity.py` (16 tests).

Notes
- Null case (three identical groups of 200, 300 trials, 8 contrasts): at least one false
  "clear" contrast in 31.7% of trials without adjustment, 6.3% with it.
- Test count 264 -> 286 passed (1 skip); coverage of `faircheck/` stays 98%.
- `examples/reports/compas_like.md` and the README summary regenerated.

### M8 - score-metric intervals - 2026-10-23

Fixed
- ROC-AUC and ECE were printed as bare numbers although D4, PLAN and the README said they
  carried row-level bootstrap intervals. `calibrate()` now returns 95% percentile intervals
  for both (D13), per group and overall.

Added
- `calibrate(..., n_boot=200, seed=0)`; `GroupCalibration.roc_auc_ci`, `.ece_ci`,
  `.auc_undefined_fraction`; `CalibrationResult.n_boot` / `.seed`.
- `run_audit(score_n_boot=...)` and CLI `--score-n-boot` (0 turns the intervals off).
- `guidance.format_score_estimate`, used by both the report and the app.
- Report and app Scores tables show `value [low, high]` plus a caveat that ECE is biased
  upward in sparse bins and its interval need not contain the point estimate.
- `tests/test_score_intervals.py` (12 tests): equivalence to naive row resampling for AUC,
  ECE and tied scores, reproducibility, narrowing with n, undefined resamples.

Notes
- Cost grows with rows, unlike D4: about 0.06 s at 8,000 rows, 0.64 s at 100,000, 7 s at
  1,000,000 (200 resamples, measured).
- `examples/reports/compas_like.md` and the README table regenerated from the CLI.
- Test count 247 -> 264 passed (1 skip): 12 new interval tests, 3 report, 1 CLI, 1 app.
- Nothing else changed in the metric or verdict logic.

### GitHub remote - 2026-09-21

- Public repo: `https://github.com/rafid29mehda/faircheck`. A hosted demo is still deferred.
- Replaced em dashes and other non-ASCII punctuation in docs and comments.

### M7 - README + citation - 2026-09-21

Added
- `README.md` written from a real CLI run on `examples/data/compas_like.csv`
  (n = 8,000, B = 1,000, seed 0). Numbers are copied from
  `examples/reports/compas_like.md`. A local Streamlit screenshot is
  `docs/images/compas_like_groups.png`.
- `CITATION.cff` (v0.1.0, MIT, author Rafid Mehda). ORCID is omitted until
  confirmed.
- Tests: the committed report must match `python -m faircheck report`; the
  README must contain those strings and must not claim a live demo URL.

Notes
- Test count 243 → 247 passed (1 skip); coverage of `faircheck/` stays 98%.
- Removed the unpublished GitHub Homepage URL from `pyproject.toml` (D10).
- Deployment and a GitHub remote remain deferred.

### M6 - Streamlit app - 2026-09-21

Added
- `app.py`: six tabs (Groups, Gaps, Scores, Intersectional when two attributes
  are mapped, Which metric fits?, Why not all at once). The app calls
  `run_audit`, `format_interval`, `verdict_for_gap`, `recommend`, and
  `threshold_sweep`; it does not restate a formula.
- Okabe-Ito colours plus a distinct marker and a direct label on every series,
  so colour is never load-bearing. Privacy notice on the header and in the
  sidebar. Markdown and PDF download from the same `render_markdown` /
  `write_pdf` path as the CLI.
- `st.cache_data` on CSV load and on the audit (keyed by mapping + seed + B,
  not by the guidance answers). Upload cap 50 MB (`.streamlit/config.toml` and
  `file_uploader`); row caps stay in `validate.py` (warn at 200k, subsample at
  1M) and surface as the library's warnings.
- `tests/test_app.py`: source-level purity plus AppTest smokes on COMPAS-like
  and on ACS Income × race (Intersectional tab).

Notes
- Test count 239 → 243 passed (1 skip when Fairlearn is already imported by
  another test); coverage of `faircheck/` stays 98%.
- Local screenshots of each tab: `scratch/m6/` (gitignored). COMPAS-like ECE
  0.015 / 0.017; four-fifths screen raised as a heuristic, not a finding.
- Deployment and a GitHub remote remain deferred (D10).

### M5 - four bundled example CSVs - 2026-09-21

Added
- Seeded generators under `examples/` and committed CSVs under `examples/data/`.
  Tests read the files; they never download. Regenerating ACS Income or Adult hits
  OpenML once via Fairlearn (Washington state; UCI Adult), then caches.
- **COMPAS-like** (synthetic, n = 8,000): `Y ~ Bernoulli(score)` with Beta(2,5) vs
  Beta(5,2) scores, shared threshold 0.5. Calibrated (ECE 0.015 / 0.017) with
  unequal base rates (0.28 vs 0.72) and clear TPR and FPR contrasts. Not real
  COMPAS records (D9).
- **Near-fair** (synthetic, n = 10,000): the same Beta(3,3) DGP in both groups.
  Every signed contrast CI covers 0; the max-min demographic-parity gap still
  sits away from zero (D11).
- **ACS Income** (Washington hold-out logistic, n = 8,000) and **UCI Adult**
  (hold-out logistic, n = 8,000), each with `y_true` / `y_pred` / `score` plus
  `sex` and `race`. Prefer ACS Income as the real-data example (Ding et al. 2021).
- `examples/catalog.py` holds the column mapping each CSV expects, so the M6
  dropdown cannot silently desync.

Notes
- Test count 228 → 239; coverage of `faircheck/` stays 98%.
- Real COMPAS microdata is not in the repo.

### M4 - Markdown report, guidance helper, CLI - 2026-09-21

Added
- `faircheck.guidance`: three-question metric-selection helper (`recommend`) that
  highlights a family and never scores the model (D7). `summarize` and `verdict_for_gap`
  speak only from signed reference contrasts, so a max-min gap whose CI excludes zero
  is not treated as evidence (D11). Formatters render undefined rates as `n/a`, never
  `NaN`.
- `faircheck.report`: `run_audit` orchestrates validate → counts → bootstrap →
  impossibility → (optional) calibration. `render_markdown` is the single source of
  truth for the CLI and the future app; it refuses to emit a `nan` token. PDF is
  derived from that Markdown via **fpdf2 2.8.8** (pure-Python wheel, Helvetica core
  fonts, no system libraries) - D6.
- `python -m faircheck report data.csv --label ... --pred ... --group ... --positive ...`
  (`--score`, `--out`, `--pdf`, `--n-boot`, `--seed`, and the three helper questions
  optional). The CLI computes no metric of its own.
- Tests 8, 9, 11: every report section present and numbers match the computed
  objects; CI-includes-zero wording is asserted on the spec case; CLI stdout matches
  the library and `python -m faircheck` writes a PDF.

Notes
- Test count 190 → 228; coverage of `faircheck/` stays 98%.
- Deliverable (hand-computed case, B = 1000, seed 0): B's selection rate is 30 pp
  lower than A's (95% CI -42 to -17). Equal-opportunity max-min is 0.200
  [0.016, 0.441] (excludes 0) but the signed TPR contrast includes 0, so the summary
  says **no clear evidence of a gap** in TPR. Disparate impact ratio 0.400 raises
  the four-fifths *screen*, with the EEOC "rule of thumb, not a legal finding"
  sentence attached.

### M3 - calibration, threshold sweep, impossibility explainer - 2026-09-21

Added
- `faircheck.calibration`: Mann-Whitney ROC-AUC (no sklearn at runtime), equal-width
  reliability bins, **positive-class** ECE (D12), and a vectorised threshold sweep
  (`score >= t`) via `searchsorted`. A group missing either class gets `roc_auc = NaN`
  with a reason, never 0.5 / 0.0. Empty bins contribute 0 to ECE.
- `faircheck.impossibility`: substitutes the user's rates into Chouldechova's identity
  and two counterfactuals (equalise PPV; equalise TPR/FPR). `tradeoff_applies` is the
  only boolean the UI should branch on. A counterfactual FPR outside [0, 1] is flagged
  `equal_ppv_feasible=False` rather than clipped - that is the identity saying those
  rates cannot co-exist.
- Tests: hand-computed two-bin ECE = 0.14; ECE grows as scores are shifted; AUC matches
  sklearn; sweep at a cut matches `confusion_counts`; identity recovers the spec case;
  perfect classifiers and equal base rates do not trigger the trade-off.

Notes
- Test count 154 → 190; coverage of `faircheck/` stays 98%, with `impossibility.py` at 100%.
- Deliverable numbers (seeded, Y ~ Bernoulli(score), Beta(2,5) vs Beta(5,2)): group A
  AUC 0.716 ECE 0.019; group B AUC 0.730 ECE 0.020; base-rate spread 0.41.

### M2 - bootstrap confidence intervals - 2026-09-21

Added
- `faircheck.bootstrap`: percentile confidence intervals for every rate, gap and contrast.
  The resampling uses the multinomial identity from D4 - a within-group row bootstrap *is* a
  `Multinomial(n_g, cells_g / n_g)` draw over the four confusion cells - so the cost is
  `O(n_boot × 4 × n_groups)` and does not depend on the number of rows. Measured: **2.5 ms
  for B = 1000 at n = 100,000, and 2.7 ms at n = 10,000,000**.
- Both resampling schemes: `stratified` (group sizes fixed, the default and the right
  conditioning for per-group rates) and `full` (dataset resampled as a whole, matching what
  Fairlearn's `MetricFrame` bootstrap does).
- Signed contrasts against a reference group, with `has_clear_contrast` returning
  `True` / `False` / `None`. This, not the max-min gap, is what supports evidence claims -
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

### M1 - skeleton, validation, metrics, CI - 2026-09-21

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
- CI covers Python 3.12 and 3.13 rather than the planned 3.11-3.13: `numpy` 2.5.x
  requires Python ≥ 3.12.
