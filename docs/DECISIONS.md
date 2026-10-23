# Decisions

Numbered, dated, with the reasoning and the alternative that was rejected. Code comments
cite these as `D<n>`. Supporting research and links live in `PLAN.md` §2.

---

## D1 - Streamlit on Streamlit Community Cloud, not Gradio on Hugging Face Spaces
**2026-09-21 · accepted**

Hugging Face deprecated the native Streamlit SDK (2025-04-30) and now requires a paid plan
to *create* any compute Space - Gradio or Docker, CPU Basic included. The only free
carve-out is ZeroGPU, which is Gradio-only and capped at two Spaces per account. Using a
GPU-allocation product to host a NumPy app would be the most fragile part of the project.

Streamlit Community Cloud is free, has no paid tier to be pushed into, and its reactive
rerun model fits the two interactive requirements (column mapping that depends on the
uploaded file, and a threshold slider) with no extra wiring.

*Rejected:* Gradio + ZeroGPU; HF Docker Space (needs PRO).
*Consequence:* charts use Altair, which Streamlit already depends on, so no extra pin.

## D2 - Fairlearn is a test oracle, not a runtime dependency
**2026-09-21 · accepted**

`faircheck/` depends only on NumPy and pandas. Fairlearn appears in
`requirements-dev.txt` and in tests, where random data is audited with both
implementations and the results compared.

Three reasons. (a) Correctness: Fairlearn coerces undefined rates to `0.0`, which this
project must not do - see D3. (b) An independent implementation cross-validated against
the reference library is a stronger demonstration than a thin wrapper around it.
(c) A lighter runtime is easier to host and slower to rot.

*Cost, accepted:* the shipped app does not use Fairlearn, so a future Fairlearn bug fix
does not automatically become ours.
*Guard:* `tests/test_core_has_no_ui.py::test_core_does_not_import_fairlearn`.

## D3 - An undefined rate is `NaN` and renders as `n/a`
**2026-09-21 · accepted**

A rate with a zero denominator has no value: TPR for a group with no positive-class rows,
precision for a group that is never selected. FairCheck returns `NaN`, propagates it
through the gaps (`max`/`min` propagate `NaN`), and renders `n/a` with a footnote.

Verified during planning: `fairlearn.metrics.true_positive_rate` returns `0.0` in this
situation, inheriting scikit-learn's `zero_division=0` default. Printing `0.0` asserts
"none of this group's positives were found", which is a false claim about real people, and
a gap computed while quietly dropping an undefined group understates the true spread.

*Consequence:* the Fairlearn oracle test compares only groups where every rate is defined,
and `test_fairlearn_disagrees_on_undefined_rates_and_that_is_deliberate` pins the
divergence so a future Fairlearn change surfaces as a failure rather than a surprise.
*Consequence:* `fails_four_fifths` returns `None`, not `False`, for an undefined ratio - a
screen that never ran must not render as a pass.

## D4 - Bootstrap via multinomial draws over confusion cells
**2026-09-21 · accepted (implementation in M2)**

Every confusion-matrix metric is a function only of the four cell counts. Resampling
`n_g` rows with replacement within group *g* therefore induces exactly a
`Multinomial(n_g, cells_g / n_g)` draw over those cells. So the whole bootstrap for all
rates and gaps is one `rng.multinomial(..., size=B)` per group: `O(B · 4 · G)`,
independent of dataset size, with no `(B, n)` index matrix.

This is exact rather than an approximation, and it will be tested against a naive
row-resampling bootstrap instead of merely asserted.

Default scheme is stratified (observed group sizes held fixed), which is the right
conditioning for per-group rate intervals. Score-based metrics (ROC-AUC, ECE) are *not*
functions of four counts, so they use genuine row-level resampling with a lower `B`
(*specified here in M2; not actually built until M8 - see D13*).

*Requires:* metrics stay shape-generic - see `AGENTS.md` invariant 4.

## D5 - Binary core with a user-chosen positive label; multi-class via one-vs-rest
**2026-09-21 · accepted**

The favorable outcome is a modelling choice, so `positive_label` is a value selected from
the label column rather than a hard-coded `1`. A column with more than two distinct values
is audited one-vs-rest against the chosen value, with a warning naming the split.

This keeps every formula in `metrics.py` exactly correct (all of them are binary
confusion-matrix quantities) while still giving multi-class users a truthful answer.

*Rejected:* true multi-class support (per-class confusion matrices and macro-averaged
gaps) - it multiplies the UI surface and the impossibility discussion without deepening
what the project demonstrates.
*Guard:* `test_flipping_the_positive_label_swaps_the_confusion_matrix`.

## D6 - Markdown is the canonical report; PDF is derived from it
**2026-09-21 · accepted; PDF via fpdf2 2.8.8 (M4)**

`report.py` produces Markdown, and the app and CLI emit byte-identical output from the
same function, so the export is covered by CI.

PDF is a *derived* artifact of that Markdown. M4 chose **fpdf2 2.8.8**: a pure-Python
wheel with no system libraries (no headless browser, no Cairo/Pango). Rendering uses
the 14 PDF core fonts (Helvetica/Courier) so nothing is bundled. `write_pdf` imports
fpdf2 lazily; metric modules never see it. Install via `pip install 'faircheck[pdf]'`
or the pin in `requirements.txt`.

Long table rows wrap character-wise (`WrapMode.CHAR`) because a markdown cell can
exceed the page width in Courier, and `multi_cell(w=0)` measures from the current x
which may already sit near the right margin. Non-Latin-1 glyphs (`×`, minus, em-dash)
are folded before Helvetica is asked to draw them.

*Rejected:* weasyprint / wkhtmltopdf (system libraries); a browser print hint as the
primary path (still available later in the Streamlit app).

## D7 - No overall fairness score or grade
**2026-09-21 · accepted**

The app teaches that calibration and equal error rates cannot both hold when base rates
differ (Chouldechova 2017; Kleinberg et al. 2016). A single headline number would have to
pick a winner among mutually incompatible criteria and would hide exactly the trade-off
the tool exists to explain.

Instead: per-metric verdicts ("clear gap" / "no clear evidence of a gap"), a plain-language
summary hedged by the confidence intervals, and a metric-selection helper that asks about
the decision context. The user confirmed they are happy with no headline grade, on the
condition that the explanation is thorough enough to stand in for one.

## D8 - `(n_groups, 4)` integer cell tables are the core data structure
**2026-09-21 · accepted**

Group order is fixed (lexicographic over the sensitive columns, observed combinations
only) so that colour assignment is stable across runs. Combinations that never occur are
absent rather than present-as-zero, because a zero-row group has no defined rates at all.

Counting is a single `bincount` over `(group, cell)` pairs rather than a groupby per
metric, which keeps the whole table one vectorised pass. Intersectional keys are tuples,
so the marginal and intersectional views share one code path.

## D9 - Bundle four examples, including both ACS income and UCI Adult
**2026-09-21 · accepted; implemented in M5**

Per the user's request, both real datasets ship. ACSIncome is the literature's intended
replacement for Adult (Ding et al. 2021 documented Adult's idiosyncrasies); Adult is more
instantly recognisable to a reader. Bundling both, and saying in the README why ACSIncome
is preferred, demonstrates the awareness that choosing only one would merely imply.

Plus the two synthetic examples: a calibrated "COMPAS-like" case with differing base rates
and unequal error rates, and a near-fair contrast case.

Real COMPAS records are **not** bundled. The synthetic case makes the same pedagogical
point without restating a contested factual claim about real defendants. Every generator
is seeded and its CSV committed, so tests and CI never touch the network.

## D11 - Evidence claims use signed contrasts against a reference group, never max-min gaps
**2026-09-21 · accepted**

A gap defined as `max_a r_a - min_a r_a` is **non-negative by construction**. Its bootstrap
interval therefore almost never contains zero, even when every group shares the same true
rate, because sampling noise alone pushes the maximum above the minimum. Reading "the CI
excludes zero" as evidence of disparity would make the tool report a gap in *every dataset
it ever sees* - the single most consequential statistical mistake available here.

Demonstrated on the specification's own hand-computed case (n = 100 per group):

| | value | 95% CI |
|---|---|---|
| Equal opportunity difference (max-min) | 0.200 | [0.016, 0.441] - excludes 0 |
| TPR contrast, B - A (signed) | -0.200 | [-0.441, +0.056] - **includes 0** |

Same data, same metric, opposite verdicts. The max-min interval looks significant; the
honest signed comparison says the data cannot resolve a 0.20 TPR gap at this sample size.

So FairCheck reports both, with different jobs. The max-min gaps are the headline summary
numbers (and match Fairlearn's definitions, so they are comparable with the literature).
The **signed contrast of each group against a reference group** is what drives every verdict
and every sentence of the plain-language summary.

The reference defaults to the **largest** group, not the extreme one. Picking the extreme
group after looking at the results is a selection effect: the winner of a noisy maximum is
optimistic, and its naive interval under-covers. `has_clear_contrast` returns `None` when
the interval is undefined, and `False` means "no clear evidence of a difference" - never
"no difference".

*Guard:* `test_max_minus_min_gap_is_positive_even_when_all_groups_are_identical` builds three
groups from one distribution and asserts that the max-min interval stays away from zero
while every signed contrast straddles it.

## D12 - ECE is on the positive-class probability, not top-label confidence
**2026-09-21 · accepted**

Guo et al. (2017) define ECE over top-label confidence `max_k p̂(y=k)`. FairCheck's score
column is `s = p̂(Y=1)`, and the question the UI asks is "will a human read this as a
probability of the favorable outcome?" Those are different quantities. We bin `s` on
equal-width [0, 1] intervals (default 10) and compare the mean score in each bin to the
observed positive rate. Empty bins contribute 0. Nixon et al. (2019) note that ECE is
binning-sensitive and biased; the bin count is part of the reported number, and ECE is
treated as descriptive, not a test.

The same honesty rule as D3 applies to ROC-AUC: a group with only one class is `n/a`
with a reason, not 0.5.

*Related.* Chouldechova's identity can return an FPR outside [0, 1] when asked to
equalise PPV at a value incompatible with a group's prevalence and FNR. That is reported
as infeasible, not clipped to 1.0.

## D10 - GitHub remote is public; hosted demo is still deferred
**2026-09-21 · revised**

A public GitHub repo exists at `https://github.com/rafid29mehda/faircheck` (created
when the user asked). Streamlit Community Cloud and any other hosted demo stay
out of scope until they ask. D1 still records the hosting research for that day.

*Earlier (same date):* the user asked not to think about hosting or a GitHub repo
yet. That part of D10 no longer holds.

## D13 - Score-metric intervals: row-level bootstrap via per-row multiplicities
**2026-10-23 · accepted (implemented in M8)**

*What was wrong.* D4, PLAN and the README all said ROC-AUC and ECE get row-level bootstrap
intervals. `calibration.py` never resampled anything, and the report printed both as bare
numbers - the one place the tool broke its own "no number without an interval" rule. Found
while auditing the project against current practice, not by a test.

*Decision.* Per group (and for the overall row), draw `n` row indices with replacement and
turn them into per-row multiplicities with one `bincount`. Scores are sorted and binned
once, so each resample is O(n) weighted counting with no re-sorting:

- **AUC** = `sum(pos_w * (neg_below + 0.5 * neg_tied)) / (W_pos * W_neg)` over distinct
  scores, the same Mann-Whitney statistic as `roc_auc` (ties count 0.5).
- **ECE** = `sum_m |sum(y) - sum(s)| / n` over bins, because `|B_m| * |rate - mean score|`
  collapses to `|sum(y) - sum(s)|` within a bin.

Default 200 resamples (`run_audit(score_n_boot=...)`, CLI `--score-n-boot`; `0` switches
intervals off and leaves `NaN`). One independent stream per group, seeded `(seed, index)`, so
the audit seed still fixes every number. Resamples that lose a class are skipped for AUC and
counted in `auc_undefined_fraction`, matching D4's treatment of undefined metrics.

*Cost (measured 2026-10-23, 200 resamples, two groups plus the overall row):* about 0.06 s
at 8,000 rows, 0.64 s at 100,000, 7 s at 1,000,000. Unlike the confusion-cell bootstrap
(D4) this grows with rows, hence the lower default and the off switch.

*Known limitation, stated in the report.* Plug-in ECE is biased upward when bins are sparse,
so its percentile interval describes the sampling variability of the estimate and can sit
above the point estimate. On the near-fair example group A reads 0.008 with interval
[0.009, 0.026]. It is **not** a calibrated interval for population ECE. A debiased or
binning-free calibration estimator is future work.

*Rejected:* DeLong's analytic AUC variance (covers AUC only; a second machinery for ECE);
the multinomial fast path (does not apply to score metrics).
*Guards:* `tests/test_score_intervals.py` - agreement with naive row resampling for AUC, ECE
and heavily tied scores (the tied-score test was added after a deliberate mutation of the
tie term went undetected), reproducibility, narrowing with n, undefined-resample handling.

## D14 - Verdicts use simultaneous (max-t) bands over the headline contrasts
**2026-10-23 · accepted (implemented in M9)**

*Problem.* Every group-versus-reference contrast was judged on its own 95% interval. With G
groups there are 4 x (G-1) of them (selection rate, TPR, FPR, precision), so false findings
add up. Measured on a null case, three identical groups of 200, 300 trials, 8 contrasts per
audit: at least one false "clear" contrast in 31.7% of trials. (The 12% quoted earlier in
the guide counted only the two TPR contrasts.)

*Decision.* Standardise each contrast by its bootstrap standard deviation. T* is the 95th
percentile, over resamples, of the largest standardised deviation from the observed value
across the whole family. The band for a contrast is `obs +/- T* sd`, and a verdict is
"clear" only if that band excludes zero. It reuses the D4 resamples, so it costs nothing
extra. On the same null case the false-finding rate drops to 6.3% (19 of 300; the Monte
Carlo error on 300 trials is about 1.3 points, so this is consistent with 5%). The mean
critical value is 2.64 for a family of 8, and 2.37 for the COMPAS-like pair (family of 4).

*Family.* `bootstrap.VERDICT_RATE_KEYS` times the non-reference groups. These four rates
are the inputs of the five gaps, so they are everything a verdict depends on. Base rate,
accuracy and FNR (which is 1 - TPR) stay marginal because no verdict uses them.

*Edge cases.* A contrast with no finite resamples is left out of the maximum and has no
verdict (`None`). A constant contrast (standard deviation 0, for example one group catching
every positive and the other none) keeps a degenerate band at its observed value. The
reference group is zero by construction.

*Wording.* A clear result reads "still clear after adjusting for N comparisons". A contrast
whose own 95% interval excludes zero but whose adjusted band does not gets a sentence saying
exactly that, so the adjustment is never silent. `adjust="none"` (CLI `--adjust none`) goes
back to judging each contrast on its own.

*What it does not do.* It covers these four rates against one reference group. It does not
cover other attributes audited in separate runs or repeated re-runs. It is a bootstrap
approximation, not an exact guarantee, and small groups make it rougher.

*Related work.* Cherian and Candes (JMLR 2024) treat fairness auditing as multiple testing
and use the bootstrap for simultaneous bounds over collections of groups. This decision uses
the standard sup-t bootstrap band, not their procedure, and I have not implemented or
compared theirs. Ferrara et al. (NeurIPS 2025) address small intersectional groups with
size-adaptive tests; that is the next item.

*Rejected.* Bonferroni and Holm (ignore that contrasts from the same data are correlated, so
they over-correct); Benjamini-Hochberg (controls the share of false discoveries instead,
which suits a later subgroup search better than a fixed family).
*Guards:* `tests/test_multiplicity.py`: null false-finding rates, band widths, a borderline
contrast that flips under adjustment, and a deliberate mutation (critical value fixed at
1.96) that made five tests fail.

## D15 - Small groups: Dirichlet posterior intervals, chosen by the smallest cell
**2026-10-23 · accepted (implemented in M10)**

*Problem.* The percentile bootstrap of D4 resamples the observed cells, so it can only
produce a cell it has already seen. A group with a zero cell gets a degenerate interval, and
one with a tiny cell gets one that is far too narrow. The old `n < 30` flag was meant to
cover this, but group size is the wrong thing to look at. Measured with
`experiments/small_group_coverage.py` (fixed truth, base rate 0.3, 95% interval for TPR):

| n | true TPR | bootstrap covers | Dirichlet(0.5) covers |
|---|---|---|---|
| 20 | 0.50 / 0.80 / 0.95 | 0.900 / 0.690 / 0.256 | 0.933 / 0.963 / 0.967 |
| 40 | 0.50 / 0.80 / 0.95 | 0.960 / 0.906 / 0.452 | 0.956 / 0.940 / 0.973 |
| 100 | 0.50 / 0.80 / 0.95 | 0.948 / 0.921 / 0.785 | 0.938 / 0.933 / 0.971 |
| 400 | 0.50 / 0.80 / 0.95 | 0.953 / 0.953 / 0.939 | 0.943 / 0.953 / 0.954 |

At n = 100 with a true TPR of 0.95 the bootstrap still misses 21% of the time, well past the
`n < 30` line. What matters is the smallest observed cell:

| smallest observed cell | cases | bootstrap | Dirichlet(0.5) |
|---|---|---|---|
| 0 | 9171 | 0.558 | 0.948 |
| 1 | 5152 | 0.936 | 0.948 |
| 2-4 | 8228 | 0.946 | 0.947 |
| 5-9 | 5836 | 0.946 | 0.943 |
| 10 and up | 19484 | 0.944 to 0.948 | 0.940 to 0.944 |

(Second table: truths drawn from a Dirichlet(0.7) over the four cells, all four verdict rates.
That setup favours the posterior by construction, so read it for *where* the bootstrap breaks,
a zero cell, and not for the size of the gap. The first table is the neutral one.)

*Decision.* A group whose smallest cell is below `MIN_CELL_FOR_BOOTSTRAP = 5` gets
`Dirichlet(cells + 0.5)` draws of its cell proportions instead of multinomial resamples. 0.5
is the Jeffreys prior: any rate that is a ratio of two cell sums then has a Beta posterior with
that prior. Rates are scale-free functions of the cells, so the draws go through the same
formulas, the same percentile intervals and the same max-t bands (D14). Groups at or above the
threshold keep the D4 bootstrap exactly, so large-group numbers and the committed report are
byte-identical to before. The posterior uses its own seed stream, so a small group never
changes another group's draws. Threshold 5 sits above the measured break (2) as a margin.

*Undefined stays undefined (D3).* The prior must not invent a rate the data cannot define: a
posterior group whose observed rate is undefined (no positives, so no TPR) keeps `NaN` in every
draw. Its interval is `n/a`, its undefined fraction is 1.0 and its verdict is `None`.

*Effect on the bundled data.* The four single-attribute examples are unchanged (no group has a
cell under 5). On ACS Income sex x race, 6 of 16 groups switch. Female x NHPI (n = 20, no true
positives found) goes from a TPR interval of [0.00, 0.00] to [0.00, 0.43]; Female x Alaska
Native (n = 2) from [1.00, 1.00] to [0.16, 1.00].

*Changes in behaviour.* Posterior groups never lose a class in a resample, so their
`rate_undefined_fraction` is 0 unless the observed rate is undefined. `--intervals bootstrap`
(`interval_method="bootstrap"`) restores the old behaviour for every group.

*Related work.* Ferrara et al. (NeurIPS 2025) combine analytic intervals for large subgroups
with a Dirichlet-multinomial Bayesian estimator for small intersectional ones. This decision
follows the same idea (a posterior for small cells) but is my own construction on this
project's cell tables, with its own coverage study. I have not compared it against their
method.

*Not done.* The posterior intervals are credible intervals under a Jeffreys prior, not
frequentist guarantees; the first table is the evidence for their coverage, over a limited
grid. Wilson or Clopper-Pearson intervals for single proportions are a fair comparison I did
not run. The `n < 30` flag stays: it says the estimate is imprecise, which is true however the
interval is built.
*Guards:* `tests/test_small_groups.py`: the rule, a zero cell, a coverage regression (adaptive
96.8% against 44.4% for the bootstrap at n = 40, TPR 0.95), undefined rates, verdict
integration, and two deliberate mutations (rule disabled; undefined mask removed) that failed
seven tests and one test respectively.

## D16 - Hosted demo on Streamlit Community Cloud
**2026-10-23 · accepted (supersedes the deferral in D10)**

D10 deferred hosting until it was asked for. It was asked for, so the app is live at
https://faircheck.streamlit.app, deployed from `main` (`app.py`, Python 3.12, no secrets).
Every push to `main` redeploys it, so push only after CI is green.

*Deployment notes.* The first build took about 50 seconds. The Python version dropdown
defaulted to 3.14, not the 3.12 the docs describe; it was set to 3.12, the version CI and the
pins are verified on. The subdomain `faircheck` was free. Before deploying, the app booted
with no errors from clean environments built from `requirements.txt` with both pip and uv
(the installer Community Cloud uses), and a 50 MB wide upload peaked near 430 MB.

*What was checked on the live URL.* The Markdown report for the COMPAS-like example is
byte-identical to `examples/reports/compas_like.md` (SHA-256 `8b3869af...17a8`, 7,538 bytes).
Near-fair gives "no clear evidence" for all four contrasts. ACS Income with a second
attribute `race` shows 16 groups, 4 with n < 30, and the Dirichlet note names the same six
groups as a local run. The hand-computed case, uploaded and mapped, gives the same selection
gap (30 points lower, CI -42 to -17) and "no clear evidence" elsewhere. The PDF download is a
valid PDF. The uploader shows the 25 MB cap. A non-CSV text file produces warnings and no
crash.

*Found on the live app, items 1 and 2 fixed afterwards (M11, part 3).*
1. An empty upload showed a raw `pandas.errors.EmptyDataError`. `validate.read_csv` already
   had a clear message ("That file is empty"), but the app's `_load_upload` called
   `pd.read_csv` directly and skipped it. The app now uses `read_csv` and shows the
   library's message (and its "header but no data rows" and "not UTF-8" messages) in an
   error box.
2. After an upload, every mapping dropdown defaulted to the first column, and the positive
   outcome to the first value (often `0`). A CSV in the README's own format arrived with
   label, prediction and group all set to `y_true`, and a positive outcome of 0 silently
   inverted the analysis. `validate.suggest_columns` now guesses roles from common column
   names (never reusing a column, skipping id-like columns when falling back), and
   `suggest_positive_label` prefers a value that reads as "yes" (1, true, yes, pos), else the
   last value in sorted order. The sidebar says these are guesses to check. Bundled examples
   keep their stored mappings. The score column is only ever matched by name, never by
   position.
3. (Still open) The browser console shows three 404s on static resources. They were not identified and
   nothing visible depends on them.

*Not tested live.* A 25 MB upload, how long a wake-up from sleep takes, phone layout and
concurrent visitors.

*Unverified platform facts.* The resource limits (about 690 MB to 2.7 GB of memory), the
12-hour sleep window and "no paid tier" come from forum posts, not Streamlit's docs. The
platform's own logging of uploads is not documented in what was checked. The README says
uploads go to a third-party host and points to the bundled examples for anything sensitive.

*Rollback.* Delete the app in the Community Cloud dashboard. The repository is unaffected.

*Stale modules after a push.* A push redeploys the app, but the running process can keep the
old version of a package it has already imported. After the upload fixes the live app showed
`ImportError: cannot import name 'suggest_columns' from 'faircheck.validate'` even though the
new file was on disk (the platform log showed the path and the missing name). A push that
changes a module under `faircheck/` that `app.py` imports may need a reboot: in the Community
Cloud dashboard open the app's menu and choose Reboot. A change to `app.py` alone did not
need one. Check the live URL after such a push.

