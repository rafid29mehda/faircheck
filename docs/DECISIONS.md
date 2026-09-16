# Decisions

Numbered, dated, with the reasoning and the alternative that was rejected. Code comments
cite these as `D<n>`. Supporting research and links live in `PLAN.md` §2.

---

## D1 — Streamlit on Streamlit Community Cloud, not Gradio on Hugging Face Spaces
**2026-09-21 · accepted**

Hugging Face deprecated the native Streamlit SDK (2025-04-30) and now requires a paid plan
to *create* any compute Space — Gradio or Docker, CPU Basic included. The only free
carve-out is ZeroGPU, which is Gradio-only and capped at two Spaces per account. Using a
GPU-allocation product to host a NumPy app would be the most fragile part of the project.

Streamlit Community Cloud is free, has no paid tier to be pushed into, and its reactive
rerun model fits the two interactive requirements (column mapping that depends on the
uploaded file, and a threshold slider) with no extra wiring.

*Rejected:* Gradio + ZeroGPU; HF Docker Space (needs PRO).
*Consequence:* charts use Altair, which Streamlit already depends on, so no extra pin.

## D2 — Fairlearn is a test oracle, not a runtime dependency
**2026-09-21 · accepted**

`faircheck/` depends only on NumPy and pandas. Fairlearn appears in
`requirements-dev.txt` and in tests, where random data is audited with both
implementations and the results compared.

Three reasons. (a) Correctness: Fairlearn coerces undefined rates to `0.0`, which this
project must not do — see D3. (b) An independent implementation cross-validated against
the reference library is a stronger demonstration than a thin wrapper around it.
(c) A lighter runtime is easier to host and slower to rot.

*Cost, accepted:* the shipped app does not use Fairlearn, so a future Fairlearn bug fix
does not automatically become ours.
*Guard:* `tests/test_core_has_no_ui.py::test_core_does_not_import_fairlearn`.

## D3 — An undefined rate is `NaN` and renders as `n/a`
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
*Consequence:* `fails_four_fifths` returns `None`, not `False`, for an undefined ratio — a
screen that never ran must not render as a pass.

## D4 — Bootstrap via multinomial draws over confusion cells
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
functions of four counts, so they use genuine row-level resampling with a lower `B`.

*Requires:* metrics stay shape-generic — see `AGENTS.md` invariant 4.

## D5 — Binary core with a user-chosen positive label; multi-class via one-vs-rest
**2026-09-21 · accepted**

The favorable outcome is a modelling choice, so `positive_label` is a value selected from
the label column rather than a hard-coded `1`. A column with more than two distinct values
is audited one-vs-rest against the chosen value, with a warning naming the split.

This keeps every formula in `metrics.py` exactly correct (all of them are binary
confusion-matrix quantities) while still giving multi-class users a truthful answer.

*Rejected:* true multi-class support (per-class confusion matrices and macro-averaged
gaps) — it multiplies the UI surface and the impossibility discussion without deepening
what the project demonstrates.
*Guard:* `test_flipping_the_positive_label_swaps_the_confusion_matrix`.

## D6 — Markdown is the canonical report; PDF is derived from it
**2026-09-21 · accepted, PDF route to be chosen in M4**

`report.py` produces Markdown, and the app and CLI emit byte-identical output from the
same function, so the export is covered by CI.

PDF export was requested and will be attempted, but as a *derived* artifact. The library
will be chosen in M4 against one hard constraint: **pure-Python wheels only, no system
libraries** (a headless browser or a Cairo/Pango stack is the classic "works locally, 500s
in production" failure). If no option meets that bar, the fallback is print-friendly CSS
plus a browser print hint, and the plan gets revised rather than the constraint dropped.

## D7 — No overall fairness score or grade
**2026-09-21 · accepted**

The app teaches that calibration and equal error rates cannot both hold when base rates
differ (Chouldechova 2017; Kleinberg et al. 2016). A single headline number would have to
pick a winner among mutually incompatible criteria and would hide exactly the trade-off
the tool exists to explain.

Instead: per-metric verdicts ("clear gap" / "no clear evidence of a gap"), a plain-language
summary hedged by the confidence intervals, and a metric-selection helper that asks about
the decision context. The user confirmed they are happy with no headline grade, on the
condition that the explanation is thorough enough to stand in for one.

## D8 — `(n_groups, 4)` integer cell tables are the core data structure
**2026-09-21 · accepted**

Group order is fixed (lexicographic over the sensitive columns, observed combinations
only) so that colour assignment is stable across runs. Combinations that never occur are
absent rather than present-as-zero, because a zero-row group has no defined rates at all.

Counting is a single `bincount` over `(group, cell)` pairs rather than a groupby per
metric, which keeps the whole table one vectorised pass. Intersectional keys are tuples,
so the marginal and intersectional views share one code path.

## D9 — Bundle four examples, including both ACS income and UCI Adult
**2026-09-21 · accepted (implementation in M5)**

Per the user's request, both real datasets ship. ACSIncome is the literature's intended
replacement for Adult (Ding et al. 2021 documented Adult's idiosyncrasies); Adult is more
instantly recognisable to a reader. Bundling both, and saying in the README why ACSIncome
is preferred, demonstrates the awareness that choosing only one would merely imply.

Plus the two synthetic examples: a calibrated "COMPAS-like" case with differing base rates
and unequal error rates, and a near-fair contrast case.

Real COMPAS records are **not** bundled. The synthetic case makes the same pedagogical
point without restating a contested factual claim about real defendants. Every generator
is seeded and its CSV committed, so tests and CI never touch the network.

## D11 — Evidence claims use signed contrasts against a reference group, never max−min gaps
**2026-09-21 · accepted**

A gap defined as `max_a r_a − min_a r_a` is **non-negative by construction**. Its bootstrap
interval therefore almost never contains zero, even when every group shares the same true
rate, because sampling noise alone pushes the maximum above the minimum. Reading "the CI
excludes zero" as evidence of disparity would make the tool report a gap in *every dataset
it ever sees* — the single most consequential statistical mistake available here.

Demonstrated on the specification's own hand-computed case (n = 100 per group):

| | value | 95% CI |
|---|---|---|
| Equal opportunity difference (max−min) | 0.200 | [0.016, 0.441] — excludes 0 |
| TPR contrast, B − A (signed) | −0.200 | [−0.441, +0.056] — **includes 0** |

Same data, same metric, opposite verdicts. The max−min interval looks significant; the
honest signed comparison says the data cannot resolve a 0.20 TPR gap at this sample size.

So FairCheck reports both, with different jobs. The max−min gaps are the headline summary
numbers (and match Fairlearn's definitions, so they are comparable with the literature).
The **signed contrast of each group against a reference group** is what drives every verdict
and every sentence of the plain-language summary.

The reference defaults to the **largest** group, not the extreme one. Picking the extreme
group after looking at the results is a selection effect: the winner of a noisy maximum is
optimistic, and its naive interval under-covers. `has_clear_contrast` returns `None` when
the interval is undefined, and `False` means "no clear evidence of a difference" — never
"no difference".

*Guard:* `test_max_minus_min_gap_is_positive_even_when_all_groups_are_identical` builds three
groups from one distribution and asserts that the max−min interval stays away from zero
while every signed contrast straddles it.

## D10 — Deployment and the git remote are deferred
**2026-09-21 · accepted**

The user asked not to think about hosting or a GitHub repo yet. Work is committed to a
local git repository so history exists, but no remote is configured and nothing is
published. D1 records the hosting research for when this is picked up.
