# FairCheck — Plan (Phase 1)

> Status: **approved 2026-09-21, Phase 2 complete.** M1–M7 done. Deployment still deferred (D10).
> Current state and working conventions: `AGENTS.md`. Decisions and rationale: `docs/DECISIONS.md`.
> Per-milestone changes: `CHANGELOG.md`.

---

## 1. Goal

A web app where anyone uploads a CSV of classification predictions and gets back a correct,
well-explained, uncertainty-aware fairness report — plus a pure-Python core library and CLI that
produce the identical report offline.

**One-line CV description:**
> *FairCheck — open-source fairness auditing tool (Python/Streamlit): disaggregated group metrics with
> bootstrap confidence intervals, per-group calibration analysis, and a metric-selection guide grounded
> in the Hardt et al. / Chouldechova impossibility results. Live demo, CLI, pytest suite, CI.*

**The two-minute test.** A professor opening the demo should, without reading the README, see:
disaggregated metrics with CIs (not point estimates), undefined metrics honestly marked `n/a`
(not silently zero), an impossibility explainer computed from *their* numbers, and hedged language
when CIs overlap zero. Those four things are what separate this from the many "fairness dashboard"
projects that print a single disparate-impact number.

---

## 2. Research notes

Legend: **[verified]** = I ran it or read the primary source during this planning session.
**[docs]** = taken from official documentation, not executed.

### 2.1 Library versions (queried from PyPI, 2026-09-21) **[verified]**

| Package | Latest | Requires Python | Note |
|---|---|---|---|
| `fairlearn` | **0.14.0** (rel. 2026-06-07) | >=3.10 | deps: `narwhals>=1.14`, `numpy>=1.24.4`, `pandas>=2.0.3`, `scikit-learn>=1.5`, `scipy>=1.9.3` |
| `streamlit` | **1.64.0** (rel. 2026-09-15) | >=3.10 | already depends on `altair>=5,<7`, `pandas`, `numpy`, `pyarrow` |
| `gradio` | 6.28.0 | >=3.10 | |
| `scikit-learn` | 1.9.1 | >=3.11 | |
| `pandas` | 3.0.6 | >=3.11 | major version bump — compatibility explicitly tested below |
| `numpy` | 2.5.3 | >=3.12 | |
| `altair` | 6.3.0 | >=3.11 | ships transitively with Streamlit |
| `pytest` | 9.1.1 | >=3.10 | |

**Compatibility probe [verified].** I built a throwaway venv in `/tmp` (nothing written to the project)
and installed `fairlearn==0.14.0` with latest transitive deps. Result:

```
python 3.13.2
fairlearn 0.14.0 | pandas 3.0.6 | numpy 2.5.3 | sklearn 1.9.1
```

So the modern stack (including the pandas 3.x major bump) works together on Python 3.13. Note the
docs say Streamlit Community Cloud *defaults* to Python 3.12 while forum reports say new apps get
3.13.x; I will pin dev to 3.12 and run CI on 3.11/3.12/3.13 so either cloud default is covered.

### 2.2 Fairlearn API — confirmed surface

- `MetricFrame(*, metrics, y_true, y_pred, sensitive_features, control_features=None, sample_params=None, n_boot=None, ci_quantiles=None, random_state=None)` — [API ref](https://fairlearn.org/v0.14/api_reference/generated/fairlearn.metrics.MetricFrame.html). Properties `overall`, `by_group`; methods `difference(method=...)`, `ratio(...)`, `group_min/max()`. `control_features` is how Fairlearn does **conditional/intersectional** slicing.
- Built-in bootstrap was added in **0.10.0** ([release notes](https://github.com/fairlearn/fairlearn/releases/tag/v0.10.0)), exposed via `n_boot` + `ci_quantiles` + `random_state`, giving `overall_ci`, `by_group_ci`, `difference_ci()`, `ratio_ci()` — [user guide](https://fairlearn.org/v0.14/user_guide/assessment/confidence_interval_estimation.html).
- `demographic_parity_difference(y_true, y_pred, *, sensitive_features, method='between_groups', sample_weight=None)` → max − min selection rate. [ref](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.demographic_parity_difference.html)
- `equalized_odds_difference(..., agg='worst_case'|'mean')` → `worst_case` (default) returns `max(TPR_diff, FPR_diff)`. [ref](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.equalized_odds_difference.html)
- Metric helpers: `count`, `selection_rate(y_true, y_pred, *, pos_label=1, ...)`, `true_positive_rate`, `false_positive_rate`, `false_negative_rate`, `true_negative_rate` — all with `pos_label`. [API index](https://fairlearn.org/v0.14/api_reference/index.html)
- Dataset loaders present in 0.14.0 **[verified by introspection]**: `fetch_adult`, `fetch_acs_income(*, cache, data_home, as_frame, return_X_y, states=None)`, `fetch_bank_marketing`, `fetch_credit_card`, `fetch_diabetes_hospital`, `fetch_boston`. The `states=` argument matters — it lets me pull one state instead of 1.66M rows.

### 2.3 Two correctness findings that shape the architecture **[verified by running code]**

**(a) Fairlearn reproduces your hand-computed case exactly.** I built the confusion cells you specified
(A: TP=40 FN=10 FP=10 TN=40; B: TP=12 FN=8 FP=8 TN=72) and ran `MetricFrame`:

```
   count  selection_rate  tpr  fpr  fnr  precision  accuracy
A  100.0             0.5  0.8  0.2  0.2        0.8      0.80
B  100.0             0.2  0.6  0.1  0.4        0.6      0.84

dp_diff=0.3000  disparate_impact_ratio=0.4000  eq_odds_diff=0.2000
eq_opp_diff=0.2000  pred_parity_diff=0.2000
```

Every target number in your spec is confirmed, and `demographic_parity_ratio` is exactly the
min/max selection-rate ratio (0.2/0.5 = 0.40). This test case is therefore safe to hard-code.

**(b) Fairlearn returns 0.0, not `n/a`, for undefined rates — so I must not delegate blindly.**

```
fairlearn TPR, group with NO positives  -> 0.0     # silently, no exception
fairlearn FPR, group with NO negatives  -> 0.0
sklearn recall_score(..., zero_division=np.nan) -> nan
```

A group with zero positives has an *undefined* TPR; reporting `0.0` reads as "this group is never
correctly identified", which is a materially misleading claim. Your spec explicitly requires `n/a`.
This is the concrete justification for FairCheck owning `metrics.py` rather than wrapping Fairlearn:
**undefined is NaN, propagated into gaps, and rendered `n/a` with a footnote.** Fairlearn becomes the
*test oracle* (dev dependency) rather than a runtime dependency — compared only on groups where every
rate is defined. See Question Q2.

**(c) Chouldechova's identity verified numerically.** `FPR = p/(1−p) · (1−PPV)/PPV · (1−FNR)` holds
on the hand-computed case: group A → (0.5/0.5)(0.2/0.8)(0.8) = 0.20 ✓; group B → (0.2/0.8)(0.4/0.6)(0.6) = 0.10 ✓.
This becomes both the impossibility explainer *and* a unit test.

### 2.4 Hosting — the important research result

**Hugging Face Spaces is no longer a viable free host for this app.**

- The native **Streamlit SDK was deprecated 2025-04-30**; Streamlit apps must now use the Docker template ([Spaces changelog](https://huggingface.co/docs/hub/en/spaces-changelog)).
- More decisive: "Static Spaces are free for everyone. **Gradio and Docker Spaces run on compute and require a paid plan to create**: PRO for personal accounts, Team or Enterprise for organizations. Free personal accounts in good standing can still host up to 2 Gradio Spaces running on ZeroGPU." ([Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview), corroborated by [hub-docs PR #2624](https://github.com/huggingface/hub-docs/pull/2624)). CPU Basic is gated too.
- ZeroGPU is **Gradio-only** and capped at 2 Spaces for eligible free accounts ([ZeroGPU docs](https://huggingface.co/docs/hub/main/en/spaces-zerogpu)).

**Streamlit Community Cloud** ([docs](https://docs.streamlit.io/deploy/streamlit-community-cloud)):
free, unlimited public apps + 1 private, deploys from GitHub, no paid tier exists. Limits as published:
~2 CPU cores max, memory **throttling from ~690 MB, ~2.7 GB ceiling**, 50 GB storage
([resource-limits FAQ](https://discuss.streamlit.io/t/faq-this-app-has-gone-over-its-resource-limits/62973)).
Python version is chosen in the app's *Advanced settings* — `runtime.txt` and `config.toml` are
**ignored** for this ([deploy docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)).
`st.file_uploader` defaults to **200 MB/file**, adjustable via `server.maxUploadSize` or the per-widget
`max_upload_size` parameter ([ref](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)).

### 2.5 Example datasets — provenance and licence

| Example | Source | Licence / terms |
|---|---|---|
| Real data | `fairlearn.datasets.fetch_acs_income(states=[...])` → ACS PUMS via Ding et al. | Data governed by [US Census terms of use](https://www.census.gov/data/developers/about/terms-of-service.html); cite [Ding et al. 2021, *Retiring Adult*](https://arxiv.org/abs/2108.04884) |
| (alternative) | `fairlearn.datasets.fetch_adult` → [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) | UCI page states **no explicit licence**; cite Becker & Kohavi, DOI [10.24432/C5XW20](https://doi.org/10.24432/C5XW20). A [Zenodo mirror](https://zenodo.org/records/7214275) labels it CC BY 4.0, which I will not treat as authoritative for the UCI original |
| COMPAS-like | **synthetic**, generated by seeded script | none — our own code, MIT |
| Near-fair | **synthetic**, generated by seeded script | none — our own code, MIT |

I recommend **ACSIncome over Adult** for the real-data example: Ding et al. reconstructed Adult from
Census sources and documented idiosyncrasies limiting its external validity, and ACSIncome is the
community's intended replacement. Choosing it (and saying why in the README) is itself a signal of
familiarity with the literature. I will **not** bundle real COMPAS data — the well-known
ProPublica-vs-Northpointe dispute is exactly what the *synthetic* calibrated example illustrates
without misrepresenting anyone's real records. See Q3.

Note: `folktables` (0.0.12) is *not* needed — Fairlearn's loader avoids the Census API entirely.
Also relevant: folktables' own terms forbid redistributing the IPUMS-derived Adult reconstruction.

### 2.6 Metric definitions — primary sources

- **Equalized odds / equal opportunity** — Hardt, Price & Srebro, *Equality of Opportunity in Supervised Learning*, NeurIPS 2016, [arXiv:1610.02413](https://arxiv.org/abs/1610.02413). Def. 2.1: Ŷ ⫫ A | Y, which in the binary case is equal TPR (y=1) **and** equal FPR (y=0). Def. 2.2 (equal opportunity): `P{Ŷ=1|A=0,Y=1} = P{Ŷ=1|A=1,Y=1}` — TPR only.
- **Impossibility (classifier form)** — Chouldechova 2017, [arXiv:1610.07524](https://arxiv.org/abs/1610.07524): `FPR = p/(1−p) · (1−PPV)/PPV · (1−FNR)`. Hence a test-fair (equal-PPV) score **cannot** have equal FPR and FNR when prevalence `p` differs between groups.
- **Impossibility (score form)** — Kleinberg, Mullainathan & Raghavan 2016, [arXiv:1609.05807](https://arxiv.org/abs/1609.05807) / [ITCS 2017](https://doi.org/10.4230/LIPIcs.ITCS.2017.43): calibration within groups (A), balance for the negative class (B), balance for the positive class (C) are jointly satisfiable only under **perfect prediction or equal base rates** — and this holds approximately too (Thm 2).
- **Four-fifths rule** — [29 CFR §1607.4(D)](https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XIV/part-1607/subject-group-ECFRdb347e844acdea6/section-1607.4): a selection rate below 4/5 of the highest group's rate "will generally be regarded by the Federal enforcement agencies as evidence of adverse impact." The [EEOC Q&A](https://www.eeoc.gov/laws/guidance/questions-and-answers-clarify-and-provide-common-interpretation-uniform-guidelines) (Q11) states it is "**a rule of thumb… not intended as a legal definition**." The regulation itself notes smaller gaps can still matter if statistically and practically significant, and larger gaps may not when based on small numbers. My UI wording will mirror this, including the small-sample caveat.
- **ECE / reliability diagrams** — Naeini, Cooper & Hauskrecht, AAAI 2015; popularised by Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017, [PDF](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) eq. (3): `ECE = Σ_m (|B_m|/n)·|acc(B_m) − conf(B_m)|`. Caveat from [Nixon et al. 2019](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf): ECE is sensitive to binning scheme and is a biased estimator — I will state bin count in the UI and treat ECE as descriptive.
- **Colour** — Okabe & Ito colourblind-safe 8-colour qualitative palette, [source](https://jfly.uni-koeln.de/color/).

---

## 3. Design options

Three genuinely different shapes, scored on the criteria that matter for a portfolio piece.

| | **A. Streamlit, tabbed, MD export** *(recommended)* | **B. Gradio, single page** | **C. Streamlit single long page + PDF** |
|---|---|---|---|
| Effort | Medium | Medium | Medium-high (PDF adds a browser/LaTeX dep) |
| Clarity for a 2-min visitor | **High** — tabs map 1:1 to report sections; column mapping stays in a persistent sidebar | Medium — Blocks/State gymnastics for dependent dropdowns (column names depend on the uploaded file) | Medium — long scroll buries the calibration and impossibility sections |
| Hosting ease | **High** — Streamlit Community Cloud, free, no Docker | **Low** — HF Docker/CPU now needs PRO; ZeroGPU is the only free path and is a GPU-shaped hack for a CPU app | High (same as A) |
| Interactivity needed (threshold slider, reactive column mapping) | Native `st.slider` + rerun model fits this exactly | Workable but more wiring | Same as A |
| Risk | **Low** | Medium-high (hosting + paid-plan risk) | Medium (headless-browser PDF is the classic deploy breakage) |
| Signals depth? | Yes, via content | Neutral | PDF is polish, not depth |

**Recommendation: A.** The deciding factor is §2.4 — Gradio's free hosting path is now ZeroGPU-only,
which is absurd for a NumPy app and would be the single most likely thing to break. Streamlit
Community Cloud is free, purpose-built, and has no paid tier to be pushed into.

Markdown export over PDF, because Markdown is byte-for-byte identical between the app and the CLI
(one code path, testable in CI) whereas PDF needs a headless browser or LaTeX and is the most common
source of "works locally, 500s in production." If you want PDF, the honest version is a
"print to PDF" browser hint plus print-friendly CSS. See Q5.

Charts: **Altair**, because Streamlit already depends on `altair>=5,<7` — zero extra pinning risk,
and it produces tooltips and accessible legends without imperative plotting code.

---

## 4. Metric specification

Notation, per group *a*: confusion cells TP, FP, FN, TN; `n = TP+FP+FN+TN`;
base rate (prevalence) `p = (TP+FN)/n`. Positive/favorable outcome is **user-selected**, so the
confusion matrix is built after relabelling — never assumed to be `1`.

**Convention: a rate whose denominator is zero is `NaN`, rendered `n/a`. A gap involving any `NaN`
is `NaN`, never silently skipped to a smaller max.** (This is finding 2.3(b).)

| Metric | Formula | Denominator (→ `n/a` when 0) | When it is the right metric |
|---|---|---|---|
| Base rate | `(TP+FN)/n` | n | Context, not fairness. Drives the impossibility section. |
| Selection rate | `(TP+FP)/n` | n | The decision's actual footprint; the quantity the four-fifths rule is about. |
| TPR / recall | `TP/(TP+FN)` | positives | Who *deserved* the favorable outcome and got it. Core for **assistive** decisions. |
| FPR | `FP/(FP+TN)` | negatives | Cost of wrongly flagging. Core for **punitive** decisions. |
| FNR | `FN/(TP+FN)` = 1−TPR | positives | Cost of being missed (e.g. undiagnosed). |
| Precision / PPV | `TP/(TP+FP)` | selected | What a flag *means* for a person in this group. |
| Accuracy | `(TP+TN)/n` | n | Reported for completeness; misleading under class imbalance, and labelled as such. |
| **Demographic parity difference** | `max_a SR_a − min_a SR_a` | — | Outcome parity is the goal itself; labels are untrustworthy (see helper). Hardt et al. 2016 §1 |
| **Disparate impact ratio** | `min_a SR_a / max_a SR_a` | max SR ≠ 0 | Screening for adverse impact; flagged when < 0.8 with the EEOC "rule of thumb, not a legal finding" wording. 29 CFR §1607.4(D) |
| **Equal opportunity difference** | `max_a TPR_a − min_a TPR_a` | — | Assistive decision, labels trusted. Hardt Def. 2.2 |
| **Equalized odds difference** | `max(ΔTPR, ΔFPR)` | — | Both error directions carry real cost. Hardt Def. 2.1; matches Fairlearn `agg='worst_case'` |
| **Predictive parity difference** | `max_a PPV_a − min_a PPV_a` | — | A human will read the score/flag as "probability of Y". Chouldechova 2017 |
| **ECE per group** | `Σ_m (|B_m|/n_g)·|ȳ(B_m) − p̄(B_m)|` | bin occupancy | Scores shown to humans as probabilities. |
| **ROC-AUC per group** | rank-based | needs both classes present | Threshold-free ranking quality; skipped per-group with a note when a group is single-class. |

**ECE detail worth stating precisely.** Guo et al. define ECE over *top-label confidence*. For a binary
task with one score column `s = p̂(Y=1)`, FairCheck computes ECE on the **positive-class probability**:
per bin, observed positive fraction vs. mean predicted score. That is the quantity relevant to
"will a human read this as a probability of Y?" and it is *not* identical to top-label ECE. The UI will
say which one it is, with the bin count. Equal-width bins on [0,1], default 10, empty bins contribute 0.

### Bootstrap — the part I want to get right

Requirement: ≥1,000 resamples, seeded, vectorized, fast enough for Community Cloud's ~690 MB throttle.

**Key insight.** Every confusion-matrix metric above is a function *only* of the four cell counts
(TP, FP, FN, TN). Resampling `n_g` rows with replacement within group *g* therefore induces exactly a
**multinomial** draw over those four cells: `cells ~ Multinomial(n_g, (TP,FP,FN,TN)/n_g)`. So the
entire bootstrap for all rate metrics and all gaps is `rng.multinomial(n_g, p_g, size=B)` per group —
cost `O(B · 4 · G)`, **independent of dataset size**, no `(B, n)` index matrix, no memory blow-up.
This is exact, not an approximation, and it is testable: I will assert it matches a naive row-resampling
bootstrap in distribution (same seed family, KS / mean-and-quantile agreement).

- Default scheme: **stratified** (group sizes held at their observed values) — the right conditioning for per-group rate CIs.
- Score-based metrics (AUC, ECE) are *not* functions of four counts, so they use genuine row-level index resampling per group, with a lower default `B` and a row cap.
- Intervals are **percentile** CIs at [2.5%, 50%, 97.5%]. Honest caveat in the UI: percentile bootstrap under-covers for very small or zero-count cells, which is exactly why cells with n < 30 are flagged.
- **Revision (M2), important.** A max−min gap is non-negative by construction, so its CI essentially never contains zero and "CI excludes 0" is *not* a valid no-gap test. Verdicts therefore come from **signed contrasts against a reference group** (default: the largest group), which can legitimately straddle zero. Both are reported; only contrasts drive wording. On the spec's own hand-computed case the equal-opportunity max−min CI is [0.016, 0.441] (excludes 0) while the signed TPR contrast is −0.200 [−0.441, +0.056] (includes 0). See DECISIONS.md D11.
- Sanity test: for a single proportion, the bootstrap CI must land close to a Clopper–Pearson/Wilson interval — a cheap, strong check that the machinery is right.
- Seeding: `numpy.random.default_rng(seed)`, seed surfaced in the UI and written into the exported report so any number in the report is reproducible.

### Plain-language summary rules

Deterministic templates, no LLM, no invented numbers:
- gap CI excludes 0 → "Group X's TPR is N points lower than Group Y's (95% CI …)."
- gap CI includes 0 → "**no clear evidence of a gap** in <metric> (CI includes zero)."
- any cell n < 30 → "estimates for Group Z rest on only n rows and are unstable."
- base rates differ → pointer to the impossibility section.

---

## 5. Repository structure

```
faircheck-dashboard/
├─ faircheck/                 # pure core: no streamlit import anywhere
│  ├─ __init__.py
│  ├─ __main__.py             # CLI entry: python -m faircheck report ...
│  ├─ validate.py             # column mapping, dtype/label/score checks -> typed errors
│  ├─ metrics.py              # confusion cells, per-group rates, gaps (NaN-aware)
│  ├─ bootstrap.py            # vectorized multinomial CIs (+ row-level for score metrics)
│  ├─ calibration.py          # ECE, reliability bins, per-group ROC-AUC, threshold sweep
│  ├─ impossibility.py        # Chouldechova identity, base-rate comparison
│  ├─ guidance.py             # metric-selection helper + plain-language summary templates
│  ├─ report.py               # Markdown report builder (single source of truth)
│  └─ types.py                # dataclasses: ColumnMapping, GroupResult, AuditResult
├─ app.py                     # Streamlit UI — calls faircheck only, zero metric logic
├─ examples/
│  ├─ make_acs_income.py      # seeded; hits network once, offline thereafter
│  ├─ make_compas_like.py     # seeded synthetic: calibrated, unequal base rates
│  ├─ make_near_fair.py       # seeded synthetic: equal base rates, near-zero gaps
│  └─ data/*.csv              # committed outputs
├─ tests/                     # pytest; fully offline
├─ .github/workflows/ci.yml
├─ requirements.txt           # app runtime, pinned ==
├─ requirements-dev.txt       # + fairlearn (oracle), pytest, ruff, mypy
├─ README.md  LICENSE (MIT)  CITATION.cff  .gitignore
└─ PLAN.md
```

Enforced boundary: a test greps `faircheck/` for `streamlit` and fails if found, and the CLI is
exercised in CI — so "no metric logic in the UI" is verified, not just promised.

---

## 6. UI wireframe

**Persistent sidebar** (visible on every tab)

```
FairCheck
─────────────────────────────
[ Upload CSV ]  or  Example ▾
   · ACS income (real, logistic regression)
   · COMPAS-like (synthetic, calibrated)
   · Near-fair (synthetic)
─────────────────────────────
COLUMN MAPPING
  True label        [ y_true   ▾ ]
  Predicted label   [ y_pred   ▾ ]
  Score (optional)  [ score    ▾ ]
  Sensitive attr.   [ sex      ▾ ]  (+ add a 2nd for intersectional)
  Positive outcome  ( ) 0   (•) 1
─────────────────────────────
Bootstrap resamples [1000]  Seed [0]
🔒 Your file is processed in memory
   only. Never stored, never logged.
[ Download report (.md) ]
```

**Header strip** — n rows · G groups · positive outcome = `>50K` · seed 0 · ⚠ 2 cells with n < 30

**Tab 1 · Groups** — per-group table, one row per group, each cell `value [low, high]`, `n/a` where
undefined with a footnote; bar chart of selection rate with CI whiskers, Okabe–Ito colour **plus**
distinct hatch/marker and direct labels so colour is never load-bearing.

**Tab 2 · Gaps** — one card per gap metric: name, value with CI, sparkline of the per-group values it
came from, plain-English "what this means", and a verdict chip (`clear gap` / `no clear evidence`).
Disparate-impact card carries the four-fifths flag and the EEOC "screening heuristic, not a legal
finding" sentence.

**Tab 3 · Scores** *(only when a score column is mapped; otherwise an explainer of what's missing)* —
per-group ROC-AUC (single-class groups shown as skipped, with reason), reliability diagram per group
with the diagonal, ECE per group with bin count stated, and a threshold slider with a live TPR/FPR
table plus a note that moving the threshold trades the two off and cannot equalise both.

**Tab 4 · Intersectional** *(only when 2 sensitive columns are mapped)* — cell grid, small-n cells
visibly marked and greyed, CIs everywhere, and a standing warning that intersectional cells are small
by construction.

**Tab 5 · Which metric fits?** — three questions (assistive vs punitive · are labels trustworthy? ·
will humans read scores as probabilities?) → highlights a metric family with a one-line reason and a
link to the relevant tab. No score, no "fairness grade."

**Tab 6 · Why not all at once** — appears with real force when base rates differ: shows the two base
rates from the user's data, substitutes them into Chouldechova's identity, and states which of
{equal PPV, equal FPR, equal FNR} must break. Cites Chouldechova 2017 and Kleinberg et al. 2016.

**Footer of every tab** — "Screening tool. Not a legal or causal conclusion."

---

## 7. Testing plan (offline, no network)

| # | Test | Why |
|---|---|---|
| 1 | **Hand-computed case** — your A/B cells → all 7 rates + DI 0.40 + EOdds 0.20, exact | Already confirmed against Fairlearn (§2.3a), so this pins intent, not just self-consistency |
| 2 | **Fairlearn oracle**, random data × multiple seeds × varying group counts | Independent implementation agreement. Restricted to groups where all rates are defined, because of §2.3(b) — with a comment explaining the deliberate divergence |
| 3 | **Deliberate divergence test** — group with no positives: FairCheck `NaN`, Fairlearn `0.0` | Locks in the behaviour your spec requires and documents *why* we diverge |
| 4 | Edge cases: single group · empty confusion cells · flipped positive label · group with no negatives · all-one-class column | Where fairness tools usually break |
| 5 | **Chouldechova identity** holds to 1e-12 on generated data | Validates rates jointly, not one at a time |
| 6 | Bootstrap: same seed → identical output; different seed → different; multinomial fast path ≡ naive row-resampling; single-proportion CI ≈ Clopper–Pearson | The performance trick must be provably equivalent |
| 7 | ECE: perfectly calibrated synthetic → ECE ≈ 0; deliberately shifted scores → ECE grows monotonically | Direction, not just a number |
| 8 | Report builder: valid Markdown, parses, contains every section, no `nan` leaking into prose, numbers match the computed objects | The export is a deliverable |
| 9 | Summary wording: CI-includes-zero case must produce hedged text | The honesty requirement, as an assertion |
| 10 | Validation: each bad input (missing values, non-binary labels, score outside [0,1], unmapped column, label with 1 unique value) raises a typed error with an actionable message | Error messages are a feature |
| 11 | CLI smoke test on a bundled example; UI-purity grep | Contract enforcement |

CI: GitHub Actions, `ubuntu-latest`, Python **3.12 / 3.13**, `pip install -r requirements-dev.txt`,
`ruff` + `ruff format --check` + `mypy` + `pytest`. No network at test time — the dataset CSVs are
committed.

> **Revision (M1).** The plan said 3.11–3.13; `numpy` 2.5.x requires Python ≥ 3.12, so the matrix is
> 3.12 (the Streamlit Community Cloud default) and 3.13. Supporting 3.11 would mean unpinning numpy,
> which costs more than the extra version is worth.

---

## 8. Deployment plan — **deferred at the user's request (2026-09-21)**

Hosting and the GitHub remote are out of scope for now (DECISIONS.md D10). The research below
stands for when this is picked up; nothing in the build depends on it, because the app is a thin
caller of a library that runs anywhere.


1. GitHub public repo, MIT, CITATION.cff.
2. **Streamlit Community Cloud** (free) from `main`, entrypoint `app.py`, `requirements.txt` at repo root, Python **3.12** selected in *Advanced settings* (not `runtime.txt` — it is ignored).
3. Guard the memory ceiling: soft warning + seeded-subsample offer above ~200k rows, hard cap ~1M rows, `max_upload_size` set explicitly rather than relying on the 200 MB default.
4. `st.cache_data` on parsing and on the audit computation, keyed by mapping + seed + B.
5. Smoke-test the live URL with all three bundled examples and one hand-made CSV before the link goes in the README.
6. README written **last**, with numbers pasted from an actual CLI run on the bundled COMPAS-like example and a real screenshot.

---

## 9. Risks and fallbacks

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| Community Cloud memory throttle (~690 MB) on a big upload | Medium | Row cap + subsample + cached parsing; multinomial bootstrap keeps CI cost O(B·4G) regardless of n |
| Cloud's Python default drifts to 3.13 and a wheel is missing | Low (stack verified on 3.13 — §2.1) | CI covers 3.11–3.13; pin in `requirements.txt`; version selectable in Advanced settings |
| Community Cloud policy changes / app sleeps | Low-Medium | Fallbacks in order: (a) another free PaaS from Streamlit's deployment wiki, (b) HF **Docker** Space if you have or get PRO, (c) HF **Gradio + ZeroGPU** Space — free for 2 Spaces, with a no-op GPU-decorated function and the real work outside it, so no quota is consumed; requires a thin Gradio front-end, which is cheap *because* the core library has no UI code. (d) Always-valid: `pip install -e . && streamlit run app.py`, plus the CLI |
| Fairlearn 0.15 changes the oracle API | Low | `fairlearn==0.14.0` pinned in dev only; it is not a runtime dependency |
| pandas 3.x behaviour change breaks parsing | Low (verified) | Pinned versions + CI |
| Bootstrap CIs misused as significance tests | Medium | Explicit wording: "no clear evidence of a gap", never "fair"; no overall score or grade |
| Four-fifths flag read as a legal finding | Medium | EEOC's own "rule of thumb, not a legal definition" language quoted, plus the small-sample caveat from the regulation |
| Scope creep → nothing deployed | Medium | Milestones are ordered so a *shippable* app exists at M6; M7 is polish |

---

## 10. Milestones

- [x] **M1 — Skeleton + validation + metrics.** Repo, packaging, pins, `.gitignore`, MIT, CI workflow. `validate.py`, `metrics.py`, `types.py`. Tests 1, 3, 4, 5, 10 passing. *Delivered 2026-09-21: 79 tests, 97% coverage, ruff + mypy clean, hand-computed table printed from FairCheck's own code.*
- [x] **M2 — Bootstrap + Fairlearn oracle.** `bootstrap.py` with the multinomial fast path; tests 2 and 6 passing, including fast-path ≡ naive equivalence and a timing number. *Delivered 2026-09-21: 154 tests, 98% coverage, per-group table with 95% CIs; B=1000 takes 2.5 ms at n=100k and 2.7 ms at n=10M. Added signed reference contrasts (D11) after finding that max−min gap CIs cannot support evidence claims.*
- [x] **M3 — Scores and calibration.** `calibration.py`: per-group ROC-AUC (graceful skip), reliability bins, ECE, threshold sweep. `impossibility.py`. Tests 5, 7 passing. *Delivered 2026-09-21: 190 tests, 98% coverage. On a seeded calibrated synthetic (Y ~ Bernoulli(score), different Beta scores by group): ECE 0.019 / 0.020, AUC 0.716 / 0.730; Chouldechova identity holds to 1e-16; equal-PPV counterfactuals that require FPR > 1 are flagged infeasible rather than printed as rates (D12).*
- [x] **M4 — Report + CLI.** `report.py`, `guidance.py`, `__main__.py`. Tests 8, 9, 11 passing. *Delivered 2026-09-21: 228 tests, 98% coverage. Generated report on the hand-computed case: selection-rate contrast 30 pp (CI −42 to −17) is a clear difference; equal-opportunity max−min 0.200 [0.016, 0.441] excludes 0 but the signed TPR contrast includes 0, so the summary hedges (D11). PDF via fpdf2 2.8.8 (D6).*
- [x] **M5 — Bundled examples (four).** Seeded generator scripts + committed CSVs for ACS income, UCI Adult, COMPAS-like and near-fair; verify the COMPAS-like example really is calibrated with unequal error rates and the near-fair one really has CIs covering zero. *Delivered 2026-09-21: 239 tests, 98% coverage. COMPAS-like (n=8000): ECE 0.015/0.017, base rates 0.28 vs 0.72, clear TPR and FPR contrasts. Near-fair (n=10000): every signed contrast CI covers 0. ACS Income (WA hold-out, n=8000) and UCI Adult (n=8000) ship as prediction CSVs; generators hit OpenML once, tests never do.*
- [x] **M6 — Streamlit app.** All six tabs, Okabe–Ito palette with non-colour encoding, privacy notice, Markdown + PDF download, caching, row caps. *Delivered 2026-09-21: 243 tests passing (1 skip), 98% coverage. `app.py` is a thin caller of `run_audit`. Local run on COMPAS-like (n=8000) shows ECE 0.015/0.017, clear TPR/FPR contrasts, four-fifths screen raised with the EEOC heuristic wording, and Markdown+PDF download. ACS Income with sex×race opens the Intersectional tab (16 groups, 4 with n<30 faded). Screenshots in `scratch/m6/`.*
- [x] **M7 — README + citation.** README written from real runs with real numbers and a real screenshot; `CITATION.cff`; final read-through for over-claiming. Deployment is deferred (§8), so this milestone ships the repo, not a URL. *Delivered 2026-09-21: 247 tests passing (1 skip). README quotes the seeded CLI report on COMPAS-like (`examples/reports/compas_like.md`); tests fail if those numbers drift. Screenshot from a local Streamlit run. No live-demo URL. ORCID omitted until confirmed.*

Each milestone ends with: run the tests, show you real output, tick the box here, **stop for your go-ahead.**

---

## 11. Questions — answered 2026-09-21

| # | Question | Answer |
|---|---|---|
| 1 | Multi-class scope | Follow the proposal: binary core, one-vs-rest via the positive-label picker (D5) |
| 2 | Fairlearn as oracle vs runtime dependency | Oracle only (D2) |
| 3 | Real-data example | **Bundle both** ACSIncome and Adult (D9) |
| 4 | Deployment / GitHub repo | Deferred, do not plan for it yet (D10) |
| 5 | PDF export | **Attempt it**, alongside Markdown (D6) |
| 6 | Overall fairness score | No headline grade, but the explanation must be thorough enough that a professor understands the trade-offs properly (D7) |
| 7 | CV one-liner | Approved as written in §1 |

One item still needs confirmation: the `LICENSE` and `pyproject.toml` currently say
**"Rafid Mehda"**. Confirm the exact name (and ORCID, for `CITATION.cff` in M7).

<details>
<summary>Original wording of the questions</summary>

1. **Multiclass.** Your goal line says "binary/multi-class" but the input spec says validate for
   "non-binary labels" and the README limitations say "binary only." My proposal: build binary-only
   internally, but let the user pick *one* label value as the positive outcome so a multiclass column
   is audited one-vs-rest — which covers the multiclass case honestly and keeps every formula above
   exactly correct. Accept, or do you want true multiclass (per-class confusion matrices)?
2. **Fairlearn as oracle vs. runtime dependency.** Because of finding §2.3(b) I want `faircheck` to
   depend only on numpy/pandas and use Fairlearn as a *test oracle*. This gives correct `n/a` handling,
   a lighter deploy, and a stronger story ("independent implementation, cross-validated"). The cost:
   Fairlearn isn't in the running app. OK?
3. **Real-data example: ACSIncome (my recommendation) or Adult?** ACSIncome is the literature's
   replacement for Adult and cites Ding et al. 2021; Adult is more instantly recognisable to a reader.
   I can also bundle both.
4. **Deployment account.** Streamlit Community Cloud needs a GitHub account connected. Confirm the
   GitHub username/org to use, and tell me if you already have HF PRO (it would reopen the Docker-Space
   option). Also: should I create the GitHub repo, or will you?
5. **PDF export.** I recommend Markdown only, plus print-friendly CSS. Confirm, or should I attempt PDF?
6. **Overall fairness score.** I deliberately plan *not* to show one, since it contradicts the
   impossibility results the app teaches. Confirm you're happy with no headline grade.
7. **CV line.** Does the one-liner in §1 match how you want to describe this on applications?

</details>
