# FairCheck report

Screening tool. Not a legal or causal conclusion. Uploaded data is processed in memory only and is never stored or logged.

- Rows audited: **8,000** of 8,000 in the file
- Groups: **2** (A, B)
- Sensitive columns: group
- Positive (favorable) outcome: `1`
- Bootstrap: 1,000 stratified resamples, seed 0, reference group **A**

## Plain-language summary

- B's Selection rate is 79 percentage points higher than A's (95% CI 77 to 80 pp).
- B's TPR (recall) is 73 percentage points higher than A's (95% CI 70 to 75 pp).
- B's FPR is 72 percentage points higher than A's (95% CI 70 to 75 pp).
- B's Precision (PPV) is 21 percentage points higher than A's (95% CI 16 to 25 pp).
- Base rates differ (A 28% vs B 72%), so calibration/predictive parity and equal error rates cannot all hold for an imperfect classifier -- see 'Why not all at once'.

## Per-group rates

Each cell is the point estimate and a 95% percentile bootstrap interval. `n/a` means the
rate's denominator was zero (for example TPR when a group has no positives).

| group | n | Base rate | Selection rate | TPR (recall) | FPR | FNR | Precision (PPV) | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 4000 | 0.280 [0.267, 0.294] | 0.108 [0.099, 0.118] | 0.211 [0.186, 0.233] | 0.068 [0.060, 0.077] | 0.789 [0.767, 0.814] | 0.546 [0.501, 0.591] | 0.730 [0.716, 0.744] |
| B | 4000 | 0.719 [0.705, 0.733] | 0.896 [0.886, 0.906] | 0.937 [0.928, 0.946] | 0.790 [0.767, 0.814] | 0.063 [0.054, 0.072] | 0.752 [0.738, 0.766] | 0.733 [0.719, 0.746] |

Accuracy is reported for completeness; it is misleading under class imbalance.

## Summary gaps

Max−min gaps are the literature's headline numbers (and match Fairlearn). They are
non-negative by construction, so a CI that excludes zero is **not** evidence of a
disparity -- sampling noise alone pushes the maximum above the minimum. The **Verdict**
column uses signed contrasts against the reference group (A), which
can contain zero. See docs/DECISIONS.md D11.

| Gap | Value [95% CI] | Per-group | Verdict | Reference |
| --- | --- | --- | --- | --- |
| Demographic parity difference | 0.787 [0.774, 0.801] | Selection rate: A 0.108, B 0.896 | clear difference vs reference | Hardt et al. 2016 (arXiv:1610.02413), for contrast |
| Disparate impact ratio | 0.121 [0.111, 0.132] | Selection rate: A 0.108, B 0.896 | clear difference vs reference | 29 CFR 1607.4(D) (four-fifths rule of thumb) |
| Equal opportunity difference | 0.726 [0.702, 0.752] | TPR (recall): A 0.211, B 0.937 | clear difference vs reference | Hardt et al. 2016, Definition 2.2 |
| Equalized odds difference | 0.726 [0.712, 0.754] | TPR (recall): A 0.211, B 0.937; FPR: A 0.068, B 0.790 | clear difference vs reference | Hardt et al. 2016, Definition 2.1 |
| Predictive parity difference | 0.206 [0.161, 0.254] | Precision (PPV): A 0.546, B 0.752 | clear difference vs reference | Chouldechova 2017 (arXiv:1610.07524) |

Disparate impact ratio 0.121 is below 0.80, so the four-fifths screen is raised. The four-fifths (80%) rule is a screening heuristic that US federal enforcement agencies use as a rule of thumb, not a legal finding. Smaller gaps can still matter, and larger gaps may not when they rest on small numbers (29 CFR 1607.4(D); EEOC Q&A).

## Which metric fits this decision?

The helper asks three questions and highlights a family. It does not score the model.

1. Is the decision *assistive* (granting a benefit) or *punitive* (imposing a penalty)?
2. Are the labels trustworthy enough that error rates (which condition on Y) are meaningful?
3. Will a human read the score as a probability of the favorable outcome?

**Highlighted:** (none yet)

**Answer the three questions to highlight a metric family.** Until then every metric is shown. None of them is a fairness grade -- different criteria can disagree, and that disagreement is the point.

## Scores

ECE here is the **positive-class** expected calibration error (mean score vs. observed
positive rate in equal-width bins on [0, 1]), not Guo et al.'s top-label ECE. Empty bins
contribute 0. The bin count is part of the number; treat ECE as descriptive.

| Group | n | ROC-AUC | ECE (10 bins) |
| --- | --- | --- | --- |
| A | 4000 | 0.701 | 0.015 |
| B | 4000 | 0.707 | 0.017 |

Moving the decision threshold trades TPR against FPR and cannot equalise both when base
rates differ. Use the CLI/UI slider for the sweep; the export records ECE and AUC only.

## Why not all at once

Chouldechova (2017) identity: `FPR = p/(1-p) · (1-PPV)/PPV · (1-FNR)`. On this table the implied FPR matches the observed FPR to numerical precision.

Base rates differ by 43.9% on this file, and the classifier is not
perfect. Predictive parity (equal PPV) and equal error rates therefore cannot hold
together.

- A: FPR would be 0.068 if PPV matched A while prevalence and FNR stayed as observed (observed FPR 0.068)
- B: equalising PPV to A's is **impossible** without also moving error rates (the identity would require an FPR outside [0, 1])
- If TPR and FPR were equalised to A's, PPV would have to be A 0.546, B 0.888.

| group | Base rate | TPR | FPR | PPV |
| --- | --- | --- | --- | --- |
| A | 0.280 | 0.211 | 0.068 | 0.546 |
| B | 0.719 | 0.937 | 0.790 | 0.752 |

Sources: Chouldechova 2017 (arXiv:1610.07524); Kleinberg, Mullainathan and Raghavan 2016
(arXiv:1609.05807).

## Methods

Stratified percentile bootstrap with 1,000 resamples and seed 0.
Because every confusion-matrix rate is a function of the four cell counts, a within-group
row resample is a multinomial draw over those cells (D4). Verdicts use signed contrasts
against A, not max−min gap CIs (D11).

- Base rate: `(TP + FN) / n`
- Selection rate: `(TP + FP) / n`
- TPR (recall): `TP / (TP + FN)` (`n/a` when the group has no positives)
- FPR: `FP / (FP + TN)` (`n/a` when the group has no negatives)
- FNR: `FN / (TP + FN)` (`n/a` when the group has no positives)
- Precision (PPV): `TP / (TP + FP)` (`n/a` when the group is never selected)
- Accuracy: `(TP + TN) / n`

- Demographic parity difference: `max_a SR_a - min_a SR_a` -- Hardt et al. 2016 (arXiv:1610.02413), for contrast
- Disparate impact ratio: `min_a SR_a / max_a SR_a` -- 29 CFR 1607.4(D) (four-fifths rule of thumb)
- Equal opportunity difference: `max_a TPR_a - min_a TPR_a` -- Hardt et al. 2016, Definition 2.2
- Equalized odds difference: `max(|dTPR|, |dFPR|)` -- Hardt et al. 2016, Definition 2.1
- Predictive parity difference: `max_a PPV_a - min_a PPV_a` -- Chouldechova 2017 (arXiv:1610.07524)


---

*Screening tool. Not a legal or causal conclusion.*
