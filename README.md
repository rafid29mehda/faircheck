# FairCheck

Correct, uncertainty-aware fairness reports for classification predictions.

FairCheck is a **web app**, a **CLI**, and a **pure-Python library**. You give it a CSV of
labels, predictions, an optional score, and one or two sensitive attributes. It returns
per-group rates with bootstrap intervals, signed disparity verdicts, per-group
calibration, and an impossibility explainer computed from *this file's* base rates.

It is a **screening tool, not a legal or causal conclusion**, and it does not assign a
fairness grade. Different criteria can disagree; that disagreement is the point
(Hardt, Price and Srebro 2016; Chouldechova 2017; Kleinberg, Mullainathan and
Raghavan 2016).

```text
FairCheck - open-source fairness auditing tool (Python/Streamlit): disaggregated
group metrics with bootstrap confidence intervals, per-group calibration analysis,
and a metric-selection guide grounded in the Hardt et al. / Chouldechova
impossibility results. CLI, pytest suite, CI.
```

Source: https://github.com/rafid29mehda/faircheck

Run it locally (below). There is no hosted demo yet.

![FairCheck Groups tab on the bundled COMPAS-like example: two groups, selection-rate
bars with interval whiskers, and a hedged plain-language summary.](docs/images/compas_like_groups.png)

The screenshot is a real local run of `streamlit run app.py` on
`examples/data/compas_like.csv` (n = 8,000, 1,000 bootstrap resamples, seed 0).
The numbers in this README are copied from that same audit, written by
`python -m faircheck report` to [`examples/reports/compas_like.md`](examples/reports/compas_like.md).

## What it will not do

- Print `0.0` for an undefined rate. A group with no positives has TPR `n/a`, not
  "none found." Fairlearn's `true_positive_rate` returns `0.0` here; FairCheck
  deliberately differs.
- Treat a max-min gap whose CI excludes zero as evidence. Max-min is non-negative
  by construction. Verdicts use **signed contrasts against a reference group**
  (largest group by default).
- Call the four-fifths (80%) rule a legal finding. It is a screening heuristic.
- Score the model. The metric-selection helper highlights a *family* from three
  questions about the decision; it is not a grade.

## Install and run

Python **3.12 or 3.13**. The pinned stack does not support 3.11 (`numpy` 2.5.x).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps

# App
.venv/bin/streamlit run app.py

# CLI - same numbers as the app, same Markdown as the download button
.venv/bin/python -m faircheck report examples/data/compas_like.csv \
  --label y_true --pred y_pred --group group --score score --positive 1 \
  --seed 0 --n-boot 1000 \
  --out report.md --pdf report.pdf
```

CSV columns needed: a true label, a predicted label, one sensitive attribute
(repeat `--group` for intersectional), and optionally a score in `[0, 1]`. The
favorable outcome is a value you choose (`--positive`), not a hard-coded `1`.
A column with more than two labels is audited **one-vs-rest** against that value.

The core library depends only on NumPy and pandas. Streamlit and fpdf2 are for
the app and PDF export. Fairlearn is a **test oracle**, not a runtime dependency.

```python
import pandas as pd

from faircheck.report import render_markdown, run_audit
from faircheck.types import ColumnMapping

frame = pd.read_csv("examples/data/compas_like.csv")
audit = run_audit(
    frame,
    ColumnMapping(
        label="y_true",
        prediction="y_pred",
        sensitive=("group",),
        positive_label=1,
        score="score",
    ),
    n_boot=1000,
    seed=0,
)
print("\n".join(audit.summary))
print(render_markdown(audit))
```

Development extras (Fairlearn oracle, pytest, ruff, mypy):

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/python -m pytest -q
```

CI (`.github/workflows/ci.yml`) runs ruff, mypy and pytest on Python 3.12 and 3.13.
Tests are offline: the example CSVs are committed.

## A real run (COMPAS-like, seed 0)

This file is **synthetic**. It is not real COMPAS records. `Y ~ Bernoulli(score)`
with Beta(2, 5) vs Beta(5, 2) scores, shared threshold 0.5, n = 4,000 per group.

Command (the committed report is the stdout of this, saved to
`examples/reports/compas_like.md`):

```bash
.venv/bin/python -m faircheck report examples/data/compas_like.csv \
  --label y_true --pred y_pred --group group --score score --positive 1 \
  --seed 0 --n-boot 1000
```

Plain-language summary (verbatim):

- B's Selection rate is 79 percentage points higher than A's (95% CI 77 to 80 pp).
- B's TPR (recall) is 73 percentage points higher than A's (95% CI 70 to 75 pp).
- B's FPR is 72 percentage points higher than A's (95% CI 70 to 75 pp).
- B's Precision (PPV) is 21 percentage points higher than A's (95% CI 16 to 25 pp).
- Base rates differ (A 28% vs B 72%), so calibration/predictive parity and equal error rates cannot all hold for an imperfect classifier -- see 'Why not all at once'.

| group | n | Base rate | Selection rate | TPR | FPR | ECE (10 bins) | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 4000 | 0.280 [0.267, 0.294] | 0.108 [0.099, 0.118] | 0.211 [0.186, 0.233] | 0.068 [0.060, 0.077] | 0.015 | 0.701 |
| B | 4000 | 0.719 [0.705, 0.733] | 0.896 [0.886, 0.906] | 0.937 [0.928, 0.946] | 0.790 [0.767, 0.814] | 0.017 | 0.707 |

Disparate impact ratio **0.121 [0.111, 0.132]**. The four-fifths screen is raised
as a heuristic, not a finding. ECE is the **positive-class** expected calibration
error (mean score vs observed positive rate, 10 equal-width bins), not Guo et al.'s
top-label ECE.

Equalising B's PPV to A's is **impossible** without also moving error rates: the
Chouldechova identity would require an FPR outside [0, 1].

The near-fair companion (`examples/data/near_fair.csv`) uses the same DGP in both
groups. Every signed contrast CI covers 0; the max-min demographic-parity gap still
sits away from zero. That pair is why verdicts never use max-min.

## Bundled examples

| File | What it is |
| --- | --- |
| `examples/data/compas_like.csv` | Synthetic. Calibrated, unequal base rates, unequal error rates. **Not real COMPAS.** |
| `examples/data/near_fair.csv` | Synthetic. Same process in both groups; signed contrasts cover 0. |
| `examples/data/acs_income.csv` | ACS PUMS Income, Washington, hold-out logistic. Prefer this as the real-data example ([Ding et al. 2021](https://arxiv.org/abs/2108.04884)). |
| `examples/data/adult.csv` | UCI Adult, hold-out logistic. More familiar; Adult has documented idiosyncrasies that ACS Income was built to avoid. |

Provenance and regeneration: [`examples/README.md`](examples/README.md).

## Limitations

- **Classification only**, binary internally. Multi-class columns are one-vs-rest
  against the chosen positive label, with a warning.
- Percentile bootstrap **under-covers** very small or zero-count cells. Groups with
  n < 30 are flagged; do not over-read a quiet cell.
- Score metrics (ROC-AUC, ECE) use row-level resampling, not the multinomial
  fast path. A group with only one class has ROC-AUC `n/a`, not 0.5.
- Intersectional cells are small by construction.
- No mitigation, no causal claims, no hosted demo yet.

Screening tool. Not a legal or causal conclusion.

Design choices are numbered in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Cite

See [`CITATION.cff`](CITATION.cff).

```bibtex
@software{mehda_faircheck_2026,
  author  = {Mehda, Rafid},
  title   = {FairCheck},
  year    = {2026},
  version = {0.1.0},
  license = {MIT},
  note    = {Uncertainty-aware fairness reports for classification predictions},
}
```

## License

[MIT](LICENSE) © 2026 Rafid Mehda.
