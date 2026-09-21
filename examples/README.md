# Example CSVs for FairCheck

These files are the committed, offline inputs for tests and the app. Tests never
download data. Regenerating ACS Income or Adult hits OpenML once (Fairlearn cache
thereafter); the two synthetic examples are pure NumPy.

```bash
.venv/bin/python -m examples.make_compas_like
.venv/bin/python -m examples.make_near_fair
.venv/bin/python -m examples.make_acs_income   # network on first run
.venv/bin/python -m examples.make_adult        # network on first run
```

| File | What it is | Sensitive columns | Provenance |
|---|---|---|---|
| `compas_like.csv` | Synthetic. Calibrated scores, unequal base rates, unequal error rates. **Not real COMPAS records.** | `group` | this repo, MIT |
| `near_fair.csv` | Synthetic. Same DGP in both groups; signed contrasts cover zero. | `group` | this repo, MIT |
| `acs_income.csv` | ACS PUMS Income, Washington state, hold-out logistic predictions. `y_true = 1` means PINCP ≥ $50,000. | `sex`, `race` | [Ding et al. 2021](https://arxiv.org/abs/2108.04884); [US Census terms](https://www.census.gov/data/developers/about/terms-of-service.html) |
| `adult.csv` | UCI Adult, hold-out logistic predictions. `y_true = 1` means income > $50,000. Prefer ACS Income (D9). | `sex`, `race` | Becker & Kohavi, DOI [10.24432/C5XW20](https://doi.org/10.24432/C5XW20) |

Shared columns: `y_true`, `y_pred`, `score` in `[0, 1]`. Predictions are `score >= 0.5`.

A seeded CLI report on COMPAS-like (`n_boot=1000`, seed 0) is committed at
[`reports/compas_like.md`](reports/compas_like.md). The README quotes it; tests
assert the file still matches `python -m faircheck report`.
